"""Soft Actor-Critic.

Maximum-entropy off-policy actor-critic: twin Q critics with target copies, a
reparameterised tanh-Gaussian actor, and a temperature that is tuned
automatically against a target entropy rather than left as a hyperparameter.

Why the pieces are there
------------------------
* **Twin critics with a min.** Single-critic Q-learning with a max-like
  bootstrap overestimates, and the overestimate compounds. Taking
  ``min(Q1, Q2)`` in the target is a cheap, strongly biased-downward estimator
  that empirically beats the alternatives.
* **Entropy in the target.** The soft Bellman backup is
  ``y = r + gamma * (1 - terminated) * (min Q'(s', a') - alpha * log pi(a'|s'))``.
  The ``alpha * log pi`` term is what makes this "soft"; drop it and this is
  just a stochastic-actor DDPG.
* **``(1 - terminated)``, never ``(1 - done)``.** A transition cut off by the
  mission time limit still has a real successor state and a real future return.
  Masking it out teaches the agent that reaching the horizon is as bad as
  losing the vehicle. The replay buffer stores the two flags separately for
  exactly this reason.
* **Automatic temperature.** ``alpha`` is optimised so the policy's average
  entropy tracks ``target_entropy`` (default ``-action_dim``). This is the part
  that depends on the tanh log-prob Jacobian correction being right: get the
  correction wrong and the measured entropy is off by a state-dependent
  constant, so the controller converges to the wrong exploration level without
  anything ever erroring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .base import Agent, TrainStats, Transition
from .buffers import ReplayBuffer
from .networks import (
    QNetwork,
    RunningMeanStd,
    SquashedGaussianPolicy,
    count_parameters,
    hard_update,
    resolve_device,
    soft_update,
    to_numpy,
    to_tensor,
)
from .ppo import _torch_load

logger = logging.getLogger(__name__)

__all__ = ["SACAgent"]


class SACAgent(Agent):
    """SAC for continuous control in ``[-1, 1]^action_dim``.

    Parameters
    ----------
    hidden:
        Hidden widths for the actor and both critics.
    lr:
        Adam learning rate, shared by actor, critics and ``log_alpha`` unless
        ``actor_lr`` / ``critic_lr`` / ``alpha_lr`` are given in ``kwargs``.
    tau:
        Polyak coefficient for the target critics.
    batch_size, buffer_size, learning_starts:
        Replay sizing. Before ``learning_starts`` env steps the agent emits
        **uniform random** actions and performs no gradient steps -- seeding the
        buffer from an untrained policy's near-deterministic output is a
        well-known way to make early SAC collapse.
    train_freq, gradient_steps:
        Do ``gradient_steps`` updates every ``train_freq`` calls to
        :meth:`update`.
    target_entropy:
        Defaults to ``-action_dim``, the standard heuristic for a bounded
        action space.
    autotune_alpha:
        When False, ``alpha`` is held at ``kwargs["alpha"]`` (default 0.2).
    device:
        Defaults to ``"cpu"``: at 36 -> 256 -> 256 -> 5 with batches of 256 the
        networks are far too small to amortise GPU/MPS launch overhead, so CPU
        is faster as well as reproducible. ``"cuda"``/``"mps"`` honoured.
    seed:
        Seeds initialisation, action sampling and replay sampling.

    Additional keyword options
    --------------------------
    ``alpha`` (initial/fixed temperature, default 0.2), ``activation``
    (default ``"relu"``), ``max_grad_norm`` (default None = off),
    ``normalize_obs`` (default False; off-policy replay predates the
    statistics, so normalisation is applied at sample time from raw stored
    observations), ``actor_lr``/``critic_lr``/``alpha_lr``.
    """

    name = "sac"
    learns = True
    uses_constraints = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_entropy: float | None = None,
        autotune_alpha: bool = True,
        device: str = "cpu",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = max(1, int(gradient_steps))
        self.autotune_alpha = bool(autotune_alpha)
        self.target_entropy = (
            float(-action_dim) if target_entropy is None else float(target_entropy)
        )
        self.device = resolve_device(device)
        self.seed = int(seed)

        self.activation = kwargs.get("activation", "relu")
        self.max_grad_norm = kwargs.get("max_grad_norm", None)
        self.normalize_obs = bool(kwargs.get("normalize_obs", False))
        self.obs_clip = float(kwargs.get("obs_clip", 10.0))
        actor_lr = float(kwargs.get("actor_lr", self.lr))
        critic_lr = float(kwargs.get("critic_lr", self.lr))
        alpha_lr = float(kwargs.get("alpha_lr", self.lr))
        alpha_init = float(kwargs.get("alpha", 0.2))

        self.config.update(
            hidden=self.hidden,
            lr=self.lr,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_entropy=self.target_entropy,
            autotune_alpha=self.autotune_alpha,
            device=str(self.device),
            seed=self.seed,
            alpha_init=alpha_init,
            normalize_obs=self.normalize_obs,
        )

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.actor = SquashedGaussianPolicy(
            obs_dim,
            action_dim,
            hidden=self.hidden,
            activation=self.activation,
            seed=self.seed,
        ).to(self.device)
        self.critic1 = QNetwork(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        self.critic2 = QNetwork(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        self.critic1_target = QNetwork(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        self.critic2_target = QNetwork(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)
        # Frozen: never optimised, and freezing keeps num_parameters honest.
        self.critic1_target.requires_grad_(False)
        self.critic2_target.requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.tensor(float(np.log(alpha_init)), device=self.device),
            requires_grad=self.autotune_alpha,
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            if self.autotune_alpha
            else None
        )

        self.buffer = ReplayBuffer(
            self.buffer_size, obs_dim, action_dim, seed=self.seed
        )
        self.obs_rms = RunningMeanStd(obs_dim) if self.normalize_obs else None

        self.total_steps = 0
        self.n_updates = 0
        self._skipped = 0

    # --- observation plumbing ------------------------------------------------
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.obs_rms is None:
            return arr
        return self.obs_rms.normalize(arr, clip=self.obs_clip)

    def _normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return x
        mean = to_tensor(self.obs_rms.mean, self.device)
        std = to_tensor(np.sqrt(self.obs_rms.var + 1e-8), self.device)
        return ((x - mean) / std).clamp(-self.obs_clip, self.obs_clip)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().item())

    # --- Agent interface -----------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.learning_starts:
            # Uniform exploration while the replay buffer fills. Deterministic
            # (evaluation) calls always use the policy, even this early, so
            # eval numbers describe the policy rather than the noise.
            return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
        obs_n = self._normalize_obs(obs)
        with torch.no_grad():
            t = to_tensor(obs_n, self.device).unsqueeze(0)
            action, _, mean_action = self.actor.sample(
                t, deterministic=deterministic, with_logprob=False
            )
        chosen = mean_action if deterministic else action
        return to_numpy(chosen.squeeze(0)).clip(-1.0, 1.0)

    def observe_transition(self, tr: Transition) -> None:
        self.buffer.add_transition(tr)
        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(tr.obs, dtype=np.float64).reshape(-1))
        self.total_steps += 1

    def reset(self) -> None:
        """SAC keeps no per-episode state; replay spans episodes by design."""

    # --- the update ----------------------------------------------------------
    def update(self) -> TrainStats:
        if self.total_steps < self.learning_starts or len(self.buffer) < self.batch_size:
            return TrainStats()
        if self.total_steps % self.train_freq != 0:
            return TrainStats()

        acc: dict[str, list[float]] = {}
        for _ in range(self.gradient_steps):
            info = self._gradient_step()
            if info is None:
                continue
            for k, v in info.items():
                acc.setdefault(k, []).append(v)
            self.n_updates += 1

        if not acc:
            return TrainStats()
        stats = {k: float(np.mean(v)) for k, v in acc.items()}
        stats.update(
            alpha=self.alpha,
            n_updates=float(self.n_updates),
            total_steps=float(self.total_steps),
            buffer_size=float(len(self.buffer)),
            skipped_updates=float(self._skipped),
        )
        return TrainStats().update(**stats)

    def _gradient_step(self) -> dict[str, float] | None:
        batch = self.buffer.sample(self.batch_size, self.device)
        obs = self._normalize_batch(batch["obs"])
        next_obs = self._normalize_batch(batch["next_obs"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        # Bootstrap mask uses terminated only -- see the module docstring.
        not_terminated = 1.0 - batch["terminated"]
        alpha = self.log_alpha.detach().exp()

        # --- critics ---------------------------------------------------------
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            target_q = torch.min(
                self.critic1_target(next_obs, next_action),
                self.critic2_target(next_obs, next_action),
            )
            target_v = target_q - alpha * next_logp
            y = rewards + self.gamma * not_terminated * target_v

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = 0.5 * (((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean())
        if not torch.isfinite(critic_loss):
            self._skipped += 1
            logger.warning("SAC: non-finite critic loss; skipping gradient step")
            return None
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if not self._clip_and_check(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.critic_optimizer,
            "critic",
        ):
            return None
        self.critic_optimizer.step()

        # --- actor -----------------------------------------------------------
        new_action, logp, _ = self.actor.sample(obs)
        q_pi = torch.min(
            self.critic1(obs, new_action), self.critic2(obs, new_action)
        )
        actor_loss = (alpha * logp - q_pi).mean()
        if not torch.isfinite(actor_loss):
            self._skipped += 1
            logger.warning("SAC: non-finite actor loss; skipping actor update")
            return None
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if not self._clip_and_check(
            list(self.actor.parameters()), self.actor_optimizer, "actor"
        ):
            return None
        self.actor_optimizer.step()

        # --- temperature -----------------------------------------------------
        alpha_loss_val = 0.0
        if self.autotune_alpha and self.alpha_optimizer is not None:
            # d/d log_alpha of -alpha * (logp + H_target); pushes alpha up when
            # the policy is less random than the target entropy asks for.
            alpha_loss = -(
                self.log_alpha.exp() * (logp.detach() + self.target_entropy)
            ).mean()
            if torch.isfinite(alpha_loss):
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_val = float(alpha_loss.item())
            else:
                self._skipped += 1
                logger.warning("SAC: non-finite alpha loss; skipping temperature step")

        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "entropy": float(-logp.detach().mean().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(y.mean().item()),
        }

    def _clip_and_check(
        self,
        params: list[nn.Parameter],
        optimizer: torch.optim.Optimizer,
        what: str,
    ) -> bool:
        """Optional grad clipping plus a hard non-finite gradient guard."""
        if self.max_grad_norm is not None:
            ok = bool(
                torch.isfinite(
                    torch.nn.utils.clip_grad_norm_(params, float(self.max_grad_norm))
                )
            )
        else:
            ok = all(
                bool(torch.isfinite(p.grad).all())
                for p in params
                if p.grad is not None
            )
        if not ok:
            self._skipped += 1
            logger.warning("SAC: non-finite %s gradient; skipping step", what)
            optimizer.zero_grad(set_to_none=True)
            return False
        return True

    # --- bookkeeping ---------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Reseed action sampling, replay sampling and torch.

        Weights are initialised in ``__init__`` from the constructor seed;
        calling this afterwards does not re-initialise them.
        """
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)
        self.actor.set_seed(self.seed)
        self.buffer.set_seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def num_parameters(self) -> int:
        base = count_parameters([self.actor, self.critic1, self.critic2])
        return base + (1 if self.autotune_alpha else 0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "config": dict(self.config),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": (
                None
                if self.alpha_optimizer is None
                else self.alpha_optimizer.state_dict()
            ),
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "noise": self.actor.noise.get_state(),
            "rng": self._rng.bit_generator.state,
            "total_steps": self.total_steps,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(
                torch.as_tensor(state["log_alpha"], device=self.device)
            )
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state.get("alpha_optimizer") is not None and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        if state.get("obs_rms") is not None and self.obs_rms is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state.get("noise") is not None:
            self.actor.noise.set_state(state["noise"])
        if state.get("rng") is not None:
            self._rng.bit_generator.state = state["rng"]
        self.total_steps = int(state.get("total_steps", 0))
        self.n_updates = int(state.get("n_updates", 0))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(_torch_load(path, self.device))
