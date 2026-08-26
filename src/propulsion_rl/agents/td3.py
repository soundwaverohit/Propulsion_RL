"""Twin Delayed Deep Deterministic policy gradient.

TD3 is DDPG plus three specific fixes, and the benchmark only means something
if all three are actually present:

1. **Clipped double-Q.** Two independent critics, and the Bellman target uses
   ``min(Q1', Q2')``. DDPG's single critic bootstraps off its own maximum and
   drifts upward without bound.
2. **Delayed policy updates.** The actor (and the target networks) update once
   every ``policy_delay`` critic updates, so the policy chases a critic that has
   had time to settle instead of amplifying its error.
3. **Target policy smoothing.** Clipped Gaussian noise is added to the target
   action, which regularises the target across a small neighbourhood and stops
   the actor from exploiting sharp, spurious peaks in the critic.

As in SAC, the bootstrap mask is ``(1 - terminated)`` and never ``(1 - done)``:
every mission here ends on a time limit, and masking truncated transitions
would train the critic to believe the horizon is a catastrophe.
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
    DeterministicPolicy,
    QNetwork,
    RunningMeanStd,
    count_parameters,
    hard_update,
    resolve_device,
    soft_update,
    to_numpy,
    to_tensor,
)
from .ppo import _torch_load

logger = logging.getLogger(__name__)

__all__ = ["TD3Agent"]


class TD3Agent(Agent):
    """TD3 for continuous control in ``[-1, 1]^action_dim``.

    Parameters
    ----------
    hidden, lr, gamma, tau:
        Network widths, Adam learning rate, discount, Polyak coefficient.
    batch_size, buffer_size, learning_starts:
        Replay sizing. Before ``learning_starts`` env steps the agent emits
        uniform random actions and takes no gradient steps.
    policy_delay:
        Actor and target updates happen every ``policy_delay`` critic updates.
    target_noise, noise_clip:
        Target policy smoothing: ``a' = clip(pi'(s') + clip(N(0, target_noise),
        -noise_clip, noise_clip), -1, 1)``.
    exploration_noise:
        Std of the Gaussian noise added to the actor output during rollout.
        Ignored when ``deterministic=True``, so evaluation is noise-free.
    device:
        Defaults to ``"cpu"``. The networks here (36 -> 256 -> 256 -> 5) are
        well below the size where GPU/MPS launch overhead pays for itself, so
        CPU is faster and bitwise reproducible. ``"cuda"``/``"mps"`` honoured.
    seed:
        Seeds initialisation, exploration noise, smoothing noise and replay
        sampling.

    Additional keyword options
    --------------------------
    ``activation`` (default ``"relu"``), ``train_freq`` (default 1),
    ``gradient_steps`` (default 1), ``max_grad_norm`` (default None = off),
    ``normalize_obs`` (default False), ``actor_lr``/``critic_lr``.
    """

    name = "td3"
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
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
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
        self.policy_delay = max(1, int(policy_delay))
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
        self.exploration_noise = float(exploration_noise)
        self.device = resolve_device(device)
        self.seed = int(seed)

        self.activation = kwargs.get("activation", "relu")
        self.train_freq = max(1, int(kwargs.get("train_freq", 1)))
        self.gradient_steps = max(1, int(kwargs.get("gradient_steps", 1)))
        self.max_grad_norm = kwargs.get("max_grad_norm", None)
        self.normalize_obs = bool(kwargs.get("normalize_obs", False))
        self.obs_clip = float(kwargs.get("obs_clip", 10.0))
        actor_lr = float(kwargs.get("actor_lr", self.lr))
        critic_lr = float(kwargs.get("critic_lr", self.lr))

        self.config.update(
            hidden=self.hidden,
            lr=self.lr,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            policy_delay=self.policy_delay,
            target_noise=self.target_noise,
            noise_clip=self.noise_clip,
            exploration_noise=self.exploration_noise,
            device=str(self.device),
            seed=self.seed,
            normalize_obs=self.normalize_obs,
        )

        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.actor = DeterministicPolicy(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        self.actor_target = DeterministicPolicy(
            obs_dim, action_dim, hidden=self.hidden, activation=self.activation
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
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.critic2, self.critic2_target)
        self.actor_target.requires_grad_(False)
        self.critic1_target.requires_grad_(False)
        self.critic2_target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

        self.buffer = ReplayBuffer(
            self.buffer_size, obs_dim, action_dim, seed=self.seed
        )
        self.obs_rms = RunningMeanStd(obs_dim) if self.normalize_obs else None

        self.total_steps = 0
        self.n_updates = 0
        self._critic_updates = 0
        self._skipped = 0
        self._last_actor_loss = float("nan")

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

    # --- Agent interface -----------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.learning_starts:
            return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
        obs_n = self._normalize_obs(obs)
        with torch.no_grad():
            t = to_tensor(obs_n, self.device).unsqueeze(0)
            action = to_numpy(self.actor(t).squeeze(0))
        if not deterministic and self.exploration_noise > 0.0:
            action = action + self._rng.normal(
                0.0, self.exploration_noise, size=self.action_dim
            ).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe_transition(self, tr: Transition) -> None:
        self.buffer.add_transition(tr)
        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(tr.obs, dtype=np.float64).reshape(-1))
        self.total_steps += 1

    def reset(self) -> None:
        """TD3 keeps no per-episode state; replay spans episodes by design."""

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
        not_terminated = 1.0 - batch["terminated"]

        # --- critics: clipped double-Q with target policy smoothing ----------
        with torch.no_grad():
            noise = to_tensor(
                self._rng.normal(0.0, self.target_noise, size=tuple(actions.shape)),
                self.device,
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(
                self.critic1_target(next_obs, next_action),
                self.critic2_target(next_obs, next_action),
            )
            y = rewards + self.gamma * not_terminated * target_q

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = 0.5 * (((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean())
        if not torch.isfinite(critic_loss):
            self._skipped += 1
            logger.warning("TD3: non-finite critic loss; skipping gradient step")
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
        self._critic_updates += 1

        # --- delayed actor + target updates ----------------------------------
        if self._critic_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            if not torch.isfinite(actor_loss):
                self._skipped += 1
                logger.warning("TD3: non-finite actor loss; skipping actor update")
            else:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self._clip_and_check(
                    list(self.actor.parameters()), self.actor_optimizer, "actor"
                ):
                    self.actor_optimizer.step()
                    self._last_actor_loss = float(actor_loss.item())
                    soft_update(self.actor, self.actor_target, self.tau)
                    soft_update(self.critic1, self.critic1_target, self.tau)
                    soft_update(self.critic2, self.critic2_target, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": self._last_actor_loss,
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
            logger.warning("TD3: non-finite %s gradient; skipping step", what)
            optimizer.zero_grad(set_to_none=True)
            return False
        return True

    # --- bookkeeping ---------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Reseed exploration noise, smoothing noise, replay sampling and torch.

        Weights are initialised in ``__init__`` from the constructor seed;
        calling this afterwards does not re-initialise them.
        """
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)
        self.buffer.set_seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def num_parameters(self) -> int:
        return count_parameters([self.actor, self.critic1, self.critic2])

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "config": dict(self.config),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "rng": self._rng.bit_generator.state,
            "total_steps": self.total_steps,
            "n_updates": self.n_updates,
            "critic_updates": self._critic_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state.get("obs_rms") is not None and self.obs_rms is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state.get("rng") is not None:
            self._rng.bit_generator.state = state["rng"]
        self.total_steps = int(state.get("total_steps", 0))
        self.n_updates = int(state.get("n_updates", 0))
        self._critic_updates = int(state.get("critic_updates", 0))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(_torch_load(path, self.device))
