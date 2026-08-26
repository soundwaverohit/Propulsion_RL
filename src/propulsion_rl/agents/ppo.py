"""Proximal Policy Optimization.

The on-policy workhorse of the benchmark and the base class the constrained
(Lagrangian) variant extends, so the update is factored into small overridable
pieces rather than one monolithic loop.

Design notes that matter for the comparison
-------------------------------------------
* The policy is an **unsquashed** diagonal Gaussian with a state-independent
  log-std. The importance ratio needs the density of the sample that was
  actually drawn, so :meth:`act` keeps the raw Gaussian sample for the buffer
  and hands the environment a clipped copy. Squashing the policy instead would
  require the tanh Jacobian correction in the ratio and changes what the
  entropy bonus means; SAC does that, PPO here deliberately does not.
* Advantages are normalised **per minibatch**, not per rollout. Both are
  defensible, but per-minibatch is what the reference implementations do and it
  is what the published PPO hyperparameters were tuned against.
* Observations are normalised by a running mean/std and the *normalised* vector
  is what goes into the buffer, so the minibatch passes see numerically the
  same inputs the rollout saw. Re-normalising at update time with drifted
  statistics is a subtle way to make the old log-probs inconsistent with the
  new ones and inflate the KL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .base import Agent, TrainStats, Transition
from .buffers import RolloutBuffer
from .networks import (
    GaussianPolicy,
    RunningMeanStd,
    ValueNetwork,
    count_parameters,
    resolve_device,
    to_numpy,
    to_tensor,
)

logger = logging.getLogger(__name__)

__all__ = ["PPOAgent"]


class PPOAgent(Agent):
    """Clipped-surrogate PPO for continuous control in ``[-1, 1]^action_dim``.

    Parameters
    ----------
    obs_dim, action_dim:
        Interface widths; 36 and 5 for this project's canonical spaces.
    hidden:
        Hidden widths for both the policy and the value network.
    lr:
        Adam learning rate, shared by policy and value parameters.
    gamma, gae_lambda:
        Discount and GAE trace decay.
    clip:
        PPO ratio clip epsilon. Also the value-clip range when
        ``clip_vloss=True``.
    epochs, minibatch, rollout_steps:
        ``epochs`` passes over each ``rollout_steps``-long rollout in
        ``minibatch``-sized chunks.
    ent_coef, vf_coef, max_grad_norm:
        Entropy bonus weight, value loss weight, global grad-norm clip.
    target_kl:
        If set, stop the epoch loop early once the approximate KL between the
        old and new policy exceeds it. ``None`` disables the check.
    device:
        Defaults to ``"cpu"``. These networks are tiny (36 -> 256 -> 256 -> 5),
        far below the size where GPU/MPS kernel launch overhead pays for
        itself, so CPU is genuinely faster here as well as bitwise
        reproducible. ``"cuda"``/``"mps"`` are honoured if asked for.
    seed:
        Seeds network initialisation, action sampling and minibatch shuffling.

    Additional keyword options
    --------------------------
    ``normalize_obs`` (default True), ``normalize_reward`` (default False),
    ``clip_vloss`` (default False), ``log_std_init`` (default 0.0),
    ``activation`` (default ``"tanh"``), ``obs_clip`` (default 10.0).
    """

    name = "ppo"
    learns = True
    #: The constrained variant is a separate agent; this one folds any cost
    #: into the reward upstream.
    uses_constraints = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        epochs: int = 10,
        minibatch: int = 64,
        rollout_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        device: str = "cpu",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip = float(clip)
        self.epochs = int(epochs)
        self.minibatch = int(minibatch)
        self.rollout_steps = int(rollout_steps)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.device = resolve_device(device)
        self.seed = int(seed)

        self.normalize_obs = bool(kwargs.get("normalize_obs", True))
        self.normalize_reward = bool(kwargs.get("normalize_reward", False))
        self.clip_vloss = bool(kwargs.get("clip_vloss", False))
        self.obs_clip = float(kwargs.get("obs_clip", 10.0))
        self.activation = kwargs.get("activation", "tanh")
        self.log_std_init = float(kwargs.get("log_std_init", 0.0))

        self.config.update(
            hidden=self.hidden,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip=self.clip,
            epochs=self.epochs,
            minibatch=self.minibatch,
            rollout_steps=self.rollout_steps,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            device=str(self.device),
            seed=self.seed,
            normalize_obs=self.normalize_obs,
            normalize_reward=self.normalize_reward,
            clip_vloss=self.clip_vloss,
        )

        # Seed *before* building the networks so initialisation is reproducible.
        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.policy = GaussianPolicy(
            obs_dim,
            action_dim,
            hidden=self.hidden,
            activation=self.activation,
            log_std_init=self.log_std_init,
            seed=self.seed,
        ).to(self.device)
        self.value_net = ValueNetwork(
            obs_dim, hidden=self.hidden, activation=self.activation
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self._learnable(), lr=self.lr, eps=1e-5)

        self.buffer = self._make_buffer()
        self.obs_rms = RunningMeanStd(obs_dim) if self.normalize_obs else None
        self.ret_rms = RunningMeanStd(1) if self.normalize_reward else None

        self._ret_acc = 0.0
        self._cached: dict[str, Any] | None = None
        self._last_next_obs_n: np.ndarray | None = None
        self._last_terminated = False
        self._pending_stats: TrainStats | None = None
        self._recompute_warned = False
        self.total_steps = 0
        self.n_updates = 0

    # --- construction hooks (the Lagrangian subclass overrides these) --------
    def _make_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(
            self.rollout_steps,
            self.obs_dim,
            self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            track_cost=self.uses_constraints,
        )

    def _learnable(self) -> list[nn.Parameter]:
        return list(self.policy.parameters()) + list(self.value_net.parameters())

    # --- observation plumbing ------------------------------------------------
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.obs_rms is None:
            return arr
        return self.obs_rms.normalize(arr, clip=self.obs_clip)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return to_tensor(x, self.device).unsqueeze(0)

    # --- Agent interface -----------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_n = self._normalize_obs(obs)
        with torch.no_grad():
            t = self._to_tensor(obs_n)
            raw_action, logp = self.policy.sample(t, deterministic=deterministic)
            value = self.value_net(t)
        raw = to_numpy(raw_action.squeeze(0))
        clipped = np.clip(raw, -1.0, 1.0)
        self._cached = {
            "obs_raw": np.asarray(obs, dtype=np.float32).reshape(-1),
            "obs_n": obs_n,
            "raw_action": raw,
            "clipped_action": clipped,
            "logp": float(logp.item()),
            "value": float(value.item()),
        }
        return clipped

    def observe_transition(self, tr: Transition) -> None:
        if not self.learns:
            return
        obs_n, raw_action, logp, value = self._resolve_cached(tr)

        reward = float(tr.reward)
        if self.ret_rms is not None:
            self._ret_acc = self._ret_acc * self.gamma + reward
            self.ret_rms.update(np.array([self._ret_acc], dtype=np.float64))
            scale = float(np.sqrt(self.ret_rms.var[0] + 1e-8))
            reward = float(np.clip(reward / max(scale, 1e-8), -10.0, 10.0))
            if tr.terminated or tr.truncated:
                self._ret_acc = 0.0

        next_obs_n = self._normalize_obs(tr.next_obs)
        # Only a truncated step needs V(next_obs): for every other step the GAE
        # recursion reads values[t+1] (or the rollout's trailing last_value).
        next_value = 0.0
        if tr.truncated and not tr.terminated:
            with torch.no_grad():
                next_value = float(self.value_net(self._to_tensor(next_obs_n)).item())

        extra = {"cost": float(tr.cost), "next_value": next_value}
        extra.update(self._extra_add_kwargs(tr, next_obs_n))
        self.buffer.add(
            obs_n,
            raw_action,
            logp,
            value,
            reward,
            bool(tr.terminated),
            bool(tr.truncated),
            **extra,
        )
        self._last_next_obs_n = next_obs_n
        self._last_terminated = bool(tr.terminated) or bool(tr.truncated)

        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(tr.obs, dtype=np.float64).reshape(-1))
        self.total_steps += 1
        self._cached = None

        if self.buffer.full:
            # The rollout is complete. Run the update here rather than wait for
            # the runner to call update() on exactly the right step: a runner
            # whose cadence does not divide rollout_steps would otherwise
            # overflow the buffer. The stats are handed to the next update()
            # call so nothing disappears from the training log.
            self._pending_stats = self._do_update()

    def _extra_add_kwargs(
        self, tr: Transition, next_obs_n: np.ndarray
    ) -> dict[str, float]:
        """Hook for the constrained subclass to add cost-value bookkeeping."""
        return {}

    def _resolve_cached(
        self, tr: Transition
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Recover ``(obs_n, raw_action, logp, value)`` for this transition.

        The fast path reuses what :meth:`act` computed. If the runner handed
        back a transition that does not match the cache -- a different obs, or
        an action the environment altered beyond the ``[-1, 1]`` clip -- the
        quantities are recomputed so the surrogate ratio stays consistent with
        the data instead of silently referring to a different state.
        """
        obs_raw = np.asarray(tr.obs, dtype=np.float32).reshape(-1)
        action = np.asarray(tr.action, dtype=np.float32).reshape(-1)
        c = self._cached
        if (
            c is not None
            and np.allclose(c["obs_raw"], obs_raw, atol=1e-6, rtol=0.0)
            and np.allclose(c["clipped_action"], action, atol=1e-6, rtol=0.0)
        ):
            return c["obs_n"], c["raw_action"], c["logp"], c["value"]

        if not self._recompute_warned:
            logger.warning(
                "PPO: transition does not match the cached act() output; "
                "recomputing log-prob and value (this is slower but correct)"
            )
            self._recompute_warned = True
        obs_n = self._normalize_obs(obs_raw)
        with torch.no_grad():
            t = self._to_tensor(obs_n)
            a = to_tensor(action, self.device).unsqueeze(0)
            logp, _ = self.policy.evaluate(t, a)
            value = self.value_net(t)
        return obs_n, action, float(logp.item()), float(value.item())

    def reset(self) -> None:
        """Clear per-episode caches. The rollout buffer deliberately survives."""
        self._cached = None
        self._ret_acc = 0.0

    # --- the update ----------------------------------------------------------
    def update(self) -> TrainStats:
        if self._pending_stats is not None:
            stats, self._pending_stats = self._pending_stats, None
            return stats
        if len(self.buffer) < self.rollout_steps:
            return TrainStats()
        return self._do_update()

    def _do_update(self) -> TrainStats:
        """Compute advantages over the finished rollout and run the epochs."""
        last_value = 0.0
        if self._last_next_obs_n is not None:
            with torch.no_grad():
                last_value = float(
                    self.value_net(self._to_tensor(self._last_next_obs_n)).item()
                )
        self.buffer.compute_returns_and_advantages(last_value, *self._last_cost_value())
        stats = self._train_epochs()
        self.buffer.reset()
        self.n_updates += 1
        return stats

    def _last_cost_value(self) -> tuple[float, ...]:
        """Trailing cost bootstrap; empty for the unconstrained agent."""
        return ()

    def _train_epochs(self) -> TrainStats:
        data = self.buffer.get(self.device)
        n = data["obs"].shape[0]
        ev = _explained_variance(data["values"], data["returns"])

        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clipfracs: list[float] = []
        skipped = 0
        epochs_run = 0
        stop = False

        for _ in range(self.epochs):
            epochs_run += 1
            order = self._rng.permutation(n)
            epoch_kls: list[float] = []
            for start in range(0, n, self.minibatch):
                idx = to_tensor(
                    order[start : start + self.minibatch], self.device, torch.int64
                )
                out = self._minibatch_loss({k: v[idx] for k, v in data.items()})
                if out is None:
                    skipped += 1
                    continue
                loss, info = out

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._learnable(), self.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    logger.warning(
                        "PPO: non-finite gradient norm; skipping this minibatch"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                self.optimizer.step()

                pg_losses.append(info["policy_loss"])
                v_losses.append(info["value_loss"])
                entropies.append(info["entropy"])
                kls.append(info["approx_kl"])
                epoch_kls.append(info["approx_kl"])
                clipfracs.append(info["clip_fraction"])

            if self.target_kl is not None and epoch_kls:
                if float(np.mean(epoch_kls)) > self.target_kl:
                    stop = True
                    break

        if skipped:
            logger.warning("PPO: skipped %d minibatch update(s) this iteration", skipped)

        return TrainStats().update(
            policy_loss=float(np.mean(pg_losses)) if pg_losses else float("nan"),
            value_loss=float(np.mean(v_losses)) if v_losses else float("nan"),
            entropy=float(np.mean(entropies)) if entropies else float("nan"),
            approx_kl=float(np.mean(kls)) if kls else float("nan"),
            clip_fraction=float(np.mean(clipfracs)) if clipfracs else float("nan"),
            explained_variance=ev,
            epochs_run=float(epochs_run),
            early_stopped=float(stop),
            skipped_minibatches=float(skipped),
            n_updates=float(self.n_updates + 1),
            total_steps=float(self.total_steps),
            log_std=float(self.policy.log_std.detach().mean().item()),
        )

    def _minibatch_loss(
        self, mb: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]] | None:
        """Clipped surrogate + value loss for one minibatch.

        Returns ``None`` when the loss is not finite, so the caller can skip the
        step rather than propagate a NaN through every parameter in the policy.
        """
        new_logp, entropy = self.policy.evaluate(mb["obs"], mb["actions"])
        log_ratio = new_logp - mb["logprobs"]
        ratio = log_ratio.exp()

        adv = mb["advantages"]
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        pg1 = -adv * ratio
        pg2 = -adv * torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
        policy_loss = torch.max(pg1, pg2).mean()

        new_value = self.value_net(mb["obs"])
        if self.clip_vloss:
            clipped_v = mb["values"] + torch.clamp(
                new_value - mb["values"], -self.clip, self.clip
            )
            value_loss = 0.5 * torch.max(
                (new_value - mb["returns"]) ** 2, (clipped_v - mb["returns"]) ** 2
            ).mean()
        else:
            value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()

        entropy_mean = entropy.mean()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

        if not torch.isfinite(loss):
            logger.warning("PPO: non-finite loss (%s); skipping minibatch", loss.item())
            return None

        with torch.no_grad():
            # Schulman's k3 estimator: low-variance and non-negative, unlike
            # the naive -mean(log_ratio).
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip).float().mean()

        return loss, {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }

    # --- bookkeeping ---------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Reseed every stream this agent owns.

        Network *initialisation* happens in ``__init__`` from the constructor
        seed; calling this later reseeds sampling and shuffling but does not
        re-initialise weights. Construct with the seed you want for a
        reproducible run.
        """
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)
        self.policy.set_seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def num_parameters(self) -> int:
        return count_parameters([self.policy, self.value_net])

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "config": dict(self.config),
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "ret_rms": None if self.ret_rms is None else self.ret_rms.state_dict(),
            "noise": self.policy.noise.get_state(),
            "rng": self._rng.bit_generator.state,
            "total_steps": self.total_steps,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("obs_rms") is not None and self.obs_rms is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state.get("ret_rms") is not None and self.ret_rms is not None:
            self.ret_rms.load_state_dict(state["ret_rms"])
        if state.get("noise") is not None:
            self.policy.noise.set_state(state["noise"])
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


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    """``torch.load`` that works across the 2.x ``weights_only`` default change."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location=device)


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """``1 - Var(returns - values) / Var(returns)``; 0 means "no better than the mean"."""
    var_y = returns.var(unbiased=False)
    if float(var_y.item()) < 1e-12:
        return float("nan")
    return float((1.0 - (returns - values).var(unbiased=False) / var_y).item())
