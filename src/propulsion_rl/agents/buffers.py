"""Rollout and replay storage for the learned agents.

Both buffers preallocate their numpy arrays at construction. Nothing here
appends to a Python list per step: a 30-year low-thrust transfer integrated at
a one-day step is on the order of 10^4 steps per episode and a training run is
millions of steps, so per-step allocation shows up in the profile.

The single most important thing in this module is
:func:`compute_gae`'s treatment of **truncation versus termination**. Every
mission in this benchmark ends on a time limit, so every episode ends truncated
rather than terminated, and conflating the two teaches the agent that running
out of clock is as bad as destroying the reactor. See that function's docstring
for the exact rule.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch

from .base import Transition
from .networks import to_tensor

logger = logging.getLogger(__name__)

__all__ = ["ReplayBuffer", "RolloutBuffer", "compute_gae"]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    next_values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation over one contiguous rollout segment.

    Parameters
    ----------
    rewards, values:
        ``(T,)`` rewards ``r_t`` and value predictions ``V(s_t)``.
    terminated:
        ``(T,)`` boolean. The MDP genuinely ended: no future return exists, so
        the bootstrap is exactly zero.
    truncated:
        ``(T,)`` boolean. The episode was cut off by a time limit (or a vector
        env reset) while the MDP was still running. The future return exists,
        we just stopped looking at it.
    next_values:
        ``(T,)`` ``V(s_{t+1})`` for the *final* observation of a truncated step.
        Only read where ``truncated`` is true; pass zeros elsewhere.
    last_value:
        ``V(s_T)`` -- the value of the state following the last stored step.
        Used only when that last step was neither terminated nor truncated,
        i.e. when the rollout was cut mid-episode by the update cadence.

    Returns
    -------
    ``(advantages, returns)``, both ``(T,)`` float32, with
    ``returns = advantages + values`` (the standard GAE value target).

    Truncation handling
    -------------------
    Two distinct things happen at an episode boundary and they must be decided
    separately:

    1. **Bootstrapping.** ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)``.
       On termination ``V(s_{t+1}) = 0``. On truncation ``V(s_{t+1})`` is the
       real value of the final observation, which we *must* bootstrap from --
       otherwise the value function learns that the last state of every episode
       is worth only its immediate reward, and that error propagates backwards
       through the whole trajectory.
    2. **Accumulation.** The GAE recursion
       ``A_t = delta_t + gamma * lambda * A_{t+1}`` must be cut at *any*
       boundary, terminated or truncated, because step ``t+1`` in the buffer
       belongs to a different episode.

    Treating truncation as termination gets (1) wrong while getting (2) right,
    which is why it is such a persistent bug: the code looks symmetric and the
    agent still learns, just toward a systematically pessimistic value function
    near the horizon. In a time-limited mission that bias sits on exactly the
    states the agent spends most of its time in.
    """
    T = int(rewards.shape[0])
    if not (
        values.shape[0] == terminated.shape[0] == truncated.shape[0] == T
        and next_values.shape[0] == T
    ):
        raise ValueError("compute_gae inputs must all have the same leading length")
    advantages = np.zeros(T, dtype=np.float64)
    values_f = values.astype(np.float64)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        if terminated[t]:
            bootstrap = 0.0          # no future exists
            carry = 0.0              # cut the recursion
        elif truncated[t]:
            bootstrap = float(next_values[t])   # future exists; we stopped looking
            carry = 0.0                          # still a different episode after t
        else:
            bootstrap = float(values_f[t + 1]) if t + 1 < T else float(last_value)
            carry = 1.0
        delta = float(rewards[t]) + gamma * bootstrap - float(values_f[t])
        last_gae = delta + gamma * gae_lambda * carry * last_gae
        advantages[t] = last_gae
    returns = advantages + values_f
    return advantages.astype(np.float32), returns.astype(np.float32)


class RolloutBuffer:
    """Fixed-length on-policy storage with GAE, plus a parallel cost channel.

    The cost channel mirrors the reward channel exactly -- its own value
    predictions, its own discount, its own GAE -- so a Lagrangian agent can ask
    for ``cost_advantages`` without this class knowing anything about
    multipliers. It costs nothing when unused: pass ``track_cost=False`` (the
    default) and the cost GAE is simply never computed.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        cost_gamma: float | None = None,
        cost_lambda: float | None = None,
        track_cost: bool = False,
        dtype: type = np.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.cost_gamma = float(gamma if cost_gamma is None else cost_gamma)
        self.cost_lambda = float(gae_lambda if cost_lambda is None else cost_lambda)
        self.track_cost = bool(track_cost)
        self.dtype = dtype

        n, o, a = self.capacity, self.obs_dim, self.action_dim
        self.obs = np.zeros((n, o), dtype=dtype)
        self.actions = np.zeros((n, a), dtype=dtype)
        self.logprobs = np.zeros(n, dtype=dtype)
        self.values = np.zeros(n, dtype=dtype)
        self.rewards = np.zeros(n, dtype=dtype)
        self.terminated = np.zeros(n, dtype=bool)
        self.truncated = np.zeros(n, dtype=bool)
        self.next_values = np.zeros(n, dtype=dtype)
        self.advantages = np.zeros(n, dtype=dtype)
        self.returns = np.zeros(n, dtype=dtype)
        # Cost channel (constrained RL). Allocated unconditionally -- it is
        # a handful of float32 vectors, and the alternative is None-checks
        # scattered through every method.
        self.costs = np.zeros(n, dtype=dtype)
        self.cost_values = np.zeros(n, dtype=dtype)
        self.next_cost_values = np.zeros(n, dtype=dtype)
        self.cost_advantages = np.zeros(n, dtype=dtype)
        self.cost_returns = np.zeros(n, dtype=dtype)

        self.pos = 0
        self.ready = False

    # --- writing -------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logprob: float,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        *,
        cost: float = 0.0,
        cost_value: float = 0.0,
        next_value: float = 0.0,
        next_cost_value: float = 0.0,
    ) -> None:
        """Append one transition.

        ``next_value`` is ``V(next_obs)`` and is only consulted when
        ``truncated`` is true; the caller should compute it exactly then and may
        pass 0.0 otherwise.
        """
        if self.pos >= self.capacity:
            raise RuntimeError(
                f"RolloutBuffer is full ({self.capacity}); call compute/get then reset"
            )
        i = self.pos
        self.obs[i] = np.asarray(obs, dtype=self.dtype).reshape(self.obs_dim)
        self.actions[i] = np.asarray(action, dtype=self.dtype).reshape(self.action_dim)
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.terminated[i] = bool(terminated)
        self.truncated[i] = bool(truncated)
        self.next_values[i] = next_value
        self.costs[i] = cost
        self.cost_values[i] = cost_value
        self.next_cost_values[i] = next_cost_value
        self.pos += 1

    # --- advantages ----------------------------------------------------------
    def compute_returns_and_advantages(
        self, last_value: float, last_cost_value: float = 0.0
    ) -> None:
        """Fill ``advantages``/``returns`` (and the cost pair if tracking)."""
        n = self.pos
        if n == 0:
            raise RuntimeError("nothing to compute: buffer is empty")
        adv, ret = compute_gae(
            self.rewards[:n],
            self.values[:n],
            self.terminated[:n],
            self.truncated[:n],
            self.next_values[:n],
            float(last_value),
            self.gamma,
            self.gae_lambda,
        )
        self.advantages[:n] = adv
        self.returns[:n] = ret
        if self.track_cost:
            c_adv, c_ret = compute_gae(
                self.costs[:n],
                self.cost_values[:n],
                self.terminated[:n],
                self.truncated[:n],
                self.next_cost_values[:n],
                float(last_cost_value),
                self.cost_gamma,
                self.cost_lambda,
            )
            self.cost_advantages[:n] = c_adv
            self.cost_returns[:n] = c_ret
        self.ready = True

    # --- reading -------------------------------------------------------------
    def get(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """The whole rollout as torch tensors on ``device``."""
        if not self.ready:
            raise RuntimeError("call compute_returns_and_advantages() before get()")
        n = self.pos
        dev = torch.device(device)

        def t(x: np.ndarray) -> torch.Tensor:
            return to_tensor(x, dev)

        out = {
            "obs": t(self.obs[:n]),
            "actions": t(self.actions[:n]),
            "logprobs": t(self.logprobs[:n]),
            "values": t(self.values[:n]),
            "advantages": t(self.advantages[:n]),
            "returns": t(self.returns[:n]),
            "rewards": t(self.rewards[:n]),
            "terminated": t(self.terminated[:n].astype(np.float32)),
            "truncated": t(self.truncated[:n].astype(np.float32)),
        }
        if self.track_cost:
            out["costs"] = t(self.costs[:n])
            out["cost_values"] = t(self.cost_values[:n])
            out["cost_advantages"] = t(self.cost_advantages[:n])
            out["cost_returns"] = t(self.cost_returns[:n])
        return out

    def iter_minibatches(
        self,
        minibatch_size: int,
        rng: np.random.Generator,
        device: torch.device | str = "cpu",
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shuffled minibatches covering the rollout exactly once."""
        data = self.get(device)
        n = self.pos
        order = rng.permutation(n)
        step = max(1, int(minibatch_size))
        for start in range(0, n, step):
            idx = to_tensor(
                order[start : start + step], torch.device(device), torch.int64
            )
            yield {k: v[idx] for k, v in data.items()}

    # --- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        self.pos = 0
        self.ready = False

    @property
    def full(self) -> bool:
        return self.pos >= self.capacity

    def __len__(self) -> int:
        return self.pos

    def __repr__(self) -> str:
        return (
            f"RolloutBuffer({self.pos}/{self.capacity}, obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, track_cost={self.track_cost})"
        )


class ReplayBuffer:
    """Circular off-policy replay with preallocated arrays and O(1) insertion.

    Stores ``cost`` next to ``reward`` so a constrained off-policy agent shares
    this class rather than forking it.

    ``terminated`` and ``truncated`` are stored **separately and on purpose**.
    The Bellman target must mask on ``terminated`` alone: a transition cut off
    by the time limit still has a real successor state, so
    ``y = r + gamma * (1 - terminated) * Q(s', a')`` is correct and
    ``(1 - done)`` is not. :meth:`sample` therefore returns ``terminated``,
    never a merged ``done``.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        seed: int = 0,
        dtype: type = np.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dtype = dtype

        n, o, a = self.capacity, self.obs_dim, self.action_dim
        self.obs = np.zeros((n, o), dtype=dtype)
        self.next_obs = np.zeros((n, o), dtype=dtype)
        self.actions = np.zeros((n, a), dtype=dtype)
        self.rewards = np.zeros(n, dtype=dtype)
        self.costs = np.zeros(n, dtype=dtype)
        self.terminated = np.zeros(n, dtype=dtype)
        self.truncated = np.zeros(n, dtype=dtype)

        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool = False,
        cost: float = 0.0,
    ) -> None:
        i = self.pos
        self.obs[i] = np.asarray(obs, dtype=self.dtype).reshape(self.obs_dim)
        self.next_obs[i] = np.asarray(next_obs, dtype=self.dtype).reshape(self.obs_dim)
        self.actions[i] = np.asarray(action, dtype=self.dtype).reshape(self.action_dim)
        self.rewards[i] = reward
        self.costs[i] = cost
        self.terminated[i] = float(bool(terminated))
        self.truncated[i] = float(bool(truncated))
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_transition(self, tr: Transition) -> None:
        """Convenience wrapper around :meth:`add` for the agent contract type."""
        self.add(
            tr.obs,
            tr.action,
            tr.reward,
            tr.next_obs,
            tr.terminated,
            tr.truncated,
            tr.cost,
        )

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Uniform sample with replacement, as torch tensors on ``device``."""
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty ReplayBuffer")
        idx = self._rng.integers(0, self.size, size=int(batch_size))
        dev = torch.device(device)

        def t(x: np.ndarray) -> torch.Tensor:
            return to_tensor(x, dev)

        return {
            "obs": t(self.obs[idx]),
            "actions": t(self.actions[idx]),
            "rewards": t(self.rewards[idx]),
            "next_obs": t(self.next_obs[idx]),
            "terminated": t(self.terminated[idx]),
            "truncated": t(self.truncated[idx]),
            "costs": t(self.costs[idx]),
        }

    def state_dict(self) -> dict[str, Any]:
        """Buffer contents, for checkpoint/resume of an off-policy run."""
        n = self.size
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "pos": self.pos,
            "size": self.size,
            "obs": self.obs[:n].copy(),
            "next_obs": self.next_obs[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "costs": self.costs[:n].copy(),
            "terminated": self.terminated[:n].copy(),
            "truncated": self.truncated[:n].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = int(state["size"])
        if int(state["obs_dim"]) != self.obs_dim or int(
            state["action_dim"]
        ) != self.action_dim:
            raise ValueError("ReplayBuffer shape mismatch on load")
        if n > self.capacity:
            raise ValueError(
                f"saved buffer holds {n} transitions, capacity is {self.capacity}"
            )
        for key in (
            "obs",
            "next_obs",
            "actions",
            "rewards",
            "costs",
            "terminated",
            "truncated",
        ):
            getattr(self, key)[:n] = np.asarray(state[key], dtype=self.dtype)
        self.size = n
        self.pos = int(state["pos"]) % self.capacity

    def clear(self) -> None:
        self.pos = 0
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer({self.size}/{self.capacity}, obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim})"
        )
