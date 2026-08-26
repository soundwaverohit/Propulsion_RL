"""Composable environment wrappers.

Every wrapper here satisfies :class:`~propulsion_rl.core.protocols.EnvProtocol`
and forwards unknown attributes to the environment it wraps, so a stack behaves
like the environment underneath it -- ``env.observation_labels`` and
``env.mission_result`` still resolve through four layers of wrapper.

Recommended order, outermost last::

    env = PropulsionEnv(...)
    env = ClipAction(env)
    env = ActionRepeat(env, 4)
    env = ConstraintWrapper(env)
    env = RecordEpisodeStatistics(env)     # sees raw rewards
    env = NormalizeObservation(env)
    env = NormalizeReward(env)

:class:`RecordEpisodeStatistics` goes *inside* :class:`NormalizeReward` so the
reported returns stay in physical units; it also prefers ``info["raw_reward"]``
when present, so the order is forgiving.
"""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.spaces import Box

LOGGER = logging.getLogger(__name__)

#: Observation entries whose *sign* carries the meaning (a constraint margin is
#: safe at >= 0 and violated at < 0). Matched against ``observation_labels``.
SIGN_PRESERVING_PATTERN = r"margin|constraint|violat"


class Wrapper:
    """Base class: delegate everything, override what you change."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation_space: Box = env.observation_space
        self.action_space: Box = env.action_space

    # Only called when normal lookup fails, so wrapper-owned attributes win.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "env":
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r} "
                "(private attributes are not forwarded)"
            )
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        """The innermost environment."""
        env = self.env
        return env.unwrapped if hasattr(env, "unwrapped") else env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}{self.env!r}>"


# --- running statistics ------------------------------------------------------
class RunningMeanStd:
    """Welford / Chan running first and second moments.

    Tracks the mean and variance (for centred normalisation) *and* the raw
    second moment ``E[x^2]`` (for zero-preserving scale-only normalisation).
    The batch form is the numerically stable parallel update, so a vector env
    feeding whole batches gets the same answer as a single env fed one row at a
    time.
    """

    __slots__ = ("mean", "var", "second_moment", "count", "shape")

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.shape = tuple(shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.second_moment = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def update(self, x: np.ndarray) -> None:
        """Fold in one sample (``shape``) or a batch (``(n,) + shape``)."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape == self.shape:
            x = x.reshape((1,) + self.shape)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_second = np.mean(x * x, axis=0)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta * delta * (self.count * batch_count / total)) / total
        self.second_moment += (batch_second - self.second_moment) * (batch_count / total)
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "second_moment": self.second_moment.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.second_moment = np.asarray(
            state.get("second_moment", np.ones_like(self.var)), dtype=np.float64
        ).copy()
        self.count = float(state["count"])
        self.shape = self.mean.shape


# --- observation normalisation ----------------------------------------------
class NormalizeObservation(Wrapper):
    """Running-statistics observation normalisation with frozen evaluation.

    Constraint margins and the sign problem
    --------------------------------------
    Roughly a third of the observation is signed margins: ``>= 0`` is safe,
    ``< 0`` is a violation, and *zero is the decision boundary*. A well-behaved
    policy spends nearly all of its time with those margins comfortably
    positive, so their running mean sits near, say, +0.85 with a small variance.
    Plain ``(x - mean) / std`` then maps "comfortably safe" to ~0 and re-centres
    the whole distribution so the network sees sign flips that mean nothing --
    the one bit that actually matters is normalised away, and it is amplified by
    the small variance into noise.

    The fix is to normalise those entries by their second moment about zero
    rather than their variance about the mean::

        safe-to-centre entries:   (x - mean) / sqrt(var + eps)
        sign-carrying entries:    x / sqrt(E[x^2] + eps)

    Scale-only normalisation still fixes the units (the point of normalising)
    while keeping zero at zero and the sign intact. Entries are selected by
    matching :data:`SIGN_PRESERVING_PATTERN` against ``env.observation_labels``;
    pass ``sign_preserving`` to override with an explicit index list, a custom
    regex, or ``()`` to disable the special case entirely.
    """

    def __init__(
        self,
        env: Any,
        *,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        sign_preserving: str | Sequence[int] | None = None,
        training: bool = True,
    ) -> None:
        super().__init__(env)
        shape = tuple(env.observation_space.shape)
        self.rms = RunningMeanStd(shape)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = bool(training)
        self.center_mask = self._resolve_mask(env, shape, sign_preserving)
        self.observation_space = Box(-self.clip, self.clip, shape, np.float32)

    @staticmethod
    def _resolve_mask(
        env: Any, shape: tuple[int, ...], sign_preserving: str | Sequence[int] | None
    ) -> np.ndarray:
        """``True`` where the entry may be mean-centred."""
        mask = np.ones(shape, dtype=bool)
        if sign_preserving is not None and not isinstance(sign_preserving, str):
            idx = np.asarray(list(sign_preserving), dtype=int)
            if idx.size:
                mask[idx] = False
            return mask

        pattern = sign_preserving if isinstance(sign_preserving, str) else SIGN_PRESERVING_PATTERN
        if not pattern:
            return mask
        labels = getattr(env, "observation_labels", None)
        if not labels:
            LOGGER.debug(
                "no observation_labels available; every entry will be mean-centred"
            )
            return mask
        rx = re.compile(pattern, re.IGNORECASE)
        for i, label in enumerate(labels):
            if i < mask.size and rx.search(label):
                mask[i] = False
        LOGGER.debug(
            "NormalizeObservation: %d of %d entries normalised sign-preservingly",
            int((~mask).sum()),
            mask.size,
        )
        return mask

    # --- training / evaluation ------------------------------------------
    def freeze(self) -> "NormalizeObservation":
        """Stop updating the statistics. Call before every evaluation run.

        Evaluating with training-time statistics is not a nicety: an evaluation
        episode that ends early re-estimates the mean from a truncated slice of
        the state distribution, which shifts the observations a trained policy
        sees and makes two agents' eval numbers incomparable.
        """
        self.training = False
        return self

    def unfreeze(self) -> "NormalizeObservation":
        self.training = True
        return self

    # --- persistence ----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        state = self.rms.state_dict()
        state["center_mask"] = self.center_mask.copy()
        state["epsilon"] = self.epsilon
        state["clip"] = self.clip
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rms.load_state_dict(state)
        if "center_mask" in state:
            self.center_mask = np.asarray(state["center_mask"], dtype=bool)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.clip = float(state.get("clip", self.clip))

    def save(self, path: str | Path) -> None:
        """Persist the statistics next to the policy checkpoint."""
        np.savez(str(path), **self.state_dict())

    def load(self, path: str | Path) -> None:
        with np.load(str(path), allow_pickle=False) as data:
            self.load_state_dict({k: data[k] for k in data.files})

    # --- api -------------------------------------------------------------
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        centred = (x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        scaled = x / np.sqrt(self.rms.second_moment + self.epsilon)
        out = np.where(self.center_mask, centred, scaled)
        np.clip(out, -self.clip, self.clip, out=out)
        return out.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.training:
            self.rms.update(obs)
        return self._normalize(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.training:
            self.rms.update(obs)
        return self._normalize(obs), reward, terminated, truncated, info


class NormalizeReward(Wrapper):
    """Scale rewards by the running standard deviation of the discounted return.

    The scale, not the location: shifting rewards changes the optimal policy in
    an episodic task with variable length, so only the divisor is learned. The
    unscaled value is left in ``info["raw_reward"]`` so episode statistics and
    the mission summary stay in physical units.
    """

    def __init__(
        self,
        env: Any,
        *,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        training: bool = True,
    ) -> None:
        super().__init__(env)
        self.rms = RunningMeanStd(())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = bool(training)
        self._discounted_return = 0.0

    def freeze(self) -> "NormalizeReward":
        self.training = False
        return self

    def unfreeze(self) -> "NormalizeReward":
        self.training = True
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.rms.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rms.load_state_dict(state)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), **self.state_dict())

    def load(self, path: str | Path) -> None:
        with np.load(str(path), allow_pickle=False) as data:
            self.load_state_dict({k: data[k] for k in data.files})

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._discounted_return = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._discounted_return = self._discounted_return * self.gamma + reward
        if self.training:
            self.rms.update(np.asarray(self._discounted_return, dtype=np.float64))
        info["raw_reward"] = reward
        scale = math.sqrt(float(self.rms.var) + self.epsilon)
        scaled = reward / scale if scale > 0.0 else reward
        if terminated or truncated:
            self._discounted_return = 0.0
        return (
            obs,
            float(min(max(scaled, -self.clip), self.clip)),
            terminated,
            truncated,
            info,
        )


# --- temporal / bookkeeping wrappers ----------------------------------------
class ActionRepeat(Wrapper):
    """Hold one action for ``k`` inner steps: cheap temporal abstraction.

    A LEO->GEO spiral is 10^4-10^5 macro-steps; nothing useful changes in one
    of them, and asking a policy to make that many independent decisions wastes
    most of the sample budget on noise. Rewards and constraint costs are summed
    over the repeat so the return is unchanged, and the loop breaks early on
    termination so an episode never over-runs its end.
    """

    def __init__(self, env: Any, k: int = 4) -> None:
        super().__init__(env)
        if k < 1:
            raise ValueError(f"ActionRepeat needs k >= 1, got {k}")
        self.k = int(k)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        total_cost = 0.0
        total_raw = 0.0
        has_raw = False
        terms_sum: dict[str, float] = {}
        obs = None
        info: dict[str, Any] = {}
        terminated = truncated = False
        repeats = 0

        for _ in range(self.k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            repeats += 1
            total_reward += reward
            total_cost += float(info.get("constraint_cost", 0.0))
            if "raw_reward" in info:
                has_raw = True
                total_raw += float(info["raw_reward"])
            terms = info.get("reward_terms")
            if terms:
                for key, value in terms.items():
                    terms_sum[key] = terms_sum.get(key, 0.0) + value
            if terminated or truncated:
                break

        info["constraint_cost"] = total_cost
        if terms_sum:
            info["reward_terms"] = terms_sum
        if has_raw:
            info["raw_reward"] = total_raw
        info["action_repeat"] = repeats
        return obs, total_reward, terminated, truncated, info


class TimeLimit(Wrapper):
    """Truncate after ``max_episode_steps`` steps of *this* wrapper.

    Independent of the environment's own step limit, and counted in whatever
    units this layer sees -- put it outside :class:`ActionRepeat` to cap macro
    decisions, inside to cap simulated steps.
    """

    def __init__(self, env: Any, max_episode_steps: int) -> None:
        super().__init__(env)
        if max_episode_steps < 1:
            raise ValueError(
                f"TimeLimit needs max_episode_steps >= 1, got {max_episode_steps}"
            )
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._elapsed = 0
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.max_episode_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info


class ClipAction(Wrapper):
    """Clip actions into ``action_space``.

    Unbounded policy heads (a Gaussian before squashing) are normal; letting a
    +7.0 through would saturate the throttle in a way the propulsion model
    never sees during a scripted baseline run, making the comparison unfair.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._low = np.asarray(env.action_space.low, dtype=np.float32)
        self._high = np.asarray(env.action_space.high, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped = np.clip(
            np.asarray(action, dtype=np.float32), self._low, self._high
        )
        return self.env.step(clipped)


class RecordEpisodeStatistics(Wrapper):
    """Accumulate return, length, constraint cost and success per episode.

    On the terminal step ``info["episode"]`` gains ``r`` (return), ``l``
    (length), ``c`` (summed constraint cost), ``v`` (steps with a violation),
    ``progress``, ``success`` and ``reason``. Recent episodes are kept in
    ``return_queue`` / ``length_queue`` / ``cost_queue`` / ``success_queue``.
    """

    def __init__(self, env: Any, deque_size: int = 100) -> None:
        super().__init__(env)
        self.return_queue: deque[float] = deque(maxlen=deque_size)
        self.length_queue: deque[int] = deque(maxlen=deque_size)
        self.cost_queue: deque[float] = deque(maxlen=deque_size)
        self.success_queue: deque[bool] = deque(maxlen=deque_size)
        self.episode_count = 0
        self._return = 0.0
        self._length = 0
        self._cost = 0.0
        self._violations = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._return = 0.0
        self._length = 0
        self._cost = 0.0
        self._violations = 0
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Prefer the physical reward when a normaliser sits underneath us.
        self._return += float(info.get("raw_reward", reward))
        self._length += 1
        cost = float(info.get("constraint_cost", 0.0))
        self._cost += cost
        if cost > 0.0:
            self._violations += 1

        if terminated or truncated:
            result = info.get("mission_result")
            success = bool(getattr(result, "success", False))
            info["episode"] = {
                "r": self._return,
                "l": self._length,
                "c": self._cost,
                "v": self._violations,
                "progress": float(info.get("progress", 0.0)),
                "success": success,
                "reason": info.get("termination_reason", ""),
            }
            self.return_queue.append(self._return)
            self.length_queue.append(self._length)
            self.cost_queue.append(self._cost)
            self.success_queue.append(success)
            self.episode_count += 1
        return obs, reward, terminated, truncated, info


class ConstraintWrapper(Wrapper):
    """Give constrained-RL agents a clean, conventionally named cost channel.

    A Lagrangian PPO implementation expects ``info["cost"]`` and a
    ``cost_limit``; the environment publishes ``info["constraint_cost"]`` and no
    budget. This bridges the two without either side learning about the other.

    Set ``strip_penalty_from_reward`` to hand the agent the *unshaped* objective
    plus the cost as a separate signal -- which is the whole point of a
    Lagrangian method, and double-counts the penalty if you forget it.
    """

    def __init__(
        self,
        env: Any,
        *,
        cost_key: str = "cost",
        strip_penalty_from_reward: bool = False,
        cost_limit: float | None = None,
    ) -> None:
        super().__init__(env)
        self.cost_key = str(cost_key)
        self.strip_penalty_from_reward = bool(strip_penalty_from_reward)
        self.cost_limit = cost_limit

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info[self.cost_key] = float(info.get("constraint_cost", 0.0))
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("constraint_cost", 0.0))
        info[self.cost_key] = cost
        if self.strip_penalty_from_reward:
            penalty = float(info.get("reward_terms", {}).get("constraint_penalty", 0.0))
            if penalty:
                reward = reward - penalty
                info["raw_reward"] = info.get("raw_reward", reward)
        return obs, reward, terminated, truncated, info


#: Alias -- the same wrapper, named for what it does rather than who wants it.
CostAsInfo = ConstraintWrapper


def wrap(
    env: Any,
    *,
    clip_action: bool = True,
    action_repeat: int = 1,
    time_limit: int | None = None,
    constraint_channel: bool = True,
    record_stats: bool = True,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    gamma: float = 0.99,
) -> Any:
    """Build the recommended stack in the recommended order."""
    if clip_action:
        env = ClipAction(env)
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    if time_limit is not None:
        env = TimeLimit(env, time_limit)
    if constraint_channel:
        env = ConstraintWrapper(env)
    if record_stats:
        env = RecordEpisodeStatistics(env)
    if normalize_obs:
        env = NormalizeObservation(env)
    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma)
    return env


__all__ = [
    "ActionRepeat",
    "ClipAction",
    "ConstraintWrapper",
    "CostAsInfo",
    "NormalizeObservation",
    "NormalizeReward",
    "RecordEpisodeStatistics",
    "RunningMeanStd",
    "TimeLimit",
    "Wrapper",
    "SIGN_PRESERVING_PATTERN",
    "wrap",
]
