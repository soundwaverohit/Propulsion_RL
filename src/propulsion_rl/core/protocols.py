"""Structural interfaces used across package boundaries.

These are ``typing.Protocol`` definitions, not base classes: they document what
the environment must look like without forcing an inheritance relationship, and
they let the Gymnasium adapter and the native environment satisfy the same
contract independently.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .spaces import Box


@runtime_checkable
class EnvProtocol(Protocol):
    """The Gymnasium ``Env`` API, five-tuple step convention.

    Any object satisfying this can be handed to the experiment runner. Keeping
    the signature byte-identical to Gymnasium means external libraries
    (stable-baselines3, CleanRL) work through a thin adapter with no shims.
    """

    observation_space: Box
    action_space: Box

    def reset(
        self, *, seed: int | None = ..., options: dict[str, Any] | None = ...
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode. Returns ``(observation, info)``."""
        ...

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one macro-step.

        Returns ``(obs, reward, terminated, truncated, info)``. ``info`` must
        carry ``"constraint_cost"`` (float) so constrained-RL agents can read it
        without knowing the environment's internals.
        """
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class VectorEnvProtocol(Protocol):
    """Batched environment interface, for on-policy algorithms that need throughput."""

    num_envs: int
    observation_space: Box
    action_space: Box

    def reset(
        self, *, seed: int | None = ..., options: dict[str, Any] | None = ...
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Auto-resetting batched step; terminal observations go in ``info``."""
        ...

    def close(self) -> None:
        ...
