"""Minimal Gymnasium-compatible spaces.

Implemented locally so the package has no hard dependency on ``gymnasium``.
The attribute names and ``sample``/``contains`` semantics match Gymnasium's
``Box`` exactly, so :mod:`propulsion_rl.envs.gym_adapter` can swap in the real
class when it is installed.
"""

from __future__ import annotations

import numpy as np


class Box:
    """A closed box in R^n, matching ``gymnasium.spaces.Box``."""

    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float32,
        seed: int | None = None,
    ) -> None:
        if shape is None:
            if np.isscalar(low) and np.isscalar(high):
                raise ValueError("shape is required when low and high are scalars")
            shape = np.broadcast(np.asarray(low), np.asarray(high)).shape
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), self.shape).copy()
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        # Bounded case only; every space in this project is bounded.
        return self._rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return bool(
            x.shape == self.shape
            and np.all(x >= self.low - 1e-6)
            and np.all(x <= self.high + 1e-6)
        )

    def __repr__(self) -> str:
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )
