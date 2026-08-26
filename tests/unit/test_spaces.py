"""Contract tests for the local ``Box`` space.

``Box`` exists so the package runs without gymnasium installed, which only
works if its semantics match gymnasium's exactly. These tests pin the parts
external code depends on: shape/dtype, ``contains`` tolerance, reproducible
sampling and value equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from propulsion_rl.core.spaces import Box
from propulsion_rl.core.types import CANONICAL_ACTION_DIM, OBS_DIM


def test_scalar_bounds_broadcast_to_the_requested_shape() -> None:
    """The action space is built from scalar bounds and a width; both bound
    arrays must materialise at full size for ``contains`` to work elementwise."""
    box = Box(-1.0, 1.0, shape=(CANONICAL_ACTION_DIM,))
    assert box.shape == (CANONICAL_ACTION_DIM,)
    assert box.dtype == np.float32
    assert box.low.shape == box.shape
    assert box.high.shape == box.shape
    assert np.all(box.low == -1.0)
    assert np.all(box.high == 1.0)


def test_shape_is_inferred_from_array_bounds() -> None:
    box = Box(np.zeros(4), np.arange(1.0, 5.0))
    assert box.shape == (4,)
    assert box.high == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_scalar_bounds_without_shape_are_rejected() -> None:
    """An unshaped scalar box is ambiguous; failing loudly beats guessing (1,)."""
    with pytest.raises(ValueError, match="shape"):
        Box(-1.0, 1.0)


def test_bounds_are_owned_copies() -> None:
    """``np.broadcast_to`` returns a read-only view; the constructor must copy,
    or two spaces built from one array would alias each other."""
    low = np.zeros(3)
    box = Box(low, np.ones(3))
    box.low[0] = -5.0
    assert low[0] == 0.0
    assert box.low.flags.writeable


def test_dtype_is_respected() -> None:
    box = Box(-1.0, 1.0, shape=(3,), dtype=np.float64)
    assert box.dtype == np.float64
    assert box.sample().dtype == np.float64


def test_contains_accepts_interior_and_boundary_points() -> None:
    box = Box(-1.0, 1.0, shape=(3,))
    assert box.contains(np.zeros(3, dtype=np.float32))
    assert box.contains(np.array([-1.0, 1.0, 0.5], dtype=np.float32))


def test_contains_rejects_out_of_bounds_and_wrong_shape() -> None:
    """Shape is part of membership: a (5,) action must not pass as a (3,) one."""
    box = Box(-1.0, 1.0, shape=(3,))
    assert not box.contains(np.array([0.0, 0.0, 1.5], dtype=np.float32))
    assert not box.contains(np.array([-2.0, 0.0, 0.0], dtype=np.float32))
    assert not box.contains(np.zeros(4, dtype=np.float32))
    assert not box.contains(np.zeros((1, 3), dtype=np.float32))


def test_contains_rejects_non_finite_values() -> None:
    """A NaN observation must never be reported as a member, otherwise a
    diverged rollout looks healthy to every downstream check."""
    box = Box(-1.0, 1.0, shape=(2,))
    assert not box.contains(np.array([np.nan, 0.0], dtype=np.float32))
    assert not box.contains(np.array([np.inf, 0.0], dtype=np.float32))


def test_contains_tolerates_float32_rounding_at_the_edge() -> None:
    """Casting a float64 bound to float32 can land a hair outside; the space
    absorbs that so a legal clipped action is not rejected."""
    box = Box(-1.0, 1.0, shape=(2,))
    assert box.contains(np.array([1.0 + 1e-7, -1.0 - 1e-7], dtype=np.float64))
    assert not box.contains(np.array([1.001, 0.0], dtype=np.float64))


def test_sample_stays_inside_the_box() -> None:
    box = Box(-2.0, 3.0, shape=(6,), seed=0)
    for _ in range(200):
        x = box.sample()
        assert x.shape == box.shape
        assert x.dtype == box.dtype
        assert box.contains(x)


def test_seeded_boxes_sample_reproducibly() -> None:
    """Reproducibility is load-bearing: an evaluation that samples random
    actions must replay identically from the same seed."""
    a = Box(-1.0, 1.0, shape=(OBS_DIM,), seed=1234)
    b = Box(-1.0, 1.0, shape=(OBS_DIM,), seed=1234)
    for _ in range(5):
        assert np.array_equal(a.sample(), b.sample())


def test_reseeding_restarts_the_stream() -> None:
    box = Box(-1.0, 1.0, shape=(4,), seed=7)
    first = [box.sample() for _ in range(3)]
    box.seed(7)
    second = [box.sample() for _ in range(3)]
    for x, y in zip(first, second):
        assert np.array_equal(x, y)


def test_different_seeds_give_different_samples() -> None:
    a = Box(-1.0, 1.0, shape=(8,), seed=1)
    b = Box(-1.0, 1.0, shape=(8,), seed=2)
    assert not np.array_equal(a.sample(), b.sample())


def test_equality_is_by_value_not_identity() -> None:
    """The vector env and the gym adapter compare spaces to check that sub-envs
    agree; identity comparison would reject two identical spaces."""
    assert Box(-1.0, 1.0, shape=(5,)) == Box(-1.0, 1.0, shape=(5,))
    assert Box(-1.0, 1.0, shape=(5,)) != Box(-1.0, 1.0, shape=(4,))
    assert Box(-1.0, 1.0, shape=(5,)) != Box(0.0, 1.0, shape=(5,))
    assert Box(-1.0, 1.0, shape=(5,)) != "not a box"


def test_seed_does_not_participate_in_equality() -> None:
    """Two identically bounded spaces are the same space regardless of the RNG
    state they happen to hold."""
    assert Box(-1.0, 1.0, shape=(3,), seed=1) == Box(-1.0, 1.0, shape=(3,), seed=2)


def test_repr_mentions_shape_and_dtype() -> None:
    text = repr(Box(-1.0, 1.0, shape=(3,)))
    assert "(3,)" in text
    assert "float32" in text
