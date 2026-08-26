"""Environments: the RL-facing surface of the benchmark.

``make_env("hall_spt100", "leo_geo_transfer")`` is the entry point the whole
experiment matrix goes through. Everything else here is optional scaffolding
around it -- wrappers, a batched runner, and Gymnasium interop.

The Gymnasium adapter is safe to import without gymnasium installed; it only
needs the package when you actually call it.
"""

from __future__ import annotations

from .gym_adapter import gym_env_id, register_gym_envs, to_gymnasium
from .propulsion_env import EnvConfig, PropulsionEnv, make_env
from .vector_env import AsyncVectorEnv, SyncVectorEnv, make_vector_env
from .wrappers import (
    ActionRepeat,
    ClipAction,
    ConstraintWrapper,
    CostAsInfo,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
    RunningMeanStd,
    TimeLimit,
    Wrapper,
    wrap,
)

__all__ = [
    "ActionRepeat",
    "AsyncVectorEnv",
    "ClipAction",
    "ConstraintWrapper",
    "CostAsInfo",
    "EnvConfig",
    "NormalizeObservation",
    "NormalizeReward",
    "PropulsionEnv",
    "RecordEpisodeStatistics",
    "RunningMeanStd",
    "SyncVectorEnv",
    "TimeLimit",
    "Wrapper",
    "gym_env_id",
    "make_env",
    "make_vector_env",
    "register_gym_envs",
    "to_gymnasium",
    "wrap",
]
