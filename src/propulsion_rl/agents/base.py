"""The agent contract.

Deliberately narrow so that a hand-written PID controller, a PPO network, and a
CEM planner are interchangeable inside the experiment matrix. Anything an
algorithm needs beyond this interface (replay buffers, optimisers, model
ensembles) is its own business.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class Transition:
    """One environment transition, the unit every learner consumes."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    cost: float = 0.0                     # constraint cost, for Lagrangian methods
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainStats:
    """Whatever the learner wants to report. Averaged and logged by the runner."""

    values: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k: str) -> float:
        return self.values[k]

    def update(self, **kw: float) -> "TrainStats":
        self.values.update(kw)
        return self


class Agent(ABC):
    """Base class for every controller, learned or scripted.

    Contract notes
    --------------
    * ``act`` must be deterministic when ``deterministic=True``. Evaluation runs
      set it; a policy that ignores the flag makes its own eval numbers noisy
      and its comparison against scripted baselines unfair.
    * ``observe_transition`` is called for every step regardless of whether the
      agent learns online. Scripted baselines ignore it.
    * ``update`` is called on the runner's cadence and returns stats. Agents
      that learn only at episode boundaries should buffer and no-op otherwise.
    * ``learns`` is what the runner uses to decide whether to spend a training
      budget on this agent at all -- keep it accurate.
    """

    name: str = "abstract"
    #: False for scripted controllers, which skip the training loop entirely.
    learns: bool = True
    #: True if the agent consumes ``Transition.cost`` (constrained RL).
    uses_constraints: bool = False

    def __init__(self, obs_dim: int, action_dim: int, **kwargs: Any) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = dict(kwargs)

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Map an observation to an action in [-1, 1]^action_dim."""

    def observe_transition(self, tr: Transition) -> None:
        """Ingest a transition. Default: ignore (scripted controllers)."""

    def update(self) -> TrainStats:
        """Run a learning step if it is time to. Default: no-op."""
        return TrainStats()

    def on_episode_end(self, episode_return: float, info: dict[str, Any]) -> None:
        """Hook for episodic algorithms (CMA-ES, evolutionary search)."""

    def reset(self) -> None:
        """Clear per-episode internal state (controller integrators, plan caches)."""

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist parameters. Default: nothing to persist."""

    def load(self, path: str | Path) -> None:
        """Restore parameters saved by :meth:`save`."""

    def set_seed(self, seed: int) -> None:
        """Seed every source of randomness the agent owns."""

    @property
    def num_parameters(self) -> int:
        """Learnable parameter count, reported in the comparison table."""
        return 0

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} learns={self.learns}>"


class ScriptedAgent(Agent):
    """Convenience base for hand-written controllers.

    These are the reference points the whole study rests on: an RL method that
    cannot beat a well-tuned scripted controller has not earned its complexity.
    """

    learns = False

    def update(self) -> TrainStats:
        return TrainStats()
