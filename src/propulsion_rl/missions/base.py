"""The mission contract.

A ``Mission`` defines the task: the gravitational environment, the initial
vehicle state, what counts as progress, what counts as done, and the shape of
the reward. Swapping missions must not require touching propulsion or agent
code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.types import (
    MISSION_OBS_DIM,
    ConstraintReport,
    HealthReport,
    TerminationReason,
    ThrusterOutput,
    VehicleState,
    pad_to,
)


@dataclass(slots=True)
class MissionResult:
    """Outcome summary produced once, at episode end. Feeds the economics model."""

    reason: TerminationReason = TerminationReason.RUNNING
    success: bool = False
    progress: float = 0.0                 # [0, 1] fraction of the goal achieved
    elapsed_s: float = 0.0
    delta_v_m_s: float = 0.0
    propellant_used_kg: float = 0.0
    payload_delivered_kg: float = 0.0
    terminal_error: float = 0.0           # mission-specific miss distance metric
    constraint_violations: int = 0
    total_constraint_cost: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RewardTerms:
    """Decomposed reward, so ablations do not require re-running the sweep.

    ``total`` is what the agent optimises. Keeping the parts separate lets the
    analysis layer attribute a policy's behaviour to progress-seeking versus
    propellant-hoarding versus constraint-dodging.
    """

    progress: float = 0.0
    efficiency: float = 0.0
    time_penalty: float = 0.0
    propellant_penalty: float = 0.0
    wear_penalty: float = 0.0
    constraint_penalty: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.efficiency
            + self.time_penalty
            + self.propellant_penalty
            + self.wear_penalty
            + self.constraint_penalty
            + self.terminal
        )

    def as_dict(self) -> dict[str, float]:
        d = {k: getattr(self, k) for k in self.__slots__}
        d["total"] = self.total
        return d


class Mission(ABC):
    """Base class for mission scenarios.

    Contract notes
    --------------
    * ``mu`` and ``frame`` fix the dynamics; the integrator reads them.
    * ``reward`` is called once per step *after* the state has been advanced,
      and must be a pure function of the arguments plus internal bookkeeping
      that ``reset`` clears. No hidden global state.
    * ``max_duration_s`` is a hard wall. The environment enforces it; missions
      should not also test for it.
    """

    name: str = "abstract"
    #: Gravitational parameter of the central body, m^3/s^2.
    mu: float = 0.0
    #: "heliocentric" or "planetocentric" -- selects the perturbation set.
    frame: str = "heliocentric"
    #: Wall-clock mission limit in seconds.
    max_duration_s: float = 0.0
    #: Seconds of simulated time per environment step (the RL macro-step).
    step_dt_s: float = 0.0

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Sample an initial vehicle state. Randomisation lives here, not in the env."""

    @abstractmethod
    def observe_raw(self, state: VehicleState) -> np.ndarray:
        """Mission-relevant observation block, normalised to ~[-1, 1].

        Must include enough of the target geometry that the task is Markovian:
        a policy that cannot see where it is going cannot be blamed for failing.
        """

    def observe(self, state: VehicleState) -> np.ndarray:
        return pad_to(self.observe_raw(state), MISSION_OBS_DIM, f"{self.name} obs")

    @abstractmethod
    def observation_labels(self) -> tuple[str, ...]:
        """Names matching :meth:`observe_raw`."""

    @abstractmethod
    def progress(self, state: VehicleState) -> float:
        """Monotone-ish [0, 1] completion measure. 1.0 means the goal is met."""

    @abstractmethod
    def reward(
        self,
        prev: VehicleState,
        state: VehicleState,
        output: ThrusterOutput,
        constraints: ConstraintReport,
        health: HealthReport,
    ) -> RewardTerms:
        """Per-step reward decomposition."""

    @abstractmethod
    def terminated(
        self, state: VehicleState, health: HealthReport, constraints: ConstraintReport
    ) -> TerminationReason:
        """Return ``RUNNING`` to continue, anything else to end the episode."""

    @abstractmethod
    def summarize(self, state: VehicleState, reason: TerminationReason) -> MissionResult:
        """Build the end-of-episode record consumed by the economics model."""

    # --- optional hooks ------------------------------------------------------
    def gravity(self, state: VehicleState) -> np.ndarray:
        """Acceleration from gravity at the current state, m/s^2.

        Default is a point-mass central body. Override to add third bodies, J2,
        or solar radiation pressure.
        """
        r = state.position_m
        rn = float(np.linalg.norm(r))
        if rn < 1.0:
            return np.zeros(3)
        return -self.mu * r / rn**3

    def eclipse(self, state: VehicleState) -> bool:
        """Whether the vehicle is shadowed. Matters for solar-powered EP."""
        return False

    def heliocentric_radius_m(self, state: VehicleState) -> float:
        """Distance to the Sun. Identical to ``radius_m`` in a heliocentric frame."""
        return state.radius_m

    def info(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} frame={self.frame}>"
