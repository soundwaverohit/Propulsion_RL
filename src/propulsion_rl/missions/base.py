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

from ..core.constants import TINY_MASS_KG
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


#: A stage whose propellant is below this fraction of its launched wet mass is
#: not a marginal design, it is the wrong vehicle: every seed fails immediately
#: and for the same trivial reason, which measures nothing.
_MIN_USEFUL_PROPELLANT_FRACTION = 0.02


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

    #: Attribute holding the vehicle's own dry mass -- everything structural
    #: *except* the propulsion system, which is accounted for separately by
    #: :meth:`account_for_propulsion`. Missions that call the tank something
    #: else (a satellite has a bus, not a stage) override this name.
    dry_mass_attr: str = "stage_dry_mass_kg"

    #: Dry mass of the propulsion system this mission is flying, kg. Zero until
    #: the environment reports it; see :meth:`account_for_propulsion`.
    propulsion_dry_mass_kg: float = 0.0

    #: Gravitational parameter of the central body, m^3/s^2.
    mu: float = 0.0
    #: "heliocentric" or "planetocentric" -- selects the perturbation set.
    frame: str = "heliocentric"
    #: Wall-clock mission limit in seconds.
    max_duration_s: float = 0.0
    #: Seconds of simulated time per environment step (the RL macro-step).
    step_dt_s: float = 0.0

    # --- mass budget ---------------------------------------------------------
    #
    # Every mission here fixes the *launched* wet mass and the payload, and lets
    # the propellant load be whatever is left over. That is what makes a
    # cross-family comparison honest: an 18 t reactor and a 25 kg Hall thruster
    # do not get to fly the same amount of xenon just because the mission author
    # picked one stage mass. The propulsion system's own mass therefore has to
    # come out of the propellant, which means the mission needs to be told what
    # it is flying -- it is handed a registry name, not a vehicle.

    @property
    def vehicle_dry_mass_kg(self) -> float:
        """Structural dry mass excluding the propulsion system, kg."""
        return float(getattr(self, self.dry_mass_attr, 0.0))

    @property
    def total_dry_mass_kg(self) -> float:
        """Everything that is not propellant or payload, kg."""
        return self.vehicle_dry_mass_kg + float(self.propulsion_dry_mass_kg)

    def account_for_propulsion(self, dry_mass_kg: float) -> None:
        """Charge *dry_mass_kg* of propulsion hardware to the mass budget.

        Called once by the environment at construction, before the first
        ``reset``. Re-derives ``propellant_capacity_kg`` so a heavier stage
        genuinely flies with less propellant.

        A mission whose budget cannot absorb the system leaves the capacity at
        the floor and reports ``False`` from :meth:`mass_budget_closes`; it is
        the sweep matrix's job to refuse that pairing, not this method's to
        raise. Deciding feasibility is cheap and happens for every cell in the
        cross product, including ones nobody intends to run.
        """
        self.propulsion_dry_mass_kg = max(float(dry_mass_kg), 0.0)
        self._rebudget_mass()

    def _rebudget_mass(self) -> None:
        """Recompute ``propellant_capacity_kg`` from the current mass split.

        The default implements the wet-mass convention shared by every mission
        in this package. A mission that sizes its tank some other way overrides
        it; one that has no tank leaves it alone.
        """
        wet = getattr(self, "wet_mass_kg", None)
        payload = getattr(self, "payload_kg", None)
        if wet is None or payload is None:
            return
        self.propellant_capacity_kg = max(
            float(wet) - self.total_dry_mass_kg - float(payload), TINY_MASS_KG
        )

    def mass_budget_closes(self) -> bool:
        """Whether this vehicle can carry a useful propellant load at all.

        False when the propulsion system and payload have eaten the entire wet
        mass -- an 18 t reactor inside a 5 t comsat. Such a pairing is a
        category error rather than a hard control problem, and the matrix
        excludes it explicitly.
        """
        wet = getattr(self, "wet_mass_kg", None)
        payload = getattr(self, "payload_kg", None)
        if wet is None or payload is None:
            return True
        usable = float(wet) - self.total_dry_mass_kg - float(payload)
        return usable > _MIN_USEFUL_PROPELLANT_FRACTION * float(wet)

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

    def observation_scales(self) -> dict[str, float]:
        """Physical scale of normalised channels, by canonical channel name.

        ``observe_raw`` divides physical quantities by a normalisation of the
        mission's own choosing, so a controller that needs the quantity back in
        SI -- Edelbaum's steering law is a function of the plane change owed in
        *radians*, not of a number between zero and one -- has to know what that
        normalisation was. Guessing it is the failure mode this exists to close:
        assuming a channel is scaled by pi/2 when the mission scaled it by the
        28.5 degree starting inclination mis-states the plane change by a factor
        of three, and the resulting steering is wrong in a way that still looks
        entirely plausible in a telemetry plot.

        Returns an empty mapping by default; controllers keep their own defaults
        for anything a mission does not publish.
        """
        return {}

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
