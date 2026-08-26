"""Shared data contracts.

Every module in this package talks through the types defined here. Physics
modules, mission modules, the environment, the economics model and the agents
must not invent parallel structures -- extend these instead.

Design rule: these are frozen-ish dataclasses of plain floats and small arrays.
No behaviour beyond derived properties and validation, so they stay cheap to
construct inside a hot RL loop (millions of steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from .constants import EPS, G0

# --- Canonical interface widths ---------------------------------------------
# Fixed so that a policy trained on one propulsion system can be evaluated
# zero-shot on another. Blocks are zero-padded, never re-ordered.
CANONICAL_ACTION_DIM = 5
MISSION_OBS_DIM = 12
VEHICLE_OBS_DIM = 8
PROPULSION_OBS_DIM = 16
OBS_DIM = MISSION_OBS_DIM + VEHICLE_OBS_DIM + PROPULSION_OBS_DIM  # 36


class PropulsionFamily(str, Enum):
    ELECTRIC = "electric"
    NUCLEAR = "nuclear"


class TerminationReason(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    OUT_OF_PROPELLANT = "out_of_propellant"
    HARDWARE_FAILURE = "hardware_failure"
    SAFETY_VIOLATION = "safety_violation"
    TIMEOUT = "timeout"
    DIVERGED = "diverged"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"      # constraint breached, recoverable
    FATAL = "fatal"            # hardware destroyed, episode over


# --- Actions -----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CanonicalCommand:
    """Propulsion-agnostic command, the only thing an agent ever emits.

    Each propulsion system maps these onto its native actuators via
    ``PropulsionSystem.decode_action``. Keeping one canonical space is what
    makes the (agent x propulsion) comparison matrix meaningful.

    Attributes
    ----------
    throttle:
        [0, 1]. Fraction of the currently *allowable* operating envelope, not of
        nameplate rating. 0 means commanded off.
    operating_point:
        [0, 1]. The Isp/thrust trade knob. 0 = high thrust / low Isp end of the
        envelope, 1 = low thrust / high Isp end. Maps to discharge voltage on an
        electric thruster and to chamber temperature on a thermal one.
    thrust_yaw:
        [-pi, pi] rad. In-plane thrust direction, measured from the local
        velocity vector, positive towards the outward radial direction.
    thrust_pitch:
        [-pi/2, pi/2] rad. Out-of-plane thrust direction, positive towards the
        orbit normal.
    thermal_margin:
        [0, 1]. Fraction of the coolant/flow-margin budget spent on protecting
        hardware instead of on performance. Radiator flow and cathode flow
        margin for EP, coolant pump speed and turbine bypass for nuclear.
    """

    throttle: float
    operating_point: float
    thrust_yaw: float
    thrust_pitch: float
    thermal_margin: float

    @staticmethod
    def from_array(a: np.ndarray) -> "CanonicalCommand":
        """Decode a raw policy output in [-1, 1]^5 into physical command units.

        Values outside [-1, 1] are clipped rather than rejected: policies with
        unbounded heads (e.g. a Gaussian before squashing) are common, and a
        hard error inside the rollout loop would be worse than saturation.
        """
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.size != CANONICAL_ACTION_DIM:
            raise ValueError(
                f"expected {CANONICAL_ACTION_DIM} action components, got {a.size}"
            )
        return CanonicalCommand(
            throttle=float((a[0] + 1.0) * 0.5),
            operating_point=float((a[1] + 1.0) * 0.5),
            thrust_yaw=float(a[2] * np.pi),
            thrust_pitch=float(a[3] * np.pi * 0.5),
            thermal_margin=float((a[4] + 1.0) * 0.5),
        )

    def to_array(self) -> np.ndarray:
        """Inverse of :meth:`from_array`, for scripted controllers and tests."""
        return np.array(
            [
                self.throttle * 2.0 - 1.0,
                self.operating_point * 2.0 - 1.0,
                self.thrust_yaw / np.pi,
                self.thrust_pitch / (np.pi * 0.5),
                self.thermal_margin * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )

    def direction_rtn(self) -> np.ndarray:
        """Unit thrust direction in the local RTN (radial/transverse/normal) frame.

        yaw is measured from the transverse (velocity-ish) axis towards radial,
        so yaw=0, pitch=0 is a pure prograde burn -- the sensible default that a
        randomly initialised policy outputting zeros will produce.
        """
        cp = np.cos(self.thrust_pitch)
        return np.array(
            [
                cp * np.sin(self.thrust_yaw),   # radial
                cp * np.cos(self.thrust_yaw),   # transverse
                np.sin(self.thrust_pitch),      # normal
            ],
            dtype=np.float64,
        )


# --- Vehicle / environment context passed down to the propulsion model -------
@dataclass(slots=True)
class StepContext:
    """Everything a propulsion system needs to know about the outside world.

    Assembled by the environment once per step and handed to the propulsion
    model, so propulsion code never reaches back into the vehicle or mission.
    """

    t_s: float                      # mission elapsed time
    dt_s: float                     # integration interval for this step
    vehicle_mass_kg: float          # current total wet mass
    available_power_w: float        # electrical power the bus can supply now
    heliocentric_radius_m: float    # for solar flux
    sink_temperature_k: float = 3.0     # radiative sink (deep space)
    eclipse: bool = False           # in planetary shadow -> no solar power
    rng: np.random.Generator | None = None   # for stochastic wear/faults


@dataclass(slots=True)
class ThrusterOutput:
    """What the propulsion system produced over one step.

    ``thrust_n`` is the achieved magnitude, which may be below what was
    commanded when power, thermal or propellant limits bind. The environment
    applies the direction from the command; the propulsion system only decides
    magnitude, flow and the resource draw.
    """

    thrust_n: float = 0.0
    mdot_kg_s: float = 0.0
    isp_s: float = 0.0
    power_draw_w: float = 0.0        # electrical draw from the bus
    thermal_power_w: float = 0.0     # reactor/plasma thermal power generated
    heat_reject_w: float = 0.0       # waste heat that must be radiated
    efficiency: float = 0.0          # total thrust efficiency, [0, 1]
    throttled_by: str = "none"       # which limit bound: power/thermal/prop/life
    events: list["Event"] = field(default_factory=list)

    @property
    def jet_power_w(self) -> float:
        """Kinetic power in the exhaust beam."""
        return 0.5 * self.mdot_kg_s * (self.isp_s * G0) ** 2


@dataclass(frozen=True, slots=True)
class Event:
    """A discrete thing that happened, for logging and reward shaping."""

    name: str
    severity: Severity
    detail: str = ""
    value: float = 0.0


@dataclass(slots=True)
class HealthReport:
    """Degradation state, consumed by both the reward and the economics model."""

    wear_fraction: float = 0.0        # [0, 1]; 1.0 == end of qualified life
    remaining_life_s: float = np.inf
    throughput_kg: float = 0.0        # propellant processed (EP life metric)
    burn_time_s: float = 0.0          # accumulated firing time
    restarts: int = 0
    degraded_efficiency: float = 1.0  # multiplier on nominal efficiency
    failed: bool = False


@dataclass(slots=True)
class Limits:
    """Static envelope of a propulsion system. Used for normalisation, for
    scripted baselines, and by the economics model for sizing."""

    max_thrust_n: float
    min_thrust_n: float
    max_power_w: float
    min_power_w: float
    isp_range_s: tuple[float, float]
    max_temperature_k: float
    qualified_life_s: float
    max_throughput_kg: float = np.inf
    max_restarts: int = 1_000_000
    min_off_time_s: float = 0.0       # forced cooldown / poison-decay dwell


# --- Constraint accounting ---------------------------------------------------
@dataclass(slots=True)
class ConstraintReport:
    """Signed margins on safety constraints.

    Convention: ``margins[i] >= 0`` is safe, ``< 0`` is a violation, and the
    magnitude is normalised by the limit so values are comparable across
    constraints. Constrained-RL agents (Lagrangian PPO and friends) consume
    ``cost`` directly; unconstrained agents see it folded into the reward.
    """

    names: tuple[str, ...] = ()
    margins: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def violated(self) -> bool:
        return bool(self.margins.size and np.any(self.margins < 0.0))

    @property
    def cost(self) -> float:
        """Total normalised violation magnitude this step (0 when all satisfied)."""
        if not self.margins.size:
            return 0.0
        return float(np.sum(np.clip(-self.margins, 0.0, None)))

    @property
    def worst(self) -> float:
        return float(np.min(self.margins)) if self.margins.size else np.inf


# --- Vehicle state -----------------------------------------------------------
@dataclass(slots=True)
class VehicleState:
    """Translational state plus mass and power bookkeeping.

    Position and velocity are in the mission's inertial frame (heliocentric
    ecliptic for interplanetary, planet-centred inertial for orbit raising).
    """

    position_m: np.ndarray            # (3,)
    velocity_m_s: np.ndarray          # (3,)
    dry_mass_kg: float
    propellant_kg: float
    payload_kg: float = 0.0
    t_s: float = 0.0
    power_generated_w: float = 0.0
    power_available_w: float = 0.0    # after housekeeping load
    delta_v_applied_m_s: float = 0.0
    propellant_used_kg: float = 0.0

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_kg + self.payload_kg

    @property
    def radius_m(self) -> float:
        return float(np.linalg.norm(self.position_m))

    @property
    def speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocity_m_s))

    @property
    def specific_energy(self) -> float:
        """Orbital specific energy, mu-free part; the caller supplies mu."""
        return 0.5 * self.speed_m_s**2

    def copy(self) -> "VehicleState":
        return replace(
            self,
            position_m=self.position_m.copy(),
            velocity_m_s=self.velocity_m_s.copy(),
        )


# --- Bill of materials, the bridge to the economics model --------------------
@dataclass(slots=True)
class BillOfMaterials:
    """Physical quantities the cost model prices.

    Propulsion systems report *what they are*, never dollars. All pricing lives
    in :mod:`propulsion_rl.economics` so cost assumptions can be swept
    independently of the physics.
    """

    system_name: str
    family: PropulsionFamily
    thruster_units: int = 1
    rated_power_w: float = 0.0
    power_source_w: float = 0.0         # solar array or reactor electrical output
    reactor_thermal_w: float = 0.0
    radiator_area_m2: float = 0.0
    dry_mass_kg: float = 0.0
    propellant_type: str = "xenon"
    tank_capacity_kg: float = 0.0
    qualified_life_s: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


# --- Per-step telemetry ------------------------------------------------------
@dataclass(slots=True)
class Telemetry:
    """Flat, append-friendly record of one environment step.

    Written once per step into the episode log; the analysis layer turns a list
    of these into a DataFrame. Keep it flat -- nested structures make the
    downstream pivot tables painful.
    """

    t_s: float = 0.0
    step: int = 0
    thrust_n: float = 0.0
    isp_s: float = 0.0
    mdot_kg_s: float = 0.0
    power_draw_w: float = 0.0
    power_available_w: float = 0.0
    efficiency: float = 0.0
    mass_kg: float = 0.0
    propellant_kg: float = 0.0
    radius_m: float = 0.0
    speed_m_s: float = 0.0
    delta_v_m_s: float = 0.0
    wear_fraction: float = 0.0
    constraint_cost: float = 0.0
    worst_margin: float = 0.0
    reward: float = 0.0
    progress: float = 0.0
    throttled_by: str = "none"
    extras: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__ if k != "extras"}
        d.update(self.extras)
        return d


def zeros_obs_block(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def pad_to(vec: np.ndarray, dim: int, name: str = "block") -> np.ndarray:
    """Right-pad (or reject) an observation block to its contracted width."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size > dim:
        raise ValueError(f"{name} emitted {v.size} values, contract allows {dim}")
    if v.size == dim:
        return v
    out = np.zeros(dim, dtype=np.float32)
    out[: v.size] = v
    return out


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if abs(den) > EPS else default
