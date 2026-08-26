"""Vehicle configuration, state construction, and the vehicle observation block.

The split of responsibility: the *mission* owns where the vehicle is going, the
*propulsion system* owns what it can do about it, and this module owns what the
vehicle physically is -- masses, tankage, and the normalised summary of its own
condition that the policy sees every step.

``vehicle_observation`` is one of three blocks concatenated into the fixed
36-wide observation vector. It must stay exactly ``VEHICLE_OBS_DIM`` values in a
stable order forever: a policy trained against a Hall thruster is evaluated
zero-shot against an NTR, and that only means anything if index 3 is the same
quantity in both runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..core.constants import G0, YEAR
from ..core.types import VEHICLE_OBS_DIM, VehicleState

logger = logging.getLogger(__name__)

__all__ = [
    "VehicleConfig",
    "build_initial_state",
    "vehicle_observation",
    "vehicle_observation_labels",
    "mass_budget",
]


@dataclass(slots=True)
class VehicleConfig:
    """Static physical description of the vehicle.

    Mass bookkeeping
    ----------------
    ``dry_mass_kg`` is the bus: structure, avionics, thermal, power electronics,
    *excluding* propellant tankage and excluding the payload. Tankage is derived
    from ``propellant_capacity_kg`` via ``structure_mass_fraction`` so that
    sizing sweeps over tank size stay self-consistent -- a 5 t xenon tank is not
    free. The propulsion system's own dry mass is reported separately through
    its :class:`~propulsion_rl.core.types.BillOfMaterials` and priced there, so
    do not double count it here.

    The ``reference_*`` fields exist purely to normalise the observation block.
    They are scale factors, not limits: nothing enforces them, and a policy
    seeing an out-of-range value is a signal that the reference is badly chosen
    rather than that the vehicle is broken.
    """

    dry_mass_kg: float
    payload_kg: float = 0.0
    propellant_capacity_kg: float = 0.0
    housekeeping_power_w: float = 0.0
    structure_mass_fraction: float = 0.10
    name: str = "generic"
    initial_propellant_kg: float | None = None      # None -> full tank
    reference_power_w: float = 1.0e4
    reference_duration_s: float = YEAR
    reference_delta_v_m_s: float = 1.0e4
    reference_isp_s: float = 2000.0
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dry_mass_kg <= 0.0:
            raise ValueError("dry_mass_kg must be positive")
        for nm in (
            "payload_kg",
            "propellant_capacity_kg",
            "housekeeping_power_w",
            "structure_mass_fraction",
        ):
            if getattr(self, nm) < 0.0:
                raise ValueError(f"{nm} must be non-negative")
        if self.initial_propellant_kg is None:
            self.initial_propellant_kg = self.propellant_capacity_kg
        elif not 0.0 <= self.initial_propellant_kg <= self.propellant_capacity_kg:
            raise ValueError(
                "initial_propellant_kg must lie within [0, propellant_capacity_kg]"
            )
        for nm in (
            "reference_power_w",
            "reference_duration_s",
            "reference_delta_v_m_s",
            "reference_isp_s",
        ):
            if getattr(self, nm) <= 0.0:
                raise ValueError(f"{nm} must be positive")

    # --- derived -------------------------------------------------------------
    @property
    def tank_mass_kg(self) -> float:
        """Tankage and supporting structure implied by the propellant capacity."""
        return self.structure_mass_fraction * self.propellant_capacity_kg

    @property
    def total_dry_mass_kg(self) -> float:
        """Bus plus tankage; what ``VehicleState.dry_mass_kg`` carries."""
        return self.dry_mass_kg + self.tank_mass_kg

    @property
    def initial_wet_mass_kg(self) -> float:
        return self.total_dry_mass_kg + self.payload_kg + float(self.initial_propellant_kg)

    @property
    def final_dry_mass_kg(self) -> float:
        """Mass once the tank is empty."""
        return self.total_dry_mass_kg + self.payload_kg

    @property
    def propellant_mass_fraction(self) -> float:
        wet = self.initial_wet_mass_kg
        return float(self.initial_propellant_kg) / wet if wet > 0.0 else 0.0

    def ideal_delta_v_m_s(self, isp_s: float | None = None) -> float:
        """Tsiolkovsky budget for a full tank at ``isp_s`` (default reference)."""
        isp = self.reference_isp_s if isp_s is None else isp_s
        m0 = self.initial_wet_mass_kg
        m1 = self.final_dry_mass_kg
        if m1 <= 0.0 or m0 <= m1:
            return 0.0
        return isp * G0 * math.log(m0 / m1)


def mass_budget(config: VehicleConfig) -> dict[str, float]:
    """Itemised mass breakdown, for reporting and for the economics model."""
    return {
        "bus_dry_kg": config.dry_mass_kg,
        "tank_kg": config.tank_mass_kg,
        "payload_kg": config.payload_kg,
        "propellant_kg": float(config.initial_propellant_kg),
        "wet_kg": config.initial_wet_mass_kg,
        "propellant_mass_fraction": config.propellant_mass_fraction,
    }


def build_initial_state(
    config: VehicleConfig,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    t0: float = 0.0,
) -> VehicleState:
    """Assemble the start-of-episode :class:`VehicleState`.

    Position and velocity are copied into fresh contiguous float64 arrays, so a
    mission can hand over a view of its own buffer without the integrator later
    writing through it.
    """
    pos = np.array(position_m, dtype=np.float64).reshape(3)
    vel = np.array(velocity_m_s, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        raise ValueError("build_initial_state: non-finite initial state vector")
    return VehicleState(
        position_m=pos,
        velocity_m_s=vel,
        dry_mass_kg=config.total_dry_mass_kg,
        propellant_kg=float(config.initial_propellant_kg),
        payload_kg=config.payload_kg,
        t_s=float(t0),
        power_generated_w=0.0,
        power_available_w=0.0,
        delta_v_applied_m_s=0.0,
        propellant_used_kg=0.0,
    )


# --- Observation block -------------------------------------------------------
_LABELS: tuple[str, ...] = (
    "mass_fraction",            # current wet mass / initial wet mass
    "propellant_fraction",      # remaining propellant / tank capacity
    "power_available_frac",     # bus power offered to propulsion / reference
    "power_generated_frac",     # raw generation / reference (eclipse shows here)
    "elapsed_time_frac",        # mission elapsed / reference duration
    "delta_v_frac",             # delta-v spent / reference budget
    "delta_v_remaining_frac",   # Tsiolkovsky capability left / reference budget
    "in_sunlight",              # +1 lit, -1 dark
)


def vehicle_observation_labels() -> tuple[str, ...]:
    """Names for the entries of :func:`vehicle_observation`, in order."""
    return _LABELS


def vehicle_observation(state: VehicleState, config: VehicleConfig) -> np.ndarray:
    """The ``VEHICLE_OBS_DIM``-wide normalised vehicle block.

    Every entry is mapped to roughly [-1, 1]: bounded fractions via ``2x - 1``
    so that a half-full tank reads 0, and the sunlight flag as a clean +-1. The
    unbounded ones (time, delta-v) are clipped one unit past the reference so a
    mission that overruns its nominal duration still produces a finite, ordered
    observation rather than an exploding input.

    ``delta_v_remaining_frac`` is the piece a policy cannot infer cheaply from
    the rest: the rocket equation applied to what is left in the tank. Without
    it, "should I burn now or save it" requires the network to learn a logarithm
    from scratch.
    """
    out = np.empty(VEHICLE_OBS_DIM, dtype=np.float32)

    m0 = config.initial_wet_mass_kg
    mass = state.total_mass_kg
    out[0] = _unit(mass / m0 if m0 > 0.0 else 0.0)

    cap = config.propellant_capacity_kg
    prop_frac = state.propellant_kg / cap if cap > 0.0 else 0.0
    out[1] = _unit(prop_frac)

    pref = config.reference_power_w
    out[2] = _unit(state.power_available_w / pref)
    out[3] = _unit(state.power_generated_w / pref)

    out[4] = _unit(state.t_s / config.reference_duration_s, hi=2.0)

    dv_ref = config.reference_delta_v_m_s
    out[5] = _unit(state.delta_v_applied_m_s / dv_ref, hi=2.0)

    m_dry = config.final_dry_mass_kg
    if state.propellant_kg > 0.0 and m_dry > 0.0:
        dv_left = config.reference_isp_s * G0 * math.log(mass / m_dry)
    else:
        dv_left = 0.0
    out[6] = _unit(dv_left / dv_ref, hi=2.0)

    out[7] = 1.0 if state.power_generated_w > 0.0 else -1.0
    return out


def _unit(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Map ``[lo, hi]`` onto [-1, 1] and clip. Non-finite input reads as -1."""
    if not math.isfinite(x):
        return -1.0
    y = 2.0 * (x - lo) / (hi - lo) - 1.0
    if y < -1.0:
        return -1.0
    return 1.0 if y > 1.0 else y
