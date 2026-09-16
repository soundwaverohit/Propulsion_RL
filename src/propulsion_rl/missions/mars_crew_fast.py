"""Crewed fast Mars transit -- where nuclear thermal should win.

Same heliocentric geometry as :mod:`~.earth_mars_cargo`, and deliberately so:
holding the astrodynamics fixed is what isolates the thing that actually
changes, which is that there are people aboard.

Two constraints follow from that, and together they invert the answer:

* **A hard trip-time limit** (~220 days by default). Arriving on day 221 is not
  a partial success, it is a failure, and the reward says so -- there is no
  arrival bonus and a full failure malus.
* **A crew dose budget that grows with time.** Galactic cosmic ray dose
  accumulates every second of the cruise. Flying slower to save propellant
  spends the one resource that cannot be replenished.

Shielding is carried as real mass inside the stage's dry mass, so it is paid for
in the rocket equation rather than in a fudge term, plus a small explicit
terminal deduction so an ablation can see it.

A solar-electric stage should *fail to close this at all*, and that is the
result, not a bug: at a few hundred kW on a 120 t vehicle the thrust-to-weight
is ~1e-4 g, which cannot produce a 200-day transfer no matter how much delta-v
capacity the tanks hold. The benchmark needs to be able to show that asymmetry.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.constants import AU, DAY, EARTH_ORBIT_PERIOD, EARTH_SMA, HOUR, MU_SUN, TINY_MASS_KG
from ..core.registry import MISSION
from ..core.types import (
    ConstraintReport,
    HealthReport,
    TerminationReason,
    VehicleState,
)
from ..spacecraft.orbit import edelbaum_delta_v, planet_state
from .base import MissionResult
from .earth_mars_cargo import MARS_SMA, _HeliocentricTransfer
from .rewards import TWO_PI, RewardConfig, RewardScales, clamp01, to_symmetric

logger = logging.getLogger(__name__)

#: Unshielded free-space GCR dose equivalent rate, Sv/day, near solar minimum.
_GCR_BASE_SV_DAY = 2.5e-3
#: Areal-density scale of the shield, kg. Dose falls as exp(-m_shield / scale).
_SHIELD_SCALE_KG = 6000.0
#: GCR flux rises with heliocentric distance (weaker solar modulation).
_GCR_RADIAL_GAIN = 0.15

#: Departure phase for a ~200 day type-I transfer: the vehicle sweeps ~150 deg
#: while Mars sweeps ~110 deg, so Mars leads by ~40 deg at departure.
FAST_TRANSIT_PHASE_RAD = math.radians(40.0)


@MISSION.register(
    "mars_crew_fast",
    frame="heliocentric",
    regime="interplanetary_rendezvous",
    favours="nuclear",
    horizon_days=220.0,
)
class MarsCrewFast(_HeliocentricTransfer):
    """Deliver a crew to Mars inside a hard trip-time and dose budget.

    Arrival tolerance is deliberately looser than the cargo mission's: a crewed
    stage arrives with several km/s of v-infinity and captures propulsively or
    aerodynamically, which is a separate leg. What is *not* loose is the clock.
    """

    name = "mars_crew_fast"

    def __init__(
        self,
        *,
        wet_mass_kg: float = 120_000.0,
        stage_dry_mass_kg: float = 25_000.0,
        shield_mass_kg: float = 5_000.0,
        payload_kg: float = 30_000.0,
        step_dt_s: float = HOUR,
        trip_time_limit_s: float = 220.0 * DAY,
        arrival_radius_m: float = 6.0e8,
        arrival_speed_m_s: float = 5_000.0,
        crew_dose_budget_sv: float = 0.28,
        launch_phase_center_rad: float = FAST_TRANSIT_PHASE_RAD,
        launch_phase_window_rad: float = math.radians(35.0),
        injection_error_m_s: float = 50.0,
        config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.wet_mass_kg = float(wet_mass_kg)
        #: Shielding is *inside* ``stage_dry_mass_kg``; it is broken out only so
        #: the economics model and the ablation can see it.
        self.shield_mass_kg = float(shield_mass_kg)
        self.stage_dry_mass_kg = float(stage_dry_mass_kg)
        self.payload_kg = float(payload_kg)
        self.propellant_capacity_kg = max(
            wet_mass_kg - stage_dry_mass_kg - payload_kg, TINY_MASS_KG
        )
        self.step_dt_s = float(step_dt_s)
        self.trip_time_limit_s = float(trip_time_limit_s)
        # The wall and the requirement are the same instant: the environment's
        # timeout *is* the mission failure, so nothing has to test for it twice.
        self.max_duration_s = float(trip_time_limit_s)
        self.arrival_radius_m = float(arrival_radius_m)
        self.arrival_speed_m_s = float(arrival_speed_m_s)
        self.crew_dose_budget_sv = float(crew_dose_budget_sv)
        self.launch_phase_center_rad = float(launch_phase_center_rad)
        self.launch_phase_window_rad = float(launch_phase_window_rad)
        self.injection_error_m_s = float(injection_error_m_s)

        self.shielded_dose_rate_sv_day = _GCR_BASE_SV_DAY * math.exp(
            -self.shield_mass_kg / _SHIELD_SCALE_KG
        )

        # Schedule dominates. Propellant still counts -- 65 t of hydrogen is not
        # free -- but nothing about this mission is worth being late for.
        self.config = config or RewardConfig(
            w_time=1.50,
            w_propellant=0.25,
            w_constraint=1.25,
            isp_reference_s=900.0,
        )
        self._scales = RewardScales(
            duration_s=self.max_duration_s,
            propellant_capacity_kg=self.propellant_capacity_kg,
            delta_v_reference_m_s=edelbaum_delta_v(EARTH_SMA, MARS_SMA, 0.0, 0.0, MU_SUN),
        )
        self._dose_sv = 0.0

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Sample a crewed departure window.

        Same randomisation contract as the cargo mission -- uniform absolute
        epoch, launch phase drawn around the fast-transit optimum, dispersed
        injection -- but with a tighter phase window, because a crewed vehicle
        launches inside a defined window rather than whenever the tug is ready.
        """
        self._earth_phase0 = float(rng.uniform(0.0, TWO_PI))
        self._launch_phase_rad = self.launch_phase_center_rad + float(
            rng.uniform(-1.0, 1.0)
        ) * self.launch_phase_window_rad
        self._mars_phase0 = self._earth_phase0 + self._launch_phase_rad
        self._body_t = -1.0e300

        er, ev = planet_state(EARTH_SMA, EARTH_ORBIT_PERIOD, 0.0, self._earth_phase0)
        pos = np.asarray(er, dtype=np.float64).copy()
        vel = np.asarray(ev, dtype=np.float64).copy()
        vel += rng.normal(0.0, self.injection_error_m_s / math.sqrt(3.0), size=3)

        state = VehicleState(
            position_m=pos,
            velocity_m_s=vel,
            dry_mass_kg=self.total_dry_mass_kg,
            propellant_kg=self.propellant_capacity_kg,
            payload_kg=self.payload_kg,
            t_s=0.0,
        )

        self._acc_t_s = 0.0
        self._dose_sv = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._prev_wear = 0.0
        self._best_miss_m = math.inf
        self._best_rel_speed_m_s = math.inf
        self._cache_t = -1.0e300
        self._cache = None

        self._dv_reference = 1.0
        self._dv_reference = max(self._remaining_delta_v(state), 1.0)
        self._phi_prev = self._potential(state)
        logger.debug(
            "mars_crew_fast reset: launch phase %.1f deg, limit %.0f days",
            math.degrees(self._launch_phase_rad),
            self.trip_time_limit_s / DAY,
        )
        return state

    # --- crew dose -----------------------------------------------------------
    def _advance(self, state: VehicleState) -> None:
        dt = state.t_s - self._acc_t_s
        if dt <= 0.0:
            return
        super()._advance(state)
        rate = (
            self.shielded_dose_rate_sv_day
            * (1.0 + _GCR_RADIAL_GAIN * (state.radius_m / AU - 1.0))
            / DAY
        )
        self._dose_sv += max(rate, 0.0) * dt

    def mission_constraints(self, state: VehicleState) -> ConstraintReport:
        """Crew dose margin. Negative means the budget has been overspent."""
        return ConstraintReport(
            names=("crew_dose",),
            margins=np.array([1.0 - self._dose_sv / self.crew_dose_budget_sv]),
        )

    def _extra_termination(self, state: VehicleState) -> TerminationReason:
        # Overspending the crew's dose budget ends the mission: there is no
        # "arrive anyway" branch that a flight surgeon would sign.
        if self._dose_sv > self.crew_dose_budget_sv:
            return TerminationReason.SAFETY_VIOLATION
        return TerminationReason.RUNNING

    def _terminal_extra(self, reason: TerminationReason) -> float:
        """Explicit deduction for the shielding mass, in ``return_scale`` units.

        The dominant penalty for shielding is already physical (it is dry mass in
        the rocket equation); this is the visible, ablatable remainder.
        """
        return -2.0 * self.shield_mass_kg / self.wet_mass_kg

    def _extra_observation(self, state: VehicleState) -> float:
        return to_symmetric(clamp01(self._dose_sv / self.crew_dose_budget_sv))

    def observation_labels(self) -> tuple[str, ...]:
        return (
            "radius_gap",
            "eccentricity",
            "radial_speed",
            "transverse_speed",
            "sin_phase_to_mars",
            "cos_phase_to_mars",
            "miss_distance",
            "relative_speed",
            "phase_rate",
            "crew_dose_fraction",
            "time_fraction",
            "progress",
        )

    def summarize(self, state: VehicleState, reason: TerminationReason) -> MissionResult:
        self._advance(state)
        success = reason is TerminationReason.SUCCESS
        d_pos, d_vel = self._miss(state)
        el = self._shape(state)
        return MissionResult(
            reason=reason,
            success=success,
            progress=self.progress(state),
            elapsed_s=state.t_s,
            delta_v_m_s=state.delta_v_applied_m_s,
            propellant_used_kg=state.propellant_used_kg,
            payload_delivered_kg=self.payload_kg if success else 0.0,
            terminal_error=d_pos,
            constraint_violations=self._violations,
            total_constraint_cost=self._constraint_cost,
            extras={
                "arrival_rel_speed_m_s": d_vel,
                "best_miss_m": self._best_miss_m,
                "best_rel_speed_m_s": self._best_rel_speed_m_s,
                "crew_dose_sv": self._dose_sv,
                "crew_dose_fraction": self._dose_sv / self.crew_dose_budget_sv,
                "trip_time_days": state.t_s / DAY,
                "trip_time_limit_days": self.trip_time_limit_s / DAY,
                "shield_mass_kg": self.shield_mass_kg,
                "launch_phase_deg": math.degrees(self._launch_phase_rad),
                "final_sma_au": el.a / AU,
                "final_eccentricity": el.e,
                "delta_v_reference_m_s": self._dv_reference,
                "propellant_remaining_kg": state.propellant_kg,
            },
        )

    def info(self) -> dict[str, float]:
        return {
            "trip_time_limit_days": self.trip_time_limit_s / DAY,
            "crew_dose_budget_sv": self.crew_dose_budget_sv,
            "dose_at_limit_sv": self.shielded_dose_rate_sv_day
            * self.trip_time_limit_s
            / DAY,
            "shielded_dose_rate_msv_day": self.shielded_dose_rate_sv_day * 1e3,
            "propellant_capacity_kg": self.propellant_capacity_kg,
            "payload_kg": self.payload_kg,
            "required_accel_m_s2": self._dv_reference / self.trip_time_limit_s,
            "steps_per_episode": self.max_duration_s / self.step_dt_s,
        }
