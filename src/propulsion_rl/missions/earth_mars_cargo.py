"""Earth to Mars cargo rendezvous -- the genuinely contested case.

Circular, coplanar Earth and Mars; the vehicle starts on Earth's heliocentric
state (patched-conic departure, zero nominal v-infinity, so the stage supplies
all of the transfer energy itself) and must *rendezvous* with Mars: match
position to within a sphere-of-influence-sized window and velocity to within a
capturable relative speed.

The interesting design constraint is the progress signal. Rewarding radius
closure alone produces a policy that flies confidently to 1.52 AU and arrives
six months after Mars has left. The potential here is an estimate of the
delta-v still owed, and it has an explicit phasing component that switches on as
the vehicle's semi-major axis approaches Mars': far from Mars' orbit the right
thing to do is raise energy, near it the right thing to do is fix the phase, and
the potential says so.

Nobody is aboard, so trip time is a cost rather than a limit. That is what makes
this contested: a nuclear thermal stage closes it in eight or nine months but
spends most of its wet mass doing it, while a solar-electric tug takes two to
three years and delivers with a third of the propellant. Which one wins is a
question for the economics model, not for the reward.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.constants import (
    AU,
    DAY,
    EARTH_ORBIT_PERIOD,
    EARTH_SMA,
    MARS_ORBIT_PERIOD,
    MARS_SMA,
    MU_SUN,
    TINY_MASS_KG,
    YEAR,
)
from ..core.registry import MISSION
from ..core.types import (
    ConstraintReport,
    HealthReport,
    TerminationReason,
    ThrusterOutput,
    VehicleState,
)
from ..spacecraft.orbit import (
    edelbaum_delta_v,
    phase_angle,
    planet_state,
    synodic_period,
)
from .base import Mission, MissionResult, RewardTerms
from .rewards import (
    TWO_PI,
    Osculating,
    RewardConfig,
    RewardScales,
    assemble_terms,
    clamp,
    clamp01,
    osculating,
    potential_shaping,
    signed_frac,
    soft_frac,
    terminal_value,
    to_symmetric,
    wrap_pi,
)

logger = logging.getLogger(__name__)

V_EARTH_CIRC = math.sqrt(MU_SUN / EARTH_SMA)      # ~29.78 km/s
V_MARS_CIRC = math.sqrt(MU_SUN / MARS_SMA)        # ~24.13 km/s
_RADIUS_SPAN = MARS_SMA - EARTH_SMA

#: Delta-v charged per radian of phase error once the vehicle is on Mars' orbit.
#: Calibrated so a half-revolution of phase error is worth ~2.3 km/s, the rough
#: cost of a phasing loop at these radii.
_PHASE_GAIN = 0.03
#: Time constant converting a close-range position error into an equivalent
#: delta-v. 30 days is the scale on which a terminal correction is cheap.
_TAU_CLOSE_S = 30.0 * DAY

#: Hohmann-optimal Earth-Mars departure phase angle (Mars ahead of Earth).
HOHMANN_PHASE_RAD = math.radians(44.3)


class _HeliocentricTransfer(Mission):
    """Shared implementation for the two Earth-Mars scenarios.

    Both share the geometry, the rendezvous potential and the observation
    layout; they differ in what they *value* (propellant versus schedule), in
    their termination rules, and in the fact that one of them carries people.
    Subclasses set the mass budget, the reward config and the tolerances.
    """

    mu = MU_SUN
    frame = "heliocentric"

    # Set by subclasses.
    payload_kg: float = 0.0
    stage_dry_mass_kg: float = 0.0
    propellant_capacity_kg: float = 1.0
    arrival_radius_m: float = 6.0e8
    arrival_speed_m_s: float = 1500.0
    config: RewardConfig

    def __init__(self) -> None:
        self._mars_phase0 = 0.0
        self._earth_phase0 = 0.0
        self._launch_phase_rad = 0.0
        self._dv_reference = 1.0
        self._phi_prev = 0.0
        self._prev_wear = 0.0
        self._acc_t_s = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._best_miss_m = math.inf
        self._best_rel_speed_m_s = math.inf
        self._cache_t = -1.0e300
        self._cache: Osculating | None = None
        self._body_t = -1.0e300
        self._mars_r = np.zeros(3)
        self._mars_v = np.zeros(3)
        self._scales = RewardScales(1.0, 1.0, 1.0)
        self.synodic_period_s = synodic_period(EARTH_ORBIT_PERIOD, MARS_ORBIT_PERIOD)

    # --- geometry ------------------------------------------------------------
    def _mars_state(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Mars' heliocentric state, memoised on ``t_s``."""
        if t_s != self._body_t:
            self._mars_r, self._mars_v = planet_state(
                MARS_SMA, MARS_ORBIT_PERIOD, t_s, self._mars_phase0
            )
            self._body_t = t_s
        return self._mars_r, self._mars_v

    def _shape(self, state: VehicleState) -> Osculating:
        if state.t_s != self._cache_t or self._cache is None:
            self._cache = osculating(state.position_m, state.velocity_m_s, self.mu)
            self._cache_t = state.t_s
        return self._cache

    def _miss(self, state: VehicleState) -> tuple[float, float]:
        """(position miss distance, relative speed) with respect to Mars."""
        mr, mv = self._mars_state(state.t_s)
        dr = state.position_m - mr
        dv = state.velocity_m_s - mv
        return float(np.linalg.norm(dr)), float(np.linalg.norm(dv))

    def _remaining_delta_v(self, state: VehicleState) -> float:
        """Estimate of the delta-v still owed to complete the rendezvous.

        Two independent estimates, taking whichever is smaller:

        ``far``
            Orbit-matching (Edelbaum between the osculating and Mars semi-major
            axes, plus a circularisation term) *plus* a phasing charge weighted
            by how close the semi-major axis already is to Mars'. The weighting
            is the anti-tail-chase device: raising ``a`` is what pays early,
            fixing the phase is what pays late.
        ``close``
            Direct terminal guidance: the relative speed plus the position error
            amortised over a 30 day correction. Both are measured as *excess*
            over the arrival tolerance, so the estimate reaches exactly zero at
            the success boundary and the potential reaches exactly 1.
        """
        el = self._shape(state)
        mr, _ = self._mars_state(state.t_s)

        if el.a > 0.0 and math.isfinite(el.a):
            far = edelbaum_delta_v(el.a, MARS_SMA, 0.0, 0.0, self.mu)
            far += 0.5 * el.e * math.sqrt(self.mu / el.a)
            w_phase = clamp01(1.0 - abs(el.a - MARS_SMA) / _RADIUS_SPAN)
            if w_phase > 0.0:
                d_theta = abs(phase_angle(state.position_m, mr))
                far += _PHASE_GAIN * V_MARS_CIRC * d_theta * w_phase
        else:
            far = self._dv_reference * 2.0

        d_pos, d_vel = self._miss(state)
        close = max(d_vel - self.arrival_speed_m_s, 0.0) + (
            max(d_pos - self.arrival_radius_m, 0.0) / _TAU_CLOSE_S
        )
        return far if far < close else close

    def _potential(self, state: VehicleState) -> float:
        phi = 1.0 - self._remaining_delta_v(state) / self._dv_reference
        return clamp(phi, -0.5, 1.0)

    def progress(self, state: VehicleState) -> float:
        return clamp01(self._potential(state))

    # --- observation ---------------------------------------------------------
    def observe_raw(self, state: VehicleState) -> np.ndarray:
        el = self._shape(state)
        mr, _ = self._mars_state(state.t_s)
        d_pos, d_vel = self._miss(state)
        # Signed phase angle from the vehicle to Mars: positive means Mars leads.
        d_theta = phase_angle(state.position_m, mr)

        # Angular-rate difference: the sign tells the policy whether it is
        # gaining or losing phase on Mars, which a static phase angle does not.
        omega_sc = el.vt / el.r if el.r > 0.0 else 0.0
        d_omega = omega_sc - TWO_PI / MARS_ORBIT_PERIOD

        return np.array(
            [
                signed_frac(el.r - EARTH_SMA, _RADIUS_SPAN),
                soft_frac(el.e, 0.25),
                signed_frac(el.vr, V_EARTH_CIRC),
                signed_frac(el.vt, V_EARTH_CIRC),
                math.sin(d_theta),
                math.cos(d_theta),
                soft_frac(d_pos, 0.5 * AU),
                soft_frac(d_vel, 10.0e3),
                soft_frac(d_omega, 4.0e-8),
                self._extra_observation(state),
                to_symmetric(clamp01(state.t_s / self.max_duration_s)),
                to_symmetric(self.progress(state)),
            ],
            dtype=np.float32,
        )

    def _extra_observation(self, state: VehicleState) -> float:
        """One mission-specific slot: propellant for cargo, crew dose for crew."""
        raise NotImplementedError

    # --- reward --------------------------------------------------------------
    def reward(
        self,
        prev: VehicleState,
        state: VehicleState,
        output: ThrusterOutput,
        constraints: ConstraintReport,
        health: HealthReport,
    ) -> RewardTerms:
        self._advance(state)
        dt = state.t_s - prev.t_s
        phi = self._potential(state)
        d_phi = potential_shaping(self._phi_prev, phi, self.config.gamma)
        self._phi_prev = phi

        d_wear = health.wear_fraction - self._prev_wear
        self._prev_wear = health.wear_fraction

        cost = constraints.cost + self.mission_constraints(state).cost
        if cost > 0.0:
            self._violations += 1
            self._constraint_cost += cost

        return assemble_terms(
            self.config,
            self._scales,
            d_potential=d_phi,
            dt_s=dt,
            propellant_kg=state.propellant_used_kg - prev.propellant_used_kg,
            d_wear=d_wear,
            constraint_cost=cost,
            isp_s=output.isp_s,
            terminal=self._terminal_multiplier(state, health, constraints),
        )

    def _terminal_multiplier(
        self, state: VehicleState, health: HealthReport, constraints: ConstraintReport
    ) -> float:
        reason = self.terminated(state, health, constraints)
        if reason is TerminationReason.RUNNING:
            if state.t_s + self.step_dt_s < self.max_duration_s:
                return 0.0
            reason = TerminationReason.TIMEOUT
        return terminal_value(reason, self.config) + self._terminal_extra(reason)

    def _terminal_extra(self, reason: TerminationReason) -> float:
        return 0.0

    def mission_constraints(self, state: VehicleState) -> ConstraintReport:
        return ConstraintReport()

    def _advance(self, state: VehicleState) -> None:
        dt = state.t_s - self._acc_t_s
        if dt <= 0.0:
            return
        self._acc_t_s = state.t_s
        d_pos, d_vel = self._miss(state)
        if d_pos < self._best_miss_m:
            self._best_miss_m = d_pos
            self._best_rel_speed_m_s = d_vel

    # --- termination ---------------------------------------------------------
    def terminated(
        self, state: VehicleState, health: HealthReport, constraints: ConstraintReport
    ) -> TerminationReason:
        if not np.all(np.isfinite(state.position_m)) or not np.all(
            np.isfinite(state.velocity_m_s)
        ):
            return TerminationReason.DIVERGED
        el = self._shape(state)
        if el.energy >= 0.0 or el.r > 4.0 * AU or el.r < 0.3 * AU:
            return TerminationReason.DIVERGED

        self._advance(state)
        extra = self._extra_termination(state)
        if extra is not TerminationReason.RUNNING:
            return extra
        if constraints.worst < -1.0:
            return TerminationReason.SAFETY_VIOLATION
        if health.failed:
            return TerminationReason.HARDWARE_FAILURE

        d_pos, d_vel = self._miss(state)
        if d_pos <= self.arrival_radius_m and d_vel <= self.arrival_speed_m_s:
            return TerminationReason.SUCCESS
        if state.propellant_kg <= TINY_MASS_KG:
            return TerminationReason.OUT_OF_PROPELLANT
        return TerminationReason.RUNNING

    def _extra_termination(self, state: VehicleState) -> TerminationReason:
        return TerminationReason.RUNNING


@MISSION.register(
    "earth_mars_cargo",
    frame="heliocentric",
    regime="interplanetary_rendezvous",
    favours="contested",
    horizon_days=1278.0,
)
class EarthMarsCargo(_HeliocentricTransfer):
    """Deliver an uncrewed cargo module to Mars. No clock, so high Isp pays."""

    name = "earth_mars_cargo"

    def __init__(
        self,
        *,
        wet_mass_kg: float = 20_000.0,
        stage_dry_mass_kg: float = 4_000.0,
        payload_kg: float = 6_000.0,
        step_dt_s: float = 0.5 * DAY,
        max_duration_s: float = 3.5 * YEAR,
        arrival_radius_m: float = 8.0e8,
        arrival_speed_m_s: float = 1500.0,
        launch_phase_center_rad: float = HOHMANN_PHASE_RAD,
        launch_phase_window_rad: float = math.radians(60.0),
        injection_error_m_s: float = 50.0,
        config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.wet_mass_kg = float(wet_mass_kg)
        self.stage_dry_mass_kg = float(stage_dry_mass_kg)
        self.payload_kg = float(payload_kg)
        self.propellant_capacity_kg = max(
            wet_mass_kg - stage_dry_mass_kg - payload_kg, TINY_MASS_KG
        )
        self.step_dt_s = float(step_dt_s)
        self.max_duration_s = float(max_duration_s)
        self.arrival_radius_m = float(arrival_radius_m)
        self.arrival_speed_m_s = float(arrival_speed_m_s)
        self.launch_phase_center_rad = float(launch_phase_center_rad)
        self.launch_phase_window_rad = float(launch_phase_window_rad)
        self.injection_error_m_s = float(injection_error_m_s)

        # Cargo values propellant over schedule, but not to the exclusion of it:
        # capital tied up in a tug for three years is a real cost.
        self.config = config or RewardConfig(
            w_time=0.20,
            w_propellant=0.45,
            isp_reference_s=2000.0,
        )
        self._scales = RewardScales(
            duration_s=self.max_duration_s,
            propellant_capacity_kg=self.propellant_capacity_kg,
            delta_v_reference_m_s=edelbaum_delta_v(EARTH_SMA, MARS_SMA, 0.0, 0.0, MU_SUN),
        )

    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Sample a departure: random epoch, random launch phase, dispersed injection.

        ``launch_phase_rad`` is the Mars-minus-Earth heliocentric phase angle at
        departure, drawn around the Hohmann-optimal 44.3 deg. Earth's absolute
        longitude is drawn uniformly, so nothing about the geometry is
        memorisable -- only the *relative* problem generalises.
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
            dry_mass_kg=self.stage_dry_mass_kg,
            propellant_kg=self.propellant_capacity_kg,
            payload_kg=self.payload_kg,
            t_s=0.0,
        )

        self._acc_t_s = 0.0
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
            "earth_mars_cargo reset: launch phase %.1f deg, dv_ref %.0f m/s",
            math.degrees(self._launch_phase_rad),
            self._dv_reference,
        )
        return state

    def _extra_observation(self, state: VehicleState) -> float:
        return to_symmetric(clamp01(state.propellant_kg / self.propellant_capacity_kg))

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
            "propellant_fraction",
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
                "arrival_day": state.t_s / DAY,
                "trip_time_days": state.t_s / DAY,
                "launch_phase_deg": math.degrees(self._launch_phase_rad),
                "final_sma_au": el.a / AU,
                "final_eccentricity": el.e,
                "delta_v_reference_m_s": self._dv_reference,
                "propellant_remaining_kg": state.propellant_kg,
                "synodic_period_days": self.synodic_period_s / DAY,
            },
        )

    def info(self) -> dict[str, float]:
        return {
            "hohmann_reference_m_s": edelbaum_delta_v(
                EARTH_SMA, MARS_SMA, 0.0, 0.0, MU_SUN
            ),
            "synodic_period_days": self.synodic_period_s / DAY,
            "propellant_capacity_kg": self.propellant_capacity_kg,
            "payload_kg": self.payload_kg,
            "steps_per_episode": self.max_duration_s / self.step_dt_s,
        }
