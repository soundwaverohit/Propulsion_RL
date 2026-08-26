"""LEO to GEO orbit raising -- the canonical electric-propulsion win.

A 400 km, 28.5 deg parking orbit (Cape latitude) to a circular equatorial GEO
slot. Impulsively this is a ~4.7 km/s Hohmann-plus-plane-change; continuously it
is a ~5.9 km/s Edelbaum spiral. Electric propulsion pays 25% more delta-v and
takes months instead of hours, and still delivers far more payload, because
delta-v is cheap when the exhaust velocity is 18 km/s.

Two things stop that from being a free lunch, and both are modelled:

* **Eclipses.** A solar-electric tug cannot thrust in shadow. The duty cycle is
  set by real geometry (:meth:`eclipse`), not by a fudge factor -- the eclipse
  fraction is large in the LEO phase and falls to almost nothing near GEO.
* **The Van Allen belts.** A slow spiral dwells for weeks in the proton and
  electron belts, accumulating dose that degrades the solar arrays that power
  the thruster. Dose is accumulated per step and reported in
  ``MissionResult.extras``; it is what makes "just fly slower" a bad answer.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.constants import (
    DAY,
    EARTH_SMA,
    GEO_RADIUS,
    HOUR,
    LEO_RADIUS,
    MU_EARTH,
    R_EARTH,
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
from ..spacecraft.orbit import edelbaum_delta_v, hohmann_delta_v
from .base import Mission, MissionResult, RewardTerms
from .rewards import (
    TWO_PI,
    Osculating,
    RewardConfig,
    RewardScales,
    assemble_terms,
    clamp,
    clamp01,
    in_cylindrical_shadow,
    osculating,
    potential_shaping,
    signed_frac,
    soft_frac,
    sun_direction_eci,
    terminal_value,
    to_symmetric,
)

logger = logging.getLogger(__name__)

# --- Van Allen dose model ----------------------------------------------------
# Two Gaussian humps in L-shell (r / R_Earth): the inner proton belt and the
# outer electron belt. Peak rates are behind a nominal ~2.5 mm aluminium shield
# and are calibrated so a ~200 day all-electric spiral accumulates a few hundred
# krad, which is the range published GEO transfer studies report.
_INNER_PEAK_KRAD_DAY = 9.0
_INNER_L = 1.5
_INNER_WIDTH = 0.45
_OUTER_PEAK_KRAD_DAY = 4.5
_OUTER_L = 4.5
_OUTER_WIDTH = 1.30
#: Dose at which the arrays have lost 1/e of their recoverable output.
_ARRAY_DOSE_SCALE_KRAD = 3000.0


def _dose_rate_krad_s(radius_m: float) -> float:
    """Absorbed dose rate, krad/s, as a function of geocentric radius."""
    ell = radius_m / R_EARTH
    zi = (ell - _INNER_L) / _INNER_WIDTH
    zo = (ell - _OUTER_L) / _OUTER_WIDTH
    per_day = _INNER_PEAK_KRAD_DAY * math.exp(-zi * zi)
    if -6.0 < zo < 6.0:
        per_day += _OUTER_PEAK_KRAD_DAY * math.exp(-zo * zo)
    return per_day / DAY


@MISSION.register(
    "leo_geo_transfer",
    frame="planetocentric",
    regime="orbit_raising",
    favours="electric",
    horizon_days=365.0,
)
class LEOtoGEOTransfer(Mission):
    """Raise a comsat from a 400 km, 28.5 deg parking orbit to a GEO slot.

    Mass budget convention: the mission fixes the *launched* wet mass and the
    payload; the propellant load is whatever is left after the stage's dry mass.
    A heavier propulsion system therefore automatically flies with less
    propellant, which is the honest way to compare families that differ by tonnes
    of power system.
    """

    name = "leo_geo_transfer"
    mu = MU_EARTH
    frame = "planetocentric"

    def __init__(
        self,
        *,
        wet_mass_kg: float = 5000.0,
        stage_dry_mass_kg: float = 1200.0,
        payload_kg: float = 1800.0,
        start_altitude_m: float = 400e3,
        start_inclination_rad: float = math.radians(28.5),
        target_radius_m: float = GEO_RADIUS,
        step_dt_s: float = HOUR,
        max_duration_s: float = YEAR,
        radius_tolerance: float = 0.004,
        inclination_tolerance_rad: float = math.radians(0.5),
        eccentricity_tolerance: float = 0.01,
        dose_soft_limit_krad: float = 500.0,
        dose_hard_limit_krad: float = 1500.0,
        min_altitude_m: float = 150e3,
        config: RewardConfig | None = None,
    ) -> None:
        self.wet_mass_kg = float(wet_mass_kg)
        self.stage_dry_mass_kg = float(stage_dry_mass_kg)
        self.payload_kg = float(payload_kg)
        self.propellant_capacity_kg = max(
            wet_mass_kg - stage_dry_mass_kg - payload_kg, TINY_MASS_KG
        )
        self.start_altitude_m = float(start_altitude_m)
        self.start_inclination_rad = float(start_inclination_rad)
        self.target_radius_m = float(target_radius_m)
        self.step_dt_s = float(step_dt_s)
        self.max_duration_s = float(max_duration_s)
        self.radius_tolerance = float(radius_tolerance)
        self.inclination_tolerance_rad = float(inclination_tolerance_rad)
        self.eccentricity_tolerance = float(eccentricity_tolerance)
        self.dose_soft_limit_krad = float(dose_soft_limit_krad)
        self.dose_hard_limit_krad = float(dose_hard_limit_krad)
        self.min_altitude_m = float(min_altitude_m)

        # Propellant is the whole story on this mission, and a slow spiral is
        # punished through radiation rather than through the clock, so the time
        # weight stays modest.
        self.config = config or RewardConfig(
            w_time=0.30,
            w_propellant=0.40,
            isp_reference_s=1800.0,
        )

        self._scales = RewardScales(
            duration_s=self.max_duration_s,
            propellant_capacity_kg=self.propellant_capacity_kg,
            delta_v_reference_m_s=1.0,
        )

        # Episode state, all cleared by reset().
        self._sun_longitude0 = 0.0
        self._dv_reference = 1.0
        self._phi_prev = 0.0
        self._prev_wear = 0.0
        self._acc_t_s = 0.0
        self._dose_krad = 0.0
        self._belt_dwell_s = 0.0
        self._eclipse_s = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._t0_s = 0.0
        self._start_a = LEO_RADIUS
        self._start_inc = self.start_inclination_rad
        self._cache_t = -1.0e300
        self._cache: Osculating | None = None

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Sample a parking orbit: dispersed injection, random node, random epoch.

        Randomised: altitude (+/-20 km), inclination (+/-0.1 deg), RAAN and
        argument of latitude (uniform), injection velocity error (~5 m/s in a
        random direction), and the epoch, which sets the Sun's right ascension
        and therefore the whole eclipse season the tug flies through.
        """
        alt = self.start_altitude_m + float(rng.uniform(-20e3, 20e3))
        r0 = R_EARTH + alt
        inc = self.start_inclination_rad + float(rng.uniform(-1.0, 1.0)) * math.radians(0.1)
        raan = float(rng.uniform(0.0, TWO_PI))
        u0 = float(rng.uniform(0.0, TWO_PI))
        self._sun_longitude0 = float(rng.uniform(0.0, TWO_PI))

        ci, si = math.cos(inc), math.sin(inc)
        cr, sr = math.cos(raan), math.sin(raan)
        node = np.array([cr, sr, 0.0])
        inplane = np.array([-sr * ci, cr * ci, si])

        cu, su = math.cos(u0), math.sin(u0)
        pos = r0 * (cu * node + su * inplane)
        vc = math.sqrt(self.mu / r0)
        vel = vc * (-su * node + cu * inplane)
        vel += rng.normal(0.0, 5.0, size=3)          # injection dispersion, m/s

        state = VehicleState(
            position_m=pos,
            velocity_m_s=vel,
            dry_mass_kg=self.stage_dry_mass_kg,
            propellant_kg=self.propellant_capacity_kg,
            payload_kg=self.payload_kg,
            t_s=0.0,
        )

        self._t0_s = 0.0
        self._acc_t_s = 0.0
        self._dose_krad = 0.0
        self._belt_dwell_s = 0.0
        self._eclipse_s = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._prev_wear = 0.0
        self._cache_t = -1.0e300
        self._cache = None

        el = self._shape(state)
        self._start_a = el.a
        self._start_inc = el.inc
        self._dv_reference = max(
            edelbaum_delta_v(el.a, self.target_radius_m, el.inc, 0.0, self.mu), 1.0
        )
        self._phi_prev = self._potential(state)
        logger.debug(
            "leo_geo_transfer reset: a0=%.1f km i0=%.3f deg dv_ref=%.1f m/s",
            el.a * 1e-3,
            math.degrees(el.inc),
            self._dv_reference,
        )
        return state

    # --- internals -----------------------------------------------------------
    def _shape(self, state: VehicleState) -> Osculating:
        """Osculating elements, memoised on ``t_s`` for the four-calls-per-step
        access pattern (observe / progress / reward / terminated)."""
        if state.t_s != self._cache_t or self._cache is None:
            self._cache = osculating(state.position_m, state.velocity_m_s, self.mu)
            self._cache_t = state.t_s
        return self._cache

    def _remaining_delta_v(self, el: Osculating) -> float:
        """Edelbaum delta-v still owed, plus an approximate circularisation cost.

        The eccentricity term is the first-order two-impulse cost of removing a
        small eccentricity at fixed ``a``; it exists so the potential cannot be
        gamed by flinging the apogee out to GEO and calling it done.
        """
        if not (el.a > 0.0) or not math.isfinite(el.a):
            return self._dv_reference * 2.0
        dv = edelbaum_delta_v(el.a, self.target_radius_m, el.inc, 0.0, self.mu)
        dv += 0.5 * el.e * math.sqrt(self.mu / el.a)
        return dv

    def _potential(self, state: VehicleState) -> float:
        """Shaping potential: fraction of the initial Edelbaum budget retired."""
        phi = 1.0 - self._remaining_delta_v(self._shape(state)) / self._dv_reference
        return clamp(phi, -0.5, 1.0)

    def _advance(self, state: VehicleState) -> None:
        """Integrate the dose and dwell accumulators. Idempotent in ``t_s``."""
        dt = state.t_s - self._acc_t_s
        if dt <= 0.0:
            return
        self._acc_t_s = state.t_s
        rate = _dose_rate_krad_s(state.radius_m)
        self._dose_krad += rate * dt
        if rate * DAY > 0.5:                      # meaningfully inside a belt
            self._belt_dwell_s += dt
        if self.eclipse(state):
            self._eclipse_s += dt

    def mission_constraints(self, state: VehicleState) -> ConstraintReport:
        """Mission-side safety margins (the propulsion system reports its own).

        Convention as in :class:`~..core.types.ConstraintReport`: >= 0 is safe.
        """
        alt_margin = (state.radius_m - R_EARTH - self.min_altitude_m) / 250e3
        dose_margin = 1.0 - self._dose_krad / self.dose_soft_limit_krad
        return ConstraintReport(
            names=("min_altitude", "radiation_dose"),
            margins=np.array([alt_margin, dose_margin]),
        )

    # --- observation ---------------------------------------------------------
    def observe_raw(self, state: VehicleState) -> np.ndarray:
        el = self._shape(state)
        sun = sun_direction_eci(state.t_s, self._sun_longitude0)
        rn = el.r if el.r > 0.0 else 1.0
        pos = state.position_m

        # Sine of the beta angle: how far the orbit plane is from edge-on to the
        # Sun. It, not the season, is what sets the eclipse fraction.
        hx = el.iy                                  # h_hat x = sin(i) sin(raan)
        hy = -el.ix                                 # h_hat y = -sin(i) cos(raan)
        hz = math.sqrt(max(1.0 - hx * hx - hy * hy, 0.0))
        beta = clamp(sun[0] * hx + sun[1] * hy + sun[2] * hz, -1.0, 1.0)
        sun_phase = clamp(
            (pos[0] * sun[0] + pos[1] * sun[1] + pos[2] * sun[2]) / rn, -1.0, 1.0
        )

        span = self.target_radius_m - LEO_RADIUS
        return np.array(
            [
                signed_frac(self.target_radius_m - el.a, span),
                signed_frac(el.inc, self.start_inclination_rad),
                soft_frac(el.ex, 0.05),
                soft_frac(el.ey, 0.05),
                el.sin_u,
                el.cos_u,
                beta,
                sun_phase,
                1.0 if self.eclipse(state) else -1.0,
                to_symmetric(clamp01(self._dose_krad / self.dose_hard_limit_krad)),
                to_symmetric(clamp01(state.t_s / self.max_duration_s)),
                to_symmetric(self.progress(state)),
            ],
            dtype=np.float32,
        )

    def observation_labels(self) -> tuple[str, ...]:
        return (
            "sma_gap",
            "inclination_gap",
            "ecc_vector_x",
            "ecc_vector_y",
            "sin_arg_lat",
            "cos_arg_lat",
            "sun_beta",
            "sun_phase",
            "in_eclipse",
            "dose_fraction",
            "time_fraction",
            "progress",
        )

    # --- task ----------------------------------------------------------------
    def progress(self, state: VehicleState) -> float:
        return clamp01(self._potential(state))

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

        mission_con = self.mission_constraints(state)
        cost = constraints.cost + mission_con.cost
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
        return terminal_value(reason, self.config)

    def terminated(
        self, state: VehicleState, health: HealthReport, constraints: ConstraintReport
    ) -> TerminationReason:
        if not np.all(np.isfinite(state.position_m)) or not np.all(
            np.isfinite(state.velocity_m_s)
        ):
            return TerminationReason.DIVERGED
        el = self._shape(state)
        # Earth's sphere of influence is ~0.93 Gm; beyond that, or on an
        # unbound orbit, this is no longer a GEO transfer.
        if el.energy >= 0.0 or el.r > 1.0e9:
            return TerminationReason.DIVERGED

        self._advance(state)
        if el.r - R_EARTH < self.min_altitude_m:
            return TerminationReason.SAFETY_VIOLATION
        if self._dose_krad > self.dose_hard_limit_krad:
            return TerminationReason.SAFETY_VIOLATION
        if constraints.worst < -1.0:
            return TerminationReason.SAFETY_VIOLATION
        if health.failed:
            return TerminationReason.HARDWARE_FAILURE

        if (
            abs(el.a - self.target_radius_m) <= self.radius_tolerance * self.target_radius_m
            and el.inc <= self.inclination_tolerance_rad
            and el.e <= self.eccentricity_tolerance
        ):
            return TerminationReason.SUCCESS

        if state.propellant_kg <= TINY_MASS_KG:
            return TerminationReason.OUT_OF_PROPELLANT
        return TerminationReason.RUNNING

    def summarize(self, state: VehicleState, reason: TerminationReason) -> MissionResult:
        self._advance(state)
        el = self._shape(state)
        success = reason is TerminationReason.SUCCESS
        elapsed = state.t_s - self._t0_s
        return MissionResult(
            reason=reason,
            success=success,
            progress=self.progress(state),
            elapsed_s=elapsed,
            delta_v_m_s=state.delta_v_applied_m_s,
            propellant_used_kg=state.propellant_used_kg,
            payload_delivered_kg=self.payload_kg if success else 0.0,
            terminal_error=self._remaining_delta_v(el),
            constraint_violations=self._violations,
            total_constraint_cost=self._constraint_cost,
            extras={
                "radiation_dose_krad": self._dose_krad,
                "array_degradation": 1.0
                - math.exp(-self._dose_krad / _ARRAY_DOSE_SCALE_KRAD),
                "belt_dwell_days": self._belt_dwell_s / DAY,
                "eclipse_fraction": (self._eclipse_s / elapsed) if elapsed > 0.0 else 0.0,
                "trip_time_days": elapsed / DAY,
                "final_sma_km": el.a * 1e-3,
                "final_inclination_deg": math.degrees(el.inc),
                "final_eccentricity": el.e,
                "edelbaum_reference_m_s": self._dv_reference,
                "propellant_remaining_kg": state.propellant_kg,
            },
        )

    # --- hooks ---------------------------------------------------------------
    def eclipse(self, state: VehicleState) -> bool:
        return in_cylindrical_shadow(
            state.position_m, sun_direction_eci(state.t_s, self._sun_longitude0), R_EARTH
        )

    def heliocentric_radius_m(self, state: VehicleState) -> float:
        """Distance to the *Sun*, m -- deliberately not the base implementation.

        ``Mission.heliocentric_radius_m`` returns ``state.radius_m``, which in a
        planetocentric frame is the distance to Earth: 6.8e6 m at LEO, i.e.
        4.5e-5 AU. Handing that to :func:`~..core.constants.solar_flux` would
        offer a solar-electric tug a rounding error's worth of power for the
        whole episode, and the failure would read as "electric propulsion is bad
        at orbit raising" rather than as a units bug.

        Earth is treated as circular about the Sun (consistent with
        ``spacecraft.orbit.planet_state``); the geocentric position contributes
        at most 42 000 km, or 2.8e-4 AU, and is included because it is free.
        """
        sun = sun_direction_eci(state.t_s, self._sun_longitude0)
        r = state.position_m
        along = float(r[0]) * sun[0] + float(r[1]) * sun[1] + float(r[2]) * sun[2]
        return EARTH_SMA - along

    def info(self) -> dict[str, float | str]:
        dv1, dv2 = hohmann_delta_v(LEO_RADIUS, self.target_radius_m, self.mu)
        return {
            "target_radius_m": self.target_radius_m,
            "edelbaum_delta_v_m_s": edelbaum_delta_v(
                LEO_RADIUS, self.target_radius_m, self.start_inclination_rad, 0.0, self.mu
            ),
            # Coplanar two-impulse yardstick; the real impulsive number for this
            # mission adds the plane change, done at apogee where it is cheapest.
            "hohmann_coplanar_delta_v_m_s": abs(dv1) + abs(dv2),
            "propellant_capacity_kg": self.propellant_capacity_kg,
            "payload_kg": self.payload_kg,
            "steps_per_episode": self.max_duration_s / self.step_dt_s,
        }
