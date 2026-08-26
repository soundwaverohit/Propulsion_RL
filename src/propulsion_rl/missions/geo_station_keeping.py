"""GEO station keeping -- long-horizon, low-signal, tightly constrained control.

Hold an assigned geostationary slot inside a +/-0.05 deg box for a decade while
three perturbations push the vehicle out of it. :meth:`GEOStationKeeping.gravity`
overrides the point-mass default with:

* **J2**, the dominant non-spherical term at GEO (~8.3e-6 m/s^2).
* **Triaxiality (J22)**, the tesseral resonance that drives the satellite
  towards the stable longitudes at 75.1E and 104.9W and away from the unstable
  ones at 14.9W and 165.1E. Peaks at ~5.6e-8 m/s^2, which is the ~2 m/s/yr
  east-west budget every GEO operator carries.
* **Luni-solar attraction**, modelled as the calibrated secular inclination
  drift it produces (~0.85 deg/yr, precessing on the 18.6 yr lunar node cycle).
  That is the ~46 m/s/yr north-south budget, and it dominates everything else.
* **Solar radiation pressure**, anti-sunward and switched off in eclipse, which
  drives the annual eccentricity cycle.

As an RL problem this is nothing like the transfers. The horizon is ~15 000
steps, the reward signal is almost flat, and the whole task is to spend as
little propellant as possible without ever leaving the box. Reward weights
reflect that: no time penalty at all (lasting is the *point*), a large
propellant weight, and an expensive deadband.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.constants import (
    EARTH_SMA,
    GEO_RADIUS,
    HOUR,
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
    soft_frac,
    sun_direction_eci,
    terminal_value,
    to_symmetric,
    wrap_pi,
)

logger = logging.getLogger(__name__)

# --- geopotential and environment constants ----------------------------------
# Not in core.constants because they are specific to Earth's gravity field
# harmonics and to the radiation environment, neither of which any other module
# needs. Values are the standard EGM/WGS-84 ones.
OMEGA_EARTH = 7.292115e-5        # rad/s, Earth rotation rate
J2 = 1.08262668e-3
J22 = 1.8155e-6
#: Longitude of the equatorial minor axis. Equilibria sit at LAMBDA_22 + n*90 deg;
#: LAMBDA_22 +/- 90 deg (75.1E, 104.9W) are the stable ones.
LAMBDA_22 = math.radians(-14.929)
#: Solar radiation pressure at 1 AU, N/m^2.
SRP_PRESSURE = 4.5605e-6
#: Mean secular inclination drift from luni-solar attraction.
LUNISOLAR_DRIFT_RAD_YEAR = math.radians(0.85)
#: Regression period of the lunar node; the drift direction precesses with it.
LUNAR_NODE_PERIOD_S = 18.613 * YEAR


@MISSION.register(
    "geo_station_keeping",
    frame="planetocentric",
    regime="station_keeping",
    favours="electric",
    horizon_days=3652.5,
)
class GEOStationKeeping(Mission):
    """Hold a GEO slot inside a deadband for the design life of the spacecraft.

    ``payload_delivered_kg`` is defined here as payload mass *kept on station*:
    the bus mass times the fraction of the design life spent inside the box. A
    transponder that is out of its slot is not delivering service, and the
    headline $/kg figure has to see that.
    """

    name = "geo_station_keeping"
    mu = MU_EARTH
    frame = "planetocentric"

    def __init__(
        self,
        *,
        wet_mass_kg: float = 3_000.0,
        bus_dry_mass_kg: float = 1_400.0,
        payload_kg: float = 800.0,
        deadband_rad: float = math.radians(0.05),
        safety_box_rad: float = math.radians(2.0),
        step_dt_s: float = 6.0 * HOUR,
        max_duration_s: float = 10.0 * YEAR,
        area_to_mass_m2_kg: float = 0.02,
        reflectivity: float = 0.3,
        availability_target: float = 0.98,
        config: RewardConfig | None = None,
    ) -> None:
        self.wet_mass_kg = float(wet_mass_kg)
        self.bus_dry_mass_kg = float(bus_dry_mass_kg)
        self.payload_kg = float(payload_kg)
        self.propellant_capacity_kg = max(
            wet_mass_kg - bus_dry_mass_kg - payload_kg, TINY_MASS_KG
        )
        self.deadband_rad = float(deadband_rad)
        self.safety_box_rad = float(safety_box_rad)
        self.step_dt_s = float(step_dt_s)
        self.max_duration_s = float(max_duration_s)
        #: Vehicle property the environment may override; the mission needs it
        #: because SRP is a force on the *spacecraft*, not on a point mass.
        self.area_to_mass_m2_kg = float(area_to_mass_m2_kg)
        self.reflectivity = float(reflectivity)
        self.availability_target = float(availability_target)

        # No time penalty: outlasting the design life is the objective, not a
        # cost. Propellant dominates, and leaving the box is expensive.
        self.config = config or RewardConfig(
            w_time=0.0,
            w_propellant=2.0,
            w_constraint=1.5,
            constraint_gain=8.0,
            isp_reference_s=2000.0,
        )
        self._scales = RewardScales(
            duration_s=self.max_duration_s,
            propellant_capacity_kg=self.propellant_capacity_kg,
            delta_v_reference_m_s=LUNISOLAR_DRIFT_RAD_YEAR
            * math.sqrt(MU_EARTH / GEO_RADIUS)
            * self.max_duration_s
            / YEAR,
        )

        # Peak triaxiality tangential acceleration at the nominal GEO radius.
        self._a22_coeff = 6.0 * self.mu * J22 * R_EARTH * R_EARTH
        self._j2_coeff = 1.5 * J2 * self.mu * R_EARTH * R_EARTH
        # Normal acceleration amplitude reproducing the secular inclination drift:
        # <di/dt> = A / (2 v) for a_n = A cos(u - u_ls).
        self._a_lunisolar = (
            2.0
            * math.sqrt(self.mu / GEO_RADIUS)
            * LUNISOLAR_DRIFT_RAD_YEAR
            / YEAR
        )
        self._a_srp = SRP_PRESSURE * (1.0 + self.reflectivity) * self.area_to_mass_m2_kg

        # Episode state.
        self._slot_longitude = 0.0
        self._sidereal0 = 0.0
        self._sun_longitude0 = 0.0
        self._lunisolar_phase0 = 1.5 * math.pi
        self._phi_prev = 0.0
        self._prev_wear = 0.0
        self._acc_t_s = 0.0
        self._t_in_box_s = 0.0
        self._excursions = 0
        self._in_box = True
        self._max_longitude_error = 0.0
        self._max_latitude_error = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._eclipse_s = 0.0
        self._cache_t = -1.0e300
        self._cache: Osculating | None = None

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Place the satellite in its slot with realistic post-handover errors.

        Randomised: the slot longitude itself (uniform, so an episode may start
        near a stable equilibrium or near an unstable one -- a materially
        different control problem), the epoch (sidereal angle, Sun longitude and
        lunar node phase), a semi-major axis error of a couple of km, a residual
        eccentricity, and a residual inclination of up to 0.02 deg.
        """
        self._slot_longitude = float(rng.uniform(-math.pi, math.pi))
        self._sidereal0 = float(rng.uniform(0.0, TWO_PI))
        self._sun_longitude0 = float(rng.uniform(0.0, TWO_PI))
        self._lunisolar_phase0 = float(rng.uniform(0.0, TWO_PI))

        d_lambda = float(rng.uniform(-0.4, 0.4)) * self.deadband_rad
        a0 = GEO_RADIUS + float(rng.normal(0.0, 2.0e3))
        inc = abs(float(rng.normal(0.0, 1.0))) * math.radians(0.02)
        raan = float(rng.uniform(0.0, TWO_PI))
        ecc = abs(float(rng.normal(0.0, 1.0))) * 2.0e-4

        # Small inclination, so right ascension ~= raan + argument of latitude.
        alpha = self._slot_longitude + d_lambda + self._sidereal0
        u0 = alpha - raan

        ci, si = math.cos(inc), math.sin(inc)
        cr, sr = math.cos(raan), math.sin(raan)
        node = np.array([cr, sr, 0.0])
        inplane = np.array([-sr * ci, cr * ci, si])
        cu, su = math.cos(u0), math.sin(u0)

        r0 = a0 * (1.0 - ecc)
        pos = r0 * (cu * node + su * inplane)
        v0 = math.sqrt(self.mu * (2.0 / r0 - 1.0 / a0))
        vel = v0 * (-su * node + cu * inplane)

        state = VehicleState(
            position_m=pos,
            velocity_m_s=vel,
            dry_mass_kg=self.bus_dry_mass_kg,
            propellant_kg=self.propellant_capacity_kg,
            payload_kg=self.payload_kg,
            t_s=0.0,
        )

        self._acc_t_s = 0.0
        self._t_in_box_s = 0.0
        self._excursions = 0
        self._in_box = True
        self._max_longitude_error = 0.0
        self._max_latitude_error = 0.0
        self._violations = 0
        self._constraint_cost = 0.0
        self._eclipse_s = 0.0
        self._prev_wear = 0.0
        self._cache_t = -1.0e300
        self._cache = None
        self._phi_prev = 0.0
        logger.debug(
            "geo_station_keeping reset: slot %.2f deg E, i0 %.4f deg",
            math.degrees(self._slot_longitude),
            math.degrees(inc),
        )
        return state

    # --- geometry ------------------------------------------------------------
    def _shape(self, state: VehicleState) -> Osculating:
        if state.t_s != self._cache_t or self._cache is None:
            self._cache = osculating(state.position_m, state.velocity_m_s, self.mu)
            self._cache_t = state.t_s
        return self._cache

    def _longitude(self, position: np.ndarray, t_s: float) -> float:
        """Geographic longitude relative to the assigned slot, rad."""
        alpha = math.atan2(float(position[1]), float(position[0]))
        theta_g = self._sidereal0 + OMEGA_EARTH * t_s
        return wrap_pi(alpha - theta_g - self._slot_longitude)

    def _box_errors(self, state: VehicleState) -> tuple[float, float, float]:
        """(longitude error, latitude, longitude drift rate) -- all rad, rad/s."""
        x, y, z = (float(c) for c in state.position_m)
        vx, vy = float(state.velocity_m_s[0]), float(state.velocity_m_s[1])
        d_lambda = self._longitude(state.position_m, state.t_s)
        rxy2 = x * x + y * y
        drift = ((x * vy - y * vx) / rxy2 - OMEGA_EARTH) if rxy2 > 1.0 else 0.0
        rn = state.radius_m
        lat = math.asin(clamp(z / rn, -1.0, 1.0)) if rn > 1.0 else 0.0
        return d_lambda, lat, drift

    def in_deadband(self, state: VehicleState) -> bool:
        d_lambda, lat, _ = self._box_errors(state)
        return abs(d_lambda) <= self.deadband_rad and abs(lat) <= self.deadband_rad

    # --- task ----------------------------------------------------------------
    def _advance(self, state: VehicleState) -> None:
        """Accumulate in-box time and excursion statistics. Idempotent in ``t_s``."""
        dt = state.t_s - self._acc_t_s
        if dt <= 0.0:
            return
        self._acc_t_s = state.t_s
        d_lambda, lat, _ = self._box_errors(state)
        self._max_longitude_error = max(self._max_longitude_error, abs(d_lambda))
        self._max_latitude_error = max(self._max_latitude_error, abs(lat))
        inside = abs(d_lambda) <= self.deadband_rad and abs(lat) <= self.deadband_rad
        if inside:
            self._t_in_box_s += dt
        elif self._in_box:
            self._excursions += 1
        self._in_box = inside
        if self.eclipse(state):
            self._eclipse_s += dt

    def progress(self, state: VehicleState) -> float:
        """Fraction of the design life already delivered inside the box.

        Unlike the transfers, progress here is *service rendered*, not distance
        closed. It rises only while the satellite is where it is supposed to be,
        so the shaping reward is literally "+x for every hour on station".
        """
        self._advance(state)
        return clamp01(self._t_in_box_s / self.max_duration_s)

    def observe_raw(self, state: VehicleState) -> np.ndarray:
        el = self._shape(state)
        d_lambda, lat, drift = self._box_errors(state)
        sun = sun_direction_eci(state.t_s, self._sun_longitude0)
        sun_ra = math.atan2(sun[1], sun[0])

        return np.array(
            [
                soft_frac(d_lambda, 2.0 * self.deadband_rad),
                soft_frac(drift, 1.0e-8),
                soft_frac(lat, 2.0 * self.deadband_rad),
                soft_frac(el.ix, math.radians(0.1)),
                soft_frac(el.iy, math.radians(0.1)),
                soft_frac(el.ex, 5.0e-4),
                soft_frac(el.ey, 5.0e-4),
                el.cos_u,
                math.sin(sun_ra),
                math.cos(sun_ra),
                to_symmetric(
                    clamp01(state.propellant_kg / self.propellant_capacity_kg)
                ),
                to_symmetric(clamp01(state.t_s / self.max_duration_s)),
            ],
            dtype=np.float32,
        )

    def observation_labels(self) -> tuple[str, ...]:
        return (
            "longitude_error",
            "longitude_drift_rate",
            "latitude",
            "inclination_vector_x",
            "inclination_vector_y",
            "ecc_vector_x",
            "ecc_vector_y",
            "cos_arg_lat",
            "sin_sun_ra",
            "cos_sun_ra",
            "propellant_fraction",
            "time_fraction",
        )

    def mission_constraints(self, state: VehicleState) -> ConstraintReport:
        d_lambda, lat, _ = self._box_errors(state)
        return ConstraintReport(
            names=("longitude_deadband", "latitude_deadband"),
            margins=np.array(
                [
                    1.0 - abs(d_lambda) / self.deadband_rad,
                    1.0 - abs(lat) / self.deadband_rad,
                ]
            ),
        )

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
        phi = clamp01(self._t_in_box_s / self.max_duration_s)
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
            # Reaching end of design life still on station is the win condition;
            # the environment reports it as a timeout, which is correct.
            reason = TerminationReason.TIMEOUT
        return terminal_value(
            reason,
            self.config,
            timeout_is_success=self.availability(state) >= self.availability_target,
        )

    def availability(self, state: VehicleState) -> float:
        """Fraction of elapsed mission time spent inside the box."""
        return (self._t_in_box_s / state.t_s) if state.t_s > 0.0 else 1.0

    def terminated(
        self, state: VehicleState, health: HealthReport, constraints: ConstraintReport
    ) -> TerminationReason:
        if not np.all(np.isfinite(state.position_m)) or not np.all(
            np.isfinite(state.velocity_m_s)
        ):
            return TerminationReason.DIVERGED
        el = self._shape(state)
        if el.energy >= 0.0 or el.r > 1.0e9 or el.r < R_EARTH:
            return TerminationReason.DIVERGED

        self._advance(state)
        # Drifting a couple of degrees off slot means intruding on a neighbour's
        # box, which is a collision hazard, not a scoring inconvenience.
        d_lambda, lat, _ = self._box_errors(state)
        if abs(d_lambda) > self.safety_box_rad or abs(lat) > self.safety_box_rad:
            return TerminationReason.SAFETY_VIOLATION
        if constraints.worst < -1.0:
            return TerminationReason.SAFETY_VIOLATION
        if health.failed:
            return TerminationReason.HARDWARE_FAILURE
        if state.propellant_kg <= TINY_MASS_KG:
            return TerminationReason.OUT_OF_PROPELLANT
        # Success is "still here at end of life"; the environment's wall decides
        # when that is, so this method never returns SUCCESS.
        return TerminationReason.RUNNING

    def summarize(self, state: VehicleState, reason: TerminationReason) -> MissionResult:
        self._advance(state)
        availability = self.availability(state)
        completed = state.t_s >= self.max_duration_s - self.step_dt_s
        success = (
            reason in (TerminationReason.TIMEOUT, TerminationReason.SUCCESS)
            and completed
            and availability >= self.availability_target
        )
        years = state.t_s / YEAR
        return MissionResult(
            reason=reason,
            success=success,
            progress=clamp01(self._t_in_box_s / self.max_duration_s),
            elapsed_s=state.t_s,
            delta_v_m_s=state.delta_v_applied_m_s,
            propellant_used_kg=state.propellant_used_kg,
            # Payload *kept on station*: an off-slot transponder sells nothing.
            payload_delivered_kg=self.payload_kg
            * clamp01(self._t_in_box_s / self.max_duration_s),
            terminal_error=math.degrees(self._max_longitude_error),
            constraint_violations=self._violations,
            total_constraint_cost=self._constraint_cost,
            extras={
                "service_availability": availability,
                "deadband_excursions": float(self._excursions),
                "on_station_years": self._t_in_box_s / YEAR,
                "max_longitude_error_deg": math.degrees(self._max_longitude_error),
                "max_latitude_error_deg": math.degrees(self._max_latitude_error),
                "slot_longitude_deg": math.degrees(self._slot_longitude),
                "delta_v_per_year_m_s": (
                    state.delta_v_applied_m_s / years if years > 0.0 else 0.0
                ),
                "propellant_per_year_kg": (
                    state.propellant_used_kg / years if years > 0.0 else 0.0
                ),
                "eclipse_fraction": (
                    self._eclipse_s / state.t_s if state.t_s > 0.0 else 0.0
                ),
                "propellant_remaining_kg": state.propellant_kg,
            },
        )

    # --- dynamics ------------------------------------------------------------
    def gravity(self, state: VehicleState) -> np.ndarray:
        """Point mass + J2 + triaxiality + luni-solar drift + SRP.

        Written in scalar arithmetic and building exactly one array: an RK
        integrator calls this several times per step for ~15 000 steps per
        episode, so the numpy overhead of the obvious vectorised version would
        dominate the whole environment.
        """
        x, y, z = (float(c) for c in state.position_m)
        r2 = x * x + y * y + z * z
        rn = math.sqrt(r2)
        if rn < 1.0:
            return np.zeros(3)

        inv_r3 = 1.0 / (r2 * rn)
        ax = -self.mu * x * inv_r3
        ay = -self.mu * y * inv_r3
        az = -self.mu * z * inv_r3

        # --- J2 ---------------------------------------------------------------
        zr = z / rn
        k_j2 = self._j2_coeff / (r2 * r2)
        f_xy = 5.0 * zr * zr - 1.0
        ax += k_j2 * (x / rn) * f_xy
        ay += k_j2 * (y / rn) * f_xy
        az += k_j2 * zr * (5.0 * zr * zr - 3.0)

        vx, vy, vz = (float(c) for c in state.velocity_m_s)

        # --- triaxiality (J22): along-track, towards the nearest stable node ---
        d_lambda = self._longitude(state.position_m, state.t_s) + self._slot_longitude
        a22 = -(self._a22_coeff / (r2 * r2)) * math.sin(2.0 * (d_lambda - LAMBDA_22))
        # Unit transverse vector: velocity with the radial part removed.
        vr = (x * vx + y * vy + z * vz) / rn
        tx, ty, tz = vx - vr * x / rn, vy - vr * y / rn, vz - vr * z / rn
        tn = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tn > 1.0:
            ax += a22 * tx / tn
            ay += a22 * ty / tn
            az += a22 * tz / tn

        # --- luni-solar: secular inclination drift, precessing with the node ---
        alpha = math.atan2(y, x)
        phase = self._lunisolar_phase0 + TWO_PI * state.t_s / LUNAR_NODE_PERIOD_S
        az += self._a_lunisolar * math.cos(alpha - phase)

        # --- solar radiation pressure -----------------------------------------
        sun = sun_direction_eci(state.t_s, self._sun_longitude0)
        if not in_cylindrical_shadow(state.position_m, sun, R_EARTH):
            ax -= self._a_srp * sun[0]
            ay -= self._a_srp * sun[1]
            az -= self._a_srp * sun[2]

        return np.array([ax, ay, az])

    def eclipse(self, state: VehicleState) -> bool:
        return in_cylindrical_shadow(
            state.position_m, sun_direction_eci(state.t_s, self._sun_longitude0), R_EARTH
        )

    def heliocentric_radius_m(self, state: VehicleState) -> float:
        """Distance to the *Sun*, m. See the note in :mod:`~.leo_geo_transfer`.

        The base class would return the geocentric radius (4.2e7 m), which is
        1/3500th of the true Sun distance and would starve every solar-powered
        bus in the sweep of power for a decade.
        """
        sun = sun_direction_eci(state.t_s, self._sun_longitude0)
        r = state.position_m
        along = float(r[0]) * sun[0] + float(r[1]) * sun[1] + float(r[2]) * sun[2]
        return EARTH_SMA - along

    def info(self) -> dict[str, float]:
        v_geo = math.sqrt(self.mu / GEO_RADIUS)
        return {
            "deadband_deg": math.degrees(self.deadband_rad),
            "north_south_budget_m_s_year": LUNISOLAR_DRIFT_RAD_YEAR * v_geo,
            "east_west_budget_m_s_year": (
                self._a22_coeff / GEO_RADIUS**4
            ) * YEAR,
            "srp_accel_m_s2": self._a_srp,
            "propellant_capacity_kg": self.propellant_capacity_kg,
            "payload_kg": self.payload_kg,
            "design_life_years": self.max_duration_s / YEAR,
            "steps_per_episode": self.max_duration_s / self.step_dt_s,
        }
