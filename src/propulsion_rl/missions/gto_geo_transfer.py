"""GTO to GEO: the transfer all-electric communications satellites actually fly.

Why this mission exists beside ``leo_geo_transfer``
---------------------------------------------------
The LEO start is the textbook orbit-raising problem, but it is not what anybody
launches an electric comsat into, and as a reinforcement-learning benchmark it
has a disqualifying property: the transfer needs ~5.9 km/s, which at
Hall-thruster acceleration is ~900 days of thrusting, which at a one-hour
macro-step is ~21,000 environment steps *per episode*. A 300,000-step training
budget is then fourteen episodes, and no model-free method learns continuous
control from fourteen trajectories. That mission is measurable by an analytical
steering law and unmeasurable by a learner, which makes it useless for comparing
the two.

A launcher drop-off in geostationary transfer orbit changes the arithmetic
rather than the physics. The remaining budget falls to roughly 1.8 km/s, the
transfer to a few hundred days, and the episode to a couple of thousand steps --
inside the range where an RL method gets enough trajectories to be judged on its
merits. Real all-electric platforms (Boeing 702SP and successors) fly exactly
this and take four to eight months doing it, so the shortened problem is also the
more realistic one.

What is different here
----------------------
Only the starting orbit and the vehicle scale. The reward, the observation, the
radiation and eclipse bookkeeping, the insertion tolerances and the economics are
inherited unchanged from :class:`LEOtoGEOTransfer`, so results on the two
missions are directly comparable and a fix to one lands on both.

The start is a standard Cape-latitude GTO: 185 km by 35,786 km at 27 degrees, an
eccentricity of 0.73. That last number is the interesting part. Edelbaum's law is
the optimum of the *circle-to-circle* problem, so at ignition it is being asked
to steer an orbit well outside its domain of validity, and it has to circularise
before its own assumptions hold. Whether a learned controller exploits that
opening is precisely the question this benchmark exists to ask -- and on the LEO
mission it could not be asked at all, because no learner could be trained there.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.constants import GEO_RADIUS, HOUR, R_EARTH, YEAR
from ..core.registry import MISSION
from ..core.types import VehicleState
from ..spacecraft.orbit import TWO_PI
from .leo_geo_transfer import LEOtoGEOTransfer
from .rewards import RewardConfig

logger = logging.getLogger(__name__)

#: Perigee altitude of a standard geostationary transfer orbit, m.
GTO_PERIGEE_ALT_M = 185e3

#: Apogee altitude, m. A launcher targets geostationary *radius* at apogee, so
#: what remains is a perigee raise and a plane change rather than an orbit raise.
GTO_APOGEE_ALT_M = 35_786e3


@MISSION.register(
    "gto_geo_transfer",
    frame="planetocentric",
    regime="orbit_raising",
    favours="electric",
    horizon_days=730.0,
)
class GTOtoGEOTransfer(LEOtoGEOTransfer):
    """Circularise and de-incline from a launcher GTO drop-off into a GEO slot.

    Mass budget follows the parent convention: the mission fixes the launched wet
    mass and the payload, and propellant is whatever is left once the stage
    structure and the propulsion system's own dry mass are deducted. The vehicle
    is smaller than the LEO stage because a GTO drop-off is what a commercial
    launcher actually delivers for this class of spacecraft.
    """

    name = "gto_geo_transfer"

    def __init__(
        self,
        *,
        wet_mass_kg: float = 3000.0,
        stage_dry_mass_kg: float = 800.0,
        payload_kg: float = 1200.0,
        perigee_altitude_m: float = GTO_PERIGEE_ALT_M,
        apogee_altitude_m: float = GTO_APOGEE_ALT_M,
        start_inclination_rad: float = math.radians(27.0),
        target_radius_m: float = GEO_RADIUS,
        step_dt_s: float = 2.0 * HOUR,
        max_duration_s: float = 2.0 * YEAR,
        radius_tolerance: float = 0.004,
        inclination_tolerance_rad: float = math.radians(0.5),
        eccentricity_tolerance: float = 0.015,
        dose_soft_limit_krad: float = 2000.0,
        dose_hard_limit_krad: float = 6000.0,
        min_altitude_m: float = 150e3,
        config: RewardConfig | None = None,
    ) -> None:
        # Set before super().__init__, because the parent constructor calls
        # _reference_start_radius() and these are what it reads.
        self.perigee_radius_m = R_EARTH + float(perigee_altitude_m)
        self.apogee_radius_m = R_EARTH + float(apogee_altitude_m)

        super().__init__(
            wet_mass_kg=wet_mass_kg,
            stage_dry_mass_kg=stage_dry_mass_kg,
            payload_kg=payload_kg,
            start_altitude_m=perigee_altitude_m,
            start_inclination_rad=start_inclination_rad,
            target_radius_m=target_radius_m,
            step_dt_s=step_dt_s,
            max_duration_s=max_duration_s,
            radius_tolerance=radius_tolerance,
            inclination_tolerance_rad=inclination_tolerance_rad,
            eccentricity_tolerance=eccentricity_tolerance,
            dose_soft_limit_krad=dose_soft_limit_krad,
            dose_hard_limit_krad=dose_hard_limit_krad,
            min_altitude_m=min_altitude_m,
            config=config,
        )

    @property
    def start_semi_major_axis_m(self) -> float:
        """Semi-major axis of the nominal drop-off ellipse, m."""
        return 0.5 * (self.perigee_radius_m + self.apogee_radius_m)

    @property
    def start_eccentricity(self) -> float:
        """Eccentricity of the nominal drop-off ellipse."""
        return (self.apogee_radius_m - self.perigee_radius_m) / (
            self.apogee_radius_m + self.perigee_radius_m
        )

    def _reference_start_radius(self) -> float:
        """Normalise ``sma_gap`` against the transfer that is actually flown.

        Inheriting the parent's LEO span would tell a controller it had roughly
        five times further to go than it does, and the steering law reads that
        span to recover its speed ratio -- it would spend the entire transfer
        behaving as though it were still at the start of one.
        """
        return self.start_semi_major_axis_m

    def _reference_eccentricity_scale(self) -> float:
        """Scale the eccentricity channels to the ellipse actually flown.

        Inheriting the parent's near-circular 0.05 would put ``tanh(0.73/0.05)``
        into the observation: a hard +/-1 with no gradient for the entire first
        half of the transfer, exactly while circularising is the whole task.
        """
        return max(self.start_eccentricity, 0.05)

    def reset(self, rng: np.random.Generator) -> VehicleState:
        """Sample a launcher drop-off: dispersed ellipse, random node and epoch.

        Randomised the way an upper stage actually disperses: perigee altitude
        (+/-15 km), apogee (+/-120 km -- injection error is far larger at the far
        end of the ellipse), inclination (+/-0.15 deg), argument of perigee, RAAN
        and epoch, the last of which sets the eclipse season.

        The spacecraft starts at perigee, which is both the conventional drop-off
        point and the demanding part of the arc: the fastest, deepest in the
        gravity well, and inside the inner radiation belt.
        """
        rp = self.perigee_radius_m + float(rng.uniform(-15e3, 15e3))
        ra = self.apogee_radius_m + float(rng.uniform(-120e3, 120e3))
        if ra < rp:
            rp, ra = ra, rp
        a0 = 0.5 * (rp + ra)
        ecc = (ra - rp) / (ra + rp)

        inc = self.start_inclination_rad + float(
            rng.uniform(-1.0, 1.0)
        ) * math.radians(0.15)
        raan = float(rng.uniform(0.0, TWO_PI))
        argp = float(rng.uniform(0.0, TWO_PI))
        self._sun_longitude0 = float(rng.uniform(0.0, TWO_PI))

        ci, si = math.cos(inc), math.sin(inc)
        cr, sr = math.cos(raan), math.sin(raan)
        cw, sw = math.cos(argp), math.sin(argp)

        # Orbit-plane basis, then the perifocal axes rotated by argument of
        # perigee: p_hat points at perigee, q_hat 90 degrees ahead of it.
        node = np.array([cr, sr, 0.0])
        inplane = np.array([-sr * ci, cr * ci, si])
        p_hat = cw * node + sw * inplane
        q_hat = -sw * node + cw * inplane

        # At perigee the velocity is purely transverse.
        pos = rp * p_hat
        v_peri = math.sqrt(self.mu * (1.0 + ecc) / (a0 * (1.0 - ecc)))
        vel = v_peri * q_hat
        vel += rng.normal(0.0, 5.0, size=3)          # injection dispersion, m/s

        state = VehicleState(
            position_m=pos,
            velocity_m_s=vel,
            dry_mass_kg=self.total_dry_mass_kg,
            propellant_kg=self.propellant_capacity_kg,
            payload_kg=self.payload_kg,
            t_s=0.0,
        )

        self._reset_episode_accumulators(state)
        logger.debug(
            "gto_geo_transfer reset: a0=%.1f km e0=%.4f i0=%.3f deg dv_ref=%.1f m/s",
            a0 * 1e-3,
            ecc,
            math.degrees(inc),
            self._dv_reference,
        )
        return state
