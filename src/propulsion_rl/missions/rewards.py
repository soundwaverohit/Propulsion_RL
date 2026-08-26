"""Shared machinery for the mission suite.

Three things live here, and they live here because the cross-mission comparison
is only legitimate if every mission uses the *same* shaping scheme:

1. **Potential-based shaping** (Ng, Harada & Russell, 1999). Progress reward is
   always ``F(s, s') = gamma * Phi(s') - Phi(s)`` for a bounded potential
   ``Phi``, which provably leaves the optimal policy unchanged. No mission is
   allowed to invent its own progress bonus.

2. **:class:`RewardConfig`** -- every weight in :class:`~.base.RewardTerms` as a
   dataclass field, so an ablation ("does the propellant penalty matter?") is a
   config change, not a code change.

3. **Normalisation.** Every term is expressed as a dimensionless fraction times
   a single ``return_scale``. The consequence is worth stating explicitly,
   because it is what makes a learning-curve plot across four missions readable:

   * With ``gamma = 1`` the shaping term telescopes over an episode to
     ``w_progress * return_scale * (Phi_final - Phi_initial)``. Missions build
     ``Phi`` to run 0 -> 1, so a completed mission returns ``~ +100`` of progress
     reward whether it took 2 500 steps or 15 000.
   * ``time_penalty`` integrates to ``-w_time * return_scale`` over the full
     ``max_duration_s``.
   * ``propellant_penalty`` integrates to ``-w_propellant * return_scale`` if the
     whole tank is spent.
   * ``terminal`` is ``+w_terminal * success_bonus * return_scale`` on success.

   So every mission's return lives in roughly ``[-400, +250]``. That is the
   point.

A note on ``gamma``: the invariance theorem is stated for a *discounted* MDP and
requires the shaping ``gamma`` to equal the agent's discount factor. Missions do
not know the agent's discount, and using ``gamma < 1`` here injects a per-step
drain of ``-(1 - gamma) * Phi`` that, over a 15 000-step station-keeping
episode, would swamp every other term. The default is therefore ``1.0``, which
is exactly potential-based shaping for the undiscounted episodic case. Set it to
the agent's discount if you want the discounted guarantee instead.

Also collected here: the handful of geometric primitives the mission potentials
are built from (osculating elements, shadow test, solar direction). They are
shared rather than duplicated across the four mission modules so that "progress"
means the same thing everywhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..core.constants import EPS, R_EARTH, YEAR
from ..core.types import TerminationReason
from .base import RewardTerms

logger = logging.getLogger(__name__)

__all__ = [
    "RewardConfig",
    "RewardScales",
    "Osculating",
    "potential_shaping",
    "assemble_terms",
    "accumulate_terms",
    "terminal_value",
    "clamp",
    "clamp01",
    "to_symmetric",
    "signed_frac",
    "soft_frac",
    "wrap_pi",
    "osculating",
    "sun_direction_eci",
    "in_cylindrical_shadow",
    "TWO_PI",
    "OBLIQUITY_RAD",
]

TWO_PI = 2.0 * math.pi
#: Mean obliquity of the ecliptic, rad. Sets the Sun's declination history and
#: therefore the eclipse season structure in an equatorial planetocentric frame.
OBLIQUITY_RAD = math.radians(23.4393)
_COS_OBL = math.cos(OBLIQUITY_RAD)
_SIN_OBL = math.sin(OBLIQUITY_RAD)


# --- small numeric helpers ---------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Hard clip. Scalar-only and branchy on purpose: this is a hot-loop helper
    and ``np.clip`` on a Python float costs ~20x more."""
    return lo if x < lo else (hi if x > hi else x)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def to_symmetric(x01: float) -> float:
    """Map a [0, 1] quantity onto the [-1, 1] observation convention."""
    return 2.0 * x01 - 1.0


def signed_frac(x: float, scale: float) -> float:
    """``x / scale`` hard-clipped to [-1, 1]. Linear where the signal matters."""
    if scale <= EPS:
        return 0.0
    v = x / scale
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


def soft_frac(x: float, scale: float) -> float:
    """``tanh(x / scale)``: accepts unbounded input, ~linear near zero.

    Preferred over :func:`signed_frac` when the quantity has no natural maximum
    (a miss distance, an eccentricity) and the policy still benefits from seeing
    gradient far outside the nominal range.
    """
    if scale <= EPS:
        return 0.0
    return math.tanh(x / scale)


def wrap_pi(angle: float) -> float:
    """Wrap an angle onto (-pi, pi]."""
    return (angle + math.pi) % TWO_PI - math.pi


# --- reward configuration ----------------------------------------------------
@dataclass(slots=True)
class RewardConfig:
    """Weights for every :class:`~.base.RewardTerms` component.

    Each ``w_*`` is the fraction of ``return_scale`` that the corresponding term
    contributes over a *whole* nominal episode, not per step. Missions override
    the handful of weights that encode what they actually value (a crewed
    transfer weights time heavily; a station-keeping mission weights propellant
    heavily and time not at all) and inherit the rest, so ablations stay
    comparable.
    """

    #: Sets the magnitude of every term. One number, one place.
    return_scale: float = 100.0
    #: Shaping discount. 1.0 == undiscounted episodic shaping; see module docs.
    gamma: float = 1.0

    w_progress: float = 1.0
    w_efficiency: float = 0.15
    w_time: float = 0.25
    w_propellant: float = 0.35
    w_wear: float = 0.10
    w_constraint: float = 1.0
    w_terminal: float = 1.0

    #: Terminal multipliers, in units of ``return_scale``.
    success_bonus: float = 1.0
    failure_malus: float = 0.5
    hardware_malus_gain: float = 1.2
    safety_malus_gain: float = 1.5
    divergence_malus_gain: float = 2.0

    #: A constraint held in violation for the entire episode costs
    #: ``w_constraint * constraint_gain * return_scale``.
    constraint_gain: float = 4.0

    #: Specific impulse that scores neutrally on the ``efficiency`` term. Set per
    #: mission to something a *reasonable* system for that mission achieves, so
    #: the term ranks operation quality rather than propulsion family.
    isp_reference_s: float = 1500.0
    #: Clip on ``isp / isp_reference_s``. Deliberately narrow: the efficiency
    #: term is a tiebreaker, and a wide clip would let Isp alone decide the
    #: benchmark before the physics got a vote.
    isp_ratio_min: float = 0.4
    isp_ratio_max: float = 1.6


@dataclass(frozen=True, slots=True)
class RewardScales:
    """Per-mission denominators that turn physical amounts into fractions.

    Fixed at :meth:`~.base.Mission.reset` time and constant for the episode.
    """

    duration_s: float
    propellant_capacity_kg: float
    delta_v_reference_m_s: float


def potential_shaping(phi_prev: float, phi_next: float, gamma: float = 1.0) -> float:
    """``F = gamma * Phi(s') - Phi(s)`` -- Ng, Harada & Russell (1999).

    Any reward of this form leaves the set of optimal policies unchanged, which
    is the only reason the mission suite is allowed dense progress reward at all.
    """
    return gamma * phi_next - phi_prev


def terminal_value(
    reason: TerminationReason, cfg: RewardConfig, *, timeout_is_success: bool = False
) -> float:
    """Terminal bonus/malus in units of ``return_scale``, before ``w_terminal``.

    Shared so that "failing" costs the same across missions and a cross-mission
    return table is not secretly comparing different failure prices.
    """
    if reason is TerminationReason.SUCCESS:
        return cfg.success_bonus
    if reason is TerminationReason.TIMEOUT:
        return cfg.success_bonus if timeout_is_success else -cfg.failure_malus
    if reason is TerminationReason.OUT_OF_PROPELLANT:
        return -cfg.failure_malus
    if reason is TerminationReason.HARDWARE_FAILURE:
        return -cfg.failure_malus * cfg.hardware_malus_gain
    if reason is TerminationReason.SAFETY_VIOLATION:
        return -cfg.failure_malus * cfg.safety_malus_gain
    if reason is TerminationReason.DIVERGED:
        return -cfg.failure_malus * cfg.divergence_malus_gain
    return 0.0


def assemble_terms(
    cfg: RewardConfig,
    scales: RewardScales,
    *,
    d_potential: float,
    dt_s: float,
    propellant_kg: float,
    d_wear: float,
    constraint_cost: float,
    isp_s: float,
    terminal: float = 0.0,
) -> RewardTerms:
    """Build a fully populated :class:`~.base.RewardTerms` from step deltas.

    Parameters
    ----------
    d_potential:
        Already-shaped progress increment, i.e. the output of
        :func:`potential_shaping`. Passing a raw progress delta would silently
        break the invariance guarantee, so missions call the shaper themselves.
    propellant_kg:
        Propellant consumed *this step*, kg.
    d_wear:
        Increase in ``HealthReport.wear_fraction`` this step.
    constraint_cost:
        Summed normalised violation magnitude (propulsion + mission).
    isp_s:
        Specific impulse actually delivered this step. ``dv/dm`` is exactly
        ``Isp * g0``, so scoring Isp *is* scoring delta-v per kg. Non-positive
        (a coast) scores neutrally rather than badly.
    terminal:
        Terminal multiplier from :func:`terminal_value`; 0 on a non-terminal step.
    """
    scale = cfg.return_scale

    if isp_s > 0.0:
        ratio = clamp(isp_s / cfg.isp_reference_s, cfg.isp_ratio_min, cfg.isp_ratio_max)
    else:
        ratio = 1.0

    # Efficiency is gated on forward progress, which makes it un-farmable: an
    # agent cannot thrust in circles at high Isp to collect it.
    gain = d_potential if d_potential > 0.0 else 0.0

    time_frac = dt_s / scales.duration_s if scales.duration_s > EPS else 0.0
    prop_frac = (
        propellant_kg / scales.propellant_capacity_kg
        if scales.propellant_capacity_kg > EPS
        else 0.0
    )

    return RewardTerms(
        progress=cfg.w_progress * scale * d_potential,
        efficiency=cfg.w_efficiency * scale * gain * ratio,
        time_penalty=-cfg.w_time * scale * time_frac,
        propellant_penalty=-cfg.w_propellant * scale * prop_frac,
        wear_penalty=-cfg.w_wear * scale * (d_wear if d_wear > 0.0 else 0.0),
        constraint_penalty=(
            -cfg.w_constraint * cfg.constraint_gain * scale * constraint_cost * time_frac
        ),
        terminal=cfg.w_terminal * scale * terminal,
    )


def accumulate_terms(acc: dict[str, float], terms: RewardTerms) -> dict[str, float]:
    """Sum a :class:`~.base.RewardTerms` into a running dict, in place."""
    for key, value in terms.as_dict().items():
        acc[key] = acc.get(key, 0.0) + value
    return acc


# --- geometry the potentials are built from ----------------------------------
@dataclass(frozen=True, slots=True)
class Osculating:
    """Everything the mission potentials need from ``(r, v)``, in one pass.

    Computed together because the intermediates (``h``, ``r . v``, ``|r|``) are
    shared; computing ``a`` and ``e`` and ``i`` separately would triple the cost
    of the hottest function in the package.

    The equinoctial-flavoured components (``ex``/``ey``, ``ix``/``iy``) are used
    instead of ``(e, argp)`` and ``(i, raan)`` because they stay well-defined and
    continuous as ``e -> 0`` and ``i -> 0``, which is exactly the regime a GEO
    slot lives in.
    """

    r: float          # |position|, m
    a: float          # semi-major axis, m (negative on a hyperbolic orbit)
    e: float          # eccentricity
    inc: float        # inclination, rad
    ex: float         # eccentricity vector, inertial x
    ey: float         # eccentricity vector, inertial y
    ix: float         # inclination vector, sin(i) cos(raan)
    iy: float         # inclination vector, sin(i) sin(raan)
    sin_u: float      # argument of latitude (true longitude if equatorial)
    cos_u: float
    vr: float         # radial speed, m/s
    vt: float         # transverse speed, m/s (h / r)
    theta: float      # true longitude atan2(y, x), rad
    energy: float     # specific orbital energy, J/kg


def osculating(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> Osculating:
    """Osculating elements from a Cartesian state. Allocation-free scalar math.

    Called several times per environment step by every mission, so it is written
    against Python floats rather than numpy vector ops.
    """
    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    vx, vy, vz = float(v_vec[0]), float(v_vec[1]), float(v_vec[2])

    rn = math.sqrt(x * x + y * y + z * z)
    if rn < 1.0:
        return Osculating(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    v2 = vx * vx + vy * vy + vz * vz
    rdotv = x * vx + y * vy + z * vz

    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)

    energy = 0.5 * v2 - mu / rn
    a = -mu / (2.0 * energy) if abs(energy) > EPS else math.inf

    # e_vec = ((v^2 - mu/r) r - (r.v) v) / mu -- cheaper than two cross products.
    k = v2 - mu / rn
    ex = (k * x - rdotv * vx) / mu
    ey = (k * y - rdotv * vy) / mu
    ez = (k * z - rdotv * vz) / mu
    e = math.sqrt(ex * ex + ey * ey + ez * ez)

    if hn > EPS:
        cos_i = clamp(hz / hn, -1.0, 1.0)
        inc = math.acos(cos_i)
        # Node vector n = z_hat x h = (-hy, hx, 0); |n| = hn sin(i).
        nx, ny = -hy, hx
        nn = math.sqrt(nx * nx + ny * ny)
        ix, iy = nx / hn, ny / hn          # = sin(i) cos(raan), sin(i) sin(raan)
        if nn > 1e-9 * hn:
            cos_u = (nx * x + ny * y) / (nn * rn)
            sin_u = z * hn / (rn * nn)
        else:
            # Equatorial: the node is undefined, so measure from the x-axis.
            cos_u, sin_u = x / rn, y / rn
    else:
        inc, ix, iy = 0.0, 0.0, 0.0
        cos_u, sin_u = x / rn, y / rn

    return Osculating(
        r=rn,
        a=a,
        e=e,
        inc=inc,
        ex=ex,
        ey=ey,
        ix=ix,
        iy=iy,
        sin_u=clamp(sin_u, -1.0, 1.0),
        cos_u=clamp(cos_u, -1.0, 1.0),
        vr=rdotv / rn,
        vt=hn / rn,
        theta=math.atan2(y, x),
        energy=energy,
    )


def sun_direction_eci(t_s: float, longitude0_rad: float) -> tuple[float, float, float]:
    """Unit vector towards the Sun in an equatorial planetocentric frame.

    Circular-Earth, mean-obliquity model: good to ~1 degree, which is far inside
    the resolution at which eclipse duty cycle actually matters here.
    """
    lam = longitude0_rad + TWO_PI * t_s / YEAR
    cl, sl = math.cos(lam), math.sin(lam)
    return cl, sl * _COS_OBL, sl * _SIN_OBL


def in_cylindrical_shadow(
    r_vec: np.ndarray, sun_hat: tuple[float, float, float], body_radius_m: float = R_EARTH
) -> bool:
    """Cylindrical (umbra-only) shadow test.

    The vehicle is shadowed when it is on the anti-sun side of the body *and*
    inside the cylinder the body casts. Ignores penumbra and the Sun's finite
    angular size; at LEO that is a ~10 s error on a ~35 min eclipse.
    """
    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    proj = x * sun_hat[0] + y * sun_hat[1] + z * sun_hat[2]
    if proj >= 0.0:
        return False
    r2 = x * x + y * y + z * z
    return (r2 - proj * proj) < body_radius_m * body_radius_m
