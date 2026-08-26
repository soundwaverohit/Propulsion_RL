"""Trajectory propagation under gravity plus a thrust command.

This module is the innermost loop of the whole benchmark: an episode is
thousands of macro-steps and a sweep is millions of episodes-worth of steps, so
everything here is written against plain Python floats rather than small numpy
arrays. A ``numpy`` 3-vector operation costs ~1 us of dispatch overhead; the
same arithmetic on unpacked floats costs ~50 ns. Only the boundary (inputs and
the returned :class:`VehicleState`) is expressed in arrays.

Physics contract
----------------
* Gravity is whatever ``mission.gravity(state)`` returns, so a mission can add
  third bodies, J2 or SRP without this module knowing. A fast path detects the
  un-overridden point-mass default and inlines it.
* The thrust direction is fixed in the *rotating* RTN frame for the duration of
  the macro-step, because that is what the agent commands. The inertial
  direction is therefore re-derived at every integrator stage; over a months-long
  low-thrust spiral, holding the inertial direction fixed instead would rotate
  the burn off prograde by tens of degrees within a single step.
* Mass is depleted *during* the step, so the thrust acceleration ``F/m`` grows
  as propellant burns. Freezing the acceleration at its start-of-step value is
  the classic silent accuracy loss in low-thrust simulations.
* A tank that empties mid-step cuts thrust at the instant it empties, not at the
  step boundary: the step is split into a burn arc and a coast arc.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.constants import EPS, TINY_MASS_KG
from ..core.types import VehicleState

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from ..missions.base import Mission

logger = logging.getLogger(__name__)

__all__ = [
    "DivergedState",
    "PropagationStatus",
    "MIN_RADIUS_M",
    "MAX_RADIUS_M",
    "rtn_basis",
    "rtn_to_inertial",
    "inertial_to_rtn",
    "propagate",
    "propagate_with_status",
    "is_diverged",
    "specific_energy",
]

# --- Divergence guards -------------------------------------------------------
#: Below this radius the point-mass potential is numerically unusable (and the
#: vehicle is inside any plausible central body anyway).
MIN_RADIUS_M = 1.0e3
#: Beyond this radius the trajectory has escaped anything the benchmark models;
#: keeping it finite stops an overflow from turning into a NaN cascade.
MAX_RADIUS_M = 1.0e15
#: Speed ceiling, a hair under c. Anything faster is an integrator blow-up.
MAX_SPEED_M_S = 2.9e8

_MIN_R2 = MIN_RADIUS_M * MIN_RADIUS_M
_MAX_R2 = MAX_RADIUS_M * MAX_RADIUS_M
_MAX_V2 = MAX_SPEED_M_S * MAX_SPEED_M_S

#: Fallback thrust direction (pure prograde) when a caller hands us a zero vector.
_DEFAULT_DIR = (0.0, 1.0, 0.0)


@dataclass(slots=True)
class DivergedState(VehicleState):
    """A :class:`VehicleState` tagged as numerically invalid.

    ``propagate`` never returns NaNs. When the integrator blows up -- radius
    collapsing to zero, an overflow, a non-finite gravity model -- it returns
    the last finite sub-state wrapped in this subclass instead. It *is* a
    ``VehicleState`` (``isinstance`` passes, ``copy`` preserves the tag), so the
    environment can keep using it while :func:`is_diverged` tells it to end the
    episode with ``TerminationReason.DIVERGED``.
    """

    diverged_reason: str = "unknown"


@dataclass(slots=True)
class PropagationStatus:
    """Diagnostics for one call to :func:`propagate`."""

    ok: bool = True
    reason: str = ""
    burn_time_s: float = 0.0
    propellant_used_kg: float = 0.0
    delta_v_m_s: float = 0.0
    cutoff: bool = False          # tank ran dry inside the step
    steps_taken: int = 0          # integrator sub-intervals actually used
    max_error: float = 0.0        # RK45 only: largest accepted normalised error


def is_diverged(state: VehicleState) -> bool:
    """True when ``state`` must not be propagated any further."""
    if isinstance(state, DivergedState):
        return True
    r = state.position_m
    v = state.velocity_m_s
    r2 = float(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v2 = float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    # NaN fails every comparison, so this catches non-finite values too.
    return not (_MIN_R2 <= r2 <= _MAX_R2 and v2 <= _MAX_V2)


def specific_energy(state: VehicleState, mu: float) -> float:
    """Keplerian specific orbital energy, J/kg. Handy for drift diagnostics."""
    r = state.position_m
    v = state.velocity_m_s
    rn = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    return float(0.5 * v2 - mu / rn)


# --- Frames ------------------------------------------------------------------
def rtn_basis(position_m: np.ndarray, velocity_m_s: np.ndarray) -> np.ndarray:
    """(3,3) matrix whose COLUMNS are the radial, transverse, normal unit vectors.

    ``R`` points away from the central body, ``N`` along the orbital angular
    momentum, and ``T = N x R`` completes the right-handed set (along-track,
    equal to the velocity direction for a circular orbit).

    Because the columns are orthonormal, the inverse rotation is the transpose;
    :func:`inertial_to_rtn` uses that.
    """
    rx, ry, rz = float(position_m[0]), float(position_m[1]), float(position_m[2])
    vx, vy, vz = float(velocity_m_s[0]), float(velocity_m_s[1]), float(velocity_m_s[2])
    ur, ut, un = _rtn_axes(rx, ry, rz, vx, vy, vz)
    out = np.empty((3, 3), dtype=np.float64)
    out[0, 0], out[1, 0], out[2, 0] = ur
    out[0, 1], out[1, 1], out[2, 1] = ut
    out[0, 2], out[1, 2], out[2, 2] = un
    return out


def rtn_to_inertial(
    vec_rtn: np.ndarray, position_m: np.ndarray, velocity_m_s: np.ndarray
) -> np.ndarray:
    """Rotate a vector from the local RTN frame into the inertial frame."""
    rx, ry, rz = float(position_m[0]), float(position_m[1]), float(position_m[2])
    vx, vy, vz = float(velocity_m_s[0]), float(velocity_m_s[1]), float(velocity_m_s[2])
    (urx, ury, urz), (utx, uty, utz), (unx, uny, unz) = _rtn_axes(
        rx, ry, rz, vx, vy, vz
    )
    a, b, c = float(vec_rtn[0]), float(vec_rtn[1]), float(vec_rtn[2])
    return np.array(
        [
            a * urx + b * utx + c * unx,
            a * ury + b * uty + c * uny,
            a * urz + b * utz + c * unz,
        ],
        dtype=np.float64,
    )


def inertial_to_rtn(
    vec_inertial: np.ndarray, position_m: np.ndarray, velocity_m_s: np.ndarray
) -> np.ndarray:
    """Rotate a vector from the inertial frame into the local RTN frame."""
    rx, ry, rz = float(position_m[0]), float(position_m[1]), float(position_m[2])
    vx, vy, vz = float(velocity_m_s[0]), float(velocity_m_s[1]), float(velocity_m_s[2])
    (urx, ury, urz), (utx, uty, utz), (unx, uny, unz) = _rtn_axes(
        rx, ry, rz, vx, vy, vz
    )
    a, b, c = float(vec_inertial[0]), float(vec_inertial[1]), float(vec_inertial[2])
    return np.array(
        [
            a * urx + b * ury + c * urz,
            a * utx + b * uty + c * utz,
            a * unx + b * uny + c * unz,
        ],
        dtype=np.float64,
    )


def _rtn_axes(
    rx: float, ry: float, rz: float, vx: float, vy: float, vz: float
) -> tuple[tuple[float, float, float], ...]:
    """Radial / transverse / normal unit vectors as three float triples.

    Degenerate cases (zero radius, purely radial motion where the angular
    momentum vanishes) fall back to an arbitrary but *consistent* right-handed
    triad so the integrator keeps producing finite numbers.
    """
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rn < EPS:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    inv = 1.0 / rn
    urx, ury, urz = rx * inv, ry * inv, rz * inv

    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    if hn < EPS * rn:
        # Radial trajectory: no orbit plane. Pick any axis not parallel to r.
        if abs(urz) < 0.9:
            ax, ay, az = 0.0, 0.0, 1.0
        else:
            ax, ay, az = 1.0, 0.0, 0.0
        hx = ury * az - urz * ay
        hy = urz * ax - urx * az
        hz = urx * ay - ury * ax
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    inv = 1.0 / hn
    unx, uny, unz = hx * inv, hy * inv, hz * inv

    utx = uny * urz - unz * ury
    uty = unz * urx - unx * urz
    utz = unx * ury - uny * urx
    return (urx, ury, urz), (utx, uty, utz), (unx, uny, unz)


# --- Gravity dispatch --------------------------------------------------------
# Detecting the un-overridden Mission.gravity lets us inline the point-mass
# acceleration instead of paying a Python call plus a numpy allocation at every
# integrator stage (four per substep). Cached per mission class; the check is a
# qualname comparison so this module never has to import missions.base and risk
# an import cycle with mission modules that import us.
_GRAVITY_FASTPATH: dict[type, bool] = {}


def _use_fast_gravity(mission: Any) -> bool:
    cls = type(mission)
    hit = _GRAVITY_FASTPATH.get(cls)
    if hit is None:
        fn = getattr(type(mission).gravity, "__func__", type(mission).gravity)
        hit = (
            getattr(fn, "__qualname__", "") == "Mission.gravity"
            and getattr(fn, "__module__", "").endswith("missions.base")
        )
        _GRAVITY_FASTPATH[cls] = hit
    return hit


# --- Butcher tableaux --------------------------------------------------------
_RK4_A = (0.0, 0.5, 0.5, 1.0)
_RK4_B = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)

# Dormand-Prince 5(4), the tableau behind ode45/RK45.
_DP_C = (0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0)
_DP_A = (
    (),
    (0.2,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
# Fifth-order solution weights (identical to the last A row: FSAL).
_DP_B5 = _DP_A[6] + (0.0,)
# b5 - b4, used directly for the embedded error estimate.
_DP_E = (
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
)


# --- Core integrator ---------------------------------------------------------
def propagate(
    state: VehicleState,
    thrust_n: float,
    thrust_dir_rtn: np.ndarray,
    mdot_kg_s: float,
    mission: "Mission",
    dt_s: float,
    substeps: int = 10,
    *,
    method: str = "rk4",
    rtol: float = 1.0e-10,
    atol: float = 1.0e-6,
    max_substeps: int = 100_000,
) -> VehicleState:
    """Advance one macro-step under gravity + constant-magnitude thrust.

    Returns a NEW VehicleState; must not mutate the input.

    Parameters
    ----------
    state:
        Start-of-step vehicle state. Never mutated.
    thrust_n:
        Achieved thrust magnitude, constant over the step (the propulsion model
        already resolved power/thermal/throttle limits).
    thrust_dir_rtn:
        (3,) unit vector in the local RTN frame, held fixed in that *rotating*
        frame for the whole step. Non-unit input is normalised.
    mdot_kg_s:
        Propellant mass flow, constant while propellant remains.
    mission:
        Supplies ``mu`` and ``gravity(state)``.
    dt_s:
        Macro-step length.
    substeps:
        Number of fixed sub-intervals for ``method="rk4"``; the initial step
        guess for ``method="rk45"``.
    method:
        ``"rk4"`` (default, fixed step, fastest) or ``"rk45"`` (adaptive
        Dormand-Prince 5(4), for long coasts and month-scale spirals where a
        fixed step silently drifts).
    rtol, atol:
        Error tolerances for ``"rk45"``. Ignored by ``"rk4"``.
    max_substeps:
        Hard cap on adaptive iterations, so a pathological state cannot hang the
        rollout loop.
    """
    return _propagate_core(
        state,
        thrust_n,
        thrust_dir_rtn,
        mdot_kg_s,
        mission,
        dt_s,
        substeps,
        method,
        rtol,
        atol,
        max_substeps,
    )[0]


def propagate_with_status(
    state: VehicleState,
    thrust_n: float,
    thrust_dir_rtn: np.ndarray,
    mdot_kg_s: float,
    mission: "Mission",
    dt_s: float,
    substeps: int = 10,
    *,
    method: str = "rk4",
    rtol: float = 1.0e-10,
    atol: float = 1.0e-6,
    max_substeps: int = 100_000,
) -> tuple[VehicleState, PropagationStatus]:
    """:func:`propagate` plus a :class:`PropagationStatus` for logging/telemetry."""
    return _propagate_core(
        state,
        thrust_n,
        thrust_dir_rtn,
        mdot_kg_s,
        mission,
        dt_s,
        substeps,
        method,
        rtol,
        atol,
        max_substeps,
    )


def _propagate_core(
    state: VehicleState,
    thrust_n: float,
    thrust_dir_rtn: np.ndarray,
    mdot_kg_s: float,
    mission: Any,
    dt_s: float,
    substeps: int,
    method: str,
    rtol: float,
    atol: float,
    max_substeps: int,
) -> tuple[VehicleState, PropagationStatus]:
    status = PropagationStatus()

    # tolist() unpacks a 3-vector in ~45 ns; three float(p[i]) conversions cost
    # ~125 ns. At a million steps that is a second of pure marshalling.
    rx, ry, rz = state.position_m.tolist()
    vx, vy, vz = state.velocity_m_s.tolist()

    # Inline of is_diverged on values we already hold: the public helper has to
    # touch numpy and costs ~0.7 us, which is 2% of a whole macro-step.
    r2 = rx * rx + ry * ry + rz * rz
    v2 = vx * vx + vy * vy + vz * vz
    if not (_MIN_R2 <= r2 <= _MAX_R2 and v2 <= _MAX_V2):
        return _diverged(state, rx, ry, rz, vx, vy, vz, state.propellant_kg,
                         state.t_s, "input_state", status)

    dt = float(dt_s)
    if dt <= 0.0:
        status.ok = True
        return state.copy(), status

    mass0 = state.total_mass_kg
    prop0 = state.propellant_kg
    thrust = float(thrust_n)
    mdot = float(mdot_kg_s)
    if not (thrust > 0.0) or not math.isfinite(thrust):
        thrust = 0.0
    if not (mdot > 0.0) or not math.isfinite(mdot):
        mdot = 0.0
    if prop0 <= TINY_MASS_KG:
        # Dry tank: no thrust can be produced regardless of what was commanded.
        thrust = 0.0
        mdot = 0.0

    # --- how long can we burn before the tank empties? -----------------------
    if mdot > 0.0:
        burn_time = prop0 / mdot
        if burn_time >= dt:
            burn_time = dt
        else:
            status.cutoff = True
    else:
        burn_time = dt if thrust > 0.0 else 0.0
    coast_time = dt - burn_time

    # --- thrust direction, fixed in the rotating RTN frame -------------------
    if thrust > 0.0:
        d = thrust_dir_rtn
        dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dn < EPS or not math.isfinite(dn):
            logger.warning("degenerate thrust direction %r; defaulting to prograde", d)
            dx, dy, dz = _DEFAULT_DIR
        else:
            inv = 1.0 / dn
            dx, dy, dz = dx * inv, dy * inv, dz * inv
    else:
        dx, dy, dz = _DEFAULT_DIR

    mu = float(getattr(mission, "mu", 0.0))
    fast = _use_fast_gravity(mission)
    scratch: VehicleState | None = None
    if not fast:
        scratch = VehicleState(
            position_m=np.empty(3, dtype=np.float64),
            velocity_m_s=np.empty(3, dtype=np.float64),
            dry_mass_kg=state.dry_mass_kg,
            propellant_kg=prop0,
            payload_kg=state.payload_kg,
            t_s=state.t_s,
        )

    nsub = substeps if substeps >= 1 else 1
    if coast_time > 0.0 and burn_time > 0.0:
        n_burn = int(round(nsub * burn_time / dt))
        n_burn = 1 if n_burn < 1 else (nsub - 1 if n_burn > nsub - 1 else n_burn)
        n_coast = nsub - n_burn
    else:
        n_burn, n_coast = nsub, nsub

    # High T/W (NTP on a comsat, a chemical kick) needs a short RK4 step;
    # the default 10 substeps of a one-hour macro-step are ~6 minutes each and
    # walk a LEO state into the planet in one burn arc.
    if thrust > 0.0 and mass0 > TINY_MASS_KG and burn_time > 0.0:
        acc = thrust / mass0
        if acc > 0.05:
            n_need = int(math.ceil(burn_time / 5.0))
            n_burn = max(n_burn, min(max(n_need, 1), 4_000))

    integrate = _rk45_arc if method == "rk45" else _rk4_arc
    if method not in ("rk4", "rk45"):
        raise ValueError(f"unknown integration method {method!r}; use 'rk4' or 'rk45'")

    t0 = state.t_s
    y = (rx, ry, rz, vx, vy, vz)

    # --- burn arc ------------------------------------------------------------
    if burn_time > 0.0:
        y, ok, nst, err = integrate(
            y, 0.0, burn_time, n_burn, mu, fast, mission, scratch,
            thrust, mdot, mass0, dx, dy, dz, t0, rtol, atol, max_substeps,
        )
        status.steps_taken += nst
        status.max_error = err if err > status.max_error else status.max_error
        if not ok:
            prop_mid = prop0 - mdot * burn_time
            return _diverged(state, *y, max(prop_mid, 0.0), t0 + burn_time,
                             "burn_arc_divergence", status)

    # --- coast arc (tank dry, or thrust was off) -----------------------------
    if coast_time > 0.0:
        if scratch is not None:
            # Hand a mass-dependent gravity model the post-burn mass.
            pr = prop0 - mdot * burn_time
            scratch.propellant_kg = pr if pr > 0.0 else 0.0
        y, ok, nst, err = integrate(
            y, burn_time, dt, n_coast, mu, fast, mission, scratch,
            0.0, 0.0, mass0 - mdot * burn_time, dx, dy, dz, t0, rtol, atol,
            max_substeps,
        )
        status.steps_taken += nst
        status.max_error = err if err > status.max_error else status.max_error
        if not ok:
            return _diverged(state, *y, 0.0, t0 + dt, "coast_arc_divergence", status)

    # --- mass / delta-v bookkeeping -----------------------------------------
    used = mdot * burn_time
    if used > prop0:
        used = prop0
    prop_new = prop0 - used
    if prop_new < TINY_MASS_KG:
        prop_new = 0.0
        used = prop0

    if thrust > 0.0 and burn_time > 0.0:
        if mdot > 0.0:
            # Exact Tsiolkovsky over the burn arc: integral of (F/m) dt with
            # m(t) linear. Cheaper *and* more accurate than summing the stages.
            m_end = mass0 - used
            dv = (thrust / mdot) * math.log(mass0 / m_end) if m_end > 0.0 else 0.0
        else:
            dv = thrust * burn_time / mass0
    else:
        dv = 0.0

    status.burn_time_s = burn_time
    status.propellant_used_kg = used
    status.delta_v_m_s = dv

    fx, fy, fz, fvx, fvy, fvz = y
    r2 = fx * fx + fy * fy + fz * fz
    v2 = fvx * fvx + fvy * fvy + fvz * fvz
    if not (_MIN_R2 <= r2 <= _MAX_R2 and v2 <= _MAX_V2):
        return _diverged(state, *y, prop_new, t0 + dt, "post_step_check", status)

    out = VehicleState(
        position_m=np.array(y[0:3], dtype=np.float64),
        velocity_m_s=np.array(y[3:6], dtype=np.float64),
        dry_mass_kg=state.dry_mass_kg,
        propellant_kg=prop_new,
        payload_kg=state.payload_kg,
        t_s=t0 + dt,
        power_generated_w=state.power_generated_w,
        power_available_w=state.power_available_w,
        delta_v_applied_m_s=state.delta_v_applied_m_s + dv,
        propellant_used_kg=state.propellant_used_kg + used,
    )
    return out, status


def _diverged(
    state: VehicleState,
    rx: float,
    ry: float,
    rz: float,
    vx: float,
    vy: float,
    vz: float,
    propellant_kg: float,
    t_s: float,
    reason: str,
    status: PropagationStatus,
) -> tuple[VehicleState, PropagationStatus]:
    """Build a finite, flagged state instead of letting NaNs escape."""
    status.ok = False
    status.reason = reason
    logger.warning("propagation diverged (%s) at t=%.3f s", reason, t_s)

    def _finite(x: float, fallback: float) -> float:
        return x if math.isfinite(x) else fallback

    src_p = state.position_m
    src_v = state.velocity_m_s
    out = DivergedState(
        position_m=np.array(
            [
                _finite(rx, float(src_p[0])),
                _finite(ry, float(src_p[1])),
                _finite(rz, float(src_p[2])),
            ],
            dtype=np.float64,
        ),
        velocity_m_s=np.array(
            [
                _finite(vx, float(src_v[0])),
                _finite(vy, float(src_v[1])),
                _finite(vz, float(src_v[2])),
            ],
            dtype=np.float64,
        ),
        dry_mass_kg=state.dry_mass_kg,
        propellant_kg=_finite(propellant_kg, state.propellant_kg),
        payload_kg=state.payload_kg,
        t_s=_finite(t_s, state.t_s),
        power_generated_w=state.power_generated_w,
        power_available_w=state.power_available_w,
        delta_v_applied_m_s=state.delta_v_applied_m_s + status.delta_v_m_s,
        propellant_used_kg=state.propellant_used_kg + status.propellant_used_kg,
        diverged_reason=reason,
    )
    return out, status


def _rk4_arc(
    y: tuple[float, float, float, float, float, float],
    t_start: float,
    t_end: float,
    nsub: int,
    mu: float,
    fast: bool,
    mission: Any,
    scratch: VehicleState | None,
    thrust: float,
    mdot: float,
    mass0: float,
    dx: float,
    dy: float,
    dz: float,
    t0: float,
    rtol: float,
    atol: float,
    max_substeps: int,
) -> tuple[tuple[float, ...], bool, int, float]:
    """Classical RK4 over ``[t_start, t_end]`` in ``nsub`` equal sub-intervals.

    ``t_start``/``t_end`` are offsets from the beginning of the macro-step; they
    matter because the mass at a stage is ``mass0 - mdot * tau``.

    The single ``for stage in range(4)`` loop keeps one copy of the derivative
    while still running entirely on unpacked floats.
    """
    rx, ry, rz, vx, vy, vz = y
    h = (t_end - t_start) / nsub
    tau = t_start
    thrusting = thrust > 0.0
    sp = sv = None
    prop_start = 0.0
    track_mass = False
    if scratch is not None:
        sp = scratch.position_m
        sv = scratch.velocity_m_s
        # A mission's gravity() may be mass-dependent (solar radiation pressure
        # is a/m), so the scratch state's propellant has to track the burn too.
        prop_start = scratch.propellant_kg
        track_mass = mdot > 0.0

    for _ in range(nsub):
        # Accumulators for the weighted sum of stage derivatives.
        ar_x = ar_y = ar_z = 0.0
        av_x = av_y = av_z = 0.0
        # Previous stage derivative (RK4 stages depend only on the one before).
        kr_x = kr_y = kr_z = 0.0
        kv_x = kv_y = kv_z = 0.0

        for stage in range(4):
            a = _RK4_A[stage]
            if a == 0.0:
                px, py, pz = rx, ry, rz
                qx, qy, qz = vx, vy, vz
                ts = tau
            else:
                ah = a * h
                px = rx + ah * kr_x
                py = ry + ah * kr_y
                pz = rz + ah * kr_z
                qx = vx + ah * kv_x
                qy = vy + ah * kv_y
                qz = vz + ah * kv_z
                ts = tau + ah

            # --- gravity ---
            if fast:
                r2 = px * px + py * py + pz * pz
                if r2 < _MIN_R2 or not (r2 < _MAX_R2):
                    return (rx, ry, rz, vx, vy, vz), False, 0, 0.0
                rn = math.sqrt(r2)
                k = -mu / (r2 * rn)
                gx, gy, gz = k * px, k * py, k * pz
            else:
                sp[0] = px
                sp[1] = py
                sp[2] = pz
                sv[0] = qx
                sv[1] = qy
                sv[2] = qz
                scratch.t_s = t0 + ts
                if track_mass:
                    pr = prop_start - mdot * ts
                    scratch.propellant_kg = pr if pr > 0.0 else 0.0
                g = mission.gravity(scratch)
                gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
                rn = -1.0

            # --- thrust, direction re-derived in the instantaneous RTN frame ---
            if thrusting:
                m_now = mass0 - mdot * ts
                if m_now > TINY_MASS_KG:
                    if rn < 0.0:
                        rn = math.sqrt(px * px + py * py + pz * pz)
                    hx = py * qz - pz * qy
                    hy = pz * qx - px * qz
                    hz = px * qy - py * qx
                    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
                    if hn > EPS * rn:
                        # Same triad as _rtn_axes, folded so that no unit vector
                        # is ever materialised: R = r/rn, N = h/hn,
                        # T = N x R = (h x r)/(hn rn). Inlined because this runs
                        # four times per substep and the tuple allocations in a
                        # helper call cost more than the arithmetic does.
                        at = thrust / m_now
                        c1 = at * dx / rn
                        c2 = at * dy / (hn * rn)
                        c3 = at * dz / hn
                        gx += c1 * px + c2 * (hy * pz - hz * py) + c3 * hx
                        gy += c1 * py + c2 * (hz * px - hx * pz) + c3 * hy
                        gz += c1 * pz + c2 * (hx * py - hy * px) + c3 * hz
                    else:
                        (urx, ury, urz), (utx, uty, utz), (unx, uny, unz) = (
                            _rtn_axes(px, py, pz, qx, qy, qz)
                        )
                        at = thrust / m_now
                        gx += at * (dx * urx + dy * utx + dz * unx)
                        gy += at * (dx * ury + dy * uty + dz * uny)
                        gz += at * (dx * urz + dy * utz + dz * unz)

            kr_x, kr_y, kr_z = qx, qy, qz
            kv_x, kv_y, kv_z = gx, gy, gz
            b = _RK4_B[stage]
            ar_x += b * kr_x
            ar_y += b * kr_y
            ar_z += b * kr_z
            av_x += b * kv_x
            av_y += b * kv_y
            av_z += b * kv_z

        rx += h * ar_x
        ry += h * ar_y
        rz += h * ar_z
        vx += h * av_x
        vy += h * av_y
        vz += h * av_z
        tau += h

        r2 = rx * rx + ry * ry + rz * rz
        if not (_MIN_R2 <= r2 <= _MAX_R2):
            return (rx, ry, rz, vx, vy, vz), False, nsub, 0.0

    return (rx, ry, rz, vx, vy, vz), True, nsub, 0.0


def _rk45_arc(
    y: tuple[float, float, float, float, float, float],
    t_start: float,
    t_end: float,
    nsub: int,
    mu: float,
    fast: bool,
    mission: Any,
    scratch: VehicleState | None,
    thrust: float,
    mdot: float,
    mass0: float,
    dx: float,
    dy: float,
    dz: float,
    t0: float,
    rtol: float,
    atol: float,
    max_substeps: int,
) -> tuple[tuple[float, ...], bool, int, float]:
    """Adaptive Dormand-Prince 5(4) over ``[t_start, t_end]``.

    Same physics as :func:`_rk4_arc`; the step size is chosen from the embedded
    fourth-order error estimate. Slower per accepted step, but it is the only
    honest way to run a months-long spiral without picking ``substeps`` by
    guesswork.
    """
    rx, ry, rz, vx, vy, vz = y
    span = t_end - t_start
    h = span / max(nsub, 1)
    h_min = span * 1e-12
    tau = t_start
    thrusting = thrust > 0.0
    sp = sv = None
    prop_start = 0.0
    track_mass = False
    if scratch is not None:
        sp = scratch.position_m
        sv = scratch.velocity_m_s
        prop_start = scratch.propellant_kg
        track_mass = mdot > 0.0

    kr = [(0.0, 0.0, 0.0)] * 7
    kv = [(0.0, 0.0, 0.0)] * 7
    accepted = 0
    iters = 0
    worst = 0.0

    while tau < t_end - h_min:
        if tau + h > t_end:
            h = t_end - tau
        iters += 1
        if iters > max_substeps:
            logger.warning("rk45 hit max_substeps=%d; giving up", max_substeps)
            return (rx, ry, rz, vx, vy, vz), False, accepted, worst

        blew_up = False
        for stage in range(7):
            if stage == 0:
                px, py, pz = rx, ry, rz
                qx, qy, qz = vx, vy, vz
            else:
                acoef = _DP_A[stage]
                px = py = pz = qx = qy = qz = 0.0
                for j, aij in enumerate(acoef):
                    if aij == 0.0:
                        continue
                    krj = kr[j]
                    kvj = kv[j]
                    px += aij * krj[0]
                    py += aij * krj[1]
                    pz += aij * krj[2]
                    qx += aij * kvj[0]
                    qy += aij * kvj[1]
                    qz += aij * kvj[2]
                px = rx + h * px
                py = ry + h * py
                pz = rz + h * pz
                qx = vx + h * qx
                qy = vy + h * qy
                qz = vz + h * qz
            ts = tau + _DP_C[stage] * h

            if fast:
                r2 = px * px + py * py + pz * pz
                if not (_MIN_R2 <= r2 <= _MAX_R2):
                    blew_up = True
                    break
                k = -mu / (r2 * math.sqrt(r2))
                gx, gy, gz = k * px, k * py, k * pz
            else:
                sp[0] = px
                sp[1] = py
                sp[2] = pz
                sv[0] = qx
                sv[1] = qy
                sv[2] = qz
                scratch.t_s = t0 + ts
                if track_mass:
                    pr = prop_start - mdot * ts
                    scratch.propellant_kg = pr if pr > 0.0 else 0.0
                g = mission.gravity(scratch)
                gx, gy, gz = float(g[0]), float(g[1]), float(g[2])

            if thrusting:
                m = mass0 - mdot * ts
                if m > TINY_MASS_KG:
                    (urx, ury, urz), (utx, uty, utz), (unx, uny, unz) = _rtn_axes(
                        px, py, pz, qx, qy, qz
                    )
                    at = thrust / m
                    gx += at * (dx * urx + dy * utx + dz * unx)
                    gy += at * (dx * ury + dy * uty + dz * uny)
                    gz += at * (dx * urz + dy * utz + dz * unz)

            kr[stage] = (qx, qy, qz)
            kv[stage] = (gx, gy, gz)

        if blew_up:
            h *= 0.25
            if h < h_min:
                return (rx, ry, rz, vx, vy, vz), False, accepted, worst
            continue

        # Fifth-order update and the embedded error estimate.
        nr_x = nr_y = nr_z = 0.0
        nv_x = nv_y = nv_z = 0.0
        er_x = er_y = er_z = 0.0
        ev_x = ev_y = ev_z = 0.0
        for s in range(7):
            b = _DP_B5[s]
            e = _DP_E[s]
            krs = kr[s]
            kvs = kv[s]
            if b != 0.0:
                nr_x += b * krs[0]
                nr_y += b * krs[1]
                nr_z += b * krs[2]
                nv_x += b * kvs[0]
                nv_y += b * kvs[1]
                nv_z += b * kvs[2]
            if e != 0.0:
                er_x += e * krs[0]
                er_y += e * krs[1]
                er_z += e * krs[2]
                ev_x += e * kvs[0]
                ev_y += e * kvs[1]
                ev_z += e * kvs[2]

        rn_x = rx + h * nr_x
        rn_y = ry + h * nr_y
        rn_z = rz + h * nr_z
        vn_x = vx + h * nv_x
        vn_y = vy + h * nv_y
        vn_z = vz + h * nv_z

        sr = atol + rtol * max(abs(rx), abs(rn_x), abs(ry), abs(rn_y), abs(rz), abs(rn_z))
        sv_ = atol + rtol * max(abs(vx), abs(vn_x), abs(vy), abs(vn_y), abs(vz), abs(vn_z))
        e0 = h * er_x / sr
        e1 = h * er_y / sr
        e2 = h * er_z / sr
        e3 = h * ev_x / sv_
        e4 = h * ev_y / sv_
        e5 = h * ev_z / sv_
        err = math.sqrt(
            (e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5) / 6.0
        )

        if not math.isfinite(err):
            h *= 0.25
            if h < h_min:
                return (rx, ry, rz, vx, vy, vz), False, accepted, worst
            continue

        if err <= 1.0:
            rx, ry, rz = rn_x, rn_y, rn_z
            vx, vy, vz = vn_x, vn_y, vn_z
            tau += h
            accepted += 1
            if err > worst:
                worst = err
            r2 = rx * rx + ry * ry + rz * rz
            if not (_MIN_R2 <= r2 <= _MAX_R2):
                return (rx, ry, rz, vx, vy, vz), False, accepted, worst
            factor = 5.0 if err < 1e-10 else min(5.0, 0.9 * err ** -0.2)
        else:
            factor = max(0.2, 0.9 * err ** -0.2)
        h *= factor
        if h < h_min:
            h = h_min

    return (rx, ry, rz, vx, vy, vz), True, accepted, worst
