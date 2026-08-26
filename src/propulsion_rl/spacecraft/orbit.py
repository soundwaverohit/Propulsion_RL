"""Orbital element utilities and closed-form transfer estimates.

Everything here is a pure function of its arguments -- no vehicle, no mission,
no hidden state -- so missions, scripted baselines, reward normalisers and the
analysis layer can all share one implementation.

Two things in this module are load-bearing beyond bookkeeping:

* :func:`edelbaum_delta_v` is the analytical optimum for a combined low-thrust
  orbit raise plus plane change. Scripted baselines are scored against it and
  rewards are normalised by it, so an error here shifts the whole leaderboard.
* :func:`cartesian_to_elements` / :func:`elements_to_cartesian` round-trip to
  machine precision *including* the near-circular and near-equatorial cases,
  where the naive textbook formulation divides by a vanishing eccentricity or
  node vector.

Angle convention: radians throughout, ``i`` in [0, pi], every other angle
wrapped to [0, 2pi).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..core.constants import MU_SUN

logger = logging.getLogger(__name__)

__all__ = [
    "OrbitalElements",
    "cartesian_to_elements",
    "elements_to_cartesian",
    "edelbaum_delta_v",
    "edelbaum_time_of_flight",
    "hohmann_delta_v",
    "hohmann_transfer_time",
    "lambert_solve",
    "lambert_delta_v",
    "synodic_period",
    "phase_angle",
    "planet_state",
    "circular_speed",
    "orbital_period",
    "true_to_eccentric",
    "eccentric_to_true",
    "true_to_mean",
    "mean_to_true",
]

TWO_PI = 2.0 * math.pi

#: Below this the eccentricity vector's direction is numerically meaningless.
CIRCULAR_TOL = 1.0e-11
#: Below this sin(i) the node vector's direction is numerically meaningless.
EQUATORIAL_TOL = 1.0e-11


def _wrap_2pi(x: float) -> float:
    x = math.fmod(x, TWO_PI)
    return x + TWO_PI if x < 0.0 else x


def _acos_clamped(x: float) -> float:
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return math.pi
    return math.acos(x)


@dataclass(slots=True)
class OrbitalElements:
    """Classical Keplerian elements plus the derived quantities everyone wants.

    ``mu`` travels with the elements so the derived properties (period, energy,
    angular momentum) are self-contained; :func:`cartesian_to_elements` fills it
    in. It defaults to the Sun because the interplanetary missions dominate;
    always pass it explicitly for planetocentric work.
    """

    a: float                    # semi-major axis, m (negative for hyperbolic)
    e: float                    # eccentricity
    i: float                    # inclination, rad, [0, pi]
    raan: float                 # right ascension of the ascending node, rad
    argp: float                 # argument of periapsis, rad
    nu: float                   # true anomaly, rad
    mu: float = MU_SUN          # gravitational parameter, m^3/s^2

    # --- derived -------------------------------------------------------------
    @property
    def p(self) -> float:
        """Semi-latus rectum, m."""
        return self.a * (1.0 - self.e * self.e)

    @property
    def rp(self) -> float:
        """Periapsis radius, m."""
        return self.a * (1.0 - self.e)

    @property
    def ra(self) -> float:
        """Apoapsis radius, m. Infinite for e >= 1."""
        return self.a * (1.0 + self.e) if self.e < 1.0 else math.inf

    @property
    def r(self) -> float:
        """Current radius at ``nu``, m."""
        return self.p / (1.0 + self.e * math.cos(self.nu))

    @property
    def energy(self) -> float:
        """Specific orbital energy, J/kg."""
        return -self.mu / (2.0 * self.a)

    @property
    def angular_momentum(self) -> float:
        """Specific angular momentum magnitude, m^2/s."""
        return math.sqrt(self.mu * self.p)

    @property
    def period(self) -> float:
        """Orbital period, s. Infinite for open orbits."""
        if self.a <= 0.0:
            return math.inf
        return TWO_PI * math.sqrt(self.a**3 / self.mu)

    @property
    def mean_motion(self) -> float:
        """Mean motion, rad/s. Zero for open orbits."""
        if self.a <= 0.0:
            return 0.0
        return math.sqrt(self.mu / self.a**3)

    @property
    def is_circular(self) -> bool:
        return self.e < 1.0e-6

    @property
    def is_equatorial(self) -> bool:
        return math.sin(self.i) < 1.0e-6

    @property
    def is_closed(self) -> bool:
        return self.e < 1.0 and self.a > 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.e, self.i, self.raan, self.argp, self.nu)

    def with_nu(self, nu: float) -> "OrbitalElements":
        """Same orbit, different point on it."""
        return OrbitalElements(self.a, self.e, self.i, self.raan, self.argp,
                               _wrap_2pi(nu), self.mu)


# --- Cartesian <-> elements --------------------------------------------------
def cartesian_to_elements(
    r: np.ndarray, v: np.ndarray, mu: float
) -> OrbitalElements:
    """Inertial state vectors -> classical elements.

    Degenerate geometries are resolved by folding the undefined angle into the
    next one down, which is both the standard convention and exactly what makes
    the round trip through :func:`elements_to_cartesian` exact:

    * circular inclined -> ``argp = 0``, ``nu`` is the argument of latitude;
    * equatorial elliptical -> ``raan = 0``, ``argp`` is the longitude of
      periapsis;
    * circular equatorial -> ``raan = argp = 0``, ``nu`` is the true longitude.
    """
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    v2 = vx * vx + vy * vy + vz * vz
    if rn <= 0.0 or not math.isfinite(rn) or not math.isfinite(v2):
        raise ValueError("cartesian_to_elements: non-physical state vector")

    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    if hn <= 0.0:
        raise ValueError("cartesian_to_elements: rectilinear state has no orbit plane")

    # Node vector n = zhat x h  (points at the ascending node).
    nx, ny = -hy, hx
    nn = math.sqrt(nx * nx + ny * ny)

    rv = rx * vx + ry * vy + rz * vz
    c1 = v2 - mu / rn
    ex = (c1 * rx - rv * vx) / mu
    ey = (c1 * ry - rv * vy) / mu
    ez = (c1 * rz - rv * vz) / mu
    e = math.sqrt(ex * ex + ey * ey + ez * ez)

    energy = 0.5 * v2 - mu / rn
    if abs(e - 1.0) < 1.0e-12:
        # Parabolic: a is infinite, keep p meaningful by faking a huge a.
        a = math.copysign(math.inf, 1.0)
    else:
        a = -mu / (2.0 * energy)

    # atan2 rather than acos(hz/hn): for a near-equatorial orbit hz/hn rounds to
    # exactly 1.0 in float64 and acos throws the inclination away entirely
    # (i = 1e-9 rad comes back as 0). atan2 keeps every digit of a small i, and
    # of pi - i for a near-retrograde one.
    i = math.atan2(nn, hz)

    equatorial = nn <= EQUATORIAL_TOL * hn
    circular = e <= CIRCULAR_TOL

    if not equatorial and not circular:
        raan = _wrap_2pi(math.atan2(ny, nx))
        argp = _acos_clamped((nx * ex + ny * ey) / (nn * e))
        if ez < 0.0:
            argp = TWO_PI - argp
        nu = _acos_clamped((ex * rx + ey * ry + ez * rz) / (e * rn))
        if rv < 0.0:
            nu = TWO_PI - nu
    elif not equatorial and circular:
        # Argument of latitude replaces argp + nu.
        raan = _wrap_2pi(math.atan2(ny, nx))
        argp = 0.0
        nu = _acos_clamped((nx * rx + ny * ry) / (nn * rn))
        if rz < 0.0:
            nu = TWO_PI - nu
    elif equatorial and not circular:
        # Longitude of periapsis replaces raan + argp. Retrograde orbits measure
        # it the other way round, which is what keeps the round trip exact.
        raan = 0.0
        argp = _wrap_2pi(math.atan2(ey, ex) if hz >= 0.0 else math.atan2(-ey, ex))
        nu = _acos_clamped((ex * rx + ey * ry + ez * rz) / (e * rn))
        if rv < 0.0:
            nu = TWO_PI - nu
    else:
        # True longitude replaces everything.
        raan = 0.0
        argp = 0.0
        nu = _wrap_2pi(math.atan2(ry, rx) if hz >= 0.0 else math.atan2(-ry, rx))

    return OrbitalElements(a=a, e=e, i=i, raan=_wrap_2pi(raan),
                           argp=_wrap_2pi(argp), nu=_wrap_2pi(nu), mu=mu)


def elements_to_cartesian(
    el: OrbitalElements, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Classical elements -> inertial position (m) and velocity (m/s)."""
    e = el.e
    p = el.p
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError(
            f"elements_to_cartesian: degenerate semi-latus rectum p={p} "
            f"(a={el.a}, e={e})"
        )

    cnu = math.cos(el.nu)
    snu = math.sin(el.nu)
    denom = 1.0 + e * cnu
    if denom <= 0.0:
        raise ValueError(
            f"elements_to_cartesian: true anomaly {el.nu} is outside the "
            f"asymptotes of a hyperbola with e={e}"
        )
    rn = p / denom
    sq = math.sqrt(mu / p)

    # Perifocal frame.
    xp, yp = rn * cnu, rn * snu
    vxp, vyp = -sq * snu, sq * (e + cnu)

    cO, sO = math.cos(el.raan), math.sin(el.raan)
    ci, si = math.cos(el.i), math.sin(el.i)
    cw, sw = math.cos(el.argp), math.sin(el.argp)

    # R3(-raan) R1(-i) R3(-argp), columns applied to the perifocal vector.
    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    r31 = sw * si
    r32 = cw * si

    pos = np.array(
        [r11 * xp + r12 * yp, r21 * xp + r22 * yp, r31 * xp + r32 * yp],
        dtype=np.float64,
    )
    vel = np.array(
        [r11 * vxp + r12 * vyp, r21 * vxp + r22 * vyp, r31 * vxp + r32 * vyp],
        dtype=np.float64,
    )
    return pos, vel


# --- Anomaly conversions -----------------------------------------------------
def true_to_eccentric(nu: float, e: float) -> float:
    """True anomaly -> eccentric anomaly (elliptic) / hyperbolic anomaly."""
    if e < 1.0:
        return _wrap_2pi(
            2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu),
                             math.sqrt(1.0 + e) * math.cos(0.5 * nu))
        )
    s = math.sqrt(e * e - 1.0) * math.sin(nu) / (1.0 + e * math.cos(nu))
    return math.asinh(s)


def eccentric_to_true(ecc_anom: float, e: float) -> float:
    """Eccentric (or hyperbolic) anomaly -> true anomaly."""
    if e < 1.0:
        return _wrap_2pi(
            2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * ecc_anom),
                             math.sqrt(1.0 - e) * math.cos(0.5 * ecc_anom))
        )
    return 2.0 * math.atan2(
        math.sqrt(e + 1.0) * math.tanh(0.5 * ecc_anom), math.sqrt(e - 1.0)
    )


def true_to_mean(nu: float, e: float) -> float:
    """True anomaly -> mean anomaly."""
    ea = true_to_eccentric(nu, e)
    if e < 1.0:
        return _wrap_2pi(ea - e * math.sin(ea))
    return e * math.sinh(ea) - ea


def mean_to_true(m: float, e: float, tol: float = 1.0e-13, max_iter: int = 60) -> float:
    """Mean anomaly -> true anomaly by Newton iteration on Kepler's equation."""
    if e < 1.0:
        m = _wrap_2pi(m)
        ea = m if e < 0.8 else math.pi
        for _ in range(max_iter):
            f = ea - e * math.sin(ea) - m
            fp = 1.0 - e * math.cos(ea)
            step = f / fp
            ea -= step
            if abs(step) < tol:
                break
        else:
            logger.warning("mean_to_true: Kepler solver hit %d iterations", max_iter)
        return eccentric_to_true(ea, e)

    h = math.asinh(m / e) if abs(m) > 1.0 else m
    for _ in range(max_iter):
        f = e * math.sinh(h) - h - m
        fp = e * math.cosh(h) - 1.0
        step = f / fp
        h -= step
        if abs(step) < tol:
            break
    return eccentric_to_true(h, e)


# --- Closed-form transfer estimates ------------------------------------------
def circular_speed(r: float, mu: float) -> float:
    """Speed on a circular orbit of radius ``r``."""
    return math.sqrt(mu / r)


def orbital_period(a: float, mu: float) -> float:
    """Keplerian period of a semi-major axis ``a``."""
    return TWO_PI * math.sqrt(a**3 / mu)


def edelbaum_delta_v(a0: float, a1: float, i0: float, i1: float, mu: float) -> float:
    """Edelbaum's combined circular orbit-raise + plane-change delta-v, m/s.

    The classic 1961 result for continuous low thrust between two circular
    orbits with an optimally steered out-of-plane component::

        dv = sqrt(V0^2 + V1^2 - 2 V0 V1 cos(pi/2 * di))

    with ``V = sqrt(mu / a)``. Two sanity anchors fall straight out of it:
    ``di = 0`` gives ``|V0 - V1|`` (pure spiral), and ``a0 = a1`` gives
    ``2 V sin(pi di / 4)`` (pure plane change) -- both far cheaper than the
    impulsive equivalents, which is the whole point of electric propulsion.

    Note this is the *ideal* cost: it assumes thrust is always available and
    always optimally pointed, so it is a lower bound on what any agent can
    achieve and therefore a fair reward normaliser.
    """
    if a0 <= 0.0 or a1 <= 0.0:
        raise ValueError("edelbaum_delta_v: semi-major axes must be positive")
    v0 = math.sqrt(mu / a0)
    v1 = math.sqrt(mu / a1)
    di = abs(i1 - i0)
    inner = v0 * v0 + v1 * v1 - 2.0 * v0 * v1 * math.cos(0.5 * math.pi * di)
    return math.sqrt(max(inner, 0.0))


def edelbaum_time_of_flight(
    delta_v_m_s: float, thrust_n: float, mass_kg: float, mdot_kg_s: float = 0.0
) -> float:
    """Time to deliver ``delta_v`` at constant thrust, s.

    With ``mdot = 0`` this is the constant-mass ``m dv / F``; otherwise it is the
    rocket equation solved for burn time, which is materially shorter because
    the vehicle gets lighter as it goes.
    """
    if thrust_n <= 0.0:
        return math.inf
    if mdot_kg_s <= 0.0:
        return mass_kg * delta_v_m_s / thrust_n
    ve = thrust_n / mdot_kg_s
    return mass_kg * (1.0 - math.exp(-delta_v_m_s / ve)) / mdot_kg_s


def hohmann_delta_v(r0: float, r1: float, mu: float) -> tuple[float, float]:
    """Two-impulse Hohmann transfer between coplanar circular orbits.

    Returns ``(dv1, dv2)`` as *signed* tangential impulses -- negative when the
    burn is retrograde, i.e. when lowering the orbit. Total cost is
    ``abs(dv1) + abs(dv2)``.
    """
    if r0 <= 0.0 or r1 <= 0.0:
        raise ValueError("hohmann_delta_v: radii must be positive")
    at = 0.5 * (r0 + r1)
    v_c0 = math.sqrt(mu / r0)
    v_c1 = math.sqrt(mu / r1)
    v_t0 = math.sqrt(mu * (2.0 / r0 - 1.0 / at))
    v_t1 = math.sqrt(mu * (2.0 / r1 - 1.0 / at))
    return v_t0 - v_c0, v_c1 - v_t1


def hohmann_transfer_time(r0: float, r1: float, mu: float) -> float:
    """Half the period of the Hohmann transfer ellipse, s."""
    at = 0.5 * (r0 + r1)
    return math.pi * math.sqrt(at**3 / mu)


# --- Lambert -----------------------------------------------------------------
def _stumpff(psi: float) -> tuple[float, float]:
    """Stumpff functions c2, c3 with the series used near psi = 0."""
    if psi > 1.0e-6:
        s = math.sqrt(psi)
        return (1.0 - math.cos(s)) / psi, (s - math.sin(s)) / (psi * s)
    if psi < -1.0e-6:
        s = math.sqrt(-psi)
        return (math.cosh(s) - 1.0) / (-psi), (math.sinh(s) - s) / (s * (-psi))
    # Truncated series; |psi| < 1e-6 makes the next term ~1e-16 of the leading one.
    return 0.5 - psi / 24.0, 1.0 / 6.0 - psi / 120.0


def lambert_solve(
    r0: np.ndarray,
    r1: np.ndarray,
    tof_s: float,
    mu: float,
    prograde: bool = True,
    tol: float = 1.0e-10,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal-variable Lambert solver: two positions and a flight time.

    Returns ``(v0, v1)``, the velocities at ``r0`` and ``r1`` on the connecting
    zero-revolution conic. Bisection on the universal variable ``psi`` -- slower
    than a Newton or Izzo scheme but monotone and unconditionally convergent,
    and this never runs inside the RL step loop.

    Raises ``ValueError`` for the geometrically degenerate cases (collinear
    positions, where the transfer plane is undefined).
    """
    r0v = np.asarray(r0, dtype=np.float64).reshape(3)
    r1v = np.asarray(r1, dtype=np.float64).reshape(3)
    n0 = float(np.linalg.norm(r0v))
    n1 = float(np.linalg.norm(r1v))
    if n0 <= 0.0 or n1 <= 0.0 or tof_s <= 0.0:
        raise ValueError("lambert_solve: radii and time of flight must be positive")

    cos_dnu = float(np.dot(r0v, r1v)) / (n0 * n1)
    cos_dnu = min(1.0, max(-1.0, cos_dnu))
    dnu = math.acos(cos_dnu)
    cross_z = float(r0v[0] * r1v[1] - r0v[1] * r1v[0])
    if prograde:
        if cross_z < 0.0:
            dnu = TWO_PI - dnu
    elif cross_z > 0.0:
        dnu = TWO_PI - dnu

    sin_dnu = math.sin(dnu)
    # The test is on sin(dnu), not on the resulting A: at dnu = pi exactly,
    # sin(dnu) is 1e-16 but A is ~1e-8 * the orbit scale, which sails past any
    # absolute tolerance and returns confident nonsense.
    if abs(sin_dnu) < 1.0e-8:
        raise ValueError(
            "lambert_solve: transfer angle is 0 or pi, so the transfer plane is "
            "undefined. Perturb the geometry or use a multi-revolution solver."
        )
    a_par = sin_dnu * math.sqrt(n0 * n1 / (1.0 - math.cos(dnu)))

    psi = 0.0
    psi_lo, psi_hi = -4.0 * math.pi, 4.0 * math.pi * math.pi
    y = n0 + n1
    sqrt_mu = math.sqrt(mu)

    for _ in range(max_iter):
        c2, c3 = _stumpff(psi)
        y = n0 + n1 + a_par * (psi * c3 - 1.0) / math.sqrt(c2)
        if a_par > 0.0 and y < 0.0:
            # Push psi up until y turns positive (Vallado's fix-up).
            for _ in range(100):
                psi += 0.1
                c2, c3 = _stumpff(psi)
                y = n0 + n1 + a_par * (psi * c3 - 1.0) / math.sqrt(c2)
                if y >= 0.0:
                    break
            psi_lo = psi
        chi = math.sqrt(y / c2)
        dt = (chi**3 * c3 + a_par * math.sqrt(y)) / sqrt_mu
        if abs(dt - tof_s) < tol * tof_s:
            break
        if dt <= tof_s:
            psi_lo = psi
        else:
            psi_hi = psi
        psi = 0.5 * (psi_lo + psi_hi)
    else:
        logger.warning(
            "lambert_solve: no convergence in %d iterations (tof=%.3e s)",
            max_iter, tof_s,
        )

    f = 1.0 - y / n0
    g = a_par * math.sqrt(y / mu)
    g_dot = 1.0 - y / n1
    v0 = (r1v - f * r0v) / g
    v1 = (g_dot * r1v - r0v) / g
    return v0, v1


def lambert_delta_v(
    r0: np.ndarray,
    v0: np.ndarray,
    r1: np.ndarray,
    v1: np.ndarray,
    tof_s: float,
    mu: float,
    prograde: bool = True,
) -> tuple[float, float]:
    """Impulsive cost of a Lambert transfer between two moving bodies.

    ``(r0, v0)`` is the departure body's state at departure, ``(r1, v1)`` the
    arrival body's state ``tof_s`` later. Returns ``(dv_depart, dv_arrive)``
    magnitudes in m/s; the arrival term is the full rendezvous cost, so drop it
    if the mission only needs a flyby.
    """
    vt0, vt1 = lambert_solve(r0, r1, tof_s, mu, prograde=prograde)
    dv0 = float(np.linalg.norm(vt0 - np.asarray(v0, dtype=np.float64).reshape(3)))
    dv1 = float(np.linalg.norm(np.asarray(v1, dtype=np.float64).reshape(3) - vt1))
    return dv0, dv1


# --- Simple ephemeris --------------------------------------------------------
def synodic_period(period_a: float, period_b: float) -> float:
    """Time between successive identical relative configurations, s.

    Infinite when the two periods match (the geometry never repeats because it
    never changes).
    """
    if period_a <= 0.0 or period_b <= 0.0:
        raise ValueError("synodic_period: periods must be positive")
    diff = abs(1.0 / period_a - 1.0 / period_b)
    return math.inf if diff < 1.0e-18 else 1.0 / diff


def phase_angle(r1: np.ndarray, r2: np.ndarray) -> float:
    """Signed angle from ``r1`` to ``r2`` about +z, in [-pi, pi].

    Positive means ``r2`` leads ``r1`` in the prograde (counter-clockwise)
    sense. Both vectors are projected onto the xy-plane, which is what the
    coplanar transfer-window logic wants; the out-of-plane component of a real
    planet's position is a fraction of a degree and irrelevant here.
    """
    x1, y1 = float(r1[0]), float(r1[1])
    x2, y2 = float(r2[0]), float(r2[1])
    return math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)


def planet_state(
    sma: float, period: float, t: float, phase0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Circular, coplanar ephemeris: position (m) and velocity (m/s) at time ``t``.

    A full JPL ephemeris is out of scope for the benchmark; Earth's and Mars'
    real eccentricities (0.017 and 0.093) shift a transfer window by days, not
    by whether it exists. Missions that care can override this with their own.
    """
    n = TWO_PI / period
    theta = phase0 + n * t
    c, s = math.cos(theta), math.sin(theta)
    speed = sma * n
    return (
        np.array([sma * c, sma * s, 0.0], dtype=np.float64),
        np.array([-speed * s, speed * c, 0.0], dtype=np.float64),
    )
