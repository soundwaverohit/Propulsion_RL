"""Conservation laws the propagator must not break.

These are the invariants that no amount of conformance testing over the
registries will catch, because they are properties of the *integrator* rather
than of any thruster or mission: an unforced two-body orbit conserves energy and
angular momentum exactly, so any drift in the reported semi-major axis is pure
numerics.

The distinction matters because RK4 is not symplectic. Its truncation error on a
Keplerian arc is systematically dissipative, not merely noisy, so too coarse a
step does not jitter the orbit -- it lowers it, every step, in the same
direction. That failure is indistinguishable from drag by eye, it survives every
determinism and finiteness check the rest of the suite makes, and it is large:
at ten sub-intervals for a one-hour step, a 400 km circular orbit loses ~287 km
of semi-major axis in a single day of simulated time, roughly forty times the
effect of the low-thrust burn the benchmark exists to measure.

:func:`~propulsion_rl.spacecraft.dynamics.substeps_for` is what keeps the step
sized against the orbit rather than the wall clock, so it is tested here beside
the conservation laws it exists to protect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propulsion_rl.core.constants import AU, GEO_RADIUS, LEO_RADIUS, MU_EARTH, MU_SUN
from propulsion_rl.core.registry import MISSION
from propulsion_rl.spacecraft import dynamics
from tests.conftest import synthetic_vehicle_state

pytestmark = pytest.mark.physics

#: One hour, the macro-step every planetocentric mission in the package uses.
HOUR_S = 3600.0

#: Thrust direction is irrelevant to an unforced arc, but ``propagate`` still
#: requires a unit vector, so give it a well-formed one.
RADIAL = np.array([1.0, 0.0, 0.0])


def _elements(state) -> tuple[float, float]:
    """Semi-major axis (m) and specific angular momentum (m^2/s) of *state*."""
    r_vec = np.asarray(state.position_m, dtype=float)
    v_vec = np.asarray(state.velocity_m_s, dtype=float)
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    sma = 1.0 / (2.0 / r - v * v / MU_EARTH)
    h = float(np.linalg.norm(np.cross(r_vec, v_vec)))
    return sma, h


def _coast(state, mission, steps: int, **kw):
    """Propagate *steps* unforced macro-steps: no thrust, no mass flow."""
    for _ in range(steps):
        state = dynamics.propagate(state, 0.0, RADIAL, 0.0, mission, HOUR_S, **kw)
    return state


@pytest.fixture
def leo_mission():
    return MISSION.make("leo_geo_transfer")


class TestUnforcedTwoBodyIsConserved:
    """An orbit with the engine off must stay where it is."""

    def test_semi_major_axis_holds_over_a_day(self, leo_mission):
        """A day of coasting must not move the orbit by a metre that matters.

        The tolerance is one part in 1e6 of the semi-major axis -- about 7 m at
        LEO. A low-thrust burn moves it by kilometres per day, so anything
        inside this band is safely below the signal.
        """
        start = synthetic_vehicle_state()
        a0, _ = _elements(start)
        substeps = dynamics.substeps_for(LEO_RADIUS, MU_EARTH, HOUR_S)

        end = _coast(start, leo_mission, 24, substeps=substeps)

        a1, _ = _elements(end)
        assert abs(a1 - a0) / a0 < 1.0e-6, (
            f"unforced orbit drifted {(a1 - a0) / 1e3:+.3f} km in 24 h with "
            f"{substeps} substeps; the integrator is adding or removing energy"
        )

    def test_angular_momentum_holds_over_a_day(self, leo_mission):
        """Central gravity applies no torque, so |r x v| is invariant.

        The bound is looser than the arithmetic precision but far tighter than
        any physical effect: a day of coasting moves it by ~4e-8 of itself,
        while the plane change this mission is scored on is order 1e-1.
        """
        start = synthetic_vehicle_state()
        _, h0 = _elements(start)
        substeps = dynamics.substeps_for(LEO_RADIUS, MU_EARTH, HOUR_S)

        end = _coast(start, leo_mission, 24, substeps=substeps)

        _, h1 = _elements(end)
        assert abs(h1 - h0) / h0 < 1.0e-7

    def test_drift_is_dissipative_and_shrinks_with_resolution(self, leo_mission):
        """Refining the step must reduce the error, and the error must be a loss.

        Pinning the *sign* is what separates this from a generic accuracy test:
        a symmetric error would average out over a sweep, while a one-sided one
        accumulates into a fake deorbit. Pinning the *trend* is what proves the
        residual is truncation error rather than a modelling term.
        """
        start = synthetic_vehicle_state()
        a0, _ = _elements(start)

        drifts = []
        for substeps in (10, 40, 160):
            a1, _ = _elements(_coast(start, leo_mission, 24, substeps=substeps))
            drifts.append(a1 - a0)

        assert drifts[0] < 0.0, "coarse RK4 should lose energy, not gain it"
        assert abs(drifts[1]) < abs(drifts[0])
        assert abs(drifts[2]) < abs(drifts[1])

    def test_the_shipped_default_would_have_deorbited_the_vehicle(self, leo_mission):
        """Regression: the fixed count this package used to pass is not safe.

        Kept as an explicit statement of the bug rather than a bare number, so
        that anyone tempted to hard-code ``substeps=10`` again sees what it cost:
        a quarter of a megametre of altitude per simulated day, with the engine
        off.
        """
        start = synthetic_vehicle_state()
        a0, _ = _elements(start)

        a1, _ = _elements(_coast(start, leo_mission, 24, substeps=10))

        assert a1 - a0 < -100e3


class TestSubstepsFor:
    """The step count must follow the orbit, not the wall clock."""

    def test_leo_hour_step_is_refined(self):
        """One hour is most of a LEO revolution, so it needs many sub-intervals."""
        n = dynamics.substeps_for(LEO_RADIUS, MU_EARTH, HOUR_S)
        period_s = 2.0 * math.pi * math.sqrt(LEO_RADIUS**3 / MU_EARTH)
        assert n >= HOUR_S * dynamics.DEFAULT_SUBSTEPS_PER_ORBIT / period_s

    def test_slow_orbits_are_not_over_refined(self):
        """A heliocentric day is a rounding error of a year: keep the floor.

        Without this the fix would pay for LEO accuracy on every interplanetary
        cruise step too, and the cruise missions are the long ones.
        """
        assert dynamics.substeps_for(AU, MU_SUN, 86_400.0) == 10
        assert dynamics.substeps_for(GEO_RADIUS, MU_EARTH, HOUR_S) == 10

    def test_scales_linearly_with_step_length(self):
        quarter = dynamics.substeps_for(LEO_RADIUS, MU_EARTH, HOUR_S / 4, minimum=1)
        whole = dynamics.substeps_for(LEO_RADIUS, MU_EARTH, HOUR_S, minimum=1)
        assert whole == pytest.approx(4 * quarter, abs=2)

    def test_is_bounded(self):
        """A pathological state must not turn one step into unbounded work."""
        n = dynamics.substeps_for(1.0, MU_EARTH, 1.0e9)
        assert n == dynamics.MAX_DERIVED_SUBSTEPS

    @pytest.mark.parametrize(
        "radius, mu",
        [(0.0, MU_EARTH), (-1.0, MU_EARTH), (float("nan"), MU_EARTH),
         (LEO_RADIUS, 0.0), (LEO_RADIUS, float("inf"))],
    )
    def test_degenerate_inputs_fall_back_to_the_floor(self, radius, mu):
        """Never raise on a bad state; leave rejecting it to the caller's guard."""
        assert dynamics.substeps_for(radius, mu, HOUR_S, minimum=7) == 7

    def test_zero_and_negative_steps_are_the_floor(self):
        assert dynamics.substeps_for(LEO_RADIUS, MU_EARTH, 0.0, minimum=3) == 3
        assert dynamics.substeps_for(LEO_RADIUS, MU_EARTH, -HOUR_S) >= 10


class TestEnvUsesTheRefinedStep:
    """The fix has to reach the environment, not just the dynamics module."""

    def test_a_prograde_burn_raises_the_orbit(self):
        """The end-to-end statement of the bug: thrust must add energy.

        Before the substep fix this assertion failed by two orders of magnitude
        in the wrong direction -- the vehicle fell out of the sky under full
        thrust -- while every other test in the suite still passed.
        """
        from propulsion_rl.core.registry import AGENT
        from propulsion_rl.envs.propulsion_env import EnvConfig, make_env

        env = make_env(
            "hall_spt100", "leo_geo_transfer",
            config=EnvConfig(max_steps=48), seed=0,
        )
        agent = AGENT.make("max_thrust", obs_dim=36, action_dim=5)
        obs, _ = env.reset(seed=0)

        a_start, _ = _elements(env._state)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            done = terminated or truncated
        a_end, _ = _elements(env._state)

        assert a_end > a_start, (
            f"semi-major axis fell {(a_end - a_start) / 1e3:.1f} km under a "
            "full prograde burn"
        )
