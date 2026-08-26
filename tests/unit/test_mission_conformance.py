"""Conformance suite run against every registered mission.

Parametrised from ``core.registry.MISSION`` at collection time. A mission
defines the task, so everything here is about the task being well posed:
a Markovian observation of fixed width, a bounded progress signal, a finite
reward, an honest termination reason, and gravity that actually points at the
central body.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propulsion_rl.core.types import (
    MISSION_OBS_DIM,
    ConstraintReport,
    HealthReport,
    TerminationReason,
    ThrusterOutput,
    VehicleState,
)
from propulsion_rl.missions.base import Mission, MissionResult, RewardTerms
from tests.conftest import OBS_ABS_MAX, make_mission


def state_signature(state: VehicleState) -> tuple:
    """Bit-exact snapshot of the parts ``reset`` is allowed to randomise."""
    return (
        tuple(np.asarray(state.position_m, dtype=float).tolist()),
        tuple(np.asarray(state.velocity_m_s, dtype=float).tolist()),
        state.dry_mass_kg,
        state.propellant_kg,
        state.payload_kg,
        state.t_s,
    )


def reset_with(name: str, seed: int) -> tuple:
    """Construct the mission fresh and reset it on a fixed seed."""
    mission = make_mission(name)
    return mission, mission.reset(np.random.default_rng(seed))


# --- structure ---------------------------------------------------------------
def test_registered_object_implements_the_abc(mission) -> None:
    """The environment calls the ABC's methods unconditionally."""
    assert isinstance(mission, Mission)
    assert isinstance(mission.name, str) and mission.name


def test_name_matches_the_registry_key(mission, mission_name) -> None:
    """Results are keyed on the registry name across the whole comparison."""
    assert mission.name.lower() == mission_name.lower()


def test_scenario_scalars_are_positive_and_coherent(mission) -> None:
    """``mu``, ``step_dt_s`` and ``max_duration_s`` set the dynamics and the RL
    horizon. A zero or negative value makes the episode length undefined and
    the integrator divide by zero."""
    assert mission.mu > 0.0, "no central body: gravity would be identically zero"
    assert math.isfinite(mission.mu)
    assert mission.step_dt_s > 0.0
    assert mission.max_duration_s > 0.0
    assert mission.max_duration_s >= mission.step_dt_s, (
        "the wall-clock limit is shorter than a single macro-step"
    )
    assert mission.frame in {"heliocentric", "planetocentric"}, (
        f"unknown frame {mission.frame!r}; the perturbation set is selected on it"
    )
    horizon = mission.max_duration_s / mission.step_dt_s
    assert 1 <= horizon <= 1e6, f"episode horizon of {horizon:.0f} steps is unusable"


# --- reset -------------------------------------------------------------------
def test_reset_returns_a_usable_vehicle_state(mission_name) -> None:
    """Everything downstream -- dynamics, mass bookkeeping, observation -- reads
    these fields on the very first step."""
    _, state = reset_with(mission_name, 0)
    assert isinstance(state, VehicleState)
    for name, vec in (("position_m", state.position_m), ("velocity_m_s", state.velocity_m_s)):
        arr = np.asarray(vec)
        assert arr.shape == (3,), f"{name} has shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{name} is not finite: {arr}"
    assert state.dry_mass_kg > 0.0
    assert state.propellant_kg > 0.0, "a mission that starts dry cannot be flown"
    assert state.payload_kg >= 0.0
    assert state.total_mass_kg > 0.0
    assert state.radius_m > 0.0, "starting at the centre of the central body"
    assert state.speed_m_s > 0.0


def test_reset_is_deterministic_for_a_fixed_seed(mission_name) -> None:
    """Seed control is the only lever the sweep has over episode variation; if
    one seed does not reproduce, no two agents are ever compared on the same
    task instance."""
    _, first = reset_with(mission_name, 20260825)
    _, second = reset_with(mission_name, 20260825)
    assert state_signature(first) == state_signature(second)


def test_reset_varies_across_seeds(mission_name) -> None:
    """Domain randomisation lives in ``reset``. With an identical start state
    for every seed, seed-averaged results measure only agent noise, and any
    policy can overfit one trajectory."""
    signatures = {state_signature(reset_with(mission_name, s)[1]) for s in range(8)}
    assert len(signatures) > 1, (
        "reset() ignores its generator: all 8 seeds give the same initial state"
    )


def test_reset_rewinds_a_reused_mission(mission_name) -> None:
    """The runner reuses one mission object across episodes; internal reward
    bookkeeping must be cleared or episode 2 inherits episode 1's totals."""
    mission = make_mission(mission_name)
    first = mission.reset(np.random.default_rng(5))
    progress_first = mission.progress(first)
    mission.reset(np.random.default_rng(9))
    again = mission.reset(np.random.default_rng(5))
    assert state_signature(again) == state_signature(first)
    assert mission.progress(again) == pytest.approx(progress_first)


# --- observation -------------------------------------------------------------
def test_observe_is_contract_width_finite_and_normalised(mission_name) -> None:
    """The mission block is the first 12 entries of the 36-wide observation.
    Un-normalised values here (metres, seconds) dwarf the other blocks."""
    mission, state = reset_with(mission_name, 3)
    for _ in range(5):
        obs = mission.observe(state)
        assert obs.shape == (MISSION_OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all(), f"non-finite observation: {obs}"
        assert np.abs(obs).max() <= OBS_ABS_MAX, (
            f"observation is not normalised to ~[-1, 1]: {obs}"
        )
        state.position_m = state.position_m * 1.05
        state.t_s += mission.step_dt_s
        state.propellant_kg *= 0.9


def test_observation_labels_match_observe_raw(mission_name) -> None:
    mission, state = reset_with(mission_name, 3)
    raw = np.asarray(mission.observe_raw(state)).reshape(-1)
    labels = mission.observation_labels()
    assert isinstance(labels, tuple)
    assert len(labels) == raw.size
    assert raw.size <= MISSION_OBS_DIM
    assert all(isinstance(name, str) and name for name in labels)
    assert len(set(labels)) == len(labels), "duplicate observation labels"


def test_observe_and_progress_do_not_mutate_the_state(mission_name) -> None:
    """Both are called several times per step by the env and the logger; a
    hidden mutation would make the physics depend on the logging settings."""
    mission, state = reset_with(mission_name, 3)
    before = state_signature(state)
    for _ in range(3):
        mission.observe(state)
        mission.observe_raw(state)
        mission.progress(state)
        mission.eclipse(state)
        mission.heliocentric_radius_m(state)
    assert state_signature(state) == before


# --- progress, reward, termination -------------------------------------------
def test_progress_is_a_bounded_fraction(mission_name) -> None:
    """Progress is compared across missions with wildly different units, which
    only works if it really is a [0, 1] completion fraction."""
    mission, state = reset_with(mission_name, 4)
    for _ in range(10):
        p = mission.progress(state)
        assert isinstance(p, float) or np.isscalar(p)
        assert math.isfinite(float(p)), f"progress is {p}"
        assert 0.0 <= float(p) <= 1.0, f"progress {p} outside [0, 1]"
        state.position_m = state.position_m * 1.1
        state.velocity_m_s = state.velocity_m_s * 0.98
        state.t_s += mission.step_dt_s


def test_reward_terms_are_finite(mission_name) -> None:
    """A single NaN in any term poisons the whole gradient; the environment
    only checks ``total``, so each part is checked here."""
    mission, prev = reset_with(mission_name, 6)
    state = prev.copy()
    state.t_s += mission.step_dt_s
    state.propellant_kg = max(state.propellant_kg - 1.0, 0.0)
    state.propellant_used_kg += 1.0
    state.delta_v_applied_m_s += 10.0
    output = ThrusterOutput(
        thrust_n=0.2, mdot_kg_s=1e-5, isp_s=2000.0, power_draw_w=4000.0,
        efficiency=0.5,
    )
    terms = mission.reward(prev, state, output, ConstraintReport(), HealthReport())
    assert isinstance(terms, RewardTerms)
    for key, value in terms.as_dict().items():
        assert math.isfinite(value), f"reward term {key} is {value}"


def test_reward_is_pure_in_its_arguments(mission_name) -> None:
    """The contract says reward is a pure function of its arguments plus
    bookkeeping that ``reset`` clears. Calling it twice on the same transition
    must therefore give the same answer -- otherwise replaying a logged episode
    reproduces a different return."""
    mission, prev = reset_with(mission_name, 6)
    state = prev.copy()
    state.t_s += mission.step_dt_s
    output = ThrusterOutput(thrust_n=0.1, mdot_kg_s=5e-6, isp_s=2000.0)
    first = mission.reward(prev, state, output, ConstraintReport(), HealthReport())
    second = mission.reward(prev, state, output, ConstraintReport(), HealthReport())
    assert first.as_dict() == second.as_dict()


def test_terminated_returns_a_termination_reason(mission_name) -> None:
    """The environment branches on identity of the enum member; a bare string
    or bool would compare unequal to every member and hang the episode."""
    mission, state = reset_with(mission_name, 8)
    reason = mission.terminated(state, HealthReport(), ConstraintReport())
    assert isinstance(reason, TerminationReason)
    assert reason is TerminationReason.RUNNING, (
        f"a freshly reset mission already reports {reason}"
    )


def test_terminated_reports_hardware_failure(mission_name) -> None:
    """A failed thruster ends the episode. If the mission ignores health, a
    dead engine coasts to the wall-clock limit collecting shaping reward."""
    mission, state = reset_with(mission_name, 8)
    reason = mission.terminated(
        state, HealthReport(failed=True, wear_fraction=1.0), ConstraintReport()
    )
    assert isinstance(reason, TerminationReason)
    assert reason is not TerminationReason.RUNNING, (
        "a failed thruster does not end the episode"
    )


def test_terminated_reports_an_empty_tank(mission_name) -> None:
    """Out of propellant with the goal unmet is a real, common outcome."""
    mission, state = reset_with(mission_name, 8)
    state.propellant_used_kg += state.propellant_kg
    state.propellant_kg = 0.0
    reason = mission.terminated(state, HealthReport(), ConstraintReport())
    assert isinstance(reason, TerminationReason)


# --- summary -----------------------------------------------------------------
@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.SUCCESS,
        TerminationReason.TIMEOUT,
        TerminationReason.OUT_OF_PROPELLANT,
        TerminationReason.HARDWARE_FAILURE,
    ],
)
def test_summarize_is_well_formed(mission_name, reason: TerminationReason) -> None:
    """The ``MissionResult`` is the only thing the economics model and the
    ranking table ever see, so every field must be trustworthy for every way an
    episode can end."""
    mission, state = reset_with(mission_name, 10)
    state.t_s += 10 * mission.step_dt_s
    state.propellant_used_kg = 25.0
    state.propellant_kg = max(state.propellant_kg - 25.0, 0.0)
    state.delta_v_applied_m_s = 500.0

    result = mission.summarize(state, reason)
    assert isinstance(result, MissionResult)
    assert result.reason is reason, (
        f"summarize was told {reason} but recorded {result.reason}"
    )
    assert 0.0 <= result.progress <= 1.0
    assert math.isfinite(result.elapsed_s) and result.elapsed_s >= 0.0
    assert math.isfinite(result.delta_v_m_s) and result.delta_v_m_s >= 0.0
    assert result.propellant_used_kg >= 0.0
    assert result.payload_delivered_kg >= 0.0
    assert result.terminal_error >= 0.0
    assert result.constraint_violations >= 0
    assert result.total_constraint_cost >= 0.0
    assert isinstance(result.success, bool)
    if result.success:
        assert reason is TerminationReason.SUCCESS, (
            f"episode marked successful while ending as {reason}"
        )
    if reason is not TerminationReason.SUCCESS:
        assert not result.success
        assert result.payload_delivered_kg == 0.0 or result.progress < 1.0


def test_summarize_carries_the_flown_totals(mission_name) -> None:
    """Delta-v and propellant in the result must be what the vehicle actually
    accumulated -- the cost model divides dollars by exactly these numbers."""
    mission, state = reset_with(mission_name, 11)
    state.t_s += 5 * mission.step_dt_s
    state.delta_v_applied_m_s = 1234.0
    state.propellant_used_kg = 56.0
    result = mission.summarize(state, TerminationReason.TIMEOUT)
    assert result.delta_v_m_s == pytest.approx(1234.0, rel=1e-6)
    assert result.propellant_used_kg == pytest.approx(56.0, rel=1e-6)
    assert result.elapsed_s == pytest.approx(state.t_s, rel=1e-6)


# --- gravity -----------------------------------------------------------------
def test_gravity_points_at_the_central_body(mission_name) -> None:
    """A sign error here reverses the two-body problem and every trajectory
    escapes on the first step."""
    mission, state = reset_with(mission_name, 12)
    for scale in (1.0, 1.5, 3.0):
        probe = state.copy()
        probe.position_m = state.position_m * scale
        g = mission.gravity(probe)
        assert np.asarray(g).shape == (3,)
        assert np.isfinite(g).all()
        assert float(np.dot(g, probe.position_m)) < 0.0, (
            "gravitational acceleration does not point inward"
        )


def test_gravity_falls_off_as_one_over_r_squared(mission_name) -> None:
    """The central term must dominate: doubling the radius quarters the pull.
    Perturbations are allowed to move it, but only by a few percent."""
    mission, state = reset_with(mission_name, 12)
    near = state.copy()
    far = state.copy()
    far.position_m = state.position_m * 2.0

    g_near = float(np.linalg.norm(mission.gravity(near)))
    g_far = float(np.linalg.norm(mission.gravity(far)))
    assert g_near > 0.0
    assert g_far == pytest.approx(g_near / 4.0, rel=0.05), (
        f"|g| went from {g_near:.6g} to {g_far:.6g} when r doubled"
    )
    expected = mission.mu / near.radius_m**2
    assert g_near == pytest.approx(expected, rel=0.05), (
        f"|g| = {g_near:.6g} but mu/r^2 = {expected:.6g}"
    )


def test_gravity_is_finite_at_the_origin(mission_name) -> None:
    """A diverged trajectory can land on the singularity; returning inf there
    turns one bad step into a NaN-poisoned episode."""
    mission, state = reset_with(mission_name, 12)
    state.position_m = np.zeros(3)
    g = mission.gravity(state)
    assert np.isfinite(g).all(), f"gravity at r=0 is {g}"


def test_optional_hooks_return_the_documented_types(mission_name) -> None:
    mission, state = reset_with(mission_name, 13)
    assert isinstance(mission.eclipse(state), (bool, np.bool_))
    r = mission.heliocentric_radius_m(state)
    assert math.isfinite(r) and r > 0.0
    assert isinstance(mission.info(), dict)
