"""Conformance suite run against every registered agent.

Parametrised from ``core.registry.AGENT`` at collection time. The agent
interface is deliberately narrow so a PID controller, a PPO network and a CEM
planner are interchangeable in the experiment matrix; these tests check that
each of them really is interchangeable -- built for the canonical (36, 5)
interface, emitting legal actions, honouring the determinism flag, and
surviving every optional hook the runner calls.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propulsion_rl.agents.base import Agent, ScriptedAgent, TrainStats, Transition
from propulsion_rl.core.types import CANONICAL_ACTION_DIM, OBS_DIM
from tests.conftest import make_agent


def observation(seed: int = 0) -> np.ndarray:
    """A legal, in-range observation of the contracted width."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, OBS_DIM).astype(np.float32)


def transition(seed: int = 0, reward: float = 1.0) -> Transition:
    """A complete transition, the unit every learner consumes."""
    return Transition(
        obs=observation(seed),
        action=np.zeros(CANONICAL_ACTION_DIM, dtype=np.float32),
        reward=reward,
        next_obs=observation(seed + 1),
        terminated=False,
        truncated=False,
        cost=0.25,
        info={"constraint_cost": 0.25},
    )


def assert_legal_action(action: np.ndarray, where: str) -> None:
    """Every action must be a finite point inside the canonical action box."""
    arr = np.asarray(action)
    assert arr.shape == (CANONICAL_ACTION_DIM,), f"{where}: shape {arr.shape}"
    assert np.issubdtype(arr.dtype, np.floating), f"{where}: dtype {arr.dtype}"
    assert np.isfinite(arr).all(), f"{where}: non-finite action {arr}"
    assert np.abs(arr).max() <= 1.0 + 1e-6, (
        f"{where}: action {arr} leaves [-1, 1]; the command decoder would clip "
        "it and the agent's own logging would disagree with what was flown"
    )


# --- construction ------------------------------------------------------------
def test_constructs_for_the_canonical_interface(agent) -> None:
    """Every agent must be buildable for (36, 5) from its registry name alone --
    that is the only call the experiment matrix makes."""
    assert isinstance(agent, Agent)
    assert agent.obs_dim == OBS_DIM
    assert agent.action_dim == CANONICAL_ACTION_DIM
    assert isinstance(agent.name, str) and agent.name
    assert isinstance(agent.learns, bool)
    assert isinstance(agent.uses_constraints, bool)
    assert isinstance(agent.num_parameters, int)
    assert agent.num_parameters >= 0


def test_name_matches_the_registry_key(agent, agent_name) -> None:
    assert agent.name.lower() == agent_name.lower()


def test_declared_action_dim_is_the_action_dim_emitted(agent_name) -> None:
    """An agent that accepts ``action_dim`` must honour it or refuse it.

    Ablations and reduced-actuator studies build agents at other widths. An
    agent that silently stores ``action_dim=3`` and then emits five components
    hands the environment a command it never asked for, and the mismatch only
    surfaces as a shape error deep inside the rollout.
    """
    try:
        other = make_agent(agent_name, obs_dim=12, action_dim=3)
    except (ValueError, TypeError, NotImplementedError):
        return  # refusing a width it cannot serve is the honest alternative
    assert other.action_dim == 3
    action = np.asarray(other.act(np.zeros(12, dtype=np.float32), deterministic=True))
    assert action.shape == (3,), (
        f"{agent_name} was built with action_dim=3 but emitted "
        f"{action.shape[0]} components; it should either honour the width or "
        "reject it at construction"
    )


# --- acting ------------------------------------------------------------------
@pytest.mark.parametrize("deterministic", [True, False])
def test_act_returns_a_legal_action(agent, deterministic: bool) -> None:
    """The environment clips out-of-range actions silently, so an agent that
    emits them trains against a command it never actually sent."""
    for seed in range(5):
        assert_legal_action(
            agent.act(observation(seed), deterministic=deterministic),
            f"{agent.name} act(deterministic={deterministic})",
        )


def test_act_handles_the_extremes_of_the_observation_space(agent) -> None:
    """Saturated observations are common early in training; a division by a
    zero-variance normaliser shows up here first."""
    for obs in (
        np.zeros(OBS_DIM, dtype=np.float32),
        np.ones(OBS_DIM, dtype=np.float32),
        -np.ones(OBS_DIM, dtype=np.float32),
    ):
        assert_legal_action(agent.act(obs, deterministic=True), "extreme observation")


def test_deterministic_mode_is_repeatable(agent) -> None:
    """Evaluation runs set ``deterministic=True``. An agent that ignores the
    flag reports noisy eval numbers and an unfair comparison against the
    scripted baselines."""
    obs = observation(42)
    first = np.asarray(agent.act(obs, deterministic=True), dtype=np.float64)
    for _ in range(4):
        again = np.asarray(agent.act(obs, deterministic=True), dtype=np.float64)
        assert np.array_equal(first, again), (
            f"deterministic act() varies across calls: {first} != {again}"
        )


def test_act_does_not_mutate_the_observation(agent) -> None:
    """The environment reuses its observation buffer; in-place normalisation
    inside the agent would corrupt the transition the learner stores."""
    obs = observation(3)
    original = obs.copy()
    agent.act(obs, deterministic=True)
    assert np.array_equal(obs, original)


def test_set_seed_makes_stochastic_behaviour_reproducible(agent_name) -> None:
    """Reproducibility is the point of ``set_seed``: two agents seeded alike
    must explore alike, or a seed sweep measures nothing."""
    observations = [observation(i) for i in range(8)]

    def rollout() -> list[np.ndarray]:
        agent = make_agent(agent_name)
        agent.set_seed(1234)
        return [
            np.asarray(agent.act(obs, deterministic=False), dtype=np.float64)
            for obs in observations
        ]

    first, second = rollout(), rollout()
    for index, (a, b) in enumerate(zip(first, second)):
        assert np.array_equal(a, b), (
            f"identically seeded agents diverged at step {index}: {a} != {b}"
        )


# --- runner hooks ------------------------------------------------------------
def test_every_runner_hook_is_callable(agent) -> None:
    """The runner calls all of these for every agent regardless of whether it
    learns; a scripted controller that forgets to inherit the no-ops crashes
    the sweep at the first step."""
    agent.reset()
    for i in range(3):
        agent.observe_transition(transition(i))
    stats = agent.update()
    assert isinstance(stats, TrainStats)
    assert isinstance(stats.values, dict)
    assert all(isinstance(k, str) for k in stats.values)
    assert all(math.isfinite(float(v)) for v in stats.values.values()), (
        f"non-finite training statistic in {stats.values}"
    )
    agent.on_episode_end(12.5, {"progress": 0.5})
    agent.reset()
    assert_legal_action(agent.act(observation(9), deterministic=True), "after hooks")


def test_terminal_transitions_are_accepted(agent) -> None:
    """Episode boundaries are where buffers get flushed; both flags must be
    handled without raising."""
    for terminated, truncated in ((True, False), (False, True)):
        tr = transition(5)
        tr.terminated = terminated
        tr.truncated = truncated
        agent.observe_transition(tr)
    agent.on_episode_end(-3.0, {})
    agent.update()


def test_scripted_agents_declare_that_they_do_not_learn(agent) -> None:
    """``learns`` decides whether the runner spends a training budget on this
    agent at all. A scripted baseline that claims to learn wastes the budget
    and, worse, is reported as a learned method in the comparison."""
    if isinstance(agent, ScriptedAgent):
        assert agent.learns is False
    if agent.learns is False:
        assert isinstance(agent.update(), TrainStats)


def test_learning_agents_have_parameters(agent) -> None:
    """An agent that claims to learn but reports zero parameters is either
    mis-declared or has an unwired network; both make its results meaningless."""
    if agent.learns:
        assert agent.num_parameters > 0, (
            f"{agent.name} declares learns=True but reports no parameters"
        )


# --- persistence -------------------------------------------------------------
def test_save_load_round_trip(agent_name, tmp_path) -> None:
    """Evaluation runs load a checkpoint written by a training run. If the
    round trip does not restore behaviour exactly, every reported score belongs
    to a different policy than the one that was trained."""
    agent = make_agent(agent_name)
    if agent.num_parameters == 0:
        pytest.skip(f"{agent_name} has no parameters to persist")
    agent.set_seed(7)
    for i in range(4):
        agent.observe_transition(transition(i))
    agent.update()

    path = tmp_path / f"{agent_name}.pt"
    agent.save(path)
    assert path.exists() or list(tmp_path.iterdir()), (
        "save() wrote nothing to the path it was given"
    )

    obs = [observation(100 + i) for i in range(4)]
    expected = [np.asarray(agent.act(o, deterministic=True)) for o in obs]

    restored = make_agent(agent_name)
    restored.load(path)
    for index, (o, want) in enumerate(zip(obs, expected)):
        got = np.asarray(restored.act(o, deterministic=True))
        assert np.allclose(got, want, atol=1e-6), (
            f"loaded policy disagrees with the saved one at step {index}: "
            f"{got} != {want}"
        )
