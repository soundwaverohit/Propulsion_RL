"""The environment loop, exercised on real (propulsion x mission) pairings.

Everything here goes through ``make_env`` -- the one function the experiment
matrix calls -- and is parametrised over a representative cross-section of the
registries, one system per family times the first couple of missions. The
invariants are the ones an RL library assumes without asking: a five-tuple
step, a fixed-width finite observation inside the declared space, a constraint
channel on every step, reproducibility under a seed, and physics that closes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from propulsion_rl.core.constants import G0
from propulsion_rl.core.spaces import Box
from propulsion_rl.core.types import CANONICAL_ACTION_DIM, OBS_DIM, CanonicalCommand
from propulsion_rl.missions.base import MissionResult
from tests.conftest import call_env_factory, find_instance, find_symbol, flatten_info

pytestmark = pytest.mark.integration

#: A steady prograde burn at mid Isp -- the simplest control that actually flies.
PROGRADE = CanonicalCommand(
    throttle=1.0,
    operating_point=0.5,
    thrust_yaw=0.0,
    thrust_pitch=0.0,
    thermal_margin=0.5,
).to_array()


def build_env(pairing: tuple[str, str], *, seed: int | None = 0,
              max_steps: int | None = None) -> Any:
    """Build one environment for a pairing, capped at *max_steps* if asked."""
    make_env = find_symbol("propulsion_rl.envs", "make_env")
    extra: dict[str, Any] = {"seed": seed}
    if max_steps is not None:
        config_cls = find_symbol("propulsion_rl.envs", "EnvConfig")
        extra["config"] = config_cls(max_steps=max_steps)
    return call_env_factory(make_env, pairing[0], pairing[1], **extra)


def rollout(env: Any, steps: int, *, seed: int | None = 0,
            action: np.ndarray | None = None) -> dict[str, list]:
    """Run up to *steps* steps of a fixed action, recording everything."""
    act = PROGRADE if action is None else action
    obs, info = env.reset(seed=seed)
    log: dict[str, list] = {
        "obs": [np.asarray(obs).copy()],
        "reward": [],
        "info": [info],
        "terminated": [],
        "truncated": [],
    }
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(act)
        log["obs"].append(np.asarray(obs).copy())
        log["reward"].append(float(reward))
        log["info"].append(info)
        log["terminated"].append(bool(terminated))
        log["truncated"].append(bool(truncated))
        if terminated or truncated:
            break
    return log


def step_rows(env: Any, log: dict[str, list]) -> list[dict[str, Any]]:
    """Per-step telemetry, from the env's own log when it keeps one."""
    telemetry = getattr(getattr(env, "unwrapped", env), "telemetry", None)
    if telemetry:
        return [record.as_row() for record in telemetry]
    return [flatten_info(info) for info in log["info"][1:]]


def series(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Numeric column *key*, or an empty list when the env does not log it."""
    if not rows or key not in rows[0]:
        return []
    return [float(row[key]) for row in rows]


# --- API shape ---------------------------------------------------------------
def test_spaces_are_the_canonical_ones(pairing) -> None:
    """A policy trained on one pairing must be loadable on any other, which is
    only true while every environment advertises exactly the same spaces."""
    env = build_env(pairing)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (CANONICAL_ACTION_DIM,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert env.action_space.dtype == np.float32
    env.close()


def test_reset_returns_an_observation_in_the_space(pairing) -> None:
    """Learners allocate buffers from ``observation_space``; an observation
    that does not fit it corrupts memory-mapped rollout storage."""
    env = build_env(pairing)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all(), f"non-finite reset observation: {obs}"
    assert env.observation_space.contains(obs), (
        f"reset observation outside the declared space: "
        f"min={obs.min()}, max={obs.max()}"
    )
    assert isinstance(info, dict)
    env.close()


def test_step_returns_the_five_tuple(pairing) -> None:
    """The Gymnasium five-tuple, byte-identical in shape, is what lets external
    libraries drive this environment through a thin adapter."""
    env = build_env(pairing)
    env.reset(seed=0)
    result = env.step(PROGRADE)
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(reward, (float, int, np.floating))
    assert math.isfinite(float(reward)), f"reward is {reward}"
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)
    assert not (terminated and truncated), (
        "an episode cannot both terminate and truncate on the same step"
    )
    env.close()


def test_constraint_cost_is_present_on_every_step(pairing) -> None:
    """Constrained-RL agents read ``info['constraint_cost']`` without knowing
    anything about the environment. One missing key mid-episode is a KeyError
    thousands of steps into a training run."""
    env = build_env(pairing)
    log = rollout(env, 25)
    for index, info in enumerate(log["info"][1:]):
        assert "constraint_cost" in info, f"no constraint_cost at step {index}"
        cost = float(info["constraint_cost"])
        assert math.isfinite(cost), f"constraint_cost is {cost} at step {index}"
        assert cost >= 0.0, f"negative constraint cost {cost} at step {index}"
    env.close()


def test_observations_stay_inside_the_space_all_episode(pairing) -> None:
    """The observation is the policy's only input; a spike outside the space
    saturates the first layer and shows up as an unexplained loss of skill."""
    env = build_env(pairing)
    log = rollout(env, 50)
    for index, obs in enumerate(log["obs"]):
        assert np.isfinite(obs).all(), f"non-finite observation at step {index}"
        assert env.observation_space.contains(obs), (
            f"observation left the space at step {index}: "
            f"min={obs.min():.4g}, max={obs.max():.4g}"
        )
    env.close()


def test_step_before_reset_is_an_error(pairing) -> None:
    """Stepping an unreset environment would silently fly an undefined state."""
    env = build_env(pairing)
    with pytest.raises((RuntimeError, AttributeError, TypeError, ValueError)):
        env.step(PROGRADE)
    env.close()


# --- episodes ----------------------------------------------------------------
def test_episode_ends_and_reports_a_mission_result(pairing) -> None:
    """Every episode must end with a well-formed ``MissionResult``: it is the
    only record the economics model and the ranking table ever see."""
    env = build_env(pairing, max_steps=12)
    log = rollout(env, 200)
    assert log["terminated"][-1] or log["truncated"][-1], (
        "episode did not finish within its own step cap"
    )
    info = log["info"][-1]
    result = find_instance(info, MissionResult)
    assert result is not None, (
        f"terminal info carries no MissionResult; keys were {sorted(info)}"
    )
    assert 0.0 <= result.progress <= 1.0
    assert result.elapsed_s > 0.0
    assert result.delta_v_m_s >= 0.0
    assert result.propellant_used_kg >= 0.0
    assert math.isfinite(result.total_constraint_cost)
    assert result.constraint_violations >= 0
    env.close()


def test_identical_seeds_give_identical_trajectories(pairing) -> None:
    """Reproducibility is the backbone of the comparison: a seeded rollout must
    replay exactly, or no two agents are ever measured on the same task."""
    first = rollout(build_env(pairing, seed=None), 30, seed=4321)
    second = rollout(build_env(pairing, seed=None), 30, seed=4321)
    assert len(first["obs"]) == len(second["obs"])
    for index, (a, b) in enumerate(zip(first["obs"], second["obs"])):
        assert np.array_equal(a, b), f"observations diverged at step {index}"
    assert first["reward"] == second["reward"]


def test_different_seeds_give_different_trajectories(pairing) -> None:
    """Without episode-to-episode variation, a seed sweep measures nothing and
    a policy can memorise one trajectory."""
    a = rollout(build_env(pairing, seed=None), 30, seed=1)
    b = rollout(build_env(pairing, seed=None), 30, seed=999)
    same = len(a["obs"]) == len(b["obs"]) and all(
        np.array_equal(x, y) for x, y in zip(a["obs"], b["obs"])
    )
    assert not same, "two different seeds produced identical trajectories"


# --- physics -----------------------------------------------------------------
def test_mass_and_propellant_are_monotone(pairing) -> None:
    """Mass only leaves the vehicle. A rising propellant trace means the
    environment is crediting mass back -- free delta-v, and every result that
    depends on it is worthless."""
    env = build_env(pairing)
    log = rollout(env, 60)
    rows = step_rows(env, log)
    masses = series(rows, "mass_kg")
    propellant = series(rows, "propellant_kg")
    if not masses and not propellant:
        pytest.skip("environment logs neither mass_kg nor propellant_kg")
    for name, trace in (("mass_kg", masses), ("propellant_kg", propellant)):
        for index, (before, after) in enumerate(zip(trace, trace[1:])):
            assert after <= before + 1e-9, (
                f"{name} rose from {before:.6f} to {after:.6f} at step {index}"
            )
        assert all(v >= 0.0 for v in trace), f"{name} went negative"
    env.close()


def test_delta_v_is_consistent_with_the_rocket_equation(pairing) -> None:
    """Tsiolkovsky ties the three things the benchmark ranks on together: the
    delta-v credited, the propellant debited and the Isp reported. If they do
    not close, either the mission looks easier than it is or the propellant
    bill is wrong -- and the cost per kilogram is wrong with it.
    """
    env = build_env(pairing)
    log = rollout(env, 60)
    rows = step_rows(env, log)
    masses = series(rows, "mass_kg")
    isps = series(rows, "isp_s")
    delta_vs = series(rows, "delta_v_m_s")
    if not (masses and isps and delta_vs) or len(masses) < 3:
        pytest.skip("environment does not log mass_kg, isp_s and delta_v_m_s")

    # Compare increments, so the pre-first-step mass never has to be guessed.
    expected = 0.0
    for previous, mass, isp in zip(masses, masses[1:], isps[1:]):
        if mass > 0.0 and previous > mass and isp > 0.0:
            expected += isp * G0 * math.log(previous / mass)
    if expected <= 0.0:
        pytest.skip("no propellant was burned in this rollout")

    reported = delta_vs[-1] - delta_vs[0]
    assert reported == pytest.approx(expected, rel=0.02), (
        f"environment credited {reported:.3f} m/s of delta-v; the rocket "
        f"equation on the propellant it burned gives {expected:.3f} m/s"
    )
    env.close()


def test_reported_thrust_and_flow_agree_with_the_mass_debited(pairing) -> None:
    """The propellant the vehicle loses must be the mass flow the thruster
    reported, integrated over the step. A mismatch means two different physics
    models are running side by side."""
    env = build_env(pairing)
    log = rollout(env, 40)
    rows = step_rows(env, log)
    masses = series(rows, "mass_kg")
    mdots = series(rows, "mdot_kg_s")
    times = series(rows, "t_s")
    if not (masses and mdots and times) or len(masses) < 3:
        pytest.skip("environment does not log mass_kg, mdot_kg_s and t_s")
    dt = times[1] - times[0]
    burned = sum(mdots[1:]) * dt
    lost = masses[0] - masses[-1]
    if burned <= 0.0:
        pytest.skip("no propellant was burned in this rollout")
    assert lost == pytest.approx(burned, rel=0.02), (
        f"vehicle lost {lost:.6g} kg but the thruster reported {burned:.6g} kg "
        "of flow"
    )
    env.close()


# --- composition -------------------------------------------------------------
def test_wrappers_compose_without_breaking_the_contract(pairing) -> None:
    """The recommended wrapper stack is several layers deep. Each layer must
    forward the five-tuple, the spaces and the constraint channel untouched,
    or an agent silently trains against a different interface than it reports.
    """
    wrap = find_symbol("propulsion_rl.envs", "wrap")
    env = build_env(pairing)
    wrapped = wrap(env)
    assert wrapped.observation_space.shape == (OBS_DIM,)
    assert wrapped.action_space.shape == (CANONICAL_ACTION_DIM,)
    obs, info = wrapped.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()
    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(PROGRADE)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
        assert math.isfinite(float(reward))
        assert "constraint_cost" in info
        if terminated or truncated:
            break
    assert wrapped.unwrapped is getattr(env, "unwrapped", env)
    wrapped.close()


@pytest.mark.slow
def test_sync_vector_env_autoreset_preserves_the_terminal_observation(
    pairing,
) -> None:
    """Generalised advantage estimation bootstraps the last step of an episode
    from V(s_terminal). If auto-reset quietly hands back the *next* episode's
    first observation instead, nothing crashes -- the critic just bootstraps
    from an unrelated state and every episode's final advantage is wrong.
    """
    vector_cls = find_symbol("propulsion_rl.envs", "SyncVectorEnv")
    env = vector_cls([lambda: build_env(pairing, seed=i, max_steps=6) for i in (0, 1)])
    assert env.num_envs == 2
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, OBS_DIM)

    actions = np.stack([PROGRADE, PROGRADE])
    for _ in range(20):
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        assert obs.shape == (2, OBS_DIM)
        assert np.isfinite(obs).all()
        assert np.asarray(rewards).shape == (2,)
        done = np.asarray(terminateds) | np.asarray(truncateds)
        if done.any():
            finals = infos.get("final_observation")
            assert finals is not None, (
                "a sub-environment finished but no final_observation was "
                f"reported; info keys were {sorted(infos)}"
            )
            mask = np.asarray(infos.get("_final_observation", done), dtype=bool)
            assert mask.tolist() == done.tolist()
            for index in np.flatnonzero(done):
                terminal = np.asarray(finals[index])
                assert terminal.shape == (OBS_DIM,)
                assert np.isfinite(terminal).all()
                assert not np.array_equal(terminal, obs[index]), (
                    "the terminal observation equals the post-reset one, so "
                    "the true final state was lost"
                )
            break
    else:
        pytest.fail("no sub-environment finished within 20 steps of a 6-step cap")
    env.close()
