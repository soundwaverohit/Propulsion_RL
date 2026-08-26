"""One short flight per propulsion family, from action to dollars.

This is the smoke test for the benchmark's headline claim: that a controller,
an electric or nuclear propulsion model, a mission and a cost model can be
composed by name and produce a technically plausible outcome with an economic
figure attached. It deliberately asserts loose, physical bounds -- the tight
invariants live in the conformance suites -- because its job is to catch a
pipeline that is wired together wrongly, not a model that is tuned wrongly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from propulsion_rl.core.types import (
    CANONICAL_ACTION_DIM,
    OBS_DIM,
    CanonicalCommand,
    PropulsionFamily,
    TerminationReason,
)
from propulsion_rl.economics.base import EconomicResult
from propulsion_rl.missions.base import MissionResult
from tests.conftest import (
    call_env_factory,
    find_instance,
    find_symbol,
    make_agent,
    make_cost_model,
    propulsion_names_in_family,
    registered,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MAX_STEPS = 40


def pick_agent() -> tuple[str, Any]:
    """A scripted baseline if one is registered, else any agent, else a
    hand-rolled prograde controller -- the study's reference point."""
    for name in registered("AGENT"):
        candidate = make_agent(name)
        if candidate.learns is False:
            return name, candidate
    names = registered("AGENT")
    if names:
        return names[0], make_agent(names[0])
    return "prograde_constant", _ProgradeController()


class _ProgradeController:
    """Full throttle along the velocity vector. Not clever, but never wrong."""

    name = "prograde_constant"
    learns = False

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        assert obs.shape == (OBS_DIM,)
        return CanonicalCommand(1.0, 0.5, 0.0, 0.0, 0.5).to_array()

    def reset(self) -> None:
        pass


def fly(family: PropulsionFamily) -> tuple[str, str, MissionResult, Any]:
    """Fly a short episode on the first system of *family*; return the result."""
    systems = propulsion_names_in_family(family.value)
    if not systems:
        pytest.skip(f"no {family.value} propulsion system registered yet")
    missions = registered("MISSION")
    if not missions:
        pytest.skip("no mission registered yet")

    make_env = find_symbol("propulsion_rl.envs", "make_env")
    config_cls = find_symbol("propulsion_rl.envs", "EnvConfig")
    propulsion_name, mission_name = systems[0], missions[0]
    env = call_env_factory(
        make_env,
        propulsion_name,
        mission_name,
        seed=17,
        config=config_cls(max_steps=MAX_STEPS),
    )

    agent_name, agent = pick_agent()
    agent.reset()
    obs, _ = env.reset(seed=17)
    total_reward = 0.0
    info: dict[str, Any] = {}
    for step in range(MAX_STEPS + 5):
        action = np.asarray(agent.act(obs, deterministic=True), dtype=np.float32)
        assert action.shape == (CANONICAL_ACTION_DIM,), (
            f"{agent_name} emitted {action.shape}, not the canonical action"
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        assert np.isfinite(obs).all(), f"observation went non-finite at step {step}"
        assert math.isfinite(total_reward), f"return went non-finite at step {step}"
        if terminated or truncated:
            break
    else:
        pytest.fail("episode never ended despite an explicit step cap")

    result = find_instance(info, MissionResult)
    assert result is not None, (
        f"{propulsion_name}/{mission_name} ended with no MissionResult"
    )
    env.close()
    return propulsion_name, mission_name, result, env


def assert_plausible(result: MissionResult, where: str) -> None:
    """Bounds no real spacecraft can violate, whatever the control policy."""
    assert isinstance(result.reason, TerminationReason)
    assert result.reason is not TerminationReason.RUNNING, f"{where}: still running"
    assert result.reason is not TerminationReason.DIVERGED, (
        f"{where}: the integrator diverged on a constant prograde burn"
    )
    assert 0.0 <= result.progress <= 1.0, f"{where}: progress {result.progress}"
    assert 0.0 < result.elapsed_s < 100.0 * 365.25 * 86400.0, (
        f"{where}: elapsed {result.elapsed_s} s"
    )
    assert 0.0 <= result.delta_v_m_s < 1.0e6, f"{where}: dv {result.delta_v_m_s}"
    assert 0.0 <= result.propellant_used_kg < 1.0e6, (
        f"{where}: propellant {result.propellant_used_kg} kg"
    )
    assert math.isfinite(result.terminal_error)


@pytest.mark.parametrize("family", list(PropulsionFamily))
def test_a_scripted_baseline_flies_and_is_costed(family: PropulsionFamily) -> None:
    """End to end for one family: fly, summarise, then price the summary.

    Both halves of the study have to survive the round trip -- a mission result
    the cost model cannot consume is as broken as physics that does not run.
    """
    propulsion_name, mission_name, result, _ = fly(family)
    where = f"{family.value}: {propulsion_name}/{mission_name}"
    assert_plausible(result, where)

    if result.propellant_used_kg > 0.0:
        assert result.delta_v_m_s > 0.0, (
            f"{where}: burned propellant without gaining any delta-v"
        )

    cost_models = registered("COST_MODEL")
    if not cost_models:
        pytest.skip("no cost model registered yet")

    from propulsion_rl.core.registry import PROPULSION

    bom = PROPULSION.make(propulsion_name).bom()
    from propulsion_rl.core.types import HealthReport

    for name in cost_models:
        economics = make_cost_model(name).evaluate(bom, result, HealthReport())
        assert isinstance(economics, EconomicResult), f"{where}/{name}"
        total = economics.breakdown.total
        assert math.isfinite(total) and total > 0.0, (
            f"{where}/{name}: total cost {total}"
        )
        assert not math.isnan(economics.cost_per_kg_delivered)
        assert not math.isnan(economics.figure_of_merit)


@pytest.mark.parametrize("family", list(PropulsionFamily))
def test_the_environment_attaches_economics_when_asked(
    family: PropulsionFamily,
) -> None:
    """``make_env(..., cost_model=...)`` is how the sweep gets dollars out of a
    rollout; the terminal info must then carry a finished ``EconomicResult``."""
    systems = propulsion_names_in_family(family.value)
    missions = registered("MISSION")
    cost_models = registered("COST_MODEL")
    if not (systems and missions and cost_models):
        pytest.skip(
            f"need a {family.value} system, a mission and a cost model; have "
            f"{len(systems)}/{len(missions)}/{len(cost_models)}"
        )
    make_env = find_symbol("propulsion_rl.envs", "make_env")
    config_cls = find_symbol("propulsion_rl.envs", "EnvConfig")
    env = call_env_factory(
        make_env,
        systems[0],
        missions[0],
        seed=3,
        cost_model=cost_models[0],
        config=config_cls(max_steps=15),
    )
    obs, _ = env.reset(seed=3)
    action = CanonicalCommand(1.0, 0.5, 0.0, 0.0, 0.5).to_array()
    info: dict[str, Any] = {}
    for _ in range(30):
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    economics = find_instance(info, EconomicResult)
    assert economics is not None, (
        f"terminal info has no EconomicResult; keys were {sorted(info)}"
    )
    assert math.isfinite(economics.breakdown.total)
    assert economics.breakdown.total > 0.0
    env.close()
