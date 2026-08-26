"""Conformance suite run against every registered cost model.

Parametrised from ``core.registry.COST_MODEL`` at collection time. Economics is
the tiebreaker of the whole study, so the headline numbers have to be
defensible: no negative dollars, no NaN dividing a failed mission, a breakdown
that adds up, and a written record of every price that went in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propulsion_rl.core.types import PropulsionFamily, TerminationReason
from propulsion_rl.economics.base import CostBreakdown, CostModel, EconomicResult
from tests.conftest import synthetic_bom, synthetic_health, synthetic_mission_result

_COST_FIELDS = (
    "thruster_capex",
    "power_system_capex",
    "reactor_capex",
    "tankage_capex",
    "integration_capex",
    "propellant_cost",
    "launch_cost",
    "operations_cost",
    "refurbishment_cost",
    "insurance_cost",
    "time_value_cost",
)


def assert_well_formed(result: EconomicResult, where: str) -> None:
    """Every published economic figure must be a real, non-negative number."""
    assert isinstance(result, EconomicResult), f"{where}: not an EconomicResult"
    breakdown = result.breakdown
    assert isinstance(breakdown, CostBreakdown), f"{where}: breakdown is wrong type"
    for field in _COST_FIELDS:
        value = getattr(breakdown, field)
        assert math.isfinite(value), f"{where}: {field} is {value}"
        assert value >= 0.0, f"{where}: {field} is negative ({value})"
    for key, value in breakdown.extras.items():
        assert math.isfinite(value), f"{where}: extras[{key!r}] is {value}"
    assert math.isfinite(breakdown.total)
    assert breakdown.total > 0.0, f"{where}: a free mission is not a cost model"
    assert 1900 < breakdown.currency_year < 2200
    for field in ("amortized_cost", "uses_remaining", "npv", "figure_of_merit"):
        value = getattr(result, field)
        assert not math.isnan(value), f"{where}: {field} is NaN"
    assert result.amortized_cost >= 0.0
    assert result.uses_remaining >= 0.0
    assert not math.isnan(result.cost_per_kg_delivered), f"{where}: NaN $/kg"
    assert not math.isnan(result.cost_per_delta_v), f"{where}: NaN $/(kg m/s)"
    assert result.cost_per_kg_delivered >= 0.0
    assert result.cost_per_delta_v >= 0.0
    assert isinstance(result.notes, dict)


# --- the breakdown itself ----------------------------------------------------
def test_breakdown_total_is_capex_plus_opex_plus_risk() -> None:
    """The headline total must be the sum of the four buckets. A term dropped
    from ``total`` would make an expensive option look cheap in the ranking."""
    breakdown = CostBreakdown(
        thruster_capex=1.0,
        power_system_capex=2.0,
        reactor_capex=4.0,
        tankage_capex=8.0,
        integration_capex=16.0,
        propellant_cost=32.0,
        launch_cost=64.0,
        operations_cost=128.0,
        refurbishment_cost=256.0,
        insurance_cost=512.0,
        time_value_cost=1024.0,
    )
    assert breakdown.capex == pytest.approx(31.0)
    assert breakdown.opex == pytest.approx(480.0)
    assert breakdown.total == pytest.approx(
        breakdown.capex
        + breakdown.opex
        + breakdown.insurance_cost
        + breakdown.time_value_cost
    )
    assert breakdown.total == pytest.approx(2047.0)


def test_breakdown_as_dict_is_flat_and_carries_the_derived_totals() -> None:
    """The sensitivity sweep pivots these dicts into a DataFrame."""
    row = CostBreakdown(thruster_capex=3.0, extras={"tooling": 1.5}).as_dict()
    assert row["thruster_capex"] == 3.0
    assert row["tooling"] == 1.5
    assert row["capex"] == pytest.approx(3.0)
    assert row["total"] == pytest.approx(3.0)
    assert "extras" not in row


# --- registered models -------------------------------------------------------
def test_registered_object_implements_the_abc(cost_model) -> None:
    assert isinstance(cost_model, CostModel)
    assert isinstance(cost_model.name, str) and cost_model.name


def test_name_matches_the_registry_key(cost_model, cost_model_name) -> None:
    assert cost_model.name.lower() == cost_model_name.lower()


def test_assumptions_are_published(cost_model) -> None:
    """Price assumptions dominate the answer, so the model must state them.
    An empty dict means the sensitivity sweep has nothing to sweep."""
    assumptions = cost_model.assumptions()
    assert isinstance(assumptions, dict)
    assert assumptions, f"{cost_model.name} publishes no cost assumptions"
    assert all(isinstance(k, str) and k for k in assumptions)
    for key, value in assumptions.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(float(value)), f"assumption {key} is {value}"


@pytest.mark.parametrize("family", list(PropulsionFamily))
def test_evaluate_prices_a_successful_mission(cost_model, family) -> None:
    """The happy path for both families: a delivered payload gets a finite,
    positive cost per kilogram."""
    bom = synthetic_bom(
        family=family,
        reactor_thermal_w=1.0e6 if family is PropulsionFamily.NUCLEAR else 0.0,
        propellant_type="hydrogen" if family is PropulsionFamily.NUCLEAR else "xenon",
    )
    result = cost_model.evaluate(bom, synthetic_mission_result(), synthetic_health())
    assert_well_formed(result, f"{cost_model.name}/{family.value}")
    assert math.isfinite(result.cost_per_kg_delivered)
    assert result.cost_per_kg_delivered > 0.0


def test_cost_per_kg_rises_as_the_delivered_payload_falls(cost_model) -> None:
    """Whatever amortisation scheme a model uses, the headline figure is a cost
    *per kilogram delivered*: deliver a quarter of the payload for the same
    hardware and the number must not improve."""
    bom, health = synthetic_bom(), synthetic_health()
    full = cost_model.evaluate(
        bom, synthetic_mission_result(payload_delivered_kg=400.0), health
    ).cost_per_kg_delivered
    quarter = cost_model.evaluate(
        bom, synthetic_mission_result(payload_delivered_kg=100.0), health
    ).cost_per_kg_delivered
    assert quarter >= full * (1.0 - 1e-9), (
        f"delivering 100 kg costs {quarter:.6g}/kg against {full:.6g}/kg "
        "for 400 kg"
    )


def test_zero_payload_yields_infinite_cost_per_kg(cost_model) -> None:
    """Dividing by a zero payload must give inf, not NaN and not a crash: the
    ranking sorts on this field, and a NaN silently sorts anywhere."""
    result = cost_model.evaluate(
        synthetic_bom(),
        synthetic_mission_result(payload_delivered_kg=0.0),
        synthetic_health(),
    )
    assert_well_formed(result, f"{cost_model.name}/zero-payload")
    assert result.cost_per_kg_delivered == math.inf, (
        f"delivering nothing costs {result.cost_per_kg_delivered} per kg"
    )


def test_failed_mission_is_priced_without_crashing(cost_model) -> None:
    """A hardware failure at t=0 is a legal outcome and the most expensive one;
    the model must still return a usable record for the comparison table."""
    result = cost_model.evaluate(
        synthetic_bom(),
        synthetic_mission_result(
            reason=TerminationReason.HARDWARE_FAILURE,
            success=False,
            progress=0.0,
            elapsed_s=0.0,
            delta_v_m_s=0.0,
            propellant_used_kg=0.0,
            payload_delivered_kg=0.0,
            terminal_error=1.0e9,
        ),
        synthetic_health(wear_fraction=1.0, remaining_life_s=0.0, failed=True),
    )
    assert_well_formed(result, f"{cost_model.name}/failed")
    assert result.cost_per_kg_delivered == math.inf
    assert result.cost_per_delta_v == math.inf, (
        "a mission with no delta-v cannot have a finite cost per delta-v"
    )


def test_out_of_propellant_mission_is_priced(cost_model) -> None:
    """Partial progress with an empty tank -- the most common real failure."""
    result = cost_model.evaluate(
        synthetic_bom(),
        synthetic_mission_result(
            reason=TerminationReason.OUT_OF_PROPELLANT,
            success=False,
            progress=0.6,
            payload_delivered_kg=0.0,
            propellant_used_kg=450.0,
        ),
        synthetic_health(),
    )
    assert_well_formed(result, f"{cost_model.name}/out-of-propellant")


def test_evaluate_does_not_mutate_its_inputs(cost_model) -> None:
    """The runner prices one episode against several cost models in turn; a
    model that edits the bill of materials would change the next model's
    answer, and the two would no longer be comparable."""
    bom = synthetic_bom()
    result = synthetic_mission_result()
    health = synthetic_health()
    before = (
        [getattr(bom, f.name) for f in bom.__dataclass_fields__.values()],
        [getattr(result, f.name) for f in result.__dataclass_fields__.values()],
        [getattr(health, f.name) for f in health.__dataclass_fields__.values()],
    )
    cost_model.evaluate(bom, result, health)
    after = (
        [getattr(bom, f.name) for f in bom.__dataclass_fields__.values()],
        [getattr(result, f.name) for f in result.__dataclass_fields__.values()],
        [getattr(health, f.name) for f in health.__dataclass_fields__.values()],
    )
    assert before == after


def test_evaluate_is_deterministic(cost_model) -> None:
    """Costing has no business being stochastic; two runs of one episode must
    produce the same dollar figure."""
    args = (synthetic_bom(), synthetic_mission_result(), synthetic_health())
    first = cost_model.evaluate(*args)
    second = cost_model.evaluate(*args)
    assert first.breakdown.as_dict() == second.breakdown.as_dict()
    assert first.cost_per_kg_delivered == second.cost_per_kg_delivered


def test_bigger_hardware_never_costs_less(cost_model) -> None:
    """Monotonicity in scale. Doubling every physical quantity while holding
    qualified life fixed must not make the stage cheaper -- a sign error or a
    misplaced reciprocal in a scaling law shows up here immediately."""
    small = synthetic_bom()
    big = synthetic_bom(
        thruster_units=small.thruster_units * 2,
        rated_power_w=small.rated_power_w * 2,
        power_source_w=small.power_source_w * 2,
        reactor_thermal_w=small.reactor_thermal_w * 2,
        radiator_area_m2=small.radiator_area_m2 * 2,
        dry_mass_kg=small.dry_mass_kg * 2,
        tank_capacity_kg=small.tank_capacity_kg * 2,
    )
    result = synthetic_mission_result()
    health = synthetic_health()
    cheap = cost_model.evaluate(small, result, health).breakdown.total
    dear = cost_model.evaluate(big, result, health).breakdown.total
    assert dear >= cheap * (1.0 - 1e-9), (
        f"doubling the hardware cut the total from {cheap:.6g} to {dear:.6g}"
    )


def test_longer_trip_never_reduces_the_time_value_cost(cost_model) -> None:
    """``time_value_cost`` is the opportunity cost of a slow transfer, which is
    the whole economic argument against low-thrust propulsion. It must rise
    with trip time, or the comparison flatters electric propulsion."""
    bom, health = synthetic_bom(), synthetic_health()
    quick = cost_model.evaluate(
        bom, synthetic_mission_result(elapsed_s=90.0 * 86400.0), health
    ).breakdown
    slow = cost_model.evaluate(
        bom, synthetic_mission_result(elapsed_s=900.0 * 86400.0), health
    ).breakdown
    assert slow.time_value_cost >= quick.time_value_cost * (1.0 - 1e-9), (
        f"a ten times longer transfer priced its time at "
        f"{slow.time_value_cost:.6g} against {quick.time_value_cost:.6g}"
    )


def test_more_propellant_never_reduces_the_propellant_bill(cost_model) -> None:
    """Propellant is priced per kilogram burned; the sign of that term decides
    whether the model rewards or punishes efficiency."""
    bom, health = synthetic_bom(), synthetic_health()
    light = cost_model.evaluate(
        bom, synthetic_mission_result(propellant_used_kg=50.0), health
    ).breakdown
    heavy = cost_model.evaluate(
        bom, synthetic_mission_result(propellant_used_kg=400.0), health
    ).breakdown
    assert heavy.propellant_cost >= light.propellant_cost * (1.0 - 1e-9)


def test_costs_are_finite_for_extreme_but_legal_inputs(cost_model) -> None:
    """Sweeps reach the edges of the design space; the cost model must not
    produce inf or NaN dollars for a very small or very large stage."""
    health = synthetic_health()
    for bom in (
        synthetic_bom(
            thruster_units=1,
            rated_power_w=1.0,
            power_source_w=1.0,
            radiator_area_m2=0.0,
            dry_mass_kg=1.0,
            tank_capacity_kg=0.0,
        ),
        synthetic_bom(
            family=PropulsionFamily.NUCLEAR,
            thruster_units=8,
            rated_power_w=5.0e8,
            power_source_w=1.0e8,
            reactor_thermal_w=5.0e8,
            radiator_area_m2=5.0e3,
            dry_mass_kg=5.0e4,
            propellant_type="hydrogen",
            tank_capacity_kg=1.0e5,
        ),
    ):
        result = cost_model.evaluate(bom, synthetic_mission_result(), health)
        assert_well_formed(result, f"{cost_model.name}/{bom.dry_mass_kg:g}kg")
        assert math.isfinite(result.breakdown.total)


def test_figure_of_merit_ranks_the_cheaper_mission_higher(cost_model) -> None:
    """``figure_of_merit`` is documented as higher-is-better and is what the
    ranking sorts on; if it tracks cost the wrong way round, the study's
    conclusion is inverted."""
    bom, health = synthetic_bom(), synthetic_health()
    good = cost_model.evaluate(bom, synthetic_mission_result(), health)
    bad = cost_model.evaluate(
        bom,
        synthetic_mission_result(
            reason=TerminationReason.TIMEOUT,
            success=False,
            progress=0.2,
            payload_delivered_kg=0.0,
            elapsed_s=1500.0 * 86400.0,
        ),
        health,
    )
    assert np.isfinite(good.figure_of_merit)
    assert good.figure_of_merit >= bad.figure_of_merit, (
        f"a failed, slow mission scores {bad.figure_of_merit:.6g} against "
        f"{good.figure_of_merit:.6g} for a successful one"
    )
