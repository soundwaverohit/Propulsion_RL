"""Conformance suite run against every registered propulsion system.

Parametrised from ``core.registry.PROPULSION`` at collection time, so a new
thruster is covered by all of this the moment it registers -- no edit here.

The invariants below are the ones that hold for *any* reaction engine, ion or
nuclear thermal: momentum bookkeeping, energy bookkeeping, the power budget,
monotone response to the throttle, and the purity/determinism guarantees the
base class promises. A model that violates one of them will still produce a
learning curve; it will just be a learning curve for physics that does not
exist, which is why these are asserted rather than eyeballed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propulsion_rl.core.constants import AU, G0
from propulsion_rl.core.types import (
    PROPULSION_OBS_DIM,
    BillOfMaterials,
    ConstraintReport,
    HealthReport,
    Limits,
    PropulsionFamily,
    ThrusterOutput,
)
from propulsion_rl.propulsion.base import PropulsionSystem
from tests.conftest import (
    OBS_ABS_MAX,
    command_grid,
    context_for,
    float_attributes,
    make_propulsion,
    nominal_command,
    output_signature,
)

#: Relative slack on identities that should hold to machine precision but pass
#: through a few floating point operations first.
RTOL = 1e-6


def fresh(name: str, seed: int = 12345):
    """A constructed, reset propulsion system -- the state every test starts in."""
    system = make_propulsion(name)
    system.reset(np.random.default_rng(seed))
    return system


def _finite_output(out: ThrusterOutput, where: str) -> None:
    for field in (
        "thrust_n",
        "mdot_kg_s",
        "isp_s",
        "power_draw_w",
        "thermal_power_w",
        "heat_reject_w",
        "efficiency",
    ):
        value = getattr(out, field)
        assert isinstance(value, (int, float)), f"{where}: {field} is not a scalar"
        assert math.isfinite(value), f"{where}: {field} is {value}"
        assert value >= 0.0, f"{where}: {field} is negative ({value})"


# --- structure ---------------------------------------------------------------
def test_registered_object_implements_the_abc(propulsion) -> None:
    """The registry is typed by convention only; everything downstream calls
    the ABC's methods, so an entry that is not a ``PropulsionSystem`` breaks
    the environment at run time rather than at import time."""
    assert isinstance(propulsion, PropulsionSystem)
    assert isinstance(propulsion.family, PropulsionFamily)
    assert isinstance(propulsion.propellant, str) and propulsion.propellant
    assert isinstance(propulsion.self_powered, bool)


def test_name_matches_the_registry_key(propulsion, propulsion_name) -> None:
    """Telemetry, results files and the comparison matrix are keyed on the
    registry name; a mismatch silently splits one system into two rows."""
    assert isinstance(propulsion.name, str) and propulsion.name
    assert propulsion.name.lower() == propulsion_name.lower()


def test_limits_are_a_coherent_envelope(propulsion) -> None:
    """``Limits`` is what every normalisation, scripted baseline and sizing
    calculation divides by, so an inverted or zero bound propagates everywhere."""
    limits = propulsion.limits()
    assert isinstance(limits, Limits)
    assert limits.max_thrust_n > 0.0
    assert 0.0 <= limits.min_thrust_n <= limits.max_thrust_n
    assert limits.max_power_w > 0.0
    assert 0.0 <= limits.min_power_w <= limits.max_power_w
    low, high = limits.isp_range_s
    assert 0.0 < low <= high, "Isp range must be positive and ordered"
    assert high < 1e6, "Isp above 1e6 s is not a chemical or nuclear thruster"
    assert limits.max_temperature_k > 0.0
    assert limits.qualified_life_s > 0.0
    assert limits.max_throughput_kg > 0.0
    assert limits.max_restarts >= 1
    assert limits.min_off_time_s >= 0.0


def test_limits_do_not_change_during_an_episode(propulsion_name) -> None:
    """The base class calls ``limits`` static. Observation normalisation is
    computed against it, so a drifting envelope makes the observation scale
    time-varying and the policy's inputs non-stationary."""
    system = fresh(propulsion_name)
    before = system.limits()
    ctx = context_for(system)
    for _ in range(20):
        system.step(nominal_command(), ctx)
        ctx.t_s += ctx.dt_s
    after = system.limits()
    assert before == after


def test_bom_is_well_formed(propulsion) -> None:
    """The bill of materials is the only bridge to the cost model; a missing
    or negative quantity there turns into a nonsense dollar figure."""
    bom = propulsion.bom()
    assert isinstance(bom, BillOfMaterials)
    assert bom.system_name
    assert isinstance(bom.family, PropulsionFamily)
    assert bom.family == propulsion.family
    assert bom.thruster_units >= 1
    assert bom.propellant_type
    for field in (
        "rated_power_w",
        "power_source_w",
        "reactor_thermal_w",
        "radiator_area_m2",
        "dry_mass_kg",
        "tank_capacity_kg",
        "qualified_life_s",
    ):
        value = getattr(bom, field)
        assert math.isfinite(value), f"bom.{field} is {value}"
        assert value >= 0.0, f"bom.{field} is negative ({value})"
    assert bom.dry_mass_kg > 0.0, "a thruster with no mass cannot be launched"
    if bom.family is PropulsionFamily.NUCLEAR:
        assert bom.reactor_thermal_w > 0.0 or bom.power_source_w > 0.0, (
            "a nuclear system must report its reactor to the cost model"
        )


def test_housekeeping_and_decode_action_are_usable(propulsion) -> None:
    """The vehicle subtracts housekeeping power before offering the rest, and
    logs the decoded setpoints; both are called every step."""
    hk = propulsion.housekeeping_power_w()
    assert math.isfinite(hk) and hk >= 0.0
    decoded = propulsion.decode_action(nominal_command())
    assert isinstance(decoded, dict) and decoded
    assert all(isinstance(k, str) for k in decoded)
    assert all(math.isfinite(float(v)) for v in decoded.values())
    assert isinstance(propulsion.info(), dict)


# --- stepping ----------------------------------------------------------------
def test_step_runs_over_the_action_space_corners(propulsion_name) -> None:
    """Saturation logic lives at the corners of the action box. Every corner,
    plus hard off and hard on, must produce a finite, non-negative output."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for command in command_grid():
        out = system.step(command, ctx)
        assert isinstance(out, ThrusterOutput)
        _finite_output(out, f"command {command}")
        assert isinstance(out.throttled_by, str)
        ctx.t_s += ctx.dt_s


def test_step_never_raises_on_absurd_inputs(propulsion_name) -> None:
    """The base class forbids raising from ``step`` for physical reasons: a
    dead bus, a massless vehicle or a decade-long step must degrade the output
    and set health flags, never kill the rollout with an exception."""
    system = fresh(propulsion_name)
    limits = system.limits()
    absurd = [
        dict(available_power_w=0.0),
        dict(available_power_w=1e-9),
        dict(vehicle_mass_kg=0.0),
        dict(dt_s=1e9),
        dict(dt_s=1e-6),
        dict(heliocentric_radius_m=100.0 * AU, eclipse=True),
        dict(heliocentric_radius_m=0.01 * AU, sink_temperature_k=2000.0),
        dict(available_power_w=1e6 * max(limits.max_power_w, 1.0)),
    ]
    for over in absurd:
        ctx = context_for(system, **over)
        out = system.step(nominal_command(), ctx)
        _finite_output(out, f"ctx {over}")


def test_step_survives_a_drained_tank(propulsion_name) -> None:
    """Running dry is a normal end-of-mission state, not an error. After the
    propellant is gone the model must keep answering with zero-ish thrust."""
    system = fresh(propulsion_name)
    ctx = context_for(system, dt_s=1e7)
    for _ in range(40):
        system.step(nominal_command(), ctx)
        ctx.t_s += ctx.dt_s
    for _ in range(5):
        out = system.step(nominal_command(), ctx)
        _finite_output(out, "drained tank")
        ctx.t_s += ctx.dt_s
    for attr, value in float_attributes(system).items():
        if "propellant" in attr or attr.endswith("tank_kg"):
            assert value >= 0.0, f"{attr} went negative ({value})"
    assert system.health().throughput_kg >= 0.0


# --- conservation and consistency -------------------------------------------
def test_thrust_equals_mdot_times_exhaust_velocity(propulsion_name) -> None:
    """T = mdot * g0 * Isp is the definition of specific impulse, not an
    approximation. If the three reported numbers do not close, either the
    delta-v the environment integrates or the propellant it debits is wrong,
    and the mission economics inherit the error."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for command in command_grid():
        out = system.step(command, ctx)
        ctx.t_s += ctx.dt_s
        if out.mdot_kg_s == 0.0:
            assert out.thrust_n == 0.0, (
                "thrust with zero mass flow implies infinite Isp"
            )
            continue
        expected = out.mdot_kg_s * G0 * out.isp_s
        assert out.thrust_n == pytest.approx(expected, rel=RTOL, abs=1e-12), (
            f"T={out.thrust_n} but mdot*g0*Isp={expected} "
            f"(mdot={out.mdot_kg_s}, Isp={out.isp_s}, cmd={command})"
        )


def test_reported_isp_stays_inside_the_declared_envelope(propulsion_name) -> None:
    """``Limits.isp_range_s`` is used to normalise observations and to size the
    vehicle; an operating point outside it means one of the two is wrong."""
    system = fresh(propulsion_name)
    low, high = system.limits().isp_range_s
    ctx = context_for(system)
    for command in command_grid():
        out = system.step(command, ctx)
        ctx.t_s += ctx.dt_s
        if out.mdot_kg_s > 0.0 and out.thrust_n > 0.0:
            assert low * 0.95 <= out.isp_s <= high * 1.05, (
                f"Isp {out.isp_s} outside declared range {(low, high)} "
                f"(cmd={command})"
            )


def test_efficiency_is_a_fraction(propulsion_name) -> None:
    """Efficiency is a ratio of powers. Above one it is free energy; below zero
    it is a thruster that consumes its own exhaust."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for command in command_grid():
        out = system.step(command, ctx)
        ctx.t_s += ctx.dt_s
        assert 0.0 <= out.efficiency <= 1.0, f"efficiency {out.efficiency}"


def test_jet_power_never_exceeds_input_power(propulsion_name) -> None:
    """Energy bookkeeping. The kinetic power in the beam cannot exceed the
    electrical power drawn or the thermal power generated -- a model that
    breaks this is inventing energy, and every Isp/thrust trade it reports is
    optimistic by an unknown factor."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for command in command_grid():
        out = system.step(command, ctx)
        ctx.t_s += ctx.dt_s
        jet = out.jet_power_w
        supplied = max(out.power_draw_w, out.thermal_power_w)
        assert jet <= supplied * (1.0 + RTOL) + 1e-9, (
            f"jet power {jet:.6g} W exceeds input power {supplied:.6g} W "
            f"(draw={out.power_draw_w:.6g}, thermal={out.thermal_power_w:.6g}, "
            f"cmd={command})"
        )


def test_reported_efficiency_matches_the_power_ratio(propulsion_name) -> None:
    """``efficiency`` is documented as *total* thrust efficiency, so it must be
    the jet-power fraction of one of the two input powers the same step
    reported -- not an unrelated component figure. Whether the denominator is
    the electrical draw or the reactor thermal power is the model's choice."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for command in (nominal_command(1.0), nominal_command(0.6, 0.9)):
        out = system.step(command, ctx)
        ctx.t_s += ctx.dt_s
        if out.jet_power_w <= 0.0:
            continue
        candidates = [
            out.jet_power_w / p
            for p in (out.power_draw_w, out.thermal_power_w)
            if p > 0.0
        ]
        assert candidates, "produced a beam with no reported input power"
        assert any(
            out.efficiency == pytest.approx(ratio, rel=0.05) for ratio in candidates
        ), (
            f"efficiency {out.efficiency} matches neither jet/electric nor "
            f"jet/thermal {candidates} (cmd={command})"
        )


# --- power budget ------------------------------------------------------------
def test_power_draw_respects_the_bus_budget(propulsion_name) -> None:
    """Contract: over-drawing the bus is a modelling bug, not a constraint
    violation. Clamp and report it in ``throttled_by`` instead."""
    system = fresh(propulsion_name)
    if system.self_powered:
        pytest.skip("self-powered system: the bus budget does not bind it")
    full = system.limits().max_power_w
    for available in (full, 0.5 * full, 0.1 * full, 1.0, 0.0):
        probe = fresh(propulsion_name)
        ctx = context_for(probe, available_power_w=available)
        out = probe.step(nominal_command(), ctx)
        assert out.power_draw_w <= available * (1.0 + RTOL) + 1e-6, (
            f"drew {out.power_draw_w:.6g} W from a {available:.6g} W bus"
        )


def test_less_power_never_increases_thrust(propulsion_name) -> None:
    """Monotone response to the power budget. A sign error in the power-limited
    branch shows up as a thruster that gets stronger as the array degrades,
    which teaches the policy to fly into eclipse."""
    system = make_propulsion(propulsion_name)
    if system.self_powered:
        pytest.skip("self-powered system: thrust is not a function of bus power")
    full = system.limits().max_power_w
    fractions = [1.0, 0.75, 0.5, 0.25, 0.05, 0.0]
    thrusts = []
    for frac in fractions:
        probe = fresh(propulsion_name)
        ctx = context_for(probe, available_power_w=frac * full)
        thrusts.append(probe.step(nominal_command(), ctx).thrust_n)
    tol = 1e-9 * max(max(thrusts), 1e-9)
    for (f_hi, t_hi), (f_lo, t_lo) in zip(
        zip(fractions, thrusts), zip(fractions[1:], thrusts[1:])
    ):
        assert t_lo <= t_hi + tol, (
            f"thrust rose from {t_hi:.6g} N at {f_hi:.0%} power to "
            f"{t_lo:.6g} N at {f_lo:.0%} power"
        )


def test_more_throttle_never_gives_less_thrust(propulsion_name) -> None:
    """Monotonicity at a fixed operating point. Without it the throttle axis of
    the action space is not interpretable and a scripted baseline cannot be
    tuned; with a non-monotone map, 'more thrust' is not a reachable intent."""
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    thrusts = []
    for level in levels:
        probe = fresh(propulsion_name)
        ctx = context_for(probe)
        thrusts.append(probe.step(nominal_command(throttle=level), ctx).thrust_n)
    tol = 1e-9 * max(max(thrusts), 1e-9)
    for (lo, t_lo), (hi, t_hi) in zip(
        zip(levels, thrusts), zip(levels[1:], thrusts[1:])
    ):
        assert t_hi >= t_lo - tol, (
            f"thrust fell from {t_lo:.6g} N at throttle {lo} to "
            f"{t_hi:.6g} N at throttle {hi}"
        )
    assert thrusts[-1] > 0.0, "full throttle at full power produced no thrust"


def test_zero_throttle_means_off(propulsion_name) -> None:
    """Commanded off means off. Cathode keeper flow may continue, but a system
    that still produces thrust -- or burns anything like a firing propellant
    load -- makes every coast phase silently expensive."""
    off = fresh(propulsion_name)
    on = fresh(propulsion_name)
    off_out = off.step(nominal_command(throttle=0.0), context_for(off))
    on_out = on.step(nominal_command(throttle=1.0), context_for(on))
    assert off_out.thrust_n == pytest.approx(0.0, abs=1e-9)
    assert off_out.mdot_kg_s <= 0.1 * on_out.mdot_kg_s + 1e-12, (
        f"zero throttle still flows {off_out.mdot_kg_s:.6g} kg/s against "
        f"{on_out.mdot_kg_s:.6g} kg/s at full throttle"
    )


# --- observation and constraints --------------------------------------------
def test_observe_is_contract_width_finite_and_normalised(propulsion_name) -> None:
    """The propulsion block occupies a fixed 16-wide slice of the 36-wide
    observation. Wrong width breaks zero-shot transfer; un-normalised values
    swamp the other blocks and stall the policy's first layer."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for _ in range(5):
        obs = system.observe(ctx)
        assert obs.shape == (PROPULSION_OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all(), f"non-finite observation: {obs}"
        assert np.abs(obs).max() <= OBS_ABS_MAX, (
            f"observation is not normalised to ~[-1, 1]: {obs}"
        )
        system.step(nominal_command(), ctx)
        ctx.t_s += ctx.dt_s


def test_observation_labels_match_observe_raw(propulsion_name) -> None:
    """Interpretability depends on the labels lining up with the values; an
    off-by-one here mislabels every plot in the results section."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    raw = np.asarray(system.observe_raw(ctx)).reshape(-1)
    labels = system.observation_labels()
    assert isinstance(labels, tuple)
    assert len(labels) == raw.size, (
        f"{len(labels)} labels for {raw.size} raw observation values"
    )
    assert raw.size <= PROPULSION_OBS_DIM
    assert all(isinstance(name, str) and name for name in labels)
    assert len(set(labels)) == len(labels), "duplicate observation labels"


def test_constraints_are_well_formed(propulsion_name) -> None:
    """Constrained-RL agents consume ``cost`` directly; a ragged or non-finite
    report makes the Lagrange multiplier diverge on the first update."""
    system = fresh(propulsion_name)
    ctx = context_for(system)
    for _ in range(10):
        report = system.constraints()
        assert isinstance(report, ConstraintReport)
        margins = np.asarray(report.margins, dtype=float).reshape(-1)
        assert len(report.names) == margins.size, (
            f"{len(report.names)} constraint names for {margins.size} margins"
        )
        assert all(isinstance(name, str) and name for name in report.names)
        assert np.isfinite(margins).all(), f"non-finite margins: {margins}"
        assert report.cost >= 0.0
        assert math.isfinite(report.cost)
        system.step(nominal_command(), ctx)
        ctx.t_s += ctx.dt_s


# --- health ------------------------------------------------------------------
def test_wear_is_monotone_and_failure_latches(propulsion_name) -> None:
    """Wear is cumulative damage: it cannot heal, and a failed thruster cannot
    come back. A non-latching failure lets a policy learn to 'reset' hardware
    by idling, which is the single most rewarding physics bug available."""
    system = fresh(propulsion_name)
    ctx = context_for(system, dt_s=1e6)
    previous = system.health()
    assert isinstance(previous, HealthReport)
    seen_failed = previous.failed
    for step in range(100):
        system.step(nominal_command(), ctx)
        ctx.t_s += ctx.dt_s
        current = system.health()
        assert 0.0 <= current.wear_fraction <= 1.0, (
            f"wear_fraction {current.wear_fraction} at step {step}"
        )
        assert current.wear_fraction >= previous.wear_fraction - 1e-12, (
            f"wear healed from {previous.wear_fraction} to "
            f"{current.wear_fraction} at step {step}"
        )
        assert current.throughput_kg >= previous.throughput_kg - 1e-12
        assert current.burn_time_s >= previous.burn_time_s - 1e-9
        assert current.restarts >= previous.restarts
        assert current.remaining_life_s >= 0.0
        assert 0.0 <= current.degraded_efficiency <= 1.0 + 1e-9
        if seen_failed:
            assert current.failed, f"failure un-latched at step {step}"
        seen_failed = seen_failed or current.failed
        previous = current


def test_propellant_throughput_tracks_mass_flow(propulsion_name) -> None:
    """Throughput is the life metric for electric propulsion, so it must be the
    integral of the mass flow the model itself reported."""
    system = fresh(propulsion_name)
    ctx = context_for(system, dt_s=1e5)
    burned = 0.0
    start = system.health().throughput_kg
    for _ in range(20):
        out = system.step(nominal_command(), ctx)
        burned += out.mdot_kg_s * ctx.dt_s
        ctx.t_s += ctx.dt_s
    reported = system.health().throughput_kg - start
    if burned > 0.0:
        assert reported == pytest.approx(burned, rel=0.02), (
            f"health reports {reported:.6g} kg processed, mass flow integrates "
            f"to {burned:.6g} kg"
        )


# --- determinism and purity --------------------------------------------------
@pytest.mark.slow
def test_two_identically_seeded_instances_match_bit_for_bit(propulsion_name) -> None:
    """Reproducibility is load-bearing for the whole comparison: seeds are the
    only control the sweep has over unit-to-unit variation, so two runs of one
    seed must be identical to the last bit, not merely close."""
    commands = [
        nominal_command(
            throttle=float(t), operating_point=float(o), thermal_margin=float(m)
        )
        for t, o, m in np.random.default_rng(7).uniform(0.0, 1.0, size=(100, 3))
    ]

    def trajectory() -> list[tuple]:
        system = make_propulsion(propulsion_name)
        system.reset(np.random.default_rng(4242))
        ctx = context_for(system, rng=np.random.default_rng(99))
        rows = []
        for command in commands:
            out = system.step(command, ctx)
            rows.append((output_signature(out), tuple(system.observe(ctx).tolist())))
            ctx.t_s += ctx.dt_s
        return rows

    first, second = trajectory(), trajectory()
    for index, (a, b) in enumerate(zip(first, second)):
        assert a == b, f"trajectories diverged at step {index}: {a} != {b}"


def test_probe_methods_do_not_mutate_state(propulsion_name) -> None:
    """``observe``, ``constraints``, ``health`` and ``bom`` are contractually
    pure reads. The environment and the logger call them a different number of
    times depending on configuration, so any hidden mutation makes an episode's
    physics depend on whether telemetry was switched on."""
    commands = [
        nominal_command(throttle=float(t), operating_point=float(o))
        for t, o in np.random.default_rng(11).uniform(0.0, 1.0, size=(25, 2))
    ]

    def trajectory(probe: bool) -> list[tuple]:
        system = make_propulsion(propulsion_name)
        system.reset(np.random.default_rng(2024))
        ctx = context_for(system, rng=np.random.default_rng(5))
        rows = []
        for command in commands:
            if probe:
                for _ in range(2):
                    system.observe(ctx)
                    system.constraints()
                    system.health()
                    system.bom()
                    system.observation_labels()
                    system.limits()
            out = system.step(command, ctx)
            if probe:
                for _ in range(2):
                    system.observe(ctx)
                    system.constraints()
                    system.health()
                    system.bom()
            rows.append(output_signature(out))
            ctx.t_s += ctx.dt_s
        return rows

    reference = trajectory(probe=False)
    probed = trajectory(probe=True)
    for index, (a, b) in enumerate(zip(reference, probed)):
        assert a == b, (
            f"calling observe/constraints/health/bom changed the trajectory at "
            f"step {index}: {a} != {b}"
        )


def test_reset_restores_the_start_of_life_state(propulsion_name) -> None:
    """``reset`` must fully rewind: a second episode on the same object has to
    behave like the first, or every sweep silently trains on worn hardware."""
    system = fresh(propulsion_name)
    ctx = context_for(system, dt_s=1e6, rng=np.random.default_rng(3))
    first = [output_signature(system.step(nominal_command(), ctx)) for _ in range(20)]
    system.reset(np.random.default_rng(12345))
    ctx = context_for(system, dt_s=1e6, rng=np.random.default_rng(3))
    second = [output_signature(system.step(nominal_command(), ctx)) for _ in range(20)]
    assert first == second, "state leaked across reset()"
