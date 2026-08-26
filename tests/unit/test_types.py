"""Contract tests for the shared data types.

These defend the vocabulary every other module speaks. They import their
subjects directly and carry no skips: if any of them fails, policies trained
against one propulsion system can no longer be evaluated against another,
which is the entire premise of the benchmark.
"""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from propulsion_rl.core.constants import G0
from propulsion_rl.core.types import (
    CANONICAL_ACTION_DIM,
    MISSION_OBS_DIM,
    OBS_DIM,
    PROPULSION_OBS_DIM,
    VEHICLE_OBS_DIM,
    CanonicalCommand,
    ConstraintReport,
    Event,
    Severity,
    Telemetry,
    ThrusterOutput,
    VehicleState,
    pad_to,
    safe_div,
    zeros_obs_block,
)
from propulsion_rl.missions.base import RewardTerms


# --- interface widths --------------------------------------------------------
def test_obs_dim_is_the_sum_of_its_blocks() -> None:
    """The 36-wide observation is exactly the three contracted blocks.

    Zero-shot transfer between propulsion systems depends on the blocks being
    concatenated, never re-ordered or re-sized independently.
    """
    assert OBS_DIM == MISSION_OBS_DIM + VEHICLE_OBS_DIM + PROPULSION_OBS_DIM
    assert OBS_DIM == 36
    assert CANONICAL_ACTION_DIM == 5


# --- canonical action mapping ------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_from_array_to_array_round_trip(seed: int) -> None:
    """The action encoding is invertible, so scripted controllers and learned
    policies address exactly the same command space."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=CANONICAL_ACTION_DIM)
    round_tripped = CanonicalCommand.from_array(a).to_array()
    assert round_tripped.dtype == np.float32
    assert round_tripped == pytest.approx(a, abs=1e-6)


def test_to_array_from_array_round_trip() -> None:
    """The inverse direction also closes, in physical units."""
    cmd = CanonicalCommand(
        throttle=0.25,
        operating_point=0.75,
        thrust_yaw=1.1,
        thrust_pitch=-0.4,
        thermal_margin=0.9,
    )
    back = CanonicalCommand.from_array(cmd.to_array())
    assert back.throttle == pytest.approx(cmd.throttle, abs=1e-6)
    assert back.operating_point == pytest.approx(cmd.operating_point, abs=1e-6)
    assert back.thrust_yaw == pytest.approx(cmd.thrust_yaw, abs=1e-6)
    assert back.thrust_pitch == pytest.approx(cmd.thrust_pitch, abs=1e-6)
    assert back.thermal_margin == pytest.approx(cmd.thermal_margin, abs=1e-6)


def test_from_array_clips_instead_of_raising() -> None:
    """An unsquashed Gaussian policy head saturates rather than killing the
    rollout: out-of-range components clip to the envelope edges."""
    cmd = CanonicalCommand.from_array(np.array([5.0, -3.0, 12.0, -9.0, 2.0]))
    assert cmd.throttle == 1.0
    assert cmd.operating_point == 0.0
    assert cmd.thrust_yaw == pytest.approx(np.pi)
    assert cmd.thrust_pitch == pytest.approx(-np.pi / 2)
    assert cmd.thermal_margin == 1.0


def test_from_array_clipping_keeps_every_field_in_range() -> None:
    """No policy output, however wild, can produce an out-of-envelope command."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        cmd = CanonicalCommand.from_array(rng.normal(0.0, 50.0, CANONICAL_ACTION_DIM))
        assert 0.0 <= cmd.throttle <= 1.0
        assert 0.0 <= cmd.operating_point <= 1.0
        assert -np.pi <= cmd.thrust_yaw <= np.pi
        assert -np.pi / 2 <= cmd.thrust_pitch <= np.pi / 2
        assert 0.0 <= cmd.thermal_margin <= 1.0


@pytest.mark.parametrize("size", [0, 1, 4, 6, 10])
def test_from_array_rejects_wrong_width(size: int) -> None:
    """A width mismatch is a wiring bug, not something to silently pad."""
    with pytest.raises(ValueError, match=str(CANONICAL_ACTION_DIM)):
        CanonicalCommand.from_array(np.zeros(size))


def test_from_array_accepts_any_shape_with_the_right_count() -> None:
    """Column vectors from a torch head with a batch axis of 1 still decode."""
    cmd = CanonicalCommand.from_array(np.zeros((1, CANONICAL_ACTION_DIM)))
    assert cmd.throttle == 0.5


def test_zero_action_is_half_throttle_prograde() -> None:
    """A freshly initialised policy emitting zeros must produce a sane command.

    Zero maps to the centre of every bounded knob and to a pure prograde burn,
    so an untrained agent gains orbital energy instead of thrashing.
    """
    cmd = CanonicalCommand.from_array(np.zeros(CANONICAL_ACTION_DIM))
    assert cmd.throttle == pytest.approx(0.5)
    assert cmd.operating_point == pytest.approx(0.5)
    assert cmd.thrust_yaw == pytest.approx(0.0)
    assert cmd.thrust_pitch == pytest.approx(0.0)
    assert cmd.thermal_margin == pytest.approx(0.5)
    assert cmd.direction_rtn() == pytest.approx([0.0, 1.0, 0.0])


def test_direction_rtn_is_unit_norm_over_the_whole_domain() -> None:
    """Thrust direction is a pure rotation: the environment multiplies it by a
    thrust magnitude, so any norm drift silently rescales the applied force."""
    for yaw in np.linspace(-np.pi, np.pi, 37):
        for pitch in np.linspace(-np.pi / 2, np.pi / 2, 19):
            cmd = CanonicalCommand(0.5, 0.5, float(yaw), float(pitch), 0.5)
            d = cmd.direction_rtn()
            assert d.shape == (3,)
            assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-12)


def test_direction_rtn_sign_conventions() -> None:
    """Yaw is measured from transverse towards radial, pitch towards the normal.

    Getting these axes crossed turns an orbit-raising burn into a plane change
    without any test noticing, so the conventions are pinned explicitly.
    """
    radial = CanonicalCommand(0.5, 0.5, np.pi / 2, 0.0, 0.5).direction_rtn()
    assert radial == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    retro = CanonicalCommand(0.5, 0.5, np.pi, 0.0, 0.5).direction_rtn()
    assert retro == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    normal = CanonicalCommand(0.5, 0.5, 0.0, np.pi / 2, 0.5).direction_rtn()
    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


# --- observation block padding -----------------------------------------------
def test_pad_to_right_pads_short_blocks() -> None:
    """Short blocks are zero-padded on the right so a system that reports fewer
    channels still lands its values at fixed observation indices."""
    out = pad_to(np.array([1.0, 2.0, 3.0]), 6, "block")
    assert out.shape == (6,)
    assert out.dtype == np.float32
    assert out == pytest.approx([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


def test_pad_to_passes_exact_width_through() -> None:
    v = pad_to(np.arange(4.0), 4)
    assert v == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert v.dtype == np.float32


def test_pad_to_rejects_over_wide_blocks() -> None:
    """Truncating would silently drop channels and shift every later index, so
    an over-wide block is an error naming the offending producer."""
    with pytest.raises(ValueError, match="thruster"):
        pad_to(np.zeros(9), 8, "thruster obs")


def test_zeros_obs_block_dtype() -> None:
    z = zeros_obs_block(PROPULSION_OBS_DIM)
    assert z.shape == (PROPULSION_OBS_DIM,)
    assert z.dtype == np.float32
    assert not z.any()


# --- constraint accounting ---------------------------------------------------
def test_constraint_report_empty_case() -> None:
    """A system with no declared constraints costs nothing and is never in
    violation; ``worst`` is +inf so a min over reports still works."""
    rep = ConstraintReport()
    assert rep.violated is False
    assert rep.cost == 0.0
    assert rep.worst == np.inf


def test_constraint_report_all_satisfied() -> None:
    rep = ConstraintReport(("a", "b"), np.array([0.5, 0.0]))
    assert rep.violated is False
    assert rep.cost == 0.0
    assert rep.worst == pytest.approx(0.0)


def test_constraint_report_cost_sums_only_violations() -> None:
    """Cost is the summed depth of violation; satisfied margins never earn a
    credit that could cancel a real breach."""
    rep = ConstraintReport(("a", "b", "c"), np.array([2.0, -0.25, -0.75]))
    assert rep.violated is True
    assert rep.cost == pytest.approx(1.0)
    assert rep.worst == pytest.approx(-0.75)


def test_constraint_report_cost_is_never_negative() -> None:
    rng = np.random.default_rng(3)
    for _ in range(100):
        rep = ConstraintReport(("x",) * 4, rng.normal(size=4))
        assert rep.cost >= 0.0
        assert rep.violated == bool(rep.cost > 0.0)


# --- vehicle state -----------------------------------------------------------
def test_total_mass_is_dry_plus_propellant_plus_payload() -> None:
    """Mass bookkeeping is the backbone of the Tsiolkovsky accounting; a
    forgotten payload term would inflate every reported delta-v."""
    s = VehicleState(
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        dry_mass_kg=800.0,
        propellant_kg=400.0,
        payload_kg=300.0,
    )
    assert s.total_mass_kg == pytest.approx(1500.0)
    s.propellant_kg -= 100.0
    assert s.total_mass_kg == pytest.approx(1400.0)


def test_vehicle_state_derived_geometry() -> None:
    s = VehicleState(
        position_m=np.array([3.0, 4.0, 0.0]),
        velocity_m_s=np.array([0.0, 6.0, 8.0]),
        dry_mass_kg=1.0,
        propellant_kg=0.0,
    )
    assert s.radius_m == pytest.approx(5.0)
    assert s.speed_m_s == pytest.approx(10.0)
    assert s.specific_energy == pytest.approx(50.0)


def test_vehicle_state_copy_is_deep_in_the_arrays() -> None:
    """The environment keeps a previous-state copy for reward shaping; aliasing
    the arrays would make every delta identically zero."""
    s = VehicleState(np.zeros(3), np.zeros(3), 1.0, 1.0)
    c = s.copy()
    c.position_m[0] = 99.0
    c.velocity_m_s[1] = -3.0
    c.propellant_kg = 0.0
    assert s.position_m[0] == 0.0
    assert s.velocity_m_s[1] == 0.0
    assert s.propellant_kg == 1.0


# --- thruster output ---------------------------------------------------------
def test_jet_power_matches_the_half_mdot_ve_squared_definition() -> None:
    """Jet power is the yardstick the efficiency conformance test measures
    against, so its definition is pinned here."""
    out = ThrusterOutput(thrust_n=0.1, mdot_kg_s=5e-6, isp_s=2000.0)
    assert out.jet_power_w == pytest.approx(0.5 * 5e-6 * (2000.0 * G0) ** 2)


def test_jet_power_is_zero_when_not_firing() -> None:
    assert ThrusterOutput().jet_power_w == 0.0


# --- reward decomposition ----------------------------------------------------
def test_reward_total_is_the_sum_of_its_parts() -> None:
    """Ablation analysis attributes behaviour to individual terms; ``total``
    must therefore be exactly their sum, with no hidden weighting."""
    terms = RewardTerms(
        progress=1.5,
        efficiency=0.25,
        time_penalty=-0.1,
        propellant_penalty=-0.3,
        wear_penalty=-0.05,
        constraint_penalty=-2.0,
        terminal=10.0,
    )
    assert terms.total == pytest.approx(1.5 + 0.25 - 0.1 - 0.3 - 0.05 - 2.0 + 10.0)


def test_reward_terms_default_to_zero() -> None:
    assert RewardTerms().total == 0.0


def test_reward_as_dict_carries_every_part_and_the_total() -> None:
    terms = RewardTerms(progress=2.0, terminal=-1.0)
    d = terms.as_dict()
    assert d["total"] == pytest.approx(terms.total)
    assert set(d) == {
        "progress",
        "efficiency",
        "time_penalty",
        "propellant_penalty",
        "wear_penalty",
        "constraint_penalty",
        "terminal",
        "total",
    }
    assert sum(v for k, v in d.items() if k != "total") == pytest.approx(d["total"])


# --- telemetry ---------------------------------------------------------------
def test_telemetry_row_is_flat_and_includes_extras() -> None:
    """The analysis layer pivots a list of these into a DataFrame, so every
    field must appear as a top-level column."""
    row = Telemetry(t_s=1.0, step=2, thrust_n=0.3, extras={"beam_v": 300.0}).as_row()
    assert row["t_s"] == 1.0
    assert row["step"] == 2
    assert row["thrust_n"] == 0.3
    assert row["beam_v"] == 300.0
    assert "extras" not in row
    assert all(not isinstance(v, dict) for v in row.values())


# --- small helpers -----------------------------------------------------------
def test_safe_div_guards_division_by_zero() -> None:
    """Rate calculations run every step with occasionally-zero denominators;
    the guard returns the default instead of an inf that poisons the reward."""
    assert safe_div(1.0, 4.0) == pytest.approx(0.25)
    assert safe_div(1.0, 0.0) == 0.0
    assert safe_div(1.0, 0.0, default=-1.0) == -1.0
    assert safe_div(1.0, -1e-18, default=7.0) == 7.0
    assert np.isfinite(safe_div(1.0, 1e-18, default=0.0))


def test_events_are_hashable_and_carry_severity() -> None:
    """Events are frozen so a log entry cannot be edited after the fact."""
    e = Event("overtemp", Severity.CRITICAL, "chamber", 2800.0)
    assert {e, Event("overtemp", Severity.CRITICAL, "chamber", 2800.0)} == {e}
    assert Severity.FATAL.value == "fatal"
    with pytest.raises(FrozenInstanceError):
        e.value = 1.0  # type: ignore[misc]


def test_severity_and_termination_reason_are_string_enums() -> None:
    """YAML configs and CSV logs round-trip these by value, so they must
    compare equal to their plain strings."""
    from propulsion_rl.core.types import PropulsionFamily, TerminationReason

    assert Severity.WARNING == "warning"
    assert TerminationReason.RUNNING == "running"
    assert PropulsionFamily.NUCLEAR == "nuclear"
    families = {f.value for f in PropulsionFamily}
    assert families == {"electric", "nuclear"}
    assert len({r.value for r in TerminationReason}) == len(list(TerminationReason))


def test_action_corners_all_decode() -> None:
    """Every corner of the action box decodes to a finite, in-range command."""
    for corner in itertools.product((-1.0, 1.0), repeat=CANONICAL_ACTION_DIM):
        cmd = CanonicalCommand.from_array(np.array(corner))
        assert np.isfinite(cmd.direction_rtn()).all()
        assert np.isfinite(cmd.to_array()).all()
