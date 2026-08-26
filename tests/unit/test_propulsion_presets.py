"""Smoke checks for the named propulsion presets promised by the benchmark.

The conformance suite owns detailed conservation and lifecycle invariants.  The
checks here guard the integration seam and use deliberately broad heritage-
hardware bands, so a calibration improvement does not turn into a brittle test.
"""

from __future__ import annotations

import pytest

import propulsion_rl.propulsion as propulsion_package
from propulsion_rl.core.registry import PROPULSION
from propulsion_rl.core.types import PropulsionFamily
from propulsion_rl.propulsion.base import PropulsionSystem


ELECTRIC_PRESETS = (
    "hall_spt100",
    "hall_hermes",
    "ion_nstar",
    "ion_next",
)
NUCLEAR_PRESETS = (
    "ntp_pewee",
    "ntp_nerva",
    "nep_brayton",
    "nep_kilopower",
)
EXPECTED_PRESETS = ELECTRIC_PRESETS + NUCLEAR_PRESETS


def test_importing_parent_package_registers_all_documented_presets() -> None:
    """The root package imports only ``propulsion`` to populate the registry."""

    missing = sorted(set(EXPECTED_PRESETS) - set(PROPULSION.names()))
    assert not missing, f"documented propulsion presets were not registered: {missing}"


@pytest.mark.parametrize(
    ("name", "family"),
    [
        *((name, PropulsionFamily.ELECTRIC) for name in ELECTRIC_PRESETS),
        *((name, PropulsionFamily.NUCLEAR) for name in NUCLEAR_PRESETS),
    ],
)
def test_preset_constructs_by_name_and_is_public(name: str, family: PropulsionFamily) -> None:
    system = PROPULSION.make(name)

    assert isinstance(system, PropulsionSystem)
    assert system.name == name
    assert system.family is family
    assert system.self_powered is (family is PropulsionFamily.NUCLEAR)

    # A registry factory may return a configured generic class or a named preset
    # subclass.  In either case callers must be able to import that concrete
    # class from ``propulsion_rl.propulsion``.
    concrete = type(system)
    assert any(
        getattr(propulsion_package, public_name, None) is concrete
        for public_name in propulsion_package.__all__
    ), f"{concrete.__name__} is registered but not public at package level"

    meta_family = PROPULSION.meta(name).get("family")
    assert getattr(meta_family, "value", meta_family) == family.value


@pytest.mark.parametrize(
    ("name", "thrust_band_n", "power_band_w", "isp_band_s"),
    [
        ("hall_spt100", (0.04, 0.20), (0.8e3, 3.0e3), (1_000.0, 2_500.0)),
        ("hall_hermes", (0.20, 2.0), (5.0e3, 40.0e3), (1_500.0, 5_000.0)),
        ("ion_nstar", (0.03, 0.25), (1.0e3, 6.0e3), (2_000.0, 6_000.0)),
        ("ion_next", (0.10, 1.0), (3.0e3, 25.0e3), (2_500.0, 8_000.0)),
    ],
)
def test_electric_presets_stay_in_broad_flight_hardware_bands(
    name: str,
    thrust_band_n: tuple[float, float],
    power_band_w: tuple[float, float],
    isp_band_s: tuple[float, float],
) -> None:
    system = PROPULSION.make(name)
    limits = system.limits()
    bom = system.bom()
    isp_midpoint = 0.5 * sum(limits.isp_range_s)

    assert thrust_band_n[0] <= limits.max_thrust_n <= thrust_band_n[1]
    assert power_band_w[0] <= limits.max_power_w <= power_band_w[1]
    assert isp_band_s[0] <= isp_midpoint <= isp_band_s[1]
    assert bom.family is PropulsionFamily.ELECTRIC
    assert bom.reactor_thermal_w == 0.0
    assert bom.rated_power_w > 0.0


@pytest.mark.parametrize("name", ("ntp_pewee", "ntp_nerva"))
def test_nuclear_thermal_presets_resemble_ground_test_scale(name: str) -> None:
    system = PROPULSION.make(name)
    limits = system.limits()
    bom = system.bom()
    isp_midpoint = 0.5 * sum(limits.isp_range_s)

    # Pewee/NERVA-class tests span hundreds of MW and tens to hundreds of kN;
    # the bands intentionally include modern derivatives around that scale.
    assert 20_000.0 <= limits.max_thrust_n <= 2.0e6
    assert 600.0 <= isp_midpoint <= 1_200.0
    assert 0.1e9 <= bom.reactor_thermal_w <= 3.0e9
    assert bom.family is PropulsionFamily.NUCLEAR
    assert bom.propellant_type.lower() in {"hydrogen", "h2", "lh2", "liquid_hydrogen"}


@pytest.mark.parametrize(
    ("name", "power_band_w", "thrust_band_n"),
    [
        ("nep_brayton", (0.1e6, 100.0e6), (0.05, 2_000.0)),
        ("nep_kilopower", (1.0e3, 100.0e3), (0.001, 20.0)),
    ],
)
def test_nuclear_electric_presets_have_plausible_power_and_thrust_scale(
    name: str,
    power_band_w: tuple[float, float],
    thrust_band_n: tuple[float, float],
) -> None:
    system = PROPULSION.make(name)
    limits = system.limits()
    bom = system.bom()
    isp_midpoint = 0.5 * sum(limits.isp_range_s)

    assert power_band_w[0] <= bom.power_source_w <= power_band_w[1]
    assert thrust_band_n[0] <= limits.max_thrust_n <= thrust_band_n[1]
    assert 1_000.0 <= isp_midpoint <= 20_000.0
    assert bom.family is PropulsionFamily.NUCLEAR
    assert bom.reactor_thermal_w > 0.0

