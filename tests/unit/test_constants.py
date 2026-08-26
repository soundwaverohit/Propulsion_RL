"""Physical sanity checks on the constants table.

Every number here is either a measured physical constant or a derived quantity
that must reproduce a textbook value. A typo in an exponent or a transposed
digit in MU_EARTH would otherwise propagate silently into every trajectory,
every delta-v and every cost-per-kilogram in the benchmark. These tests
re-derive the well-known answers from the stored constants.
"""

from __future__ import annotations

import math

import pytest

from propulsion_rl.core import constants as C
from propulsion_rl.core.constants import (
    AU,
    G0,
    GEO_RADIUS,
    HOUR,
    LEO_RADIUS,
    MARS_SMA,
    MU_EARTH,
    MU_SUN,
    SOLAR_CONSTANT_1AU,
    isp_to_ve,
    solar_flux,
    ve_to_isp,
)


def circular_speed(mu: float, radius: float) -> float:
    """Speed on a circular orbit of the given radius -- the classic sqrt(mu/r)."""
    return math.sqrt(mu / radius)


# --- solar flux --------------------------------------------------------------
@pytest.mark.physics
def test_solar_flux_at_one_au_is_the_solar_constant() -> None:
    """The inverse-square law must be anchored at 1 AU, or every solar-electric
    power budget in the study is scaled by a constant factor."""
    assert solar_flux(AU) == pytest.approx(SOLAR_CONSTANT_1AU, rel=1e-12)


@pytest.mark.physics
def test_solar_flux_at_mars_is_about_43_percent_of_earths() -> None:
    """Published figure: ~586 W/m^2 at Mars' mean distance, 43% of Earth's.

    This is the single number that decides whether solar-electric propulsion is
    viable beyond 1.5 AU, so it is checked against the literature value.
    """
    ratio = solar_flux(MARS_SMA) / solar_flux(AU)
    assert ratio == pytest.approx(0.431, abs=0.005)
    assert solar_flux(MARS_SMA) == pytest.approx(586.0, rel=0.02)


def test_solar_flux_follows_an_inverse_square() -> None:
    assert solar_flux(2 * AU) == pytest.approx(solar_flux(AU) / 4.0, rel=1e-12)
    assert solar_flux(0.5 * AU) == pytest.approx(solar_flux(AU) * 4.0, rel=1e-12)


def test_solar_flux_is_monotone_decreasing_and_finite_at_zero() -> None:
    """The zero-distance guard keeps a diverged trajectory from producing an
    inf power budget that then contaminates the whole episode with NaNs."""
    distances = [1e-3, 1e6, 0.1 * AU, AU, 10 * AU, 100 * AU]
    fluxes = [solar_flux(d) for d in distances]
    assert all(math.isfinite(f) for f in fluxes)
    assert all(a > b for a, b in zip(fluxes, fluxes[1:]))
    assert math.isfinite(solar_flux(0.0))


# --- specific impulse --------------------------------------------------------
def test_isp_and_exhaust_velocity_round_trip() -> None:
    """Isp and effective exhaust velocity are the same quantity in different
    units; the pair of converters must be exact inverses."""
    for isp in (300.0, 1600.0, 3000.0, 5000.0, 9000.0):
        assert ve_to_isp(isp_to_ve(isp)) == pytest.approx(isp, rel=1e-12)
    for ve in (3_000.0, 30_000.0, 50_000.0):
        assert isp_to_ve(ve_to_isp(ve)) == pytest.approx(ve, rel=1e-12)


@pytest.mark.physics
def test_isp_conversion_matches_known_hardware() -> None:
    """NSTAR at 3120 s exhausts at ~30.6 km/s; NERVA at 850 s at ~8.3 km/s."""
    assert isp_to_ve(3120.0) == pytest.approx(30_600.0, rel=0.01)
    assert isp_to_ve(850.0) == pytest.approx(8_336.0, rel=0.01)
    assert G0 == pytest.approx(9.80665, rel=1e-12)


# --- astrodynamics -----------------------------------------------------------
@pytest.mark.physics
def test_circular_velocity_at_leo_is_7_67_km_s() -> None:
    """A 400 km circular orbit runs at 7.67 km/s. Wrong by a factor of ten and
    every orbit-raising mission becomes trivially easy or impossible."""
    v = circular_speed(MU_EARTH, LEO_RADIUS)
    assert v == pytest.approx(7_672.6, rel=1e-3)
    assert v / 1000.0 == pytest.approx(7.67, abs=0.01)


@pytest.mark.physics
def test_circular_velocity_at_geo_is_3_07_km_s() -> None:
    v = circular_speed(MU_EARTH, GEO_RADIUS)
    assert v == pytest.approx(3_074.7, rel=1e-3)
    assert v / 1000.0 == pytest.approx(3.07, abs=0.01)


@pytest.mark.physics
def test_geo_radius_matches_a_sidereal_day_period() -> None:
    """GEO is defined by the sidereal day, so the stored radius and MU_EARTH
    must agree with 86164 s to better than a part in a thousand."""
    period = 2 * math.pi * math.sqrt(GEO_RADIUS**3 / MU_EARTH)
    assert period == pytest.approx(86_164.1, rel=1e-3)


@pytest.mark.physics
def test_earth_orbital_velocity_is_29_78_km_s() -> None:
    """Derived from MU_SUN and AU: the textbook 29.78 km/s. This cross-checks
    two independent constants against one another."""
    v = circular_speed(MU_SUN, AU)
    assert v == pytest.approx(29_784.0, rel=1e-3)
    assert v / 1000.0 == pytest.approx(29.78, abs=0.02)


@pytest.mark.physics
def test_earth_year_follows_from_kepler_third_law() -> None:
    """MU_SUN, EARTH_SMA and EARTH_ORBIT_PERIOD are three constants bound by
    one law; storing all three means they can disagree, so check they do not."""
    period = 2 * math.pi * math.sqrt(C.EARTH_SMA**3 / MU_SUN)
    assert period == pytest.approx(C.EARTH_ORBIT_PERIOD, rel=1e-3)


@pytest.mark.physics
def test_mars_year_follows_from_kepler_third_law() -> None:
    """Mars' 687-day year must fall out of MARS_SMA and MU_SUN."""
    period = 2 * math.pi * math.sqrt(MARS_SMA**3 / MU_SUN)
    assert period == pytest.approx(C.MARS_ORBIT_PERIOD, rel=1e-3)
    assert period / C.DAY == pytest.approx(687.0, rel=1e-3)


def test_body_radii_and_orbit_radii_are_ordered() -> None:
    """Sanity ordering that catches a swapped assignment: LEO is above the
    Earth's surface, GEO is above LEO, and Mars is the smaller planet."""
    assert C.R_MARS < C.R_EARTH < LEO_RADIUS < GEO_RADIUS
    assert LEO_RADIUS == pytest.approx(C.R_EARTH + 400e3)
    assert C.MU_MARS < MU_EARTH < MU_SUN
    assert MU_SUN / MU_EARTH == pytest.approx(332_946.0, rel=1e-3)
    assert AU < MARS_SMA < 2 * AU


def test_time_units() -> None:
    assert C.HOUR == 3600.0
    assert C.DAY == 24 * C.HOUR
    assert C.YEAR / C.DAY == pytest.approx(365.25)


# --- propellants -------------------------------------------------------------
@pytest.mark.physics
def test_ion_masses_are_the_atomic_weights_times_one_amu() -> None:
    """Beam current and thrust both scale with ion mass; an error here would
    show up as an efficiency that quietly exceeds unity."""
    assert C.M_XENON / C.AMU == pytest.approx(131.293, rel=1e-6)
    assert C.M_KRYPTON / C.AMU == pytest.approx(83.798, rel=1e-6)
    assert C.M_ARGON / C.AMU == pytest.approx(39.948, rel=1e-6)
    assert C.M_IODINE / C.AMU == pytest.approx(126.904, rel=1e-6)
    assert C.M_ARGON < C.M_KRYPTON < C.M_IODINE < C.M_XENON


@pytest.mark.physics
def test_ionization_energies_follow_the_periodic_trend() -> None:
    """Iodine ionises most easily, argon hardest; that ordering is why xenon
    and iodine are the propellants of choice."""
    assert C.IONIZATION_EV_IODINE < C.IONIZATION_EV_XENON
    assert C.IONIZATION_EV_XENON < C.IONIZATION_EV_KRYPTON
    assert C.IONIZATION_EV_KRYPTON < C.IONIZATION_EV_ARGON
    assert all(
        5.0 < ev < 25.0
        for ev in (
            C.IONIZATION_EV_XENON,
            C.IONIZATION_EV_KRYPTON,
            C.IONIZATION_EV_ARGON,
            C.IONIZATION_EV_IODINE,
        )
    )


@pytest.mark.physics
def test_hydrogen_properties_are_physical() -> None:
    """Molar mass of H2 is 2.016 g/mol and gamma sits below the 5/3 monatomic
    limit -- the two numbers that set nuclear thermal Isp."""
    assert C.M_MOLAR_H2 == pytest.approx(2.016e-3, rel=1e-3)
    assert 1.0 < C.GAMMA_H2 < 5.0 / 3.0
    assert 60.0 < C.H2_DENSITY_LIQUID < 80.0


# --- nuclear -----------------------------------------------------------------
@pytest.mark.physics
def test_fission_energy_is_about_200_mev() -> None:
    """3.204e-11 J per fission is 200 MeV; this ties the nuclear power model to
    the elementary charge stored in the same table."""
    mev = C.U235_FISSION_ENERGY_J / C.ELEMENTARY_CHARGE / 1e6
    assert mev == pytest.approx(200.0, rel=1e-3)


@pytest.mark.physics
def test_delayed_neutron_groups_sum_to_beta_eff() -> None:
    """Reactor kinetics is governed by the total delayed fraction. If the six
    group betas do not add up to BETA_EFF the point-kinetics model and the
    reactivity limits disagree, and prompt-critical excursions go unflagged.
    """
    total = sum(C.DELAYED_BETA)
    assert total == pytest.approx(C.BETA_EFF, rel=1e-2)
    assert C.BETA_EFF == pytest.approx(0.0065, rel=1e-9)


def test_delayed_neutron_data_is_well_formed() -> None:
    """Six groups, ordered from the long-lived precursor to the short-lived one,
    with strictly positive constants."""
    assert len(C.DELAYED_BETA) == 6
    assert len(C.DELAYED_LAMBDA) == 6
    assert all(b > 0.0 for b in C.DELAYED_BETA)
    assert all(lam > 0.0 for lam in C.DELAYED_LAMBDA)
    assert list(C.DELAYED_LAMBDA) == sorted(C.DELAYED_LAMBDA)
    assert 0.0 < C.NEUTRON_GEN_TIME < 1e-2
    assert 0.0 < C.DECAY_HEAT_FRACTION < 0.15


@pytest.mark.physics
def test_xenon_135_half_life_backs_out_to_9_14_hours() -> None:
    """Xe-135 poisoning drives the restart dead-band after a shutdown; the
    decay constant must reproduce the published 9.14 h half-life."""
    half_life_h = math.log(2) / C.XE135_DECAY_CONST / HOUR
    assert half_life_h == pytest.approx(9.14, rel=1e-9)


@pytest.mark.physics
def test_iodine_135_half_life_backs_out_to_6_57_hours() -> None:
    """I-135 feeds the Xe-135 chain; its 6.57 h half-life sets the delay before
    the poison peak."""
    half_life_h = math.log(2) / C.I135_DECAY_CONST / HOUR
    assert half_life_h == pytest.approx(6.57, rel=1e-9)


@pytest.mark.physics
def test_iodine_decays_faster_than_xenon_and_yields_are_plausible() -> None:
    """The precursor must be shorter-lived than its daughter for the classic
    post-shutdown xenon peak to exist at all."""
    assert C.I135_DECAY_CONST > C.XE135_DECAY_CONST
    assert 0.05 < C.I135_YIELD < 0.07
    assert 0.05 < C.XE135_YIELD < 0.07
    assert C.XE135_SIGMA_A > 1e-19  # ~2.6 Mbarn, the largest in reactor physics


# --- numerical hygiene -------------------------------------------------------
def test_tolerances_are_small_and_positive() -> None:
    assert 0.0 < C.EPS < 1e-9
    assert 0.0 < C.TINY_MASS_KG < 1e-3


def test_universal_constants_match_si_definitions() -> None:
    """These are exact SI-defined values since 2019; a rounded copy here would
    silently disagree with any external cross-check."""
    assert C.BOLTZMANN == 1.380649e-23
    assert C.ELEMENTARY_CHARGE == 1.602176634e-19
    assert C.AVOGADRO == 6.02214076e23
    assert C.R_UNIVERSAL == pytest.approx(C.BOLTZMANN * C.AVOGADRO, rel=1e-9)
    assert C.STEFAN_BOLTZMANN == pytest.approx(5.670374419e-8, rel=1e-9)
