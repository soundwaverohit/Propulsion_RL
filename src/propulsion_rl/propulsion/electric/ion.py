"""Gridded-ion engine presets.

NSTAR flew on Deep Space 1 and Dawn (2.3 kW, 92 mN, 3100 s, 235 kg Xe). NEXT-C
is the throttleable successor (up to ~6.9 kW flight, 236 mN, 4190 s) and is the
high-Isp electric end of the comparison matrix.
"""

from __future__ import annotations

import math
from typing import Any

from ...core.constants import HOUR
from ...core.registry import PROPULSION
from ...core.types import PropulsionFamily
from .electrostatic import ElectrostaticDesign, ElectrostaticThruster

# Brophy, J. R. et al., "The Ion Propulsion System for Dawn", AIAA-2003-4542.
# NSTAR 30-cm: 2.3 kW PPU input, 92.7 mN, 3120 s, 235 kg throughput.
NSTAR = ElectrostaticDesign(
    name="ion_nstar",
    kind="ion",
    voltage_min_v=650.0,
    voltage_max_v=1_100.0,
    voltage_nominal_v=1_100.0,
    mdot_anode_max_kg_s=3.1e-6,
    rated_power_w=2_300.0,
    mass_utilisation=0.90,
    eta_beam=0.78,
    eta_ppu=0.93,
    divergence_half_angle_rad=math.radians(10.0),
    doubles_ratio=0.20,
    cathode_fraction=0.10,
    cathode_fraction_hi=0.20,
    accel_voltage_v=180.0,
    grid_gap_m=5.8e-4,
    screen_hole_diameter_m=1.91e-3,
    open_area_m2=0.047,
    perveance_fraction=0.30,
    thermal_mass_kg=8.0,
    thermal_area_m2=0.10,
    max_temperature_k=520.0,
    qualified_life_s=30_000.0 * HOUR,
    max_throughput_kg=235.0,
    erosion_depth_limit_m=0.8e-3,
    reference_erosion_rate_m_s=8.0e-12,
    sputter_threshold_ev=35.0,
    dry_mass_kg=80.0,
    radiator_area_m2=3.0,
    housekeeping_w=35.0,
)

# Patterson, M. J. and Benson, S., "NEXT Ion Propulsion System Development
# Status and Performance", AIAA-2007-5199. NEXT-C ~6.9 kW, 236 mN, 4190 s.
NEXT = ElectrostaticDesign(
    name="ion_next",
    kind="ion",
    voltage_min_v=275.0,
    voltage_max_v=1_800.0,
    voltage_nominal_v=1_800.0,
    mdot_anode_max_kg_s=8.0e-6,
    rated_power_w=6_900.0,
    mass_utilisation=0.92,
    eta_beam=0.82,
    eta_ppu=0.94,
    divergence_half_angle_rad=math.radians(9.0),
    doubles_ratio=0.10,
    cathode_fraction=0.08,
    cathode_fraction_hi=0.16,
    accel_voltage_v=200.0,
    grid_gap_m=6.5e-4,
    screen_hole_diameter_m=1.91e-3,
    open_area_m2=0.079,
    perveance_fraction=0.30,
    thermal_mass_kg=14.0,
    thermal_area_m2=0.16,
    max_temperature_k=540.0,
    qualified_life_s=48_000.0 * HOUR,
    max_throughput_kg=800.0,
    erosion_depth_limit_m=1.0e-3,
    reference_erosion_rate_m_s=5.0e-12,
    sputter_threshold_ev=35.0,
    dry_mass_kg=120.0,
    radiator_area_m2=8.0,
    housekeeping_w=50.0,
)


class GriddedIonEngine(ElectrostaticThruster):
    """Two-grid ion engine. Constructed from an :class:`ElectrostaticDesign`."""


def _factory(design: ElectrostaticDesign):
    def make(**_kwargs: Any) -> GriddedIonEngine:
        return GriddedIonEngine(design)

    make.__name__ = design.name
    make.__qualname__ = design.name
    return make


PROPULSION.add(
    NSTAR.name,
    _factory(NSTAR),
    family=PropulsionFamily.ELECTRIC,
    power_w=NSTAR.rated_power_w,
    kind="ion",
)
PROPULSION.add(
    NEXT.name,
    _factory(NEXT),
    family=PropulsionFamily.ELECTRIC,
    power_w=NEXT.rated_power_w,
    kind="ion",
)

__all__ = ["GriddedIonEngine", "NSTAR", "NEXT"]
