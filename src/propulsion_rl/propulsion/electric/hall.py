"""Hall-effect thruster presets.

SPT-100 is the flight-heritage 1.35 kW Fakel design (many GEO comsats, Dawn's
attitude control is a cousin). HERMeS/AEPS is the 12.5 kW NASA/Aerojet string
flying on Gateway PPE -- the high-power end of the Hall family the comparison
needs, so a months-long LEO-GEO spiral is not only a smallsat problem.
"""

from __future__ import annotations

import math
from typing import Any

from ...core.constants import HOUR
from ...core.registry import PROPULSION
from ...core.types import PropulsionFamily
from .electrostatic import ElectrostaticDesign, ElectrostaticThruster

# Manzella, D. et al., "Performance Evaluation of the SPT-100 Thruster at
# Fakel and NASA LeRC", IEPC-93-094 / NASA TM-106401. 1.35 kW, 83 mN, ~1600 s.
SPT100 = ElectrostaticDesign(
    name="hall_spt100",
    kind="hall",
    voltage_min_v=200.0,
    voltage_max_v=400.0,
    voltage_nominal_v=300.0,
    mdot_anode_max_kg_s=5.3e-6,
    rated_power_w=1_350.0,
    mass_utilisation=0.90,
    eta_beam=0.72,
    eta_ppu=0.93,
    divergence_half_angle_rad=math.radians(18.0),
    doubles_ratio=0.12,
    cathode_fraction=0.08,
    cathode_fraction_hi=0.18,
    thermal_mass_kg=3.5,
    thermal_area_m2=0.10,
    max_temperature_k=750.0,
    qualified_life_s=9_000.0 * HOUR,
    max_throughput_kg=150.0,
    erosion_depth_limit_m=2.0e-3,
    reference_erosion_rate_m_s=6.0e-11,
    sputter_threshold_ev=50.0,
    dry_mass_kg=25.0,
    radiator_area_m2=1.5,
    housekeeping_w=25.0,
)

# Hofer, R. et al., "Development and Qualification of the HERMeS Hall Thruster
# in Support of the AEPS" / NASA AEPS 12.5 kW string. ~588 mN, ~2800-3000 s.
HERMES = ElectrostaticDesign(
    name="hall_hermes",
    kind="hall",
    voltage_min_v=300.0,
    voltage_max_v=800.0,
    voltage_nominal_v=500.0,
    mdot_anode_max_kg_s=2.1e-5,
    rated_power_w=12_500.0,
    mass_utilisation=0.92,
    eta_beam=0.80,
    eta_ppu=0.95,
    divergence_half_angle_rad=math.radians(12.0),
    doubles_ratio=0.08,
    cathode_fraction=0.07,
    cathode_fraction_hi=0.16,
    thermal_mass_kg=25.0,
    thermal_area_m2=0.22,
    max_temperature_k=850.0,
    qualified_life_s=50_000.0 * HOUR,
    max_throughput_kg=2_000.0,
    erosion_depth_limit_m=3.5e-3,
    reference_erosion_rate_m_s=2.0e-11,
    sputter_threshold_ev=45.0,
    dry_mass_kg=180.0,
    radiator_area_m2=12.0,
    housekeeping_w=80.0,
)


class HallThruster(ElectrostaticThruster):
    """Hall-effect thruster string. Constructed from an :class:`ElectrostaticDesign`."""


def _factory(design: ElectrostaticDesign):
    def make(**_kwargs: Any) -> HallThruster:
        return HallThruster(design)

    make.__name__ = design.name
    make.__qualname__ = design.name
    return make


PROPULSION.add(
    SPT100.name,
    _factory(SPT100),
    family=PropulsionFamily.ELECTRIC,
    power_w=SPT100.rated_power_w,
    kind="hall",
)
PROPULSION.add(
    HERMES.name,
    _factory(HERMES),
    family=PropulsionFamily.ELECTRIC,
    power_w=HERMES.rated_power_w,
    kind="hall",
)

__all__ = ["HallThruster", "SPT100", "HERMES"]
