"""Electric propulsion: Hall-effect and gridded-ion engines.

Importing this package registers the four flight-heritage presets in
:data:`~propulsion_rl.core.registry.PROPULSION`.
"""

from __future__ import annotations

from .common import (
    EfficiencyBreakdown,
    ThermalNode,
    WearAccumulator,
    beam_velocity,
    isp_from_beam,
    thrust_from_power,
    total_efficiency,
)
from .electrostatic import ElectrostaticDesign, ElectrostaticThruster
from .hall import HERMES, SPT100, HallThruster
from .ion import NEXT, NSTAR, GriddedIonEngine

__all__ = [
    "ElectrostaticDesign",
    "ElectrostaticThruster",
    "HallThruster",
    "GriddedIonEngine",
    "SPT100",
    "HERMES",
    "NSTAR",
    "NEXT",
    "EfficiencyBreakdown",
    "ThermalNode",
    "WearAccumulator",
    "beam_velocity",
    "isp_from_beam",
    "thrust_from_power",
    "total_efficiency",
]
