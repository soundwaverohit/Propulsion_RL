"""Nuclear propulsion: thermal rockets and nuclear-electric stages.

Importing this package registers the four nuclear presets in
:data:`~propulsion_rl.core.registry.PROPULSION`.
"""

from __future__ import annotations

from .nep import BRAYTON, KILOPOWER, NEPDesign, NuclearElectricStage
from .ntp import NERVA, PEWEE, NTPDesign, NuclearThermalRocket, ntp_isp
from .reactor import (
    FissionReactor,
    ReactorDesign,
    ReactorState,
    solve_inhour,
    way_wigner_fraction,
)

__all__ = [
    "FissionReactor",
    "ReactorDesign",
    "ReactorState",
    "solve_inhour",
    "way_wigner_fraction",
    "NuclearThermalRocket",
    "NTPDesign",
    "ntp_isp",
    "PEWEE",
    "NERVA",
    "NuclearElectricStage",
    "NEPDesign",
    "KILOPOWER",
    "BRAYTON",
]
