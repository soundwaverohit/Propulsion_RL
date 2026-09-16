"""Mission scenarios.

Importing this package registers all five scenarios in
:data:`propulsion_rl.core.registry.MISSION`, so an experiment sweep names them
by string and never imports them directly.

The suite is chosen to span the regimes rather than to cover them evenly. Two
missions where high specific impulse decides the outcome, one where
thrust-to-weight decides it, and one where the answer is genuinely open:

============================  ==================  ================  ==================
Mission                       Frame               Horizon           Expected winner
============================  ==================  ================  ==================
``leo_geo_transfer``          planetocentric      ~2.5 years        electric
``gto_geo_transfer``          planetocentric      ~4-12 months      electric
``earth_mars_cargo``          heliocentric        ~1-3 years        contested
``mars_crew_fast``            heliocentric        <= 220 days       nuclear thermal
``geo_station_keeping``       planetocentric      10 years          electric
============================  ==================  ================  ==================

All four share the reward machinery in :mod:`~.rewards`, which is what makes
their returns comparable on one axis.
"""

from __future__ import annotations

from .base import Mission, MissionResult, RewardTerms
from .earth_mars_cargo import EarthMarsCargo
from .geo_station_keeping import GEOStationKeeping
from .gto_geo_transfer import GTOtoGEOTransfer
from .leo_geo_transfer import LEOtoGEOTransfer
from .mars_crew_fast import MarsCrewFast
from .rewards import RewardConfig, RewardScales

__all__ = [
    "Mission",
    "MissionResult",
    "RewardTerms",
    "RewardConfig",
    "RewardScales",
    "LEOtoGEOTransfer",
    "GTOtoGEOTransfer",
    "EarthMarsCargo",
    "MarsCrewFast",
    "GEOStationKeeping",
]
