"""Economics: the post-processing cost layer.

Importing this package registers every cost model in
:data:`propulsion_rl.core.registry.COST_MODEL`, so a sweep can select one by
string from a YAML config.

The layer is deliberately small and strictly downstream. It consumes a finished
:class:`~propulsion_rl.missions.base.MissionResult`, a
:class:`~propulsion_rl.core.types.BillOfMaterials` and a
:class:`~propulsion_rl.core.types.HealthReport`, and produces dollars. It never
touches the physics and by default never enters the reward.

Module map
----------
:mod:`~propulsion_rl.economics.prices`
    ``PriceBook``: every dollar figure, cited and dated, in constant 2026 USD.
    Three reference books -- ``BASELINE``, ``OPTIMISTIC``, ``CONSERVATIVE`` --
    bracket the answer.
:mod:`~propulsion_rl.economics.cost_model`
    ``reference`` / ``reusable`` / ``conservative`` cost models.
:mod:`~propulsion_rl.economics.metrics`
    Campaign-level LCOT, Pareto fronts, hypervolume, bootstrap intervals.
:mod:`~propulsion_rl.economics.sensitivity`
    Sweeps, tornado charts and Monte Carlo over the uncertain prices.

Quick start::

    from propulsion_rl.core.registry import COST_MODEL
    model = COST_MODEL.make("reference")
    econ = model.evaluate(bom, mission_result, health, payload_kg=10_000.0)
    print(econ.cost_per_kg_delivered)
"""

from __future__ import annotations

from . import cost_model, metrics, prices, sensitivity  # noqa: F401
from .base import CostBreakdown, CostModel, EconomicResult
from .cost_model import (
    ConservativeCostModel,
    FigureOfMeritWeights,
    ReferenceCostModel,
    ReusableCostModel,
)
from .metrics import (
    BootstrapCI,
    LCOTResult,
    ParetoFront,
    bootstrap_ci,
    campaign_summary,
    dominance_rank,
    hypervolume,
    levelized_cost_of_transport,
    pareto_front,
)
from .prices import (
    BASELINE,
    CONSERVATIVE,
    OPTIMISTIC,
    PRICE_BOOKS,
    UNCERTAIN_PARAMETERS,
    PriceBook,
    get_price_book,
)
from .sensitivity import monte_carlo, sweep, tornado

__all__ = [
    # contracts
    "CostModel",
    "CostBreakdown",
    "EconomicResult",
    # prices
    "PriceBook",
    "BASELINE",
    "OPTIMISTIC",
    "CONSERVATIVE",
    "PRICE_BOOKS",
    "UNCERTAIN_PARAMETERS",
    "get_price_book",
    # cost models
    "ReferenceCostModel",
    "ReusableCostModel",
    "ConservativeCostModel",
    "FigureOfMeritWeights",
    # metrics
    "ParetoFront",
    "LCOTResult",
    "BootstrapCI",
    "pareto_front",
    "dominance_rank",
    "hypervolume",
    "levelized_cost_of_transport",
    "bootstrap_ci",
    "campaign_summary",
    # sensitivity
    "sweep",
    "tornado",
    "monte_carlo",
]
