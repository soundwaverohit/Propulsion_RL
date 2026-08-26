"""The economics contract.

Economics is a post-processor, by design. It consumes a finished
:class:`~propulsion_rl.missions.base.MissionResult` plus the propulsion
system's :class:`~propulsion_rl.core.types.BillOfMaterials` and produces cost
figures. It never participates in the physics and, by default, never enters the
reward -- the study is technical-first, with cost as the tiebreaker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.types import BillOfMaterials, HealthReport
from ..missions.base import MissionResult


@dataclass(slots=True)
class CostBreakdown:
    """Where the money went, in constant-year dollars.

    ``cost_per_kg_delivered`` is the headline figure the comparison ranks on:
    it folds performance, trip time and hardware cost into one number that is
    comparable across a 30 kW solar-electric tug and a 500 MW nuclear stage.
    """

    currency_year: int = 2026
    # Capital
    thruster_capex: float = 0.0
    power_system_capex: float = 0.0
    reactor_capex: float = 0.0
    tankage_capex: float = 0.0
    integration_capex: float = 0.0
    # Recurring
    propellant_cost: float = 0.0
    launch_cost: float = 0.0
    operations_cost: float = 0.0
    refurbishment_cost: float = 0.0
    # Risk / finance
    insurance_cost: float = 0.0
    time_value_cost: float = 0.0       # opportunity cost of a long transfer
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def capex(self) -> float:
        return (
            self.thruster_capex
            + self.power_system_capex
            + self.reactor_capex
            + self.tankage_capex
            + self.integration_capex
        )

    @property
    def opex(self) -> float:
        return (
            self.propellant_cost
            + self.launch_cost
            + self.operations_cost
            + self.refurbishment_cost
        )

    @property
    def total(self) -> float:
        return self.capex + self.opex + self.insurance_cost + self.time_value_cost

    def as_dict(self) -> dict[str, float]:
        d = {k: getattr(self, k) for k in self.__slots__ if k != "extras"}
        d.update(self.extras)
        d.update(capex=self.capex, opex=self.opex, total=self.total)
        return d


@dataclass(slots=True)
class EconomicResult:
    """Headline economics for one episode."""

    breakdown: CostBreakdown
    cost_per_kg_delivered: float = float("inf")
    cost_per_delta_v: float = float("inf")     # $ per (kg payload * m/s)
    amortized_cost: float = 0.0                # after spreading capex over uses
    uses_remaining: float = 0.0
    npv: float = 0.0
    figure_of_merit: float = 0.0               # higher is better, for ranking
    notes: dict[str, Any] = field(default_factory=dict)


class CostModel(ABC):
    """Base class for costing schemes.

    Multiple implementations are expected -- an optimistic reusable-hardware
    model and a conservative expendable one bracket the answer. The comparison
    should report both rather than pretend one set of price assumptions is true.
    """

    name: str = "abstract"

    @abstractmethod
    def evaluate(
        self,
        bom: BillOfMaterials,
        result: MissionResult,
        health: HealthReport,
        **kwargs: Any,
    ) -> EconomicResult:
        """Price one completed mission."""

    def assumptions(self) -> dict[str, Any]:
        """Every price and rate this model used, for the sensitivity sweep."""
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
