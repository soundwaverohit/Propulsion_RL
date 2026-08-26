"""Which price assumptions is the answer actually made of?

This module exists so the report can say plainly, with numbers, which
conclusions are robust and which are artefacts of an assumed xenon price or an
assumed reactor cost. Three tools, in increasing order of ambition:

:func:`sweep`
    One parameter, a list of values, the resulting metric. The plot you put in
    an appendix.
:func:`tornado`
    Every documented uncertain price, moved to each end of its documented range,
    ranked by how much it moves ``cost_per_kg_delivered``. The plot you put in
    the results section, because it is the one that tells the reader which two
    or three numbers the whole study rests on.
:func:`monte_carlo`
    All the uncertain prices varied together, giving a distribution of $/kg
    rather than a point. The number to quote when someone asks "and what is the
    error bar on that".

All three take a ``cost_model_factory``: a callable mapping a
:class:`~propulsion_rl.economics.prices.PriceBook` to a configured
:class:`~propulsion_rl.economics.base.CostModel`. That indirection is what lets
the same sweep run against the reference, reusable and conservative models
without this module knowing anything about them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.types import BillOfMaterials, HealthReport
from ..missions.base import MissionResult
from .base import CostModel, EconomicResult
from .prices import BASELINE, UNCERTAIN_PARAMETERS, PriceBook

LOGGER = logging.getLogger(__name__)

INF = float("inf")

#: ``PriceBook -> CostModel``
CostModelFactory = Callable[[PriceBook], CostModel]

#: Metrics a sweep can report. Each maps an EconomicResult to one float.
METRICS: dict[str, Callable[[EconomicResult], float]] = {
    "cost_per_kg_delivered": lambda r: r.cost_per_kg_delivered,
    "cost_per_delta_v": lambda r: r.cost_per_delta_v,
    "total": lambda r: r.breakdown.total,
    "capex": lambda r: r.breakdown.capex,
    "opex": lambda r: r.breakdown.opex,
    "npv": lambda r: r.npv,
    "figure_of_merit": lambda r: r.figure_of_merit,
    "amortized_cost": lambda r: r.amortized_cost,
    "uses_remaining": lambda r: r.uses_remaining,
}


def _metric_fn(metric: str | Callable[[EconomicResult], float]):
    if callable(metric):
        return metric
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}. Known: {sorted(METRICS)}")
    return METRICS[metric]


@dataclass(slots=True)
class SweepResult:
    """One-at-a-time sensitivity of one metric to one parameter."""

    parameter: str
    values: np.ndarray
    metric: str
    outcomes: np.ndarray
    baseline_value: float = float("nan")
    baseline_outcome: float = float("nan")
    #: The full EconomicResult at each point, if the caller wants the breakdowns.
    results: list[EconomicResult] = field(default_factory=list)

    @property
    def elasticity(self) -> float:
        """Log-log slope of metric vs parameter across the swept range.

        1.0 means the metric is exactly proportional to the parameter (double
        the price, double the cost); 0.0 means the parameter does not matter.
        Computed only over finite, positive points; ``nan`` when fewer than two
        such points exist.
        """
        v = np.asarray(self.values, dtype=np.float64)
        y = np.asarray(self.outcomes, dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(y) & (v > 0) & (y > 0)
        if int(np.sum(ok)) < 2:
            return float("nan")
        return float(np.polyfit(np.log(v[ok]), np.log(y[ok]), 1)[0])


def sweep(
    cost_model_factory: CostModelFactory,
    bom: BillOfMaterials,
    result: MissionResult,
    health: HealthReport,
    parameter: str,
    values: Sequence[float],
    *,
    base_prices: PriceBook = BASELINE,
    metric: str | Callable[[EconomicResult], float] = "cost_per_kg_delivered",
    keep_results: bool = False,
    **eval_kwargs: Any,
) -> SweepResult:
    """One-at-a-time sensitivity: vary ``parameter``, hold everything else.

    ``parameter`` is a field name on :class:`PriceBook`. Everything else in the
    book is held at ``base_prices``, so this measures a partial derivative, not a
    plausible alternative world -- prices move together in reality (a launch
    price collapse and a xenon price collapse have the same underlying cause more
    often than not). Use :func:`monte_carlo` for the joint picture and this for
    attribution.
    """
    fn = _metric_fn(metric)
    vals = np.asarray(list(values), dtype=np.float64)
    outcomes = np.empty(vals.size, dtype=np.float64)
    kept: list[EconomicResult] = []

    for i, v in enumerate(vals):
        prices = base_prices.with_values(**{parameter: float(v)})
        model = cost_model_factory(prices)
        econ = model.evaluate(bom, result, health, **eval_kwargs)
        outcomes[i] = fn(econ)
        if keep_results:
            kept.append(econ)

    base_model = cost_model_factory(base_prices)
    base_econ = base_model.evaluate(bom, result, health, **eval_kwargs)
    return SweepResult(
        parameter=parameter,
        values=vals,
        metric=metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        outcomes=outcomes,
        baseline_value=float(getattr(base_prices, parameter)),
        baseline_outcome=float(fn(base_econ)),
        results=kept,
    )


@dataclass(slots=True)
class TornadoBar:
    """One parameter's contribution to the spread in the metric."""

    parameter: str
    low_price: float
    high_price: float
    low_outcome: float
    high_outcome: float
    baseline_outcome: float

    @property
    def swing(self) -> float:
        """Absolute change in the metric across the parameter's full range."""
        if not (math.isfinite(self.low_outcome) and math.isfinite(self.high_outcome)):
            return INF
        return abs(self.high_outcome - self.low_outcome)

    @property
    def swing_fraction(self) -> float:
        """Swing as a fraction of the baseline metric. The readable version."""
        if not math.isfinite(self.baseline_outcome) or abs(self.baseline_outcome) < 1e-12:
            return INF
        return self.swing / abs(self.baseline_outcome)

    @property
    def direction(self) -> str:
        """``'+'`` when a higher price means a higher metric, else ``'-'``."""
        if not math.isfinite(self.swing):
            return "?"
        return "+" if self.high_outcome >= self.low_outcome else "-"


@dataclass(slots=True)
class TornadoResult:
    """Ranked one-at-a-time sensitivities. Bars are sorted by swing, descending."""

    bars: list[TornadoBar]
    metric: str
    baseline_outcome: float

    def top(self, k: int = 5) -> list[TornadoBar]:
        return self.bars[:k]

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "parameter": b.parameter,
                "low_price": b.low_price,
                "high_price": b.high_price,
                "low_outcome": b.low_outcome,
                "high_outcome": b.high_outcome,
                "swing": b.swing,
                "swing_fraction": b.swing_fraction,
                "direction": b.direction,
            }
            for b in self.bars
        ]

    def format_table(self, k: int = 12, width: int = 30) -> str:
        """ASCII tornado chart, so a sweep can log its own conclusion."""
        bars = self.top(k)
        if not bars:
            return "(no sensitivities)"
        finite = [b.swing_fraction for b in bars if math.isfinite(b.swing_fraction)]
        scale = max(finite) if finite else 1.0
        lines = [
            f"{'parameter':<38} {'low':>12} {'high':>12} {'swing %':>9}  bar",
            "-" * (38 + 12 + 12 + 9 + width + 8),
        ]
        for b in bars:
            frac = b.swing_fraction
            n = int(round(width * frac / scale)) if math.isfinite(frac) and scale else 0
            lines.append(
                f"{b.parameter:<38} {b.low_outcome:>12,.0f} {b.high_outcome:>12,.0f} "
                f"{100 * frac:>8.1f}%  {'#' * max(n, 0)}{b.direction}"
            )
        return "\n".join(lines)


def tornado(
    cost_model_factory: CostModelFactory,
    bom: BillOfMaterials,
    result: MissionResult,
    health: HealthReport,
    *,
    base_prices: PriceBook = BASELINE,
    parameters: Sequence[str] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    metric: str | Callable[[EconomicResult], float] = "cost_per_kg_delivered",
    drop_insensitive: bool = True,
    **eval_kwargs: Any,
) -> TornadoResult:
    """Rank every uncertain price by how much it moves the metric.

    Each parameter is moved to the low and high end of its documented range from
    :data:`~propulsion_rl.economics.prices.UNCERTAIN_PARAMETERS` with all others
    held at ``base_prices``, and the bars are sorted by the resulting swing.

    Read the output as: *the top two or three bars are what this study is
    actually about*. If reactor cost dominates for the nuclear pairing, then the
    nuclear result is a statement about an unknown reactor price and should be
    reported as a range, not a number. If launch price dominates for the electric
    pairing, then the electric result is a statement about the launch market.
    Either way, saying so is more useful than a single confident figure.

    ``drop_insensitive`` removes bars with zero swing -- a xenon price bar on a
    hydrogen stage carries no information and only makes the chart longer.
    """
    fn = _metric_fn(metric)
    ranges = dict(ranges if ranges is not None else UNCERTAIN_PARAMETERS)
    names = list(parameters) if parameters is not None else list(ranges)

    base_econ = cost_model_factory(base_prices).evaluate(
        bom, result, health, **eval_kwargs
    )
    baseline_outcome = float(fn(base_econ))

    bars: list[TornadoBar] = []
    for name in names:
        if name not in ranges:
            LOGGER.warning("no documented range for %r; skipping", name)
            continue
        lo_price, hi_price = ranges[name]
        try:
            lo_econ = cost_model_factory(
                base_prices.with_values(**{name: float(lo_price)})
            ).evaluate(bom, result, health, **eval_kwargs)
            hi_econ = cost_model_factory(
                base_prices.with_values(**{name: float(hi_price)})
            ).evaluate(bom, result, health, **eval_kwargs)
        except KeyError:
            LOGGER.warning("%r is not a PriceBook field; skipping", name)
            continue
        bar = TornadoBar(
            parameter=name,
            low_price=float(lo_price),
            high_price=float(hi_price),
            low_outcome=float(fn(lo_econ)),
            high_outcome=float(fn(hi_econ)),
            baseline_outcome=baseline_outcome,
        )
        if drop_insensitive and bar.swing <= 0.0:
            continue
        bars.append(bar)

    bars.sort(
        key=lambda b: (b.swing if math.isfinite(b.swing) else INF), reverse=True
    )
    return TornadoResult(
        bars=bars,
        metric=metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        baseline_outcome=baseline_outcome,
    )


@dataclass(slots=True)
class MonteCarloResult:
    """Distribution of the metric under joint price uncertainty."""

    samples: np.ndarray
    metric: str
    n: int
    n_non_finite: int = 0
    parameters: tuple[str, ...] = ()
    #: Sampled price values, ``(n, len(parameters))``, for post-hoc regression.
    draws: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def finite(self) -> np.ndarray:
        return self.samples[np.isfinite(self.samples)]

    def percentiles(
        self, qs: Sequence[float] = (5.0, 25.0, 50.0, 75.0, 95.0)
    ) -> dict[float, float]:
        f = self.finite
        if f.size == 0:
            return {q: float("nan") for q in qs}
        vals = np.percentile(f, list(qs))
        return {float(q): float(v) for q, v in zip(qs, np.atleast_1d(vals))}

    @property
    def mean(self) -> float:
        f = self.finite
        return float(np.mean(f)) if f.size else float("nan")

    def rank_correlation(self) -> list[tuple[str, float]]:
        """Spearman rank correlation of each price with the metric, ranked.

        The Monte Carlo answer to the same question the tornado asks, but with
        every price moving at once. Where the two disagree, trust this one --
        the tornado's one-at-a-time assumption is the weaker of the two.
        Computed without scipy: Pearson correlation of the ranks.
        """
        f_mask = np.isfinite(self.samples)
        if self.draws.size == 0 or int(np.sum(f_mask)) < 3:
            return []
        y = _rankdata(self.samples[f_mask])
        out: list[tuple[str, float]] = []
        for j, name in enumerate(self.parameters):
            x = _rankdata(self.draws[f_mask, j])
            sx, sy = np.std(x), np.std(y)
            rho = 0.0 if sx < 1e-12 or sy < 1e-12 else float(
                np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy)
            )
            out.append((name, rho))
        out.sort(key=lambda t: abs(t[1]), reverse=True)
        return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Small helper so scipy stays optional."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average ties.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


def monte_carlo(
    cost_model_factory: CostModelFactory,
    bom: BillOfMaterials,
    result: MissionResult,
    health: HealthReport,
    *,
    n: int = 2_000,
    base_prices: PriceBook = BASELINE,
    parameters: Sequence[str] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    metric: str | Callable[[EconomicResult], float] = "cost_per_kg_delivered",
    rng: np.random.Generator | int | None = None,
    distribution: str = "triangular",
    **eval_kwargs: Any,
) -> MonteCarloResult:
    """Vary every uncertain price at once; return the distribution of the metric.

    ``distribution`` is one of:

    ``triangular`` (default)
        ``triangular(low, mode=baseline, high)`` over each documented range. The
        standard choice for expert-elicited bounds: it respects the fact that the
        baseline is the best estimate and the ends are the surprises, and it has
        no tail beyond the elicited range, which is the correct behaviour when
        the range came from judgement rather than from data.
    ``uniform``
        Flat over the range. Use when you want the range to speak for itself
        with no weight on the baseline. Gives wider, more pessimistic intervals.
    ``loguniform``
        Flat in log space. Appropriate for the reactor cost specifically, whose
        range spans an order of magnitude and where a factor is more meaningful
        than a difference.

    Prices are drawn independently, which is the model's main weakness: in the
    real world launch prices, hardware prices and interest rates are correlated,
    and independent draws therefore understate the tails. The percentiles here
    should be read as a lower bound on the true spread.
    """
    fn = _metric_fn(metric)
    ranges = dict(ranges if ranges is not None else UNCERTAIN_PARAMETERS)
    names = tuple(parameters) if parameters is not None else tuple(ranges)
    names = tuple(nm for nm in names if nm in ranges)
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    draws = np.empty((n, len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        lo, hi = ranges[name]
        lo, hi = float(lo), float(hi)
        if hi < lo:
            lo, hi = hi, lo
        if distribution == "uniform":
            draws[:, j] = generator.uniform(lo, hi, size=n)
        elif distribution == "loguniform":
            if lo <= 0.0:
                LOGGER.warning(
                    "loguniform needs a positive low bound for %r; using uniform", name
                )
                draws[:, j] = generator.uniform(lo, hi, size=n)
            else:
                draws[:, j] = np.exp(
                    generator.uniform(math.log(lo), math.log(hi), size=n)
                )
        elif distribution == "triangular":
            mode = float(getattr(base_prices, name))
            mode = min(max(mode, lo), hi)
            draws[:, j] = generator.triangular(lo, mode, hi, size=n)
        else:
            raise ValueError(
                f"distribution must be triangular/uniform/loguniform, got {distribution!r}"
            )

    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        prices = base_prices.with_values(
            **{nm: float(draws[i, j]) for j, nm in enumerate(names)}
        )
        econ = cost_model_factory(prices).evaluate(bom, result, health, **eval_kwargs)
        samples[i] = fn(econ)

    return MonteCarloResult(
        samples=samples,
        metric=metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
        n=n,
        n_non_finite=int(np.sum(~np.isfinite(samples))),
        parameters=names,
        draws=draws,
    )


__all__ = [
    "CostModelFactory",
    "METRICS",
    "SweepResult",
    "TornadoBar",
    "TornadoResult",
    "MonteCarloResult",
    "sweep",
    "tornado",
    "monte_carlo",
]
