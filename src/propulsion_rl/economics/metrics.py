"""Portfolio metrics: campaigns, Pareto fronts and confidence intervals.

Where :mod:`propulsion_rl.economics.cost_model` prices one episode, this module
aggregates over many -- many seeds, many episodes, many (RL method x propulsion
system) pairings.

The headline result of the whole study is a Pareto front of
``(trip time, $/kg delivered, mission success rate)`` with each point labelled by
its pairing, so :func:`pareto_front` is the most important function here and is
written to be correct rather than clever.

Every headline number carries an interval. Comparing RL methods on single seeds
is the single most common way studies of this kind reach a wrong conclusion, so
:func:`bootstrap_ci` exists and should be used on anything that goes in a table.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import numpy as np

from ..core.constants import EPS
from .base import EconomicResult

LOGGER = logging.getLogger(__name__)

INF = float("inf")

Sense = str  # "min" | "max"


# --- record extraction --------------------------------------------------------
def _get(record: Any, key: str) -> float:
    """Pull one objective value out of a record, dict-first then attribute."""
    if isinstance(record, Mapping):
        if key in record:
            return float(record[key])
        raise KeyError(f"record is missing objective {key!r}: keys={sorted(record)}")
    if hasattr(record, key):
        return float(getattr(record, key))
    raise KeyError(f"record {type(record).__name__} has no objective {key!r}")


def objective_matrix(
    records: Sequence[Any] | np.ndarray, objectives: Sequence[str] | None = None
) -> np.ndarray:
    """``(n, k)`` float matrix of objective values.

    Accepts a sequence of mappings, a sequence of objects with matching
    attributes, a pandas DataFrame (duck-typed, pandas is never imported), or a
    ready-made 2D array. Non-finite entries are preserved; the domination logic
    below handles them explicitly.
    """
    if isinstance(records, np.ndarray):
        m = np.atleast_2d(np.asarray(records, dtype=np.float64))
        return m
    # Duck-typed DataFrame support without importing pandas, which is optional.
    if hasattr(records, "columns") and hasattr(records, "to_numpy"):
        if objectives is None:
            return np.asarray(records.to_numpy(), dtype=np.float64)
        return np.asarray(records[list(objectives)].to_numpy(), dtype=np.float64)
    if objectives is None:
        raise ValueError("objectives must be given for a sequence of records")
    seq = list(records)
    if not seq:
        return np.zeros((0, len(objectives)), dtype=np.float64)
    return np.array(
        [[_get(r, k) for k in objectives] for r in seq], dtype=np.float64
    )


def _to_maximisation(matrix: np.ndarray, senses: Sequence[Sense]) -> np.ndarray:
    """Flip minimised columns so every column is 'bigger is better'.

    Non-finite values are mapped to ``-inf`` (worst possible) so that a record
    with ``inf`` dollars per kilogram -- a failed mission -- is dominated by any
    record with a finite cost, and never dominates anything. ``inf`` minus
    ``inf`` never arises because the comparison is done on the flipped values
    directly.
    """
    if matrix.shape[1] != len(senses):
        raise ValueError(
            f"got {matrix.shape[1]} objective columns but {len(senses)} senses"
        )
    out = np.array(matrix, dtype=np.float64, copy=True)
    for j, sense in enumerate(senses):
        s = str(sense).lower()
        if s in ("min", "minimize", "minimise", "-1", "lower"):
            col = out[:, j]
            flipped = np.where(np.isfinite(col), -col, -np.inf)
            out[:, j] = flipped
        elif s in ("max", "maximize", "maximise", "1", "+1", "higher"):
            col = out[:, j]
            out[:, j] = np.where(np.isfinite(col), col, -np.inf)
        else:
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    return out


def _dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean ``(n, n)`` matrix: ``out[i, j]`` is True when i dominates j.

    Maximisation convention. i dominates j when i is at least as good in every
    objective and strictly better in at least one. Ties in every objective are
    mutual non-domination, so duplicate points both stay on the front.
    """
    ge = np.all(a[:, None, :] >= b[None, :, :], axis=2)
    gt = np.any(a[:, None, :] > b[None, :, :], axis=2)
    return ge & gt


class ParetoFront(NamedTuple):
    """Result of :func:`pareto_front`. Unpacks as ``mask, front``."""

    #: Boolean array over the input records, True for non-dominated points.
    mask: np.ndarray
    #: The non-dominated records themselves, sorted by the first objective in
    #: its own sense (ascending for ``min``, descending for ``max``).
    front: list[Any]


def _usable_mask(matrix: np.ndarray) -> np.ndarray:
    """Rows whose every objective is finite, if any such row exists.

    A record with ``inf`` dollars per kilogram delivered nothing. Under the
    textbook rule it would still land on the front whenever it happens to be
    best on some other axis -- the fastest trip is the one that gave up
    immediately -- and a reader looking at that front would be actively
    misled. So non-finite records are set aside whenever a fully finite record
    exists, and only kept when nothing else is available (in which case the
    front is degenerate and the caller needs to know that too).
    """
    finite = np.all(np.isfinite(matrix), axis=1)
    return finite if finite.any() else np.ones(matrix.shape[0], dtype=bool)


def pareto_front(
    records: Sequence[Any] | np.ndarray,
    objectives: Sequence[str],
    senses: Sequence[Sense],
    *,
    exclude_non_finite: bool = True,
) -> ParetoFront:
    """Non-dominated set over ``objectives``.

    Parameters
    ----------
    records:
        Sequence of mappings or objects, or an ``(n, k)`` array, or a DataFrame.
    objectives:
        Keys to read, e.g. ``("trip_time_days", "cost_per_kg", "success_rate")``.
    senses:
        ``"min"`` or ``"max"`` per objective, e.g. ``("min", "min", "max")``.
    exclude_non_finite:
        Drop records with any ``inf``/``nan`` objective, provided at least one
        fully finite record exists. Default True; see :func:`_usable_mask` for
        why. Set False for the textbook rule, where a non-finite value is merely
        the worst possible value in that objective.

    Returns
    -------
    ParetoFront
        ``mask`` over the input order, and ``front``, the surviving records
        sorted along the first objective so the result plots directly as a line.

    Notes
    -----
    * Duplicates and ties both survive: two identical points do not dominate each
      other, which is the standard convention and keeps the mask stable under
      re-ordering of the input.
    * O(n^2) and vectorised. At the scale of this study -- a few thousand
      (method x propulsion x seed) points -- that is microseconds.
    """
    objectives = list(objectives)
    senses = list(senses)
    if len(objectives) != len(senses):
        raise ValueError(
            f"{len(objectives)} objectives but {len(senses)} senses; they must match"
        )
    matrix = objective_matrix(records, objectives)
    n = matrix.shape[0]
    if n == 0:
        return ParetoFront(np.zeros(0, dtype=bool), [])

    flipped = _to_maximisation(matrix, senses)
    dom = _dominates(flipped, flipped)
    mask = ~np.any(dom, axis=0)  # not dominated by anybody
    if exclude_non_finite:
        usable = _usable_mask(matrix)
        n_dropped = int(np.sum(mask & ~usable))
        if n_dropped:
            LOGGER.info(
                "pareto_front: excluded %d non-dominated record(s) with a "
                "non-finite objective (failed missions)",
                n_dropped,
            )
        mask = mask & usable

    seq = list(records) if not isinstance(records, np.ndarray) else list(matrix)
    idx = np.flatnonzero(mask)
    # Sort along the first objective in its natural sense.
    first_sense = str(senses[0]).lower()
    keys = matrix[idx, 0]
    keys = np.where(np.isfinite(keys), keys, np.inf if "min" in first_sense else -np.inf)
    order = np.argsort(keys, kind="stable")
    if "min" not in first_sense:
        order = order[::-1]
    front = [seq[int(i)] for i in idx[order]]
    return ParetoFront(mask, front)


def dominance_rank(
    records: Sequence[Any] | np.ndarray,
    objectives: Sequence[str],
    senses: Sequence[Sense],
    *,
    exclude_non_finite: bool = True,
) -> np.ndarray:
    """Non-dominated sorting rank per record. 0 is the Pareto front.

    The NSGA-II convention: rank 0 is the non-dominated set, rank 1 is what
    would be non-dominated once rank 0 is removed, and so on. Useful for scoring
    a pairing that is close to the front but not on it -- "second front" is a
    meaningfully different statement from "dominated".

    With ``exclude_non_finite`` (the default), records with a non-finite
    objective are ranked strictly behind every finite record, so ``rank == 0``
    agrees exactly with :func:`pareto_front`'s mask.
    """
    matrix = objective_matrix(records, list(objectives))
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    flipped = _to_maximisation(matrix, list(senses))
    dom = _dominates(flipped, flipped)

    ranks = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    if exclude_non_finite:
        usable = _usable_mask(matrix)
        remaining &= usable
    current = 0
    while remaining.any():
        sub = dom[np.ix_(remaining, remaining)]
        non_dom_local = ~np.any(sub, axis=0)
        idx = np.flatnonzero(remaining)[non_dom_local]
        if idx.size == 0:  # pragma: no cover - impossible for a strict order
            LOGGER.error("dominance cycle detected; assigning remaining rank %d", current)
            ranks[remaining] = current
            break
        ranks[idx] = current
        remaining[idx] = False
        current += 1
    # Anything set aside as non-finite sits behind every real front.
    ranks[ranks < 0] = current
    return ranks


# --- hypervolume --------------------------------------------------------------
def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact 2D hypervolume, minimisation, reference point dominated by all."""
    if points.size == 0:
        return 0.0
    # Keep only points that beat the reference in both objectives.
    keep = np.all(points < reference[None, :], axis=1)
    pts = points[keep]
    if pts.size == 0:
        return 0.0
    # Non-dominated filter, then sweep.
    nd = ~np.any(
        np.all(pts[:, None, :] <= pts[None, :, :], axis=2)
        & np.any(pts[:, None, :] < pts[None, :, :], axis=2),
        axis=0,
    )
    pts = pts[nd]
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    total = 0.0
    for i in range(pts.shape[0]):
        x_next = pts[i + 1, 0] if i + 1 < pts.shape[0] else reference[0]
        total += (x_next - pts[i, 0]) * (reference[1] - pts[i, 1])
    return float(total)


def hypervolume(
    records: Sequence[Any] | np.ndarray,
    objectives: Sequence[str],
    senses: Sequence[Sense],
    reference: Sequence[float],
) -> float:
    """Dominated hypervolume with respect to a reference point. Higher is better.

    Two and three objectives only, which is all this study needs: the headline
    front is ``(trip time, $/kg, success rate)``.

    ``reference`` must be worse than every point you want counted, in the
    original (unflipped) units -- e.g. ``(1400 days, 200_000 $/kg, 0.0)`` for
    ``("min", "min", "max")``. Points that do not beat the reference in every
    objective contribute nothing, which is the standard convention and is why the
    reference must be chosen once and held fixed across every pairing being
    compared. Changing it changes the ranking.

    Implementation: convert to minimisation, then for 3D slice along the third
    objective and integrate exact 2D hypervolumes. Exact, and O(n^2) in 2D /
    O(n^3) in 3D -- fine at the scale of this study, where a front has tens of
    points, and not intended for large populations.
    """
    objectives = list(objectives)
    senses = list(senses)
    k = len(objectives)
    if k not in (2, 3):
        raise NotImplementedError(
            f"hypervolume supports 2 or 3 objectives, got {k}. "
            "For more, use dominance_rank or a dedicated WFG implementation."
        )
    matrix = objective_matrix(records, objectives)
    if matrix.shape[0] == 0:
        return 0.0
    # Flip to maximisation, then negate to get a clean minimisation problem.
    maxed = _to_maximisation(matrix, senses)
    pts = -maxed  # minimisation; +inf marks a useless (non-finite) point
    ref = -_to_maximisation(
        np.asarray(reference, dtype=np.float64).reshape(1, k), senses
    )[0]

    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return 0.0

    if k == 2:
        return _hypervolume_2d(pts, ref)

    keep = np.all(pts < ref[None, :], axis=1)
    pts = pts[keep]
    if pts.size == 0:
        return 0.0
    order = np.argsort(pts[:, 2], kind="stable")
    pts = pts[order]
    total = 0.0
    for i in range(pts.shape[0]):
        z_next = pts[i + 1, 2] if i + 1 < pts.shape[0] else ref[2]
        thickness = z_next - pts[i, 2]
        if thickness <= 0.0:
            continue
        total += _hypervolume_2d(pts[: i + 1, :2], ref[:2]) * thickness
    return float(total)


# --- levelised cost of transport ---------------------------------------------
@dataclass(slots=True)
class LCOTResult:
    """Campaign-level levelised cost of transport."""

    lcot_usd_per_kg: float = INF
    pv_cost_usd: float = 0.0
    pv_mass_kg: float = 0.0
    n_missions: int = 0
    learning_slope: float = 0.9
    discount_rate: float = 0.07
    cadence_years: float = 1.0
    #: Per-mission learning multipliers actually applied (Wright's law).
    learning_factors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Per-mission discounted costs, in campaign order.
    pv_costs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: dict[str, Any] = field(default_factory=dict)


def levelized_cost_of_transport(
    results: Iterable[EconomicResult],
    *,
    learning_slope: float = 0.90,
    discount_rate: float = 0.07,
    cadence_years: float = 1.0,
    learnable_components: Sequence[str] = ("capex", "refurbishment_cost"),
    include_failures: bool = True,
) -> LCOTResult:
    """LCOT: dollars per kilogram delivered, over a campaign of N missions.

    The single-mission ``cost_per_kg_delivered`` charges the first article's
    price to the first flight, which is the right number for a demonstration and
    the wrong number for a programme. LCOT is the programme number::

        LCOT = sum_i PV(cost_i) / sum_i PV(mass_i)

    with two effects the single-mission figure cannot see:

    **Learning.** Wright's law: the i-th unit costs ``i ** log2(slope)`` times the
    first. Only ``learnable_components`` are discounted this way -- capital and
    refurbishment learn, propellant and launch commodity prices do not (they have
    their own, separate learning that belongs in the price book, not here). A 90%
    slope means the 10th vehicle costs 76% of the first and the 100th costs 47%.

    **Time.** Missions fly ``cadence_years`` apart and both cost and delivered
    mass are discounted to the campaign start. Discounting the *mass* as well as
    the cost is deliberate: it is what makes LCOT a levelised price rather than a
    simple average, exactly as in the levelised cost of energy it is named after.

    ``include_failures`` controls whether zero-delivery missions are carried.
    Leave it True. A campaign's economics include the flights that did not work,
    and dropping them is how a study accidentally reports the cost of the
    successes only.

    Returns ``inf`` for a campaign that delivered nothing, never ``nan``.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        return LCOTResult(n_missions=0, notes={"empty": True})

    slope = float(learning_slope)
    if not 0.5 <= slope <= 1.0:
        LOGGER.warning(
            "learning slope %.3f is outside the usual 0.85-0.95 aerospace band", slope
        )
    b = math.log2(max(slope, 1e-6))

    learnable_set = set(learnable_components)
    pv_costs = np.zeros(n, dtype=np.float64)
    factors = np.zeros(n, dtype=np.float64)
    pv_mass = 0.0
    dropped = 0

    for i, res in enumerate(results):
        breakdown = res.breakdown
        delivered = 0.0
        # EconomicResult does not carry delivered mass directly; recover it from
        # the identity total / cost_per_kg, which is exact by construction.
        cpk = res.cost_per_kg_delivered
        total = breakdown.total
        if math.isfinite(cpk) and cpk > EPS:
            delivered = total / cpk
        if delivered <= EPS and not include_failures:
            dropped += 1
            continue

        d = breakdown.as_dict()
        learnable = sum(float(d.get(k, 0.0)) for k in learnable_set)
        fixed = total - learnable
        factor = (i + 1.0) ** b
        factors[i] = factor
        cost_i = fixed + learnable * factor

        years = i * cadence_years
        discount = (1.0 + discount_rate) ** (-years)
        pv_costs[i] = cost_i * discount
        pv_mass += delivered * discount

    pv_cost = float(np.sum(pv_costs))
    lcot = pv_cost / pv_mass if pv_mass > EPS else INF
    return LCOTResult(
        lcot_usd_per_kg=lcot,
        pv_cost_usd=pv_cost,
        pv_mass_kg=float(pv_mass),
        n_missions=n,
        learning_slope=slope,
        discount_rate=float(discount_rate),
        cadence_years=float(cadence_years),
        learning_factors=factors,
        pv_costs=pv_costs,
        notes={
            "learnable_components": sorted(learnable_set),
            "dropped_failures": dropped,
            "wright_exponent": b,
        },
    )


# --- bootstrap ----------------------------------------------------------------
@dataclass(slots=True)
class BootstrapCI:
    """A point estimate with a percentile bootstrap interval."""

    point: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n_samples: int = 0
    n_resamples: int = 0
    alpha: float = 0.05
    method: str = "percentile"

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.lo:.4g}, {self.hi:.4g}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n_samples": self.n_samples,
            "n_resamples": self.n_resamples,
            "alpha": self.alpha,
            "method": self.method,
        }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n: int = 10_000,
    alpha: float = 0.05,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    rng: np.random.Generator | int | None = None,
    method: str = "percentile",
    drop_non_finite: bool = True,
) -> BootstrapCI:
    """Non-parametric bootstrap confidence interval over seeds.

    Every headline number in this study must carry an interval. Comparing RL
    methods on single seeds is how these studies go wrong: the seed-to-seed
    spread of a policy-gradient method on a hard control task routinely exceeds
    the difference between methods, and a table of point estimates hides that
    completely.

    Parameters
    ----------
    values:
        One value per seed (or per episode). Typically 5-20 numbers.
    n:
        Resamples. 10,000 is enough for a 95% interval; the Monte Carlo error on
        the endpoints is then well below the sampling error being estimated.
    alpha:
        Two-sided level. 0.05 gives a 95% interval.
    statistic:
        Defaults to the mean. Pass ``np.median`` for a robust centre, or a
        closure for anything else. It must accept a 1D array.
    method:
        ``"percentile"`` (default) or ``"basic"`` (the pivotal interval,
        ``2*theta - upper``/``2*theta - lower``). Percentile is the usual choice
        and is transformation-respecting; basic has better coverage for skewed
        statistics.
    drop_non_finite:
        Failed missions produce ``inf`` dollars per kilogram. Bootstrapping a
        mean that includes ``inf`` gives ``inf`` for every resample and tells you
        nothing, so non-finite values are dropped by default and their count is
        recorded. **Report that count** -- a method with a great $/kg on its 3
        successful seeds out of 10 is not a better method.

    Returns
    -------
    BootstrapCI
        With ``nan`` endpoints and a warning if there are fewer than two usable
        values -- a single seed genuinely has no interval, and faking one would
        be worse than admitting it.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n_raw = arr.size
    n_non_finite = int(np.sum(~np.isfinite(arr)))
    if drop_non_finite:
        arr = arr[np.isfinite(arr)]

    stat = statistic if statistic is not None else (lambda x: float(np.mean(x)))
    out = BootstrapCI(
        n_samples=int(arr.size), n_resamples=int(n), alpha=float(alpha), method=method
    )
    if arr.size == 0:
        LOGGER.warning("bootstrap_ci: no finite values (%d dropped)", n_non_finite)
        return out
    out.point = float(stat(arr))
    if arr.size == 1:
        LOGGER.warning(
            "bootstrap_ci: only one usable value; an interval over one seed is "
            "not a thing. Reporting the point estimate with nan bounds."
        )
        return out

    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    k = arr.size
    # Chunk so a large n x k index array cannot blow up memory.
    chunk = max(1, int(2e7 // max(k, 1)))
    stats = np.empty(n, dtype=np.float64)
    done = 0
    while done < n:
        take = min(chunk, n - done)
        idx = generator.integers(0, k, size=(take, k))
        sample = arr[idx]
        try:
            stats[done : done + take] = np.apply_along_axis(stat, 1, sample)
        except Exception:  # pragma: no cover - user statistic misbehaving
            stats[done : done + take] = [stat(row) for row in sample]
        done += take

    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(stats, [lo_q, hi_q])
    if method == "basic":
        lo, hi = 2.0 * out.point - hi, 2.0 * out.point - lo
    elif method != "percentile":
        raise ValueError(f"method must be 'percentile' or 'basic', got {method!r}")
    out.lo, out.hi = float(lo), float(hi)
    return out


def bootstrap_ci_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n: int = 10_000,
    alpha: float = 0.05,
    *,
    paired: bool = False,
    statistic: Callable[[np.ndarray], float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> BootstrapCI:
    """Bootstrap interval on ``stat(a) - stat(b)``, for comparing two pairings.

    Set ``paired=True`` when the two arms were run on the *same* seeds, which is
    the right way to run this comparison: pairing removes the seed variance and
    is typically several times tighter than the unpaired interval. An interval
    that straddles zero means the two methods are not distinguishable at this
    sample size -- say so, rather than ranking them anyway.
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    stat = statistic if statistic is not None else (lambda v: float(np.mean(v)))

    if paired:
        if x.size != y.size:
            raise ValueError(
                f"paired comparison needs equal lengths, got {x.size} and {y.size}"
            )
        return bootstrap_ci(
            x - y, n=n, alpha=alpha, statistic=stat, rng=rng, method="percentile"
        )

    xf, yf = x[np.isfinite(x)], y[np.isfinite(y)]
    out = BootstrapCI(
        n_samples=int(xf.size + yf.size), n_resamples=int(n), alpha=float(alpha)
    )
    if xf.size < 2 or yf.size < 2:
        LOGGER.warning("bootstrap_ci_diff: need >=2 finite values per arm")
        if xf.size and yf.size:
            out.point = float(stat(xf) - stat(yf))
        return out
    out.point = float(stat(xf) - stat(yf))
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        diffs[i] = stat(
            xf[generator.integers(0, xf.size, xf.size)]
        ) - stat(yf[generator.integers(0, yf.size, yf.size)])
    lo, hi = np.percentile(diffs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    out.lo, out.hi = float(lo), float(hi)
    return out


def success_rate_ci(
    successes: Sequence[bool] | np.ndarray, alpha: float = 0.05
) -> BootstrapCI:
    """Wilson score interval for a mission success rate.

    Not a bootstrap: for a binomial proportion the Wilson interval has far
    better small-sample coverage than either the normal approximation or a
    bootstrap, and it never produces the degenerate ``[0, 0]`` interval that a
    bootstrap gives for 0/10 successes -- which matters here, because "this
    pairing never succeeded" is a result that still needs an upper bound.
    """
    s = np.asarray(successes, dtype=np.float64).reshape(-1)
    n_obs = s.size
    out = BootstrapCI(n_samples=int(n_obs), n_resamples=0, alpha=float(alpha),
                      method="wilson")
    if n_obs == 0:
        return out
    p = float(np.mean(s > 0.5))
    out.point = p
    # 1 - alpha/2 normal quantile, computed without scipy.
    z = math.sqrt(2.0) * _erfinv(1.0 - alpha)
    denom = 1.0 + z * z / n_obs
    centre = (p + z * z / (2.0 * n_obs)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n_obs + z * z / (4.0 * n_obs * n_obs))
        / denom
    )
    out.lo, out.hi = float(max(centre - half, 0.0)), float(min(centre + half, 1.0))
    return out


def _erfinv(x: float) -> float:
    """Inverse error function, Newton-refined Winitzki approximation.

    Avoids a scipy import for one scalar. Accurate to ~1e-12 after refinement,
    which is far more than a confidence-interval endpoint needs.
    """
    if x <= -1.0:
        return -INF
    if x >= 1.0:
        return INF
    a = 0.147
    ln1 = math.log(1.0 - x * x)
    t1 = 2.0 / (math.pi * a) + ln1 / 2.0
    y = math.copysign(math.sqrt(max(math.sqrt(t1 * t1 - ln1 / a) - t1, 0.0)), x)
    for _ in range(3):
        err = math.erf(y) - x
        y -= err / (2.0 / math.sqrt(math.pi) * math.exp(-y * y))
    return y


# --- convenience --------------------------------------------------------------
def campaign_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    cost_key: str = "cost_per_kg_delivered",
    time_key: str = "trip_time_days",
    success_key: str = "success",
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Bootstrap the three headline axes for one (method x propulsion) pairing.

    Returns a dict ready to become one row of the results table and one point on
    the Pareto front, with an interval on every number.
    """
    costs = np.array([float(r.get(cost_key, INF)) for r in records], dtype=np.float64)
    times = np.array([float(r.get(time_key, np.nan)) for r in records], dtype=np.float64)
    wins = np.array([bool(r.get(success_key, False)) for r in records])

    cost_ci = bootstrap_ci(costs, alpha=alpha, rng=rng)
    time_ci = bootstrap_ci(times, alpha=alpha, rng=rng)
    rate_ci = success_rate_ci(wins, alpha=alpha)
    return {
        "n_seeds": len(records),
        "n_failed": int(np.sum(~np.isfinite(costs))),
        cost_key: cost_ci.point,
        f"{cost_key}_lo": cost_ci.lo,
        f"{cost_key}_hi": cost_ci.hi,
        time_key: time_ci.point,
        f"{time_key}_lo": time_ci.lo,
        f"{time_key}_hi": time_ci.hi,
        "success_rate": rate_ci.point,
        "success_rate_lo": rate_ci.lo,
        "success_rate_hi": rate_ci.hi,
    }


__all__ = [
    "ParetoFront",
    "LCOTResult",
    "BootstrapCI",
    "objective_matrix",
    "pareto_front",
    "dominance_rank",
    "hypervolume",
    "levelized_cost_of_transport",
    "bootstrap_ci",
    "bootstrap_ci_diff",
    "success_rate_ci",
    "campaign_summary",
]
