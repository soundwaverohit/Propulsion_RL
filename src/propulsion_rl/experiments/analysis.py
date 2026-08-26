"""Turning a sweep into the answer, without inventing a winner.

The failure mode this module is built against is not a bug, it is a habit: run
a matrix, take the argmax of a column of noisy means over five seeds, and
announce a pairing. With ~14 agents x 8 propulsion systems x 4 missions the
matrix contains hundreds of comparisons, and at five seeds per cell a handful of
them will look decisive by chance alone. Every function here that names a winner
also reports whether the margin survives contact with the uncertainty.

Concretely:

* Aggregates are means with **percentile bootstrap confidence intervals**, never
  bare means, and never a single seed.
* Comparisons between two agents are **paired across seeds** -- replicate 3 of
  one cell shares its environment seeds and its torch seed with replicate 3 of
  the other (see :mod:`propulsion_rl.experiments.runner`), so the difference is
  computed within a seed and the between-seed variance cancels. With N=5 that is
  the difference between a usable test and a decorative one.
* Every comparison carries a bootstrap CI on the difference, an effect size
  (Cohen's dz for paired data), and a **Holm-Bonferroni adjusted** p-value over
  the whole family of comparisons in the matrix. Holm is used rather than plain
  Bonferroni because it is uniformly more powerful and just as valid, and
  rather than FDR because the claim being made ("this pairing is best") is a
  family-wise claim.
* When a difference does not survive, the verdict column says so in words.
  ``not significant`` is a result, and the most common correct one.

The interaction analysis is deliberately a first-class output rather than a
footnote: "is the best RL method the same across propulsion systems?" is the
research question, and a main-effects table cannot answer it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Metric -> whether larger is better. Anything not listed is assumed
#: larger-is-better, which is the wrong assumption for a cost, so add costs.
METRIC_DIRECTION: dict[str, bool] = {
    "success_rate": True,
    "return": True,
    "progress": True,
    "payload_delivered_kg": True,
    "figure_of_merit": True,
    "trip_time_days": False,
    "delta_v_m_s": False,
    "propellant_kg": False,
    "wear_fraction": False,
    "constraint_violations": False,
    "constraint_cost": False,
    "terminal_error": False,
    "cost_per_kg_delivered": False,
    "cost_total": False,
    "cost_per_delta_v": False,
    "act_ms_per_step": False,
    "wall_train_s": False,
    "throttled_frac": False,
}

#: The headline table's columns, in the order a reader wants them.
HEADLINE_METRICS: tuple[str, ...] = (
    "success_rate",
    "trip_time_days",
    "delta_v_m_s",
    "propellant_kg",
    "wear_fraction",
    "constraint_violations",
    "cost_per_kg_delivered",
)

#: Weights for the composite normalised score used by the heatmap and by
#: cross-mission aggregation, where raw units are not commensurable.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "success_rate": 0.40,
    "trip_time_days": 0.20,
    "propellant_kg": 0.15,
    "constraint_violations": 0.15,
    "cost_per_kg_delivered": 0.10,
}

GROUP_COLS = ("mission", "propulsion", "agent")
_DEFAULT_N_BOOT = 10_000


# --- loading -----------------------------------------------------------------
def load_sweep(
    sweep: str, results_dir: str | Path = "results", *, drop_failed: bool = True
) -> pd.DataFrame:
    """Load one sweep's tidy rows.

    Prefers ``results.parquet``, then ``results.csv``, then the append-only
    ``results.jsonl``, then a scan of ``cells/*/row.json`` -- so a sweep that
    was killed mid-flight is still analysable from whatever survived.
    """
    d = Path(results_dir) / sweep
    if not d.exists():
        raise FileNotFoundError(f"no sweep at {d}")
    df: pd.DataFrame | None = None
    for name, reader in (
        ("results.parquet", pd.read_parquet),
        ("results.csv", pd.read_csv),
        ("results.jsonl", lambda p: pd.read_json(p, lines=True)),
    ):
        p = d / name
        if p.exists():
            try:
                df = reader(p)
                logger.debug("loaded %s (%d rows)", p, len(df))
                break
            except Exception:
                logger.warning("could not read %s; trying the next format", p)
    if df is None or df.empty:
        rows = []
        for rj in sorted((d / "cells").glob("*/row.json")):
            import json

            with __import__("contextlib").suppress(Exception):
                rows.append(json.loads(rj.read_text(encoding="utf-8")))
        df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"sweep '{sweep}' contains no results")

    if "status" not in df:
        df["status"] = "ok"
    n_failed = int((df["status"] != "ok").sum())
    if n_failed:
        logger.warning(
            "sweep '%s': %d/%d cells failed; they are %s from the analysis",
            sweep, n_failed, len(df), "excluded" if drop_failed else "kept",
        )
    if drop_failed:
        df = df[df["status"] == "ok"].copy()
    if "pairing" not in df and {"agent", "propulsion"} <= set(df.columns):
        df["pairing"] = df["agent"] + "@" + df["propulsion"]
    df.attrs["sweep"] = sweep
    df.attrs["results_dir"] = str(results_dir)
    df.attrs["n_failed"] = n_failed
    return df.reset_index(drop=True)


def load_curves(sweep: str, results_dir: str | Path = "results") -> pd.DataFrame:
    """All per-cell learning curves, concatenated and tagged with cell identity."""
    d = Path(results_dir) / sweep / "cells"
    frames = []
    for cell in sorted(d.glob("*/curve.csv")):
        try:
            c = pd.read_csv(cell)
        except Exception:
            continue
        import json

        meta_p = cell.parent / "row.json"
        meta = {}
        if meta_p.exists():
            with __import__("contextlib").suppress(Exception):
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
        for k in ("agent", "propulsion", "mission", "cost_model", "seed", "run_id"):
            c[k] = meta.get(k, cell.parent.name)
        frames.append(c)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["pairing"] = out["agent"].astype(str) + "@" + out["propulsion"].astype(str)
    return out


# --- bootstrap ---------------------------------------------------------------
def _bootstrap_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    return x[idx].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI of the mean.

    Delegates to :func:`propulsion_rl.economics.metrics.bootstrap_ci` when that
    module is present and returns something of a recognisable shape, so the two
    halves of the project report identical intervals; otherwise computes it
    here. A single observation returns a degenerate interval rather than NaN --
    it is honest about having no information about the spread.
    """
    x = np.asarray([v for v in values if _finite(v)], dtype=np.float64)
    if x.size == 0:
        return (float("nan"),) * 3
    m = float(x.mean())
    if x.size == 1:
        return m, m, m
    ext = _external_bootstrap_ci(x, alpha, n_boot)
    if ext is not None:
        return m, ext[0], ext[1]
    boots = _bootstrap_means(x, n_boot, np.random.default_rng(seed))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return m, float(lo), float(hi)


_EXTERNAL_CI_STATE: dict[str, Any] = {"checked": False, "fn": None}


def _external_bootstrap_ci(
    x: np.ndarray, alpha: float, n_boot: int
) -> tuple[float, float] | None:
    """Use :func:`propulsion_rl.economics.metrics.bootstrap_ci` when available.

    Its ``BootstrapCI`` result carries ``lo``/``hi``; several calling
    conventions are tried so a signature change in the economics module
    degrades to the local implementation instead of taking the analysis down.
    """
    if not _EXTERNAL_CI_STATE["checked"]:
        _EXTERNAL_CI_STATE["checked"] = True
        try:
            from propulsion_rl.economics.metrics import bootstrap_ci as ext

            _EXTERNAL_CI_STATE["fn"] = ext
        except Exception:
            logger.debug("economics.metrics.bootstrap_ci unavailable; using local")
    fn = _EXTERNAL_CI_STATE["fn"]
    if fn is None:
        return None
    for call in (
        lambda: fn(x, n_boot, alpha),
        lambda: fn(x, alpha=alpha),
        lambda: fn(x),
    ):
        try:
            out = call()
        except Exception:
            continue
        lo, hi = getattr(out, "lo", None), getattr(out, "hi", None)
        if lo is None or hi is None:
            vals = np.asarray(out, dtype=np.float64).reshape(-1)
            if vals.size == 2:
                lo, hi = vals[0], vals[1]
            elif vals.size == 3:
                lo, hi = vals[1], vals[2]
            else:
                continue
        if _finite(lo) and _finite(hi):
            return float(lo), float(hi)
    _EXTERNAL_CI_STATE["fn"] = None  # do not keep paying for a bad signature
    return None


# --- aggregation -------------------------------------------------------------
def aggregate(
    df: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    group: Sequence[str] = GROUP_COLS,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Mean + bootstrap CI per group, for each metric. One row per group."""
    group = [g for g in group if g in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for keys, sub in df.groupby(list(group), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group, keys))
        row["n_seeds"] = int(len(sub))
        for m in metrics:
            mean, lo, hi = bootstrap_ci(
                sub[m].to_numpy(dtype=float, na_value=np.nan),
                alpha=alpha, n_boot=n_boot,
            )
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def rank_pairings(
    df: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    *,
    sort_by: str = "score",
    alpha: float = 0.05,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """The headline table: one row per (mission, propulsion, agent).

    Every metric is reported as mean with a bootstrap CI. ``score`` is the
    composite normalised score (see :func:`normalized_score`) used to order the
    table, because there is no single natural unit in which success rate, trip
    time and dollars per kilogram are comparable, and pretending otherwise by
    ranking on one of them alone is how a study ends up recommending a policy
    that arrives fast and never succeeds.
    """
    agg = aggregate(df, metrics, GROUP_COLS, alpha=alpha, n_boot=n_boot)
    if agg.empty:
        return agg
    agg = normalized_score(agg, metrics=metrics)
    agg["pairing"] = agg["agent"].astype(str) + "@" + agg["propulsion"].astype(str)
    # Compute budget columns belong in the headline: they are what stops a
    # planner's win from being read as a free one.
    for col in ("act_ms_per_step", "wall_train_s", "train_env_steps",
                "planning_sim_steps", "num_parameters"):
        if col in df.columns:
            agg = agg.merge(
                df.groupby(list(GROUP_COLS), dropna=False)[col].mean().rename(col),
                left_on=list(GROUP_COLS), right_index=True, how="left",
            )
    key = sort_by if sort_by in agg.columns else f"{sort_by}_mean"
    if key in agg.columns:
        asc = not METRIC_DIRECTION.get(sort_by, True)
        agg = agg.sort_values(["mission", "propulsion", key],
                              ascending=[True, True, asc])
    return agg.reset_index(drop=True)


def normalized_score(
    agg: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    weights: Mapping[str, float] | None = None,
    *,
    within: Sequence[str] = ("mission",),
) -> pd.DataFrame:
    """Add a ``score`` column in [0, 1]: weighted, direction-corrected min-max.

    Normalisation happens **within a mission** by default. Trip times differ by
    two orders of magnitude between GEO station keeping and a Mars cargo run, so
    a score normalised across missions would measure the mission, not the
    controller.
    """
    weights = dict(weights or DEFAULT_SCORE_WEIGHTS)
    out = agg.copy()
    cols = [m for m in metrics if f"{m}_mean" in out.columns and m in weights]
    if not cols:
        out["score"] = np.nan
        return out
    within = [w for w in within if w in out.columns]
    parts = []
    for _, sub in (out.groupby(list(within), sort=False) if within else [((), out)]):
        sub = sub.copy()
        total_w = 0.0
        acc = np.zeros(len(sub))
        for m in cols:
            v = sub[f"{m}_mean"].to_numpy(dtype=float)
            finite = np.isfinite(v)
            if finite.sum() == 0:
                continue
            lo, hi = np.nanmin(v[finite]), np.nanmax(v[finite])
            unit = np.full(len(v), 0.5) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
            unit = np.where(finite, unit, 0.0)
            if not METRIC_DIRECTION.get(m, True):
                unit = 1.0 - unit
            w = float(weights[m])
            acc += w * unit
            total_w += w
        sub["score"] = acc / total_w if total_w > 0 else np.nan
        parts.append(sub)
    return pd.concat(parts).loc[out.index]


# --- the direct answers ------------------------------------------------------
def best_agent_per_propulsion(
    df: pd.DataFrame,
    *,
    metric: str = "score",
    by_mission: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Which controller should you fly on each propulsion system?

    One row per propulsion system (or per propulsion x mission when
    ``by_mission``), naming the winner, the runner-up, the paired difference
    with its bootstrap CI, the Holm-adjusted p-value and an explicit verdict.
    A winner whose margin does not clear the CI is reported as a tie, because
    that is what the data says.
    """
    return _winner_table(
        df, group=("propulsion",) + (("mission",) if by_mission else ()),
        over="agent", metric=metric, alpha=alpha,
    )


def best_propulsion_per_mission(
    df: pd.DataFrame,
    *,
    metric: str = "score",
    per_agent: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Which propulsion system should fly each mission?

    By default the controller is marginalised out (best achievable per
    propulsion system, averaged over the agents that were run), which is the
    hardware question. ``per_agent=True`` keeps the controller fixed instead.
    """
    return _winner_table(
        df, group=("mission",) + (("agent",) if per_agent else ()),
        over="propulsion", metric=metric, alpha=alpha,
    )


def _winner_table(
    df: pd.DataFrame,
    *,
    group: Sequence[str],
    over: str,
    metric: str,
    alpha: float,
) -> pd.DataFrame:
    higher_better = METRIC_DIRECTION.get(metric, True)
    per_seed = _per_seed_metric(df, metric)
    group = [g for g in group if g in per_seed.columns]
    rows = []
    comparisons: list[dict[str, Any]] = []
    for keys, sub in per_seed.groupby(list(group), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        means = sub.groupby(over)["value"].mean()
        if means.empty:
            continue
        order = means.sort_values(ascending=not higher_better)
        winner = str(order.index[0])
        runner = str(order.index[1]) if len(order) > 1 else ""
        row: dict[str, Any] = dict(zip(group, keys))
        m, lo, hi = bootstrap_ci(
            sub.loc[sub[over] == winner, "value"].to_numpy(float), alpha=alpha
        )
        row.update({
            "metric": metric, "winner": winner, "winner_mean": m,
            "winner_lo": lo, "winner_hi": hi,
            "runner_up": runner, "n_candidates": int(len(order)),
        })
        if runner:
            cmp_ = paired_comparison(
                sub[sub[over] == winner], sub[sub[over] == runner],
                value="value", pair_on="seed", higher_is_better=higher_better,
                alpha=alpha,
            )
            row.update({
                "runner_up_mean": float(order.iloc[1]),
                "diff": cmp_["diff"], "diff_lo": cmp_["diff_lo"],
                "diff_hi": cmp_["diff_hi"], "effect_size": cmp_["effect_size"],
                "p_value": cmp_["p_value"], "n_pairs": cmp_["n_pairs"],
                "test": cmp_["test"],
            })
            comparisons.append(row)
        else:
            row.update({"runner_up_mean": np.nan, "diff": np.nan,
                        "p_value": np.nan, "n_pairs": 0, "test": "none"})
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm over the family of "winner beats runner-up" claims in this table.
    mask = out["p_value"].notna()
    out.loc[mask, "p_holm"] = holm_bonferroni(out.loc[mask, "p_value"].to_numpy())
    out["significant"] = out["p_holm"].notna() & (out["p_holm"] < alpha)
    out["verdict"] = [_verdict(r, alpha) for _, r in out.iterrows()]
    return out.reset_index(drop=True)


def _verdict(row: pd.Series, alpha: float) -> str:
    if not row.get("runner_up"):
        return f"{row['winner']} (only candidate)"
    p = row.get("p_holm")
    lo, hi = row.get("diff_lo", np.nan), row.get("diff_hi", np.nan)
    ci = f"[{lo:+.3g}, {hi:+.3g}]" if _finite(lo) and _finite(hi) else "[n/a]"
    if _finite(p) and p < alpha:
        return (f"{row['winner']} > {row['runner_up']} "
                f"(diff {row['diff']:+.3g} {ci}, Holm p={p:.3g}, dz="
                f"{row.get('effect_size', float('nan')):.2f})")
    pstr = f"Holm p={p:.3g}" if _finite(p) else "p unavailable"
    return (f"TIE: {row['winner']} vs {row['runner_up']} not separable "
            f"(diff {row['diff']:+.3g} {ci}, {pstr}) -- do not report a winner")


def _per_seed_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Long-form (group cols, seed, value) frame for one metric.

    ``score`` is not a per-seed quantity in the raw results, so it is computed
    per seed here: normalise within (mission, seed) so that every seed
    contributes its own complete ranking and the pairing survives.
    """
    keep = [c for c in ("mission", "propulsion", "agent", "cost_model", "seed")
            if c in df.columns]
    if metric != "score":
        if metric not in df.columns:
            raise KeyError(f"metric '{metric}' not in the results")
        out = df[keep + [metric]].rename(columns={metric: "value"})
        return out.dropna(subset=["value"])
    parts = []
    by = [c for c in ("mission", "seed") if c in df.columns]
    for _, sub in df.groupby(by, sort=False) if by else [((), df)]:
        agg = sub[keep].copy()
        for m in DEFAULT_SCORE_WEIGHTS:
            agg[f"{m}_mean"] = sub[m] if m in sub.columns else np.nan
        scored = normalized_score(agg, metrics=tuple(DEFAULT_SCORE_WEIGHTS),
                                  within=())
        parts.append(scored[keep + ["score"]].rename(columns={"score": "value"}))
    return pd.concat(parts, ignore_index=True).dropna(subset=["value"])


# --- statistics --------------------------------------------------------------
def paired_comparison(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    value: str = "value",
    pair_on: str = "seed",
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_boot: int = _DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """Compare two conditions, paired on seed where possible.

    Returns the signed difference (oriented so positive always means "a is
    better"), a bootstrap CI on that difference, an effect size, and a p-value.
    Paired seeds get a paired t-test and a paired bootstrap; unpaired data falls
    back to Welch, which does not assume equal variances and is the right
    default when the two conditions are different algorithms.
    """
    from scipy import stats

    sign = 1.0 if higher_is_better else -1.0
    if pair_on in a.columns and pair_on in b.columns:
        merged = a[[pair_on, value]].merge(
            b[[pair_on, value]], on=pair_on, suffixes=("_a", "_b")
        )
    else:
        merged = pd.DataFrame()
    if len(merged) >= 2:
        d = sign * (merged[f"{value}_a"].to_numpy(float)
                    - merged[f"{value}_b"].to_numpy(float))
        d = d[np.isfinite(d)]
        n = d.size
        rng = np.random.default_rng(12345)
        boots = _bootstrap_means(d, n_boot, rng)
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
        if n > 1 and sd > 0:
            t_stat, p = stats.ttest_rel(merged[f"{value}_a"], merged[f"{value}_b"])
            p = float(p)
        else:
            # Zero variance: either identical (p=1) or a deterministic gap. A
            # deterministic gap across all seeds is real, so report the
            # bootstrap's own p-value rather than a t-test that cannot run.
            p = 1.0 if abs(float(np.mean(d))) < 1e-12 else 0.0
        dz = float(np.mean(d) / sd) if sd > 0 else (
            0.0 if abs(float(np.mean(d))) < 1e-12 else math.inf
        )
        return {"diff": float(np.mean(d)), "diff_lo": float(lo), "diff_hi": float(hi),
                "effect_size": dz, "p_value": p, "n_pairs": int(n), "test": "paired",
                "bootstrap_p": _bootstrap_p(boots)}

    xa = a[value].to_numpy(float)
    xb = b[value].to_numpy(float)
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    if xa.size < 2 or xb.size < 2:
        return {"diff": sign * (float(np.mean(xa)) - float(np.mean(xb)))
                if xa.size and xb.size else np.nan,
                "diff_lo": np.nan, "diff_hi": np.nan, "effect_size": np.nan,
                "p_value": np.nan, "n_pairs": 0, "test": "insufficient-data",
                "bootstrap_p": np.nan}
    t_stat, p = stats.ttest_ind(xa, xb, equal_var=False)
    rng = np.random.default_rng(12345)
    boots = sign * (_bootstrap_means(xa, n_boot, rng) - _bootstrap_means(xb, n_boot, rng))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    s_pooled = math.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2.0)
    d = sign * (float(np.mean(xa)) - float(np.mean(xb)))
    return {"diff": d, "diff_lo": float(lo), "diff_hi": float(hi),
            "effect_size": d / s_pooled if s_pooled > 0 else np.nan,
            "p_value": float(p), "n_pairs": 0, "test": "welch",
            "bootstrap_p": _bootstrap_p(boots)}


def _bootstrap_p(boots: np.ndarray) -> float:
    """Two-sided bootstrap p: how often the resampled difference changes sign."""
    if boots.size == 0:
        return float("nan")
    frac = float(np.mean(boots <= 0.0))
    return float(min(1.0, 2.0 * min(frac, 1.0 - frac)))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Uniformly more powerful than Bonferroni at identical family-wise error
    control, which matters when the family is the several hundred comparisons a
    full matrix generates.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def compare_agents(
    df: pd.DataFrame,
    *,
    metric: str = "score",
    within: Sequence[str] = ("mission", "propulsion"),
    alpha: float = 0.05,
    reference: str | None = None,
) -> pd.DataFrame:
    """All pairwise agent comparisons inside each (mission, propulsion) cell.

    Holm correction is applied over the whole returned family -- every
    comparison in the matrix, not per cell -- because the winner is chosen by
    looking at all of them. ``reference`` restricts the family to comparisons
    against one agent (e.g. the scripted baseline), which is both the more
    interesting question and a much smaller family, so it has more power.
    """
    per_seed = _per_seed_metric(df, metric)
    within = [w for w in within if w in per_seed.columns]
    higher_better = METRIC_DIRECTION.get(metric, True)
    rows = []
    for keys, sub in per_seed.groupby(list(within), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        agents = sorted(sub["agent"].unique())
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                if reference is not None and reference not in (a, b):
                    continue
                # Orient so "a" is the non-reference challenger when asked.
                lhs, rhs = (b, a) if reference == a else (a, b)
                res = paired_comparison(
                    sub[sub["agent"] == lhs], sub[sub["agent"] == rhs],
                    higher_is_better=higher_better, alpha=alpha,
                )
                row = dict(zip(within, keys))
                row.update({"agent_a": lhs, "agent_b": rhs, "metric": metric})
                row.update(res)
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mask = out["p_value"].notna()
    out.loc[mask, "p_holm"] = holm_bonferroni(out.loc[mask, "p_value"].to_numpy(),
                                              alpha=alpha)
    out["significant"] = out["p_holm"].notna() & (out["p_holm"] < alpha)
    out["conclusion"] = np.where(
        out["significant"],
        out["agent_a"] + " beats " + out["agent_b"],
        "not significant",
    )
    return out.sort_values("p_holm", na_position="last").reset_index(drop=True)


# --- Pareto ------------------------------------------------------------------
def pareto_table(
    df: pd.DataFrame,
    *,
    objectives: Sequence[str] = ("trip_time_days", "cost_per_kg_delivered",
                                 "success_rate"),
    by_mission: bool = True,
) -> pd.DataFrame:
    """Non-dominated pairings over (trip time, $/kg, success rate).

    There is no single best pairing: a nuclear stage that arrives in 90 days at
    ten times the cost per kilogram and an electric tug that takes two years for
    a tenth of the price are both correct answers to different questions. The
    Pareto front is the honest form of the recommendation, and the ``on_front``
    column is what a decision-maker actually reads.
    """
    agg = aggregate(df, metrics=objectives, group=GROUP_COLS, n_boot=500)
    if agg.empty:
        return agg
    agg["pairing"] = agg["agent"].astype(str) + "@" + agg["propulsion"].astype(str)
    frames = []
    groups = agg.groupby("mission", sort=True) if by_mission and "mission" in agg \
        else [("all", agg)]
    senses = tuple("max" if METRIC_DIRECTION.get(o, True) else "min"
                   for o in objectives)
    for mission, sub in groups:
        cols = [f"{o}_mean" for o in objectives]
        s = sub.copy()
        s["on_front"] = _pareto_mask(s[cols], senses)
        s["mission"] = mission
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["mission", "on_front"], ascending=[True, False])


_PARETO_STATE: dict[str, Any] = {"checked": False, "fn": None}


def _pareto_mask(frame: pd.DataFrame, senses: Sequence[str]) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``frame`` under ``senses``.

    Delegates to :func:`propulsion_rl.economics.metrics.pareto_front` so the two
    halves of the project agree on the definition of dominance (including how
    non-finite objectives are handled), with a local O(n^2) implementation as
    the fallback -- n is the number of pairings, so the cost is irrelevant.
    """
    cols = list(frame.columns)
    ext = _external_pareto(frame, cols, senses)
    if ext is not None:
        return ext
    pts = frame.to_numpy(dtype=float)
    flip = np.array([-1.0 if s == "max" else 1.0 for s in senses])
    oriented = pts * flip
    finite = np.all(np.isfinite(oriented), axis=1)
    mask = np.zeros(len(pts), dtype=bool)
    idx = np.where(finite)[0] if finite.any() else np.arange(len(pts))
    sub = oriented[idx]
    keep = np.ones(len(sub), dtype=bool)
    for i in range(len(sub)):
        dominated = np.all(sub <= sub[i], axis=1) & np.any(sub < sub[i], axis=1)
        if dominated.any():
            keep[i] = False
    mask[idx] = keep
    return mask


def _external_pareto(
    frame: pd.DataFrame, objectives: Sequence[str], senses: Sequence[str]
) -> np.ndarray | None:
    if not _PARETO_STATE["checked"]:
        _PARETO_STATE["checked"] = True
        try:
            from propulsion_rl.economics.metrics import pareto_front

            _PARETO_STATE["fn"] = pareto_front
        except Exception:
            logger.debug("economics.metrics.pareto_front unavailable; using local")
    fn = _PARETO_STATE["fn"]
    if fn is None:
        return None
    n = len(frame)
    try:
        out = fn(frame.to_dict("records"), list(objectives), list(senses))
    except Exception:
        logger.debug("economics.metrics.pareto_front rejected the call; using local",
                     exc_info=True)
        _PARETO_STATE["fn"] = None
        return None
    mask = _coerce_mask(getattr(out, "mask", out), n)
    if mask is None:
        _PARETO_STATE["fn"] = None
    return mask


def _coerce_mask(out: Any, n: int) -> np.ndarray | None:
    """Accept a boolean mask or an index array."""
    arr = np.asarray(out)
    if arr.dtype == bool and arr.shape == (n,):
        return arr
    if arr.ndim == 1 and arr.size and np.issubdtype(arr.dtype, np.integer):
        if arr.max() < n and arr.min() >= 0:
            mask = np.zeros(n, dtype=bool)
            mask[arr] = True
            return mask
    return None


# --- the research question ---------------------------------------------------
@dataclass
class InteractionResult:
    """Is the best controller the same on every propulsion system?"""

    mission: str = "all"
    metric: str = "score"
    best_per_propulsion: pd.DataFrame = field(default_factory=pd.DataFrame)
    anova: pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    kendall_w: float = float("nan")
    n_distinct_winners: int = 0
    interaction_p: float = float("nan")
    interaction_eta2: float = float("nan")
    verdict: str = ""

    def as_text(self) -> str:
        lines = [f"interaction analysis [{self.mission} / {self.metric}]",
                 f"  distinct winners across propulsion systems: "
                 f"{self.n_distinct_winners}",
                 f"  Kendall's W (agreement of agent rankings): "
                 f"{self.kendall_w:.3f}",
                 f"  agent x propulsion interaction: p={self.interaction_p:.4g}, "
                 f"partial eta^2={self.interaction_eta2:.3f}",
                 f"  VERDICT: {self.verdict}"]
        return "\n".join(lines)


def interaction_effect(
    df: pd.DataFrame,
    *,
    metric: str = "score",
    mission: str | None = None,
    alpha: float = 0.05,
) -> InteractionResult:
    """Test whether the agent ranking depends on the propulsion system.

    This is the study's research question, so it gets three independent views
    rather than one number:

    1. **The winners table.** Which agent tops each propulsion system. If it is
       the same agent everywhere, there is no interaction worth the name,
       whatever the F-test says.
    2. **A two-way ANOVA** on agent, propulsion and their interaction, with a
       partial eta-squared so a statistically detectable but tiny interaction is
       not oversold. Requires >= 2 seeds per cell; with fewer, the residual has
       no degrees of freedom and the p-value is reported as NaN rather than
       fabricated.
    3. **Rank agreement.** Kendall's W across propulsion systems over the agent
       ranking, plus the pairwise Spearman matrix. W near 1 means one controller
       ordering holds everywhere -- the "just use PPO" answer. Low W with a
       significant interaction is the interesting result: the right controller
       depends on the hardware.
    """
    data = df if mission is None else df[df["mission"] == mission]
    per_seed = _per_seed_metric(data, metric)
    if per_seed.empty:
        return InteractionResult(mission=mission or "all", metric=metric,
                                 verdict="no data")
    higher_better = METRIC_DIRECTION.get(metric, True)

    cell = per_seed.groupby(["propulsion", "agent"])["value"].mean().unstack()
    if cell.shape[0] < 2 or cell.shape[1] < 2:
        return InteractionResult(
            mission=mission or "all", metric=metric,
            verdict="need at least 2 propulsion systems and 2 agents",
        )
    winners = (cell.idxmax(axis=1) if higher_better else cell.idxmin(axis=1))
    best = pd.DataFrame({
        "propulsion": winners.index,
        "best_agent": winners.to_numpy(),
        "best_value": [cell.loc[p, a] for p, a in winners.items()],
    })
    # How much is on the table by picking per-propulsion instead of globally?
    global_best = (cell.mean(axis=0).idxmax() if higher_better
                   else cell.mean(axis=0).idxmin())
    best["global_best_agent"] = global_best
    best["value_if_global"] = [cell.loc[p, global_best] for p in winners.index]
    best["gain_from_specialising"] = (
        (best["best_value"] - best["value_if_global"]) * (1 if higher_better else -1)
    )

    anova = two_way_anova(per_seed, value="value", a="agent", b="propulsion")
    inter = anova[anova["source"] == "agent:propulsion"]
    p_int = float(inter["p"].iloc[0]) if len(inter) else float("nan")
    eta_int = float(inter["partial_eta_sq"].iloc[0]) if len(inter) else float("nan")

    w, spearman = _rank_agreement(cell, higher_better)
    n_winners = int(best["best_agent"].nunique())

    verdict = _interaction_verdict(n_winners, len(cell), w, p_int, eta_int, alpha,
                                   float(best["gain_from_specialising"].mean()))
    return InteractionResult(
        mission=mission or "all", metric=metric, best_per_propulsion=best,
        anova=anova, rank_correlation=spearman, kendall_w=w,
        n_distinct_winners=n_winners, interaction_p=p_int,
        interaction_eta2=eta_int, verdict=verdict,
    )


def _interaction_verdict(
    n_winners: int, n_prop: int, w: float, p: float, eta: float,
    alpha: float, mean_gain: float,
) -> str:
    sig = _finite(p) and p < alpha
    if n_winners == 1:
        return (
            "NO INTERACTION: the same agent wins on every propulsion system. "
            "Choose the controller once and vary the hardware."
        )
    if sig and _finite(eta) and eta >= 0.06:
        return (
            f"REAL INTERACTION: {n_winners} different agents win across {n_prop} "
            f"propulsion systems (Kendall's W={w:.2f}, p={p:.3g}, partial "
            f"eta^2={eta:.2f}); specialising the controller to the hardware is "
            f"worth {mean_gain:+.3g} on this metric. The pairing matters."
        )
    if sig:
        return (
            f"DETECTABLE BUT SMALL: the interaction is significant (p={p:.3g}) but "
            f"partial eta^2={eta:.2f} is negligible; {n_winners} nominal winners "
            f"across {n_prop} systems is probably ranking noise, not a real "
            f"hardware-controller pairing effect."
        )
    return (
        f"NOT SIGNIFICANT: {n_winners} nominal winners across {n_prop} propulsion "
        f"systems, but the agent x propulsion interaction does not clear "
        f"significance (p={p:.3g}, Kendall's W={w:.2f}). The apparent pairing "
        f"effect is consistent with seed noise -- do not claim one."
    )


def two_way_anova(
    long: pd.DataFrame, *, value: str = "value", a: str = "agent",
    b: str = "propulsion",
) -> pd.DataFrame:
    """Two-way ANOVA with interaction, unweighted-means for unbalanced cells.

    The design is balanced by construction (every cell gets the same seeds), but
    a failed cell unbalances it, so cell means are used for the effect sums of
    squares with the harmonic mean cell count. That is the standard unweighted
    means analysis; it is approximate when unbalanced, and the returned
    ``balanced`` flag says which case you are in.
    """
    from scipy import stats

    d = long[[a, b, value]].dropna()
    if d.empty:
        return pd.DataFrame()
    cells = d.groupby([a, b])[value]
    cell_mean = cells.mean().unstack()
    counts = cells.count().unstack()
    if cell_mean.isna().to_numpy().any():
        # Missing cells make the interaction undefined; drop offending rows/cols.
        cell_mean = cell_mean.dropna(axis=0, how="any").dropna(axis=1, how="any")
        counts = counts.loc[cell_mean.index, cell_mean.columns]
    if cell_mean.shape[0] < 2 or cell_mean.shape[1] < 2:
        return pd.DataFrame()

    na, nb = cell_mean.shape
    n_arr = counts.to_numpy(dtype=float)
    balanced = bool(np.all(n_arr == n_arr.flat[0]))
    n_h = float(na * nb / np.sum(1.0 / np.clip(n_arr, 1e-9, None)))

    M = cell_mean.to_numpy(dtype=float)
    grand = float(M.mean())
    row_m, col_m = M.mean(axis=1), M.mean(axis=0)
    ss_a = n_h * nb * float(np.sum((row_m - grand) ** 2))
    ss_b = n_h * na * float(np.sum((col_m - grand) ** 2))
    resid = M - row_m[:, None] - col_m[None, :] + grand
    ss_ab = n_h * float(np.sum(resid**2))

    within = 0.0
    df_w = 0
    for (ka, kb), grp in d.groupby([a, b]):
        if ka not in cell_mean.index or kb not in cell_mean.columns:
            continue
        v = grp[value].to_numpy(float)
        within += float(np.sum((v - v.mean()) ** 2))
        df_w += max(0, v.size - 1)

    rows = []
    ms_w = within / df_w if df_w > 0 else float("nan")
    for source, ss, dfn in (
        (a, ss_a, na - 1), (b, ss_b, nb - 1), (f"{a}:{b}", ss_ab, (na - 1) * (nb - 1))
    ):
        ms = ss / dfn if dfn > 0 else float("nan")
        f = ms / ms_w if _finite(ms_w) and ms_w > 0 else float("nan")
        p = float(stats.f.sf(f, dfn, df_w)) if _finite(f) and df_w > 0 else float("nan")
        rows.append({
            "source": source, "ss": ss, "df": dfn, "ms": ms, "F": f, "p": p,
            "partial_eta_sq": ss / (ss + within) if (ss + within) > 0 else float("nan"),
        })
    rows.append({"source": "residual", "ss": within, "df": df_w, "ms": ms_w,
                 "F": float("nan"), "p": float("nan"),
                 "partial_eta_sq": float("nan")})
    out = pd.DataFrame(rows)
    out.attrs["balanced"] = balanced
    if not balanced:
        logger.warning(
            "unbalanced ANOVA design (failed cells?); using unweighted means with "
            "harmonic n=%.2f -- the p-values are approximate", n_h,
        )
    return out


def _rank_agreement(cell: pd.DataFrame, higher_better: bool) -> tuple[float, pd.DataFrame]:
    """Kendall's W over agent rankings, plus the pairwise Spearman matrix."""
    from scipy import stats

    ranks = cell.rank(axis=1, ascending=not higher_better)
    m, n = ranks.shape  # m = judges (propulsion systems), n = items (agents)
    if m < 2 or n < 2:
        return float("nan"), pd.DataFrame()
    R = ranks.sum(axis=0).to_numpy(float)
    s = float(np.sum((R - R.mean()) ** 2))
    w = 12.0 * s / (m**2 * (n**3 - n)) if n > 1 else float("nan")

    idx = list(cell.index)
    sp = pd.DataFrame(np.eye(m), index=idx, columns=idx)
    for i in range(m):
        for j in range(i + 1, m):
            with __import__("contextlib").suppress(Exception):
                rho = float(stats.spearmanr(ranks.iloc[i], ranks.iloc[j]).statistic)
                sp.iloc[i, j] = sp.iloc[j, i] = rho
    return float(w), sp


# --- compute fairness --------------------------------------------------------
def compute_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    """What each agent spent, so a win can be read against its price.

    Environment steps are equalised by the runner; wall-clock and per-decision
    inference cost are not, and cannot be. This table is how the reader tells
    "better policy" from "more compute".
    """
    cols = {
        "train_env_steps": "mean", "eval_env_steps": "mean",
        "wall_train_s": "mean", "wall_eval_s": "mean", "act_ms_per_step": "mean",
        "planning_sim_steps": "mean", "n_updates": "mean",
        "num_parameters": "max",
    }
    have = {k: v for k, v in cols.items() if k in df.columns}
    if not have or "agent" not in df.columns:
        return pd.DataFrame()
    out = df.groupby("agent").agg(have)
    if "act_ms_per_step" in out.columns:
        floor = out["act_ms_per_step"].replace(0, np.nan).min()
        out["inference_cost_vs_cheapest"] = out["act_ms_per_step"] / floor
    budget = out["train_env_steps"] if "train_env_steps" in out else None
    if budget is not None:
        learners = budget[budget > 0]
        out["equal_budget_ok"] = (
            bool(learners.nunique() <= 1) if len(learners) else True
        )
    return out.reset_index()


# --- reporting ---------------------------------------------------------------
def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not _finite(mean):
        return "n/a"
    if not (_finite(lo) and _finite(hi)):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} [{lo:.{digits}g}, {hi:.{digits}g}]"


def format_table(agg: pd.DataFrame, metrics: Sequence[str] = HEADLINE_METRICS
                 ) -> pd.DataFrame:
    """Collapse mean/lo/hi triples into readable ``x [lo, hi]`` strings."""
    keep = [c for c in ("mission", "propulsion", "agent", "pairing", "n_seeds",
                        "score") if c in agg.columns]
    out = agg[keep].copy()
    if "score" in out.columns:
        out["score"] = out["score"].map(lambda v: f"{v:.3f}" if _finite(v) else "n/a")
    for m in metrics:
        if f"{m}_mean" not in agg.columns:
            continue
        out[m] = [
            format_ci(r[f"{m}_mean"], r.get(f"{m}_lo", np.nan), r.get(f"{m}_hi", np.nan))
            for _, r in agg.iterrows()
        ]
    return out


def summary_report(df: pd.DataFrame, *, metric: str = "score") -> str:
    """One text block answering the study's questions, caveats included."""
    lines: list[str] = []
    sweep = df.attrs.get("sweep", "sweep")
    lines.append(f"=== {sweep}: {len(df)} rows, "
                 f"{df['seed'].nunique() if 'seed' in df else 0} seeds ===")
    n_failed = df.attrs.get("n_failed", 0)
    if n_failed:
        lines.append(f"!! {n_failed} cells failed and are excluded")
    if "seed" in df and df["seed"].nunique() < 3:
        lines.append(
            "!! fewer than 3 seeds: confidence intervals are decorative, and no "
            "claim of a winner in this report should be believed"
        )
    lines.append("")
    lines.append("-- best agent per propulsion system --")
    for _, r in best_agent_per_propulsion(df, metric=metric).iterrows():
        lines.append(f"  {r['propulsion']:<16} {r['verdict']}")
    lines.append("")
    lines.append("-- best propulsion per mission --")
    for _, r in best_propulsion_per_mission(df, metric=metric).iterrows():
        lines.append(f"  {r['mission']:<20} {r['verdict']}")
    lines.append("")
    inter = interaction_effect(df, metric=metric)
    lines.append(inter.as_text())
    return "\n".join(lines)


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "HEADLINE_METRICS",
    "METRIC_DIRECTION",
    "InteractionResult",
    "aggregate",
    "best_agent_per_propulsion",
    "best_propulsion_per_mission",
    "bootstrap_ci",
    "compare_agents",
    "compute_budget_table",
    "format_ci",
    "format_table",
    "holm_bonferroni",
    "interaction_effect",
    "load_curves",
    "load_sweep",
    "normalized_score",
    "paired_comparison",
    "pareto_table",
    "rank_pairings",
    "summary_report",
    "two_way_anova",
]
