"""Figures for a sweep. Matplotlib is imported lazily, inside the functions.

Two rules shape this module. First, importing
:mod:`propulsion_rl.experiments.plots` must never fail on a headless machine or
one without matplotlib: plotting is a reporting convenience, and a training node
that cannot import the package because it lacks a font cache is a real outage.
So every import happens inside a function, behind :func:`_pyplot`, and the
backend is forced to ``Agg`` before ``pyplot`` is touched.

Second, every figure states its own uncertainty. Learning curves get bootstrap
CI bands across seeds rather than a single seed's jagged line; the pairing
heatmap is annotated with the seed count it rests on. A figure that looks
confident about five noisy seeds is a worse artefact than no figure.

All output lands in ``results/<sweep>/figures/``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FIGSIZE = (10.0, 6.0)
_DPI = 130


def _pyplot() -> Any:
    """Import pyplot with a non-interactive backend, or raise a clear error."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is not installed; install the 'viz' extra "
            "(pip install 'propulsion-rl[viz]') to generate figures"
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": _DPI, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })
    return plt


def figures_dir(sweep: str, results_dir: str | Path = "results") -> Path:
    d = Path(results_dir) / sweep / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _pyplot().close(fig)
    logger.info("wrote %s", path)
    return path


def _boot_band(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return (np.nan,) * 3
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def _slug(*parts: Any) -> str:
    s = "-".join(str(p) for p in parts if p not in (None, ""))
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s) or "figure"


# --- learning curves ---------------------------------------------------------
def learning_curves(
    curves: pd.DataFrame,
    out_dir: str | Path,
    *,
    metric: str = "val_return",
    split_by: str = "propulsion",
    mission: str | None = None,
) -> list[Path]:
    """One figure per propulsion system: validation metric vs environment steps.

    The x axis is environment steps, not wall-clock and not updates, because
    that is the budget the comparison equalises. Bands are bootstrap 95% CIs
    across seeds; a line with no band had one seed and should not be trusted.
    """
    plt = _pyplot()
    out_dir = Path(out_dir)
    if curves.empty or metric not in curves.columns:
        logger.warning("no learning curves to plot (metric=%s)", metric)
        return []
    df = curves if mission is None else curves[curves["mission"] == mission]
    df = df[df["env_step"] > 0] if (df["env_step"] > 0).any() else df
    paths: list[Path] = []
    groups = df.groupby(split_by, sort=True) if split_by in df.columns else [("all", df)]
    for key, sub in groups:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        drawn = 0
        for agent, g in sub.groupby("agent", sort=True):
            pts = []
            for step, gg in g.groupby("env_step", sort=True):
                m, lo, hi = _boot_band(gg[metric].to_numpy(float))
                pts.append((step, m, lo, hi, gg["seed"].nunique()))
            if not pts:
                continue
            arr = np.array([[p[0], p[1], p[2], p[3]] for p in pts], dtype=float)
            n_seeds = max(p[4] for p in pts)
            line, = ax.plot(arr[:, 0], arr[:, 1], marker="o", ms=3,
                            label=f"{agent} (n={n_seeds})")
            if n_seeds > 1:
                ax.fill_between(arr[:, 0], arr[:, 2], arr[:, 3], alpha=0.18,
                                color=line.get_color(), linewidth=0)
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xlabel("environment steps (the equalised budget)")
        ax.set_ylabel(metric.replace("_", " "))
        title = f"learning curves - {key}"
        ax.set_title(title + (f" / {mission}" if mission else ""))
        ax.legend(fontsize=7, ncol=2, frameon=False)
        paths.append(_save(fig, out_dir / f"learning_curve_{_slug(key, mission)}.png"))
    return paths


# --- pairing heatmap ---------------------------------------------------------
def pairing_heatmap(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metric: str = "score",
    mission: str | None = None,
    annotate: bool = True,
) -> list[Path]:
    """Agent x propulsion grid coloured by normalised score.

    This is the picture the whole study is for: if the bright cells form
    vertical stripes, one controller wins everywhere and the pairing does not
    matter; if they scatter, the pairing does. Empty cells are pairings the
    matrix deliberately excluded as physically implausible -- see
    ``results/<sweep>/excluded.csv`` for the reason each one is missing.
    """
    from . import analysis as A

    plt = _pyplot()
    out_dir = Path(out_dir)
    data = df if mission is None else df[df["mission"] == mission]
    if data.empty:
        return []
    paths: list[Path] = []
    missions = [mission] if mission else sorted(data["mission"].dropna().unique())
    for miss in missions:
        sub = data[data["mission"] == miss] if "mission" in data.columns else data
        per_seed = A._per_seed_metric(sub, metric)
        if per_seed.empty:
            continue
        grid = per_seed.pivot_table(index="agent", columns="propulsion",
                                    values="value", aggfunc="mean")
        counts = per_seed.pivot_table(index="agent", columns="propulsion",
                                      values="value", aggfunc="count")
        grid = grid.sort_index()
        fig, ax = plt.subplots(
            figsize=(1.0 + 0.9 * grid.shape[1], 1.2 + 0.42 * grid.shape[0])
        )
        masked = np.ma.masked_invalid(grid.to_numpy(dtype=float))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=45, ha="right")
        ax.set_yticks(range(grid.shape[0]), grid.index)
        ax.set_title(f"{metric} by pairing - {miss}\n(grey = excluded pairing)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=metric, shrink=0.85)
        if annotate:
            vals = grid.to_numpy(dtype=float)
            vmid = np.nanmean(vals) if np.isfinite(vals).any() else 0.0
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = vals[i, j]
                    if not np.isfinite(v):
                        continue
                    n = counts.to_numpy()[i, j] if counts is not None else 0
                    ax.text(j, i, f"{v:.2f}\nn={int(n)}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v < vmid else "black")
        paths.append(_save(fig, out_dir / f"heatmap_{_slug(metric, miss)}.png"))
    return paths


# --- Pareto ------------------------------------------------------------------
def pareto_scatter(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    x: str = "trip_time_days",
    y: str = "cost_per_kg_delivered",
    size: str = "success_rate",
    mission: str | None = None,
) -> list[Path]:
    """Trip time against cost per kilogram, marker size = success rate.

    Front members are labelled and joined; dominated pairings are drawn faint.
    The label is the pairing, because the recommendation is a pairing.
    """
    from . import analysis as A

    plt = _pyplot()
    out_dir = Path(out_dir)
    table = A.pareto_table(df, objectives=(x, y, size))
    if table.empty:
        return []
    paths: list[Path] = []
    missions = [mission] if mission else sorted(table["mission"].dropna().unique())
    for miss in missions:
        sub = table[table["mission"] == miss]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        xs = sub[f"{x}_mean"].to_numpy(float)
        ys = sub[f"{y}_mean"].to_numpy(float)
        ss = sub[f"{size}_mean"].to_numpy(float)
        front = sub["on_front"].to_numpy(bool)
        sizes = 30 + 220 * np.nan_to_num(ss, nan=0.0)
        ax.scatter(xs[~front], ys[~front], s=sizes[~front], c="#b0b0b0",
                   alpha=0.55, edgecolors="none", label="dominated")
        ax.scatter(xs[front], ys[front], s=sizes[front], c="#c0392b",
                   edgecolors="black", linewidths=0.5, label="Pareto front", zorder=3)
        order = np.argsort(xs[front])
        if order.size > 1:
            ax.plot(xs[front][order], ys[front][order], "--", color="#c0392b",
                    linewidth=1.0, alpha=0.7, zorder=2)
        for i in np.where(front)[0]:
            if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
                continue
            ax.annotate(sub["pairing"].iloc[i], (xs[i], ys[i]), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points")
        if np.isfinite(ys).any() and np.nanmax(ys) > 0 and np.nanmin(ys[ys > 0]) > 0:
            if np.nanmax(ys) / max(np.nanmin(ys[ys > 0]), 1e-9) > 50:
                ax.set_yscale("log")
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(f"Pareto front - {miss}  (marker size = {size})")
        ax.legend(frameon=False, fontsize=8)
        paths.append(_save(fig, out_dir / f"pareto_{_slug(miss)}.png"))
    return paths


# --- trajectory --------------------------------------------------------------
def trajectory_plot(
    telemetry: pd.DataFrame | str | Path,
    out_path: str | Path,
    *,
    title: str = "trajectory",
) -> Path | None:
    """Orbit radius, speed and accumulated delta-v against mission time.

    Draws a true in-plane trajectory when the telemetry carries positions or a
    polar angle; otherwise the radius-time history, which is what a low-thrust
    spiral is actually read from.
    """
    plt = _pyplot()
    df = _read_telemetry(telemetry)
    if df is None or df.empty:
        return None
    t_days = _time_days(df)

    xcol, ycol = _first_present(df, ("x_m", "pos_x", "r_x")), _first_present(
        df, ("y_m", "pos_y", "r_y")
    )
    theta = _first_present(df, ("theta", "true_anomaly", "arg_latitude", "phase"))
    have_plane = (xcol and ycol) or (theta and "radius_m" in df)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    ax = axes[0][0]
    if have_plane:
        if xcol and ycol:
            X, Y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
        else:
            r = df["radius_m"].to_numpy(float)
            th = df[theta].to_numpy(float)
            X, Y = r * np.cos(th), r * np.sin(th)
        scale = max(np.nanmax(np.abs(np.r_[X, Y])), 1.0)
        unit, div = ("Gm", 1e9) if scale > 1e9 else ("Mm", 1e6)
        ax.plot(X / div, Y / div, lw=1.0)
        ax.plot([0], [0], marker="*", ms=12, color="#e0a800")
        ax.set_xlabel(f"x [{unit}]")
        ax.set_ylabel(f"y [{unit}]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("in-plane trajectory")
    else:
        ax.plot(t_days, df.get("radius_m", pd.Series(dtype=float)) / 1e6, lw=1.2)
        ax.set_xlabel("mission time [days]")
        ax.set_ylabel("orbit radius [Mm]")
        ax.set_title("radius history (spiral)")

    for axis, col, label, scale in (
        (axes[0][1], "speed_m_s", "speed [km/s]", 1e3),
        (axes[1][0], "delta_v_m_s", "accumulated $\\Delta v$ [km/s]", 1e3),
        (axes[1][1], "mass_kg", "vehicle mass [kg]", 1.0),
    ):
        if col in df.columns:
            axis.plot(t_days, df[col].to_numpy(float) / scale, lw=1.2)
        axis.set_xlabel("mission time [days]")
        axis.set_ylabel(label)
    if "propellant_kg" in df.columns:
        tw = axes[1][1].twinx()
        tw.plot(t_days, df["propellant_kg"], color="#c0392b", lw=1.0, ls="--")
        tw.set_ylabel("propellant [kg]", color="#c0392b")
        tw.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, Path(out_path))


# --- telemetry dashboard -----------------------------------------------------
def telemetry_dashboard(
    telemetry: pd.DataFrame | str | Path,
    out_path: str | Path,
    *,
    title: str = "episode telemetry",
) -> Path | None:
    """Thrust, Isp, power, wear and temperature against mission time.

    This is the panel that explains *why* a pairing scored the way it did: a
    policy that looks great on delta-v while pinning the thruster against its
    thermal limit is visible here and nowhere else in the tables. Panels whose
    channel is missing from the telemetry are skipped rather than faked.
    """
    plt = _pyplot()
    df = _read_telemetry(telemetry)
    if df is None or df.empty:
        return None
    t = _time_days(df)

    temp_col = _first_present(
        df, ("temperature_k", "chamber_temperature_k", "wall_temperature_k",
             "temp_k", "T_k")
    )
    panels: list[tuple[str, list[tuple[str, str, float]]]] = [
        ("thrust [mN]", [("thrust_n", "thrust", 1e-3)]),
        ("specific impulse [s]", [("isp_s", "Isp", 1.0)]),
        ("power [kW]", [("power_draw_w", "draw", 1e3),
                        ("power_available_w", "available", 1e3)]),
        ("efficiency / throttle-limit", [("efficiency", "efficiency", 1.0)]),
        ("wear fraction", [("wear_fraction", "wear", 1.0)]),
        ("constraint cost", [("constraint_cost", "cost", 1.0),
                             ("worst_margin", "worst margin", 1.0)]),
    ]
    if temp_col:
        panels.insert(4, ("temperature [K]", [(temp_col, "temperature", 1.0)]))
    panels = [p for p in panels if any(c in df.columns for c, _, _ in p[1])]
    if not panels:
        return None

    n = len(panels)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 2.1 * rows + 1.0), sharex=True)
    flat = np.atleast_1d(axes).ravel()
    for ax, (label, series) in zip(flat, panels):
        for col, name, scale in series:
            if col in df.columns:
                ax.plot(t, df[col].to_numpy(float) / scale, lw=1.0, label=name)
        ax.set_ylabel(label, fontsize=8)
        if len(series) > 1:
            ax.legend(fontsize=6.5, frameon=False)
    # Mark the steps where some limit bound the commanded thrust.
    if "throttled_by" in df.columns:
        limited = df["throttled_by"].astype(str).isin(["none", "", "nan"]).to_numpy()
        for ax in flat[:len(panels)]:
            ax.fill_between(t, *ax.get_ylim(), where=~limited, color="#e74c3c",
                            alpha=0.08, linewidth=0, step="mid")
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for ax in flat[max(0, len(panels) - 2):len(panels)]:
        ax.set_xlabel("mission time [days]")
    fig.suptitle(f"{title}\n(red shading: a limit was binding)", fontsize=10)
    fig.tight_layout()
    return _save(fig, Path(out_path))


# --- compute-fairness figure -------------------------------------------------
def compute_vs_performance(
    df: pd.DataFrame, out_dir: str | Path, *, metric: str = "score"
) -> list[Path]:
    """Per-decision inference cost against performance, per agent.

    The figure exists to stop "wins on score" from being read as "wins". A
    planner far to the right bought its position with compute the deployed
    system may not have.
    """
    from . import analysis as A

    plt = _pyplot()
    if "act_ms_per_step" not in df.columns:
        return []
    per_seed = A._per_seed_metric(df, metric)
    if per_seed.empty:
        return []
    perf = per_seed.groupby("agent")["value"].mean()
    cost = df.groupby("agent")["act_ms_per_step"].mean()
    joined = pd.concat([perf.rename("perf"), cost.rename("cost")], axis=1).dropna()
    if joined.empty:
        return []
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.scatter(joined["cost"], joined["perf"], s=60, c="#2c3e50")
    for name, r in joined.iterrows():
        ax.annotate(name, (r["cost"], r["perf"]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    if (joined["cost"] > 0).all() and joined["cost"].max() / max(
        joined["cost"].min(), 1e-9
    ) > 20:
        ax.set_xscale("log")
    ax.set_xlabel("inference cost [ms per decision]")
    ax.set_ylabel(f"{metric} (mean over cells)")
    ax.set_title("performance against the compute it cost\n"
                 "(equal environment-step budget; unequal compute)")
    return [_save(fig, Path(out_dir) / f"compute_vs_{_slug(metric)}.png")]


# --- orchestration -----------------------------------------------------------
def plot_all(
    sweep: str,
    results_dir: str | Path = "results",
    *,
    metric: str = "score",
    max_dashboards: int = 12,
) -> list[Path]:
    """Every figure for a sweep, into ``results/<sweep>/figures/``."""
    from . import analysis as A

    out = figures_dir(sweep, results_dir)
    df = A.load_sweep(sweep, results_dir)
    paths: list[Path] = []

    for fn, args in (
        (pairing_heatmap, dict(metric=metric)),
        (pareto_scatter, {}),
        (compute_vs_performance, dict(metric=metric)),
    ):
        try:
            paths.extend(fn(df, out, **args))
        except Exception:
            logger.exception("%s failed; continuing", fn.__name__)

    try:
        curves = A.load_curves(sweep, results_dir)
        if not curves.empty:
            paths.extend(learning_curves(curves, out))
    except Exception:
        logger.exception("learning_curves failed; continuing")

    cells = Path(results_dir) / sweep / "cells"
    tele = sorted(cells.glob("*/telemetry_best.csv"))
    if len(tele) > max_dashboards:
        # Prefer one exemplar per pairing rather than the first N alphabetically.
        tele = _representative(tele, df, max_dashboards)
    for p in tele:
        name = p.parent.name
        try:
            r = telemetry_dashboard(p, out / f"telemetry_{_slug(name)}.png",
                                    title=f"telemetry - {name}")
            if r:
                paths.append(r)
            r = trajectory_plot(p, out / f"trajectory_{_slug(name)}.png",
                                title=f"trajectory - {name}")
            if r:
                paths.append(r)
        except Exception:
            logger.exception("telemetry figures for %s failed; continuing", name)
    logger.info("wrote %d figures to %s", len(paths), out)
    return paths


def _representative(paths: Sequence[Path], df: pd.DataFrame, k: int) -> list[Path]:
    """Pick up to k telemetry files spread across pairings, best cells first."""
    if "run_id" not in df.columns:
        return list(paths)[:k]
    rank = {}
    score_col = "success_rate" if "success_rate" in df.columns else None
    for _, r in df.iterrows():
        v = float(r[score_col]) if score_col and np.isfinite(
            pd.to_numeric(r[score_col], errors="coerce")
        ) else 0.0
        rank[str(r["run_id"])] = (str(r.get("pairing", "")), v)
    chosen: list[Path] = []
    seen: set[str] = set()
    scored = sorted(
        paths, key=lambda p: -rank.get(p.parent.name.split("-")[-1], ("", 0.0))[1]
    )
    for p in scored:
        pairing = rank.get(p.parent.name.split("-")[-1], ("", 0.0))[0]
        if pairing and pairing in seen:
            continue
        seen.add(pairing)
        chosen.append(p)
        if len(chosen) >= k:
            break
    return chosen


# --- small helpers -----------------------------------------------------------
def _read_telemetry(src: pd.DataFrame | str | Path) -> pd.DataFrame | None:
    if isinstance(src, pd.DataFrame):
        return src
    p = Path(src)
    if not p.exists():
        logger.warning("no telemetry at %s", p)
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        logger.exception("could not read telemetry %s", p)
        return None


def _time_days(df: pd.DataFrame) -> np.ndarray:
    if "t_s" in df.columns:
        return df["t_s"].to_numpy(float) / 86400.0
    if "step" in df.columns:
        return df["step"].to_numpy(float)
    return np.arange(len(df), dtype=float)


def _first_present(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for n in names:
        if n in df.columns and np.isfinite(
            pd.to_numeric(df[n], errors="coerce")
        ).any():
            return n
    return None


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


__all__ = [
    "compute_vs_performance",
    "figures_dir",
    "learning_curves",
    "pairing_heatmap",
    "pareto_scatter",
    "plot_all",
    "telemetry_dashboard",
    "trajectory_plot",
]
