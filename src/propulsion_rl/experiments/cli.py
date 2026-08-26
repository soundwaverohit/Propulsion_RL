"""``propulsion-rl`` command line entry point.

Subcommands
-----------
``list``      what is registered: propulsion systems, missions, agents, costs.
``run``       execute a sweep from a YAML config.
``eval``      re-evaluate one saved checkpoint under the held-out test protocol.
``analyze``   ranking, significance, Pareto front and the interaction analysis.
``plot``      every figure for a sweep.
``demo``      the whole pipeline on a tiny sweep, in under a minute.

This is the one place in the package that writes to stdout; everything under
``propulsion_rl`` logs through :mod:`logging` instead, so a sweep can be piped
into a file without the library fighting over the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS = "results"


# --- console -----------------------------------------------------------------
class Console:
    """Thin wrapper over ``rich`` that degrades to ``print`` when it is absent."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._rich = None
        try:
            from rich.console import Console as RichConsole

            self._rich = RichConsole(highlight=False, soft_wrap=False)
        except Exception:
            pass

    def print(self, *args: Any, **kw: Any) -> None:
        if self.quiet:
            return
        if self._rich is not None:
            self._rich.print(*args, **kw)
        else:
            text = " ".join(str(a) for a in args)
            for tag in ("[bold]", "[/bold]", "[dim]", "[/dim]", "[red]", "[/red]",
                        "[green]", "[/green]", "[yellow]", "[/yellow]", "[cyan]",
                        "[/cyan]", "[bold red]", "[bold green]", "[bold cyan]"):
                text = text.replace(tag, "")
            print(text)

    def rule(self, title: str = "") -> None:
        if self.quiet:
            return
        if self._rich is not None:
            self._rich.rule(title)
        else:
            print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))

    def table(self, df: Any, title: str = "", max_rows: int = 60,
              floatfmt: str = "{:.4g}") -> None:
        """Render a DataFrame, as a rich table when possible."""
        if self.quiet or df is None or len(df) == 0:
            if not self.quiet:
                self.print(f"[dim](no rows for {title or 'table'})[/dim]")
            return
        shown = df.head(max_rows)
        if self._rich is None:
            import pandas as pd

            with pd.option_context("display.width", 200,
                                   "display.max_columns", 40,
                                   "display.max_colwidth", 46):
                if title:
                    print(f"\n{title}")
                print(shown.to_string(index=False))
            if len(df) > max_rows:
                print(f"... {len(df) - max_rows} more rows")
            return
        from rich.table import Table

        t = Table(title=title or None, header_style="bold cyan",
                  show_lines=False, title_justify="left")
        for c in shown.columns:
            t.add_column(str(c), overflow="fold", max_width=42)
        for _, row in shown.iterrows():
            t.add_row(*[_fmt(v, floatfmt) for v in row])
        self._rich.print(t)
        if len(df) > max_rows:
            self.print(f"[dim]... {len(df) - max_rows} more rows[/dim]")


def _fmt(v: Any, floatfmt: str) -> str:
    import numpy as np

    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f != f:
            return "-"
        return floatfmt.format(f)
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


# --- config discovery --------------------------------------------------------
def _repo_root() -> Path:
    # src/propulsion_rl/experiments/cli.py -> repo root
    return Path(__file__).resolve().parents[3]


def find_config(name: str) -> Path:
    """Resolve a config by path, or by name against ``configs/``."""
    p = Path(name)
    if p.exists():
        return p
    for cand in (
        Path.cwd() / "configs" / name,
        Path.cwd() / "configs" / f"{name}.yaml",
        _repo_root() / "configs" / name,
        _repo_root() / "configs" / f"{name}.yaml",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"no config '{name}'; looked in ./configs and {_repo_root() / 'configs'}"
    )


# --- subcommands -------------------------------------------------------------
def cmd_list(args: argparse.Namespace, con: Console) -> int:
    import pandas as pd

    from ..core.registry import AGENT, COST_MODEL, MISSION, PROPULSION
    from .matrix import PLAUSIBILITY_RULES, propulsion_family

    for label, reg in (("propulsion systems", PROPULSION), ("missions", MISSION),
                       ("agents", AGENT), ("cost models", COST_MODEL)):
        rows = []
        for n in reg.names():
            meta = reg.meta(n)
            row = {"name": n}
            if reg is PROPULSION:
                row["family"] = propulsion_family(n)
            row.update({k: v for k, v in meta.items() if k != "family"})
            rows.append(row)
        con.table(pd.DataFrame(rows), title=f"{label} ({len(reg)})")
    if args.rules:
        con.rule("plausibility rules (pairings the matrix refuses to run)")
        for r in PLAUSIBILITY_RULES:
            con.print(f"[bold]{r.name}[/bold]\n  {r.reason}\n")
    return 0


def cmd_run(args: argparse.Namespace, con: Console) -> int:
    from .matrix import ExperimentMatrix, SweepConfig
    from .runner import RunnerConfig, run_sweep

    cfg_path = find_config(args.config)
    sweep = SweepConfig.from_yaml(cfg_path)
    if args.name:
        sweep.name = args.name
    if args.results_dir:
        sweep.results_dir = args.results_dir
    matrix = ExperimentMatrix.from_config(sweep)

    con.rule(f"sweep '{sweep.name}'  ({cfg_path})")
    if sweep.description:
        con.print(f"[dim]{sweep.description.strip()}[/dim]\n")
    con.print(matrix.summary())

    if args.dry_run:
        con.print("\n[yellow]--dry-run: nothing executed[/yellow]")
        return 0

    rc = RunnerConfig.from_sweep(
        sweep, workers=args.workers, force=args.force,
        results_dir=args.results_dir, log_level=args.log_level,
    )
    t0 = time.perf_counter()

    def _tick(row: dict, done: int, total: int) -> None:
        status = row.get("status", "?")
        mark = "[green]ok[/green]" if status == "ok" else "[bold red]FAILED[/bold red]"
        sr = row.get("success_rate")
        extra = f" success={sr:.2f}" if isinstance(sr, float) and sr == sr else ""
        con.print(f"  [{done}/{total}] {mark} {row.get('pairing', '?')} "
                  f"/{row.get('mission', '?')} seed={row.get('seed', '?')}{extra}")

    df = run_sweep(matrix, rc, on_row=_tick)
    n_failed = int((df["status"] != "ok").sum()) if "status" in df else 0
    con.rule("done")
    con.print(f"{len(df)} rows in {time.perf_counter() - t0:.1f}s -> "
              f"{rc.sweep_dir / 'results.csv'}")
    if n_failed:
        con.print(f"[bold red]{n_failed} cells failed[/bold red] "
                  f"(tracebacks in {rc.sweep_dir}/cells/*/status.json)")
    if not args.no_analyze:
        _print_analysis(df, con, metric=args.metric)
    return 0


def cmd_eval(args: argparse.Namespace, con: Console) -> int:
    import pandas as pd

    from .runner import RunnerConfig, evaluate_checkpoint

    row = evaluate_checkpoint(
        args.checkpoint, episodes=args.episodes, seed=args.seed,
        cfg=RunnerConfig(results_dir=args.results_dir or _DEFAULT_RESULTS),
    )
    con.rule(f"evaluation of {args.checkpoint}")
    keys = ["agent", "propulsion", "mission", "cost_model", "seed", "n_episodes",
            "success_rate", "return", "progress", "trip_time_days", "delta_v_m_s",
            "propellant_kg", "wear_fraction", "constraint_violations",
            "cost_per_kg_delivered", "termination_reason_mode"]
    con.table(pd.DataFrame([{k: row.get(k) for k in keys if k in row}]),
              title="held-out test episodes (deterministic, frozen normalisation)")
    return 0


def cmd_analyze(args: argparse.Namespace, con: Console) -> int:
    from . import analysis as A

    df = A.load_sweep(args.sweep, args.results_dir or _DEFAULT_RESULTS)
    _print_analysis(df, con, metric=args.metric, full=True, out=args.out)
    return 0


def _print_analysis(
    df: Any, con: Console, *, metric: str = "score", full: bool = False,
    out: str | None = None,
) -> None:
    from . import analysis as A

    con.rule("headline: ranked pairings")
    ranked = A.rank_pairings(df)
    con.table(A.format_table(ranked), title="mean [95% bootstrap CI] over seeds",
              max_rows=40 if full else 20)

    con.rule("best agent per propulsion system")
    bap = A.best_agent_per_propulsion(df, metric=metric)
    for _, r in bap.iterrows():
        col = "green" if r.get("significant") else "yellow"
        con.print(f"  [bold]{r['propulsion']:<16}[/bold] [{col}]{r['verdict']}[/{col}]")

    con.rule("best propulsion per mission")
    bpm = A.best_propulsion_per_mission(df, metric=metric)
    for _, r in bpm.iterrows():
        col = "green" if r.get("significant") else "yellow"
        con.print(f"  [bold]{r['mission']:<20}[/bold] [{col}]{r['verdict']}[/{col}]")

    con.rule("Pareto front: trip time vs $/kg vs success rate")
    par = A.pareto_table(df)
    if len(par):
        cols = [c for c in ("mission", "pairing", "trip_time_days_mean",
                            "cost_per_kg_delivered_mean", "success_rate_mean",
                            "on_front") if c in par.columns]
        con.table(par[par["on_front"]][cols] if not full else par[cols],
                  title="non-dominated pairings" if not full else "all pairings")

    con.rule("the research question: is the best method the same everywhere?")
    inter = A.interaction_effect(df, metric=metric)
    con.print(inter.as_text())
    if len(inter.best_per_propulsion):
        con.table(inter.best_per_propulsion, title="winner by propulsion system")
    if full and len(inter.anova):
        con.table(inter.anova, title="two-way ANOVA (agent x propulsion)")

    if full:
        con.rule("pairwise agent comparisons (Holm-corrected)")
        cmp_ = A.compare_agents(df, metric=metric)
        cols = [c for c in ("mission", "propulsion", "agent_a", "agent_b", "diff",
                            "diff_lo", "diff_hi", "effect_size", "p_value",
                            "p_holm", "conclusion") if c in cmp_.columns]
        con.table(cmp_[cols], title="sorted by adjusted p-value", max_rows=30)
        n_sig = int(cmp_["significant"].sum()) if len(cmp_) else 0
        con.print(f"  {n_sig}/{len(cmp_)} comparisons survive Holm correction")

        con.rule("compute budget (equal env steps, unequal everything else)")
        con.table(A.compute_budget_table(df), title="per agent")

    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(p, index=False)
        con.print(f"\nwrote ranked table -> {p}")


def cmd_plot(args: argparse.Namespace, con: Console) -> int:
    from .plots import plot_all

    paths = plot_all(args.sweep, args.results_dir or _DEFAULT_RESULTS,
                     metric=args.metric)
    con.rule(f"{len(paths)} figures")
    for p in paths:
        con.print(f"  {p}")
    return 0


def cmd_demo(args: argparse.Namespace, con: Console) -> int:
    """Run the smoke sweep end to end so a new user sees the whole pipeline."""
    con.rule("propulsion-rl demo")
    con.print("[dim]Running the smoke sweep: a handful of cells, a few seeds, "
              "tiny budgets. This is the whole pipeline in miniature.[/dim]\n")
    run_args = argparse.Namespace(
        config=args.config, workers=args.workers, force=args.force,
        dry_run=False, results_dir=args.results_dir, name=args.name,
        log_level=args.log_level, no_analyze=False, metric="score",
    )
    rc = cmd_run(run_args, con)
    if rc != 0:
        return rc
    sweep_name = args.name or Path(find_config(args.config)).stem
    if not args.no_plots:
        try:
            cmd_plot(argparse.Namespace(sweep=sweep_name, metric="score",
                                        results_dir=args.results_dir), con)
        except Exception as exc:
            con.print(f"[yellow]figures skipped: {exc}[/yellow]")
    con.rule("next steps")
    results = args.results_dir or _DEFAULT_RESULTS
    con.print(
        f"  propulsion-rl analyze --sweep {sweep_name} --results-dir {results}\n"
        f"  propulsion-rl plot    --sweep {sweep_name} --results-dir {results}\n"
        f"  propulsion-rl run --config configs/electric_vs_nuclear.yaml --workers 8"
    )
    return 0


# --- argument parsing --------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="propulsion-rl",
        description="A benchmark pairing RL methods with spacecraft propulsion.",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--quiet", action="store_true", help="suppress stdout output")
    sub = p.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list", help="show the registries")
    lst.add_argument("--rules", action="store_true",
                     help="also print the pairing plausibility rules")
    lst.set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="execute a sweep")
    run.add_argument("--config", required=True,
                     help="path, or a name resolved against configs/")
    run.add_argument("--workers", type=int, default=None,
                     help="processes for cell-level parallelism (default: config)")
    run.add_argument("--force", action="store_true",
                     help="re-run cells whose results already exist")
    run.add_argument("--dry-run", action="store_true",
                     help="print the plan, including excluded pairings, and stop")
    run.add_argument("--results-dir", default=None)
    run.add_argument("--name", default=None, help="override the sweep name")
    run.add_argument("--metric", default="score")
    run.add_argument("--no-analyze", action="store_true",
                     help="skip the summary tables after the sweep")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="re-evaluate a saved checkpoint")
    ev.add_argument("--checkpoint", required=True,
                    help="a cell directory, or its best/ or final/ subdirectory")
    ev.add_argument("--episodes", type=int, default=None)
    ev.add_argument("--seed", type=int, default=None)
    ev.add_argument("--results-dir", default=None)
    ev.set_defaults(func=cmd_eval)

    an = sub.add_parser("analyze", help="rank, test and summarise a sweep")
    an.add_argument("--sweep", required=True)
    an.add_argument("--results-dir", default=None)
    an.add_argument("--metric", default="score")
    an.add_argument("--out", default=None, help="write the ranked table to a CSV")
    an.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plot", help="generate every figure for a sweep")
    pl.add_argument("--sweep", required=True)
    pl.add_argument("--results-dir", default=None)
    pl.add_argument("--metric", default="score")
    pl.set_defaults(func=cmd_plot)

    dm = sub.add_parser("demo", help="fast end-to-end smoke run")
    dm.add_argument("--config", default="smoke.yaml")
    dm.add_argument("--workers", type=int, default=None)
    dm.add_argument("--force", action="store_true")
    dm.add_argument("--results-dir", default=None)
    dm.add_argument("--name", default=None)
    dm.add_argument("--no-plots", action="store_true")
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    con = Console(quiet=args.quiet)
    # Importing the package populates the registries.
    import propulsion_rl  # noqa: F401

    try:
        return int(args.func(args, con) or 0)
    except KeyboardInterrupt:
        con.print("\n[yellow]interrupted; partial results are on disk[/yellow]")
        return 130
    except FileNotFoundError as exc:
        con.print(f"[bold red]{exc}[/bold red]")
        return 2
    except Exception as exc:  # noqa: BLE001 -- top level, report and exit
        logger.exception("command failed")
        con.print(f"[bold red]{type(exc).__name__}: {exc}[/bold red]")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
