"""The part of the benchmark that produces the answer.

:mod:`~propulsion_rl.experiments.matrix` decides which (agent x propulsion x
mission) cells are worth running, :mod:`~propulsion_rl.experiments.runner`
executes them under an explicit fair-comparison protocol,
:mod:`~propulsion_rl.experiments.analysis` turns the rows into ranked,
significance-tested conclusions, :mod:`~propulsion_rl.experiments.plots` draws
them, and :mod:`~propulsion_rl.experiments.cli` is the ``propulsion-rl``
command.

Submodules are resolved lazily: importing this package must not drag in pandas,
scipy or matplotlib on a training node that only needs the runner.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExperimentMatrix",
    "ExperimentSpec",
    "PlausibilityRule",
    "RunnerConfig",
    "SweepConfig",
    "analysis",
    "cli",
    "load_sweep",
    "matrix",
    "plots",
    "run_cell",
    "run_sweep",
    "runner",
]

_LAZY: dict[str, tuple[str, str | None]] = {
    "matrix": (".matrix", None),
    "runner": (".runner", None),
    "analysis": (".analysis", None),
    "plots": (".plots", None),
    "cli": (".cli", None),
    "ExperimentMatrix": (".matrix", "ExperimentMatrix"),
    "ExperimentSpec": (".matrix", "ExperimentSpec"),
    "PlausibilityRule": (".matrix", "PlausibilityRule"),
    "SweepConfig": (".matrix", "SweepConfig"),
    "RunnerConfig": (".runner", "RunnerConfig"),
    "run_cell": (".runner", "run_cell"),
    "run_sweep": (".runner", "run_sweep"),
    "load_sweep": (".analysis", "load_sweep"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    from importlib import import_module

    mod = import_module(module_name, __name__)
    return mod if attr is None else getattr(mod, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
