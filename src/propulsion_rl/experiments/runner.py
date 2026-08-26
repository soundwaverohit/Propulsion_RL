"""The fair-comparison engine.

This module exists because comparison studies of this shape usually go wrong in
the same five places, and every one of them produces a paper that reports a
winner that is not real. The protocol below is the study's central claim; the
rest of the package is instrumentation for it.

The protocol
------------

**1. Equal budget is measured in environment steps.**
Every learning agent in a cell group gets exactly ``spec.train_steps``
interactions with the environment -- not equal wall-clock, not an equal number
of gradient updates. Wall-clock rewards whoever ran on the faster machine;
equal-updates rewards whoever chose the largest batch. Environment steps are
the resource the science is about. Evaluation steps are counted separately
(``eval_env_steps``) and never charged against the training budget, so an agent
cannot buy performance by evaluating more often.

Because "cheap per step" and "cheap overall" are different claims, three
compute figures are recorded for every cell and all three are reported:
``train_env_steps`` (the budget), ``wall_train_s`` (real time), and
``act_seconds`` / ``planning_sim_steps`` (inference and planning compute). A
planner that wins only by spending 100x the compute per decision has not won in
a way that matters, and the table has to make that visible rather than hide it
behind a single reward column.

**2. Scripted agents get zero training budget and the identical evaluation.**
``agent.learns is False`` means no training loop at all -- giving a PID
controller 20k steps of "training" it ignores would just be a slower way to run
the same controller. It is evaluated on exactly the same held-out seeds, with
the same ``deterministic=True`` flag, through the same code path. Scripted
baselines are the reference the whole study rests on, so their numbers must be
produced by the same machinery, not by a shortcut.

**3. Seeds are paired across cells.**
A cell is repeated over N seed replicates (default 5). Replicate ``k`` uses the
same environment seed stream, the same agent seed and the same torch seed in
*every* cell, so ``ppo@hall_spt100`` replicate 3 and ``sac@hall_spt100``
replicate 3 saw the same initial conditions and the same episode sequence. That
makes the seed-level differences paired, which is what
:mod:`propulsion_rl.experiments.analysis` needs to run paired tests: paired
tests on 5 seeds have real power, unpaired ones do not. Single-seed results are
never reported; every headline number is a mean with a bootstrap CI.

**4. Three disjoint seed bands: train, validation, test.**
Training episodes draw seeds from ``[0, 4e8)``. Checkpoint selection ("which
policy was best?") evaluates on the *validation* band ``[7e8, 8e8)``. The
number that gets reported comes from a final run on the *test* band
``[9e8, 1e9)``, which nothing was selected on. Selecting the best checkpoint on
the same episodes you then report is test-set peeking, and with 5 seeds and a
noisy environment it manufactures winners reliably. Disjointness is asserted at
runtime, not assumed.

Every evaluation runs with ``deterministic=True`` and with observation
normalisation **frozen** and synchronised from the training environment.
Normalisation statistics that keep updating during evaluation leak information
across evaluation episodes and make results depend on evaluation order.

**5. Final and best policies are both kept and both reported.**
``final_*`` columns are the policy at the end of the budget; ``best_*`` columns
are the checkpoint that scored highest on validation. The headline columns are
whichever ``report_policy`` selects (default: best). Reporting only the best
checkpoint of an unstable method overstates it; reporting only the final policy
understates it. Both are in the row.

Operationally: cells are content-addressed (see
:class:`~propulsion_rl.experiments.matrix.ExperimentSpec`) and skipped when
already complete, a crashed cell is logged and marked failed without taking the
sweep with it, and every finished row is flushed to disk immediately -- a sweep
that dies 90% of the way through is still 90% of a result.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import pickle
import platform
import random
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.base import Agent, Transition
from ..core.registry import AGENT, COST_MODEL
from ..core.types import CANONICAL_ACTION_DIM, OBS_DIM
from .matrix import PROTOCOL_VERSION, ExperimentMatrix, ExperimentSpec, SweepConfig

logger = logging.getLogger(__name__)

# --- seed bands --------------------------------------------------------------
# Disjoint by construction and asserted at runtime. Training never sees an
# episode drawn from the validation or test bands.
TRAIN_SEED_LO, TRAIN_SEED_HI = 0, 400_000_000
VAL_SEED_LO, VAL_SEED_HI = 700_000_000, 800_000_000
TEST_SEED_LO, TEST_SEED_HI = 900_000_000, 1_000_000_000
_SEEDS_PER_REPLICATE = 10_000  # ceiling on episodes per eval band per replicate

#: Offsets that separate the different roles a replicate seed plays, so the
#: agent's parameter init is not correlated with the environment's episode
#: sampling for the same replicate.
_AGENT_SEED_OFFSET = 5_000_003
_TORCH_SEED_OFFSET = 7_000_019
_TRAIN_STREAM_OFFSET = 11


def train_episode_seeds(replicate: int, n: int) -> list[int]:
    """Training episode seeds for a replicate. Identical in every cell."""
    rng = np.random.default_rng([_TRAIN_STREAM_OFFSET, int(replicate)])
    return [int(x) for x in rng.integers(TRAIN_SEED_LO, TRAIN_SEED_HI, size=max(n, 1))]


def val_episode_seeds(replicate: int, n: int) -> list[int]:
    """Held-out validation seeds: used for checkpoint selection only."""
    base = VAL_SEED_LO + int(replicate) * _SEEDS_PER_REPLICATE
    return [base + i for i in range(n)]


def test_episode_seeds(replicate: int, n: int) -> list[int]:
    """Held-out test seeds: nothing is ever selected on these."""
    base = TEST_SEED_LO + int(replicate) * _SEEDS_PER_REPLICATE
    return [base + i for i in range(n)]


def _assert_disjoint(train: Sequence[int], val: Sequence[int], test: Sequence[int]) -> None:
    ts, vs, xs = set(train), set(val), set(test)
    for a, b, an, bn in ((ts, vs, "train", "val"), (ts, xs, "train", "test"),
                         (vs, xs, "val", "test")):
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"{an}/{bn} episode seeds overlap ({len(overlap)} shared) -- the "
                f"evaluation is contaminated; refusing to produce numbers"
            )


# --- configuration -----------------------------------------------------------
@dataclass
class RunnerConfig:
    """Execution options. Everything that affects the *numbers* lives in the
    spec (and therefore in the run_id); this holds only how the work is done."""

    sweep_name: str = "default"
    results_dir: str = "results"
    workers: int = 1
    force: bool = False
    save_telemetry: bool = True
    report_policy: str = "best"          # "best" | "final"
    selection_metric: str = "return"     # what the best checkpoint is best at
    normalize_obs: bool = True
    #: Reward normalisation is applied to the *training* environment only.
    #: Evaluation always sees raw rewards, so returns stay comparable.
    normalize_reward: bool = True
    max_episode_steps: int | None = None
    update_every: int = 1                # agent.update() cadence, env steps
    fail_fast: bool = False
    keep_checkpoints: bool = True
    progress: bool = True
    log_level: str = "INFO"

    @staticmethod
    def from_sweep(cfg: SweepConfig, **overrides: Any) -> "RunnerConfig":
        rc = RunnerConfig(
            sweep_name=cfg.name,
            results_dir=cfg.results_dir,
            workers=int(cfg.workers),
            save_telemetry=bool(cfg.save_telemetry),
            report_policy=cfg.report_policy,
            selection_metric=cfg.selection_metric,
        )
        env = dict(cfg.env or {})
        if "max_episode_steps" in env:
            rc.max_episode_steps = int(env["max_episode_steps"])
        for k, v in overrides.items():
            if v is not None and hasattr(rc, k):
                setattr(rc, k, v)
        return rc

    @property
    def sweep_dir(self) -> Path:
        return Path(self.results_dir) / self.sweep_name

    def cell_dir(self, spec: ExperimentSpec) -> Path:
        return self.sweep_dir / "cells" / spec.slug


# --- episode / evaluation records --------------------------------------------
@dataclass
class EpisodeRecord:
    """One evaluation episode, reduced to the columns the study ranks on."""

    seed: int = 0
    steps: int = 0
    ret: float = 0.0
    success: bool = False
    progress: float = 0.0
    delta_v_m_s: float = 0.0
    propellant_kg: float = 0.0
    trip_time_days: float = 0.0
    wear_fraction: float = 0.0
    constraint_violations: int = 0
    constraint_cost: float = 0.0
    terminal_error: float = 0.0
    payload_delivered_kg: float = 0.0
    throttle_mean: float = 0.0
    throttle_saturated_frac: float = 0.0
    throttle_off_frac: float = 0.0
    operating_point_mean: float = 0.0
    throttled_frac: float = 0.0
    throttled_by_top: str = "none"
    cost_per_kg_delivered: float = float("nan")
    cost_total: float = float("nan")
    cost_per_delta_v: float = float("nan")
    figure_of_merit: float = float("nan")
    termination_reason: str = "unknown"
    telemetry: list[dict[str, Any]] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("telemetry", None)
        return d


@dataclass
class EvalSummary:
    """Aggregate over one evaluation pass (a fixed set of held-out seeds)."""

    episodes: list[EpisodeRecord] = field(default_factory=list)
    env_steps: int = 0
    wall_s: float = 0.0
    act_s: float = 0.0

    #: Episode field -> column name. ``ret`` is published as ``return`` and
    #: ``steps`` as ``episode_steps`` so the row never carries two spellings of
    #: the same number.
    _MEAN_FIELDS = (
        "progress", "delta_v_m_s", "propellant_kg", "trip_time_days",
        "wear_fraction", "constraint_violations", "constraint_cost",
        "terminal_error", "payload_delivered_kg", "throttle_mean",
        "throttle_saturated_frac", "throttle_off_frac", "operating_point_mean",
        "throttled_frac",
    )
    _ECON_FIELDS = ("cost_per_kg_delivered", "cost_total", "cost_per_delta_v",
                    "figure_of_merit")

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return _mean([float(e.success) for e in self.episodes])

    @property
    def mean_return(self) -> float:
        return _mean([e.ret for e in self.episodes])

    def metrics(self, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {
            f"{prefix}n_episodes": self.n,
            f"{prefix}success_rate": self.success_rate,
            f"{prefix}return": self.mean_return,
            f"{prefix}return_std": _std([e.ret for e in self.episodes]),
        }
        for f in self._MEAN_FIELDS:
            out[f"{prefix}{f}"] = _mean([getattr(e, f) for e in self.episodes])
        # Economics can be inf/nan for a failed episode; average over the finite
        # ones and report how many contributed, rather than emitting nan.
        for f in self._ECON_FIELDS:
            vals = [getattr(e, f) for e in self.episodes]
            finite = [v for v in vals if _finite(v)]
            out[f"{prefix}{f}"] = _mean(finite) if finite else float("nan")
            out[f"{prefix}{f}_n_finite"] = len(finite)
        reasons = [e.termination_reason for e in self.episodes]
        out[f"{prefix}termination_reason_mode"] = _mode(reasons)
        out[f"{prefix}throttled_by_mode"] = _mode(
            [e.throttled_by_top for e in self.episodes]
        )
        return out

    def best_episode(self, metric: str = "return") -> EpisodeRecord | None:
        """The episode whose telemetry is worth keeping on disk.

        Success first, then the selection metric: a beautiful trace of a failed
        episode is less useful than the trace of the run that worked.
        """
        if not self.episodes:
            return None
        return max(
            self.episodes,
            key=lambda e: (float(e.success), _episode_value(e, metric)),
        )


def _episode_value(e: EpisodeRecord, metric: str) -> float:
    if metric == "success":
        return float(e.success)
    if metric == "progress":
        return e.progress
    if metric == "neg_cost_per_kg":
        v = e.cost_per_kg_delivered
        return -v if _finite(v) else -1e18
    if metric == "neg_trip_time":
        return -e.trip_time_days
    return e.ret


def _selection_value(summary: EvalSummary, metric: str) -> float:
    """Scalar the checkpoint selector maximises, computed on validation only."""
    if summary.n == 0:
        return -math.inf
    if metric == "success":
        # Ties on success rate broken by return, so a 0%-success early policy
        # still makes progress up the ladder.
        return summary.success_rate * 1e6 + summary.mean_return
    if metric == "progress":
        return _mean([e.progress for e in summary.episodes])
    if metric == "neg_cost_per_kg":
        vals = [e.cost_per_kg_delivered for e in summary.episodes if
                _finite(e.cost_per_kg_delivered)]
        return -_mean(vals) if vals else -math.inf
    return summary.mean_return


# --- environment plumbing ----------------------------------------------------
_WRAPPER_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _WRAPPER_WARNED:
        _WRAPPER_WARNED.add(key)
        logger.warning(msg, *args)


def _base_env(env: Any) -> Any:
    """Innermost environment, walking the wrapper chain."""
    seen = 0
    while hasattr(env, "env") and seen < 32:
        env = env.env
        seen += 1
    return getattr(env, "unwrapped", env)


#: YAML spellings the runner accepts for a field ``EnvConfig`` calls something
#: else. Keeps sweep configs readable without coupling them to the env's names.
_ENV_CONFIG_ALIASES = {"max_episode_steps": "max_steps"}


def _build_env_config(env_config: Mapping[str, Any]) -> Any:
    """Construct an ``EnvConfig`` from YAML, tolerating unknown keys.

    The env module is owned by another part of the project, so its field set can
    change under us. Unknown keys are dropped with a warning instead of taking
    the sweep down at cell 400.
    """
    if not env_config:
        return None
    from propulsion_rl.envs.propulsion_env import EnvConfig

    resolved = {_ENV_CONFIG_ALIASES.get(k, k): v for k, v in env_config.items()}
    fields = set(getattr(EnvConfig, "__dataclass_fields__", {}))
    if not fields:
        return EnvConfig(**resolved)
    good = {k: v for k, v in resolved.items() if k in fields}
    unknown = sorted(set(resolved) - set(good))
    if unknown:
        _warn_once(
            "envcfg", "EnvConfig has no fields %s; ignoring them", unknown
        )
    return EnvConfig(**good) if good else None


def build_env(
    spec: ExperimentSpec, *, seed: int, training: bool, cfg: RunnerConfig
) -> Any:
    """Create one environment, wrapped identically for train and eval.

    The only intentional difference is reward normalisation, which is applied to
    the training environment alone: the agent may learn against a rescaled
    reward, but every reported return is raw, or the numbers would not be
    comparable across cells.
    """
    from propulsion_rl.envs.propulsion_env import make_env

    env_config = dict(spec.env_config)
    prop_kwargs = env_config.pop("propulsion_kwargs", None)
    mission_kwargs = env_config.pop("mission_kwargs", None)
    vehicle = env_config.pop("vehicle", None)
    max_steps = env_config.get("max_episode_steps", cfg.max_episode_steps)
    if spec.cost_model not in COST_MODEL:
        raise KeyError(f"unknown cost model {spec.cost_model!r}")

    env = make_env(
        spec.propulsion,
        spec.mission,
        config=_build_env_config(env_config),
        propulsion_kwargs=prop_kwargs,
        mission_kwargs=mission_kwargs,
        vehicle=vehicle,
        cost_model=spec.cost_model,
        seed=seed,
    )
    env = _apply_wrappers(env, training=training, cfg=cfg, max_steps=max_steps)
    return env


def _apply_wrappers(
    env: Any, *, training: bool, cfg: RunnerConfig, max_steps: int | None
) -> Any:
    """Wrap in a fixed order, skipping any wrapper that is unavailable.

    Order matters: the time limit is innermost so truncation is part of the raw
    episode, statistics are recorded before any reward rescaling, and
    normalisation sits outermost so its frozen statistics are the last thing the
    agent sees.
    """
    try:
        from propulsion_rl.envs import wrappers as W
    except Exception:
        _warn_once("wrappers", "propulsion_rl.envs.wrappers unavailable; running "
                   "unwrapped (no observation normalisation)")
        return env

    def _try(name: str, *args: Any, **kw: Any) -> None:
        nonlocal env
        klass = getattr(W, name, None)
        if klass is None:
            _warn_once(f"w:{name}", "wrapper %s not available; skipping", name)
            return
        try:
            env = klass(env, *args, **kw)
        except Exception as exc:
            _warn_once(f"w!{name}", "wrapper %s could not be applied (%s); skipping",
                       name, exc)

    if max_steps:
        _try("TimeLimit", int(max_steps))
    _try("RecordEpisodeStatistics")
    if cfg.normalize_obs:
        _try("NormalizeObservation")
    if training and cfg.normalize_reward:
        _try("NormalizeReward")
    return env


def _find_wrapper(env: Any, class_name: str) -> Any | None:
    seen = 0
    while env is not None and seen < 32:
        if type(env).__name__ == class_name:
            return env
        env = getattr(env, "env", None)
        seen += 1
    return None


def set_norm_training(env: Any, training: bool) -> bool:
    """Freeze or unfreeze observation-normalisation statistics.

    The wrapper is owned elsewhere, so several plausible spellings of "stop
    updating" are attempted. Returns whether one of them worked -- the caller
    records that in the result row, because an evaluation whose normaliser kept
    adapting is not the protocol this module claims to implement.
    """
    w = _find_wrapper(env, "NormalizeObservation")
    if w is None:
        return False
    for attr, val in (("set_training", training), ("freeze", None),
                      ("training", training), ("update_running_mean", training),
                      ("_update", training)):
        if not hasattr(w, attr):
            continue
        member = getattr(w, attr)
        if callable(member):
            if attr == "freeze":
                fn = getattr(w, "freeze" if not training else "unfreeze", None)
                if fn is None:
                    continue
                fn()
            else:
                member(val)
        else:
            setattr(w, attr, val)
        return True
    return False


def get_norm_state(env: Any) -> dict[str, Any] | None:
    """Serialisable observation-normalisation statistics, or None."""
    w = _find_wrapper(env, "NormalizeObservation")
    if w is None:
        return None
    for attr in ("get_state", "state_dict"):
        fn = getattr(w, attr, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return {"kind": attr, "data": fn()}
    rms = getattr(w, "obs_rms", None) or getattr(w, "rms", None) or w
    out = {}
    for k in ("mean", "var", "count", "std", "n"):
        v = getattr(rms, k, None)
        if v is not None:
            out[k] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v)
    return {"kind": "rms", "data": out} if out else None


def set_norm_state(env: Any, state: Mapping[str, Any] | None) -> bool:
    """Inverse of :func:`get_norm_state`. Used to sync eval to train stats."""
    if not state:
        return False
    w = _find_wrapper(env, "NormalizeObservation")
    if w is None:
        return False
    kind, data = state.get("kind"), state.get("data")
    if kind in ("get_state", "state_dict"):
        for attr in ("set_state", "load_state_dict"):
            fn = getattr(w, attr, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    fn(data)
                    return True
        return False
    rms = getattr(w, "obs_rms", None) or getattr(w, "rms", None) or w
    ok = False
    for k, v in (data or {}).items():
        if hasattr(rms, k):
            cur = getattr(rms, k)
            with contextlib.suppress(Exception):
                setattr(rms, k, np.asarray(v, dtype=np.float64)
                        if isinstance(cur, np.ndarray) else type(cur)(v))
                ok = True
    return ok


def sync_normalization(src: Any, dst: Any) -> bool:
    """Copy training normalisation statistics onto the (frozen) eval env."""
    return set_norm_state(dst, get_norm_state(src))


# --- info extraction ---------------------------------------------------------
def _as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort flattening of a dataclass / mapping / plain object."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            out[f.name] = getattr(obj, f.name, None)
        return out
    if hasattr(obj, "__slots__"):
        return {k: getattr(obj, k, None) for k in obj.__slots__}
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if not math.isnan(f) else default


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# --- the rollout primitives --------------------------------------------------
def run_episode(
    env: Any,
    agent: Agent,
    *,
    seed: int,
    deterministic: bool,
    collect_telemetry: bool = False,
    learn: bool = False,
    max_steps: int = 1_000_000,
    step_budget: int | None = None,
    counters: dict[str, float] | None = None,
) -> tuple[EpisodeRecord, int]:
    """Run one episode. Returns the record and the environment steps consumed.

    ``learn`` controls whether transitions are fed to the agent and updates are
    run; evaluation passes it as False so an evaluation can never move the
    policy. ``step_budget`` lets the training loop stop mid-episode exactly on
    the budget, which is what keeps "equal budget" literally true rather than
    approximately true.
    """
    counters = counters if counters is not None else {}
    agent.reset()
    obs, info = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32)

    rec = EpisodeRecord(seed=int(seed))
    throttles: list[float] = []
    op_points: list[float] = []
    throttled_by: dict[str, int] = {}
    ret = 0.0
    cost_sum = 0.0
    violations = 0
    steps = 0
    last_info: dict[str, Any] = dict(info or {})

    while steps < max_steps:
        if step_budget is not None and steps >= step_budget:
            break
        t0 = time.perf_counter()
        action = agent.act(obs, deterministic=deterministic)
        counters["act_s"] = counters.get("act_s", 0.0) + (time.perf_counter() - t0)
        counters["act_calls"] = counters.get("act_calls", 0.0) + 1

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        clipped = np.clip(a, -1.0, 1.0)
        if clipped.size >= 2:
            throttles.append(float((clipped[0] + 1.0) * 0.5))
            op_points.append(float((clipped[1] + 1.0) * 0.5))

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        info = dict(info or {})
        steps += 1
        ret += float(reward)
        cost = _num(info.get("constraint_cost", 0.0))
        cost_sum += cost
        if cost > 0.0:
            violations += 1
        tb = str(info.get("throttled_by", "none"))
        throttled_by[tb] = throttled_by.get(tb, 0) + 1

        if learn:
            agent.observe_transition(
                Transition(
                    obs=obs, action=a, reward=float(reward), next_obs=next_obs,
                    terminated=bool(terminated), truncated=bool(truncated),
                    cost=cost, info=info,
                )
            )
        obs = next_obs
        last_info = info
        if terminated or truncated:
            break

    rec.steps = steps
    rec.ret = ret
    rec.constraint_cost = cost_sum
    rec.throttle_mean = _mean(throttles)
    rec.throttle_saturated_frac = _mean([float(t > 0.99) for t in throttles])
    rec.throttle_off_frac = _mean([float(t < 0.01) for t in throttles])
    rec.operating_point_mean = _mean(op_points)
    n_throttled = sum(v for k, v in throttled_by.items() if k not in ("none", ""))
    rec.throttled_frac = n_throttled / steps if steps else 0.0
    limiting = {k: v for k, v in throttled_by.items() if k not in ("none", "")}
    rec.throttled_by_top = max(limiting, key=limiting.get) if limiting else "none"

    _fill_from_info(rec, last_info, fallback_violations=violations)
    if collect_telemetry:
        rec.telemetry = _extract_telemetry(env)
        if rec.wear_fraction == 0.0 and rec.telemetry:
            rec.wear_fraction = _num(rec.telemetry[-1].get("wear_fraction", 0.0))
    else:
        # Wear lives in telemetry for most propulsion models; pull just the tail
        # so the column is populated without keeping the whole trace.
        tail = _extract_telemetry(env, tail=1)
        if rec.wear_fraction == 0.0 and tail:
            rec.wear_fraction = _num(tail[-1].get("wear_fraction", 0.0))
    return rec, steps


def _fill_from_info(
    rec: EpisodeRecord, info: Mapping[str, Any], fallback_violations: int
) -> None:
    """Populate the mission and economics columns from the terminal ``info``."""
    mr = info.get("mission_result")
    rec.termination_reason = str(
        _enum_value(info.get("termination_reason"))
        or _enum_value(_get(mr, "reason"))
        or "unknown"
    )
    if mr is not None:
        rec.success = bool(_get(mr, "success", False))
        rec.progress = _num(_get(mr, "progress"))
        rec.delta_v_m_s = _num(_get(mr, "delta_v_m_s"))
        rec.propellant_kg = _num(_get(mr, "propellant_used_kg"))
        rec.trip_time_days = _num(_get(mr, "elapsed_s")) / 86400.0
        rec.terminal_error = _num(_get(mr, "terminal_error"))
        rec.payload_delivered_kg = _num(_get(mr, "payload_delivered_kg"))
        rec.constraint_violations = int(
            _num(_get(mr, "constraint_violations"), fallback_violations)
        )
        extras = _get(mr, "extras") or {}
        if isinstance(extras, Mapping) and "wear_fraction" in extras:
            rec.wear_fraction = _num(extras["wear_fraction"])
    else:
        rec.progress = _num(info.get("progress"))
        rec.constraint_violations = fallback_violations

    econ = info.get("economics")
    if econ is not None:
        rec.cost_per_kg_delivered = _float_or_nan(_get(econ, "cost_per_kg_delivered"))
        rec.cost_per_delta_v = _float_or_nan(_get(econ, "cost_per_delta_v"))
        rec.figure_of_merit = _float_or_nan(_get(econ, "figure_of_merit"))
        bd = _get(econ, "breakdown")
        rec.cost_total = _float_or_nan(_get(bd, "total"))


def _enum_value(v: Any) -> str | None:
    if v is None:
        return None
    return str(getattr(v, "value", v))


def _float_or_nan(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _extract_telemetry(env: Any, tail: int | None = None) -> list[dict[str, Any]]:
    base = _base_env(env)
    tel = getattr(base, "telemetry", None)
    if not tel:
        return []
    seq = list(tel)[-tail:] if tail else list(tel)
    rows = []
    for t in seq:
        if hasattr(t, "as_row"):
            with contextlib.suppress(Exception):
                rows.append(t.as_row())
                continue
        rows.append(_as_dict(t))
    return rows


def evaluate(
    env: Any,
    agent: Agent,
    seeds: Sequence[int],
    *,
    collect_telemetry: bool = False,
    max_steps: int = 1_000_000,
) -> EvalSummary:
    """Evaluate on a fixed seed set with ``deterministic=True`` and no learning.

    The caller is responsible for having frozen and synchronised normalisation
    before calling; :func:`run_cell` always does.
    """
    summary = EvalSummary()
    counters: dict[str, float] = {}
    t0 = time.perf_counter()
    for s in seeds:
        rec, steps = run_episode(
            env, agent, seed=int(s), deterministic=True,
            collect_telemetry=collect_telemetry, learn=False,
            max_steps=max_steps, counters=counters,
        )
        summary.episodes.append(rec)
        summary.env_steps += steps
    summary.wall_s = time.perf_counter() - t0
    summary.act_s = counters.get("act_s", 0.0)
    return summary


# --- the cell ----------------------------------------------------------------
def run_cell(spec: ExperimentSpec, cfg: RunnerConfig) -> dict[str, Any]:
    """Train (if the agent learns), evaluate, checkpoint, and return a tidy row.

    Raises nothing to the caller by design contract of :func:`_run_cell_safe`;
    call that from a worker.
    """
    t_start = time.perf_counter()
    cell = cfg.cell_dir(spec)
    (cell / "final").mkdir(parents=True, exist_ok=True)
    (cell / "best").mkdir(parents=True, exist_ok=True)

    replicate = int(spec.seed)
    seed_all(replicate)

    agent = make_agent(spec)
    learns = bool(getattr(agent, "learns", True))
    # The registry's metadata is a hint; the agent object is the authority.
    train_steps = int(spec.train_steps) if learns else 0
    if not learns and spec.train_steps:
        logger.debug("%s is scripted; zeroing its %d-step budget",
                     spec.agent, spec.train_steps)

    max_ep_steps = int(cfg.max_episode_steps or spec.env_config.get(
        "max_episode_steps", 1_000_000))

    val_seeds = val_episode_seeds(replicate, spec.val_episodes)
    tst_seeds = test_episode_seeds(replicate, spec.eval_episodes)
    n_train_eps = max(1, train_steps // max(1, min(max_ep_steps, 1000)) + 2)
    trn_seeds = train_episode_seeds(replicate, n_train_eps * 4)
    _assert_disjoint(trn_seeds, val_seeds, tst_seeds)

    train_env = build_env(spec, seed=trn_seeds[0], training=True, cfg=cfg)
    eval_env = build_env(spec, seed=val_seeds[0], training=False, cfg=cfg)
    # The evaluation environment's normaliser never updates. Full stop.
    froze = set_norm_training(eval_env, False)
    if cfg.normalize_obs and not froze:
        _warn_once(
            "freeze",
            "could not freeze observation normalisation on the eval env; "
            "eval statistics may drift (column norm_frozen records this)",
        )

    curve: list[dict[str, Any]] = []
    counters: dict[str, float] = {}
    best_value = -math.inf
    best_summary: EvalSummary | None = None
    best_at_step = 0
    n_updates = 0
    update_s = 0.0
    train_stats: dict[str, float] = {}
    eval_env_steps = 0
    eval_wall_s = 0.0

    def _do_val(step: int) -> None:
        nonlocal best_value, best_summary, best_at_step, eval_env_steps, eval_wall_s
        synced = sync_normalization(train_env, eval_env)
        set_norm_training(eval_env, False)
        s = evaluate(eval_env, agent, val_seeds, max_steps=max_ep_steps)
        eval_env_steps += s.env_steps
        eval_wall_s += s.wall_s
        value = _selection_value(s, cfg.selection_metric)
        curve.append({
            "env_step": step,
            "wall_s": time.perf_counter() - t_start,
            "val_return": s.mean_return,
            "val_success_rate": s.success_rate,
            "val_progress": _mean([e.progress for e in s.episodes]),
            "val_constraint_cost": _mean([e.constraint_cost for e in s.episodes]),
            "val_selection_value": value,
            "norm_synced": bool(synced),
            "is_best": bool(value > best_value),
        })
        if value > best_value:
            best_value, best_summary, best_at_step = value, s, step
            save_checkpoint(cell / "best", agent, train_env, replicate, step)

    t_train0 = time.perf_counter()
    if train_steps > 0:
        _do_val(0)  # a baseline point, so a learning curve has a left edge
        done = 0
        ep_i = 0
        next_eval = spec.eval_interval or train_steps
        while done < train_steps:
            seed = trn_seeds[ep_i % len(trn_seeds)]
            ep_i += 1
            remaining = train_steps - done
            rec, steps = run_episode(
                train_env, agent, seed=int(seed), deterministic=False,
                learn=True, max_steps=max_ep_steps,
                step_budget=remaining, counters=counters,
            )
            # Updates run on the runner's cadence, identical for every agent;
            # an agent that only learns at episode boundaries no-ops here.
            n_up, dt, stats = _run_updates(agent, steps, cfg.update_every)
            n_updates += n_up
            update_s += dt
            train_stats.update(stats)
            with contextlib.suppress(Exception):
                agent.on_episode_end(rec.ret, {"steps": steps})
            done += steps
            if steps == 0:  # an env that refuses to step would spin forever
                logger.error("cell %s: episode produced 0 steps; aborting training",
                             spec.run_id)
                break
            if spec.eval_interval and done >= next_eval:
                _do_val(done)
                while done >= next_eval:
                    next_eval += spec.eval_interval
        train_env_steps = done
    else:
        train_env_steps = 0
    train_wall_s = time.perf_counter() - t_train0

    save_checkpoint(cell / "final", agent, train_env, replicate, train_env_steps)
    final_val = None
    if train_steps > 0:
        _do_val(train_env_steps)
        final_val = curve[-1]
    else:
        _do_val(0)

    # --- the reported number: test band, run once, never selected on ---------
    sync_normalization(train_env, eval_env)
    set_norm_training(eval_env, False)
    final_test = evaluate(
        eval_env, agent, tst_seeds, collect_telemetry=cfg.save_telemetry,
        max_steps=max_ep_steps,
    )
    eval_env_steps += final_test.env_steps
    eval_wall_s += final_test.wall_s

    best_test = final_test
    reloaded = False
    if train_steps > 0 and best_at_step != train_env_steps:
        best_agent = make_agent(spec)
        if load_checkpoint(cell / "best", best_agent, eval_env):
            reloaded = True
            set_norm_training(eval_env, False)
            best_test = evaluate(
                eval_env, best_agent, tst_seeds,
                collect_telemetry=cfg.save_telemetry, max_steps=max_ep_steps,
            )
            eval_env_steps += best_test.env_steps
            eval_wall_s += best_test.wall_s
    reported = best_test if cfg.report_policy == "best" else final_test

    for e in (train_env, eval_env):
        with contextlib.suppress(Exception):
            e.close()

    # --- assemble the row ----------------------------------------------------
    act_s = counters.get("act_s", 0.0) + final_test.act_s
    act_calls = max(1.0, counters.get("act_calls", 0.0) + final_test.n)
    row: dict[str, Any] = spec.as_row()
    row.update({
        "status": "ok",
        "learns": learns,
        "protocol_version": PROTOCOL_VERSION,
        "report_policy": cfg.report_policy,
        "selection_metric": cfg.selection_metric,
        # --- budget and compute, all three reported -------------------------
        "train_env_steps": train_env_steps,
        "eval_env_steps": eval_env_steps,
        "n_updates": n_updates,
        "wall_train_s": train_wall_s,
        "wall_eval_s": eval_wall_s,
        "wall_total_s": time.perf_counter() - t_start,
        "wall_update_s": update_s,
        "act_seconds": act_s,
        "act_ms_per_step": 1000.0 * act_s / act_calls,
        "planning_sim_steps": _planning_compute(agent, train_stats),
        "num_parameters": int(getattr(agent, "num_parameters", 0) or 0),
        # --- protocol provenance --------------------------------------------
        "n_val_seeds": len(val_seeds),
        "n_test_seeds": len(tst_seeds),
        "val_seed_lo": val_seeds[0],
        "test_seed_lo": tst_seeds[0],
        "best_at_step": best_at_step,
        "best_reloaded": reloaded,
        "norm_frozen": bool(froze) or not cfg.normalize_obs,
        "uses_constraints": bool(getattr(agent, "uses_constraints", False)),
    })
    row.update(reported.metrics(""))
    row.update(final_test.metrics("final_"))
    row.update(best_test.metrics("best_"))
    row.update({f"train_{k}": v for k, v in train_stats.items()})

    _write_cell_outputs(cell, spec, cfg, row, curve, reported, final_val)
    return row


def _run_updates(
    agent: Agent, steps: int, every: int
) -> tuple[int, float, dict[str, float]]:
    """Call ``agent.update()`` on the runner's fixed cadence.

    Identical for every agent: the number of *opportunities* to learn is part of
    the equal-budget contract, and what an algorithm does with an opportunity is
    the algorithm's business.
    """
    n = 0
    stats: dict[str, float] = {}
    t0 = time.perf_counter()
    for _ in range(max(1, steps // max(1, every))):
        try:
            st = agent.update()
        except Exception:
            logger.exception("agent.update() raised; continuing without it")
            break
        n += 1
        vals = getattr(st, "values", None)
        if isinstance(vals, Mapping):
            for k, v in vals.items():
                with contextlib.suppress(TypeError, ValueError):
                    stats[str(k)] = float(v)
    return n, time.perf_counter() - t0, stats


def _planning_compute(agent: Agent, stats: Mapping[str, float]) -> float:
    """Inner-model simulation steps a planner burned, when it reports them.

    CEM-MPC and friends spend orders of magnitude more compute per decision than
    a feed-forward policy. ``act_seconds`` always captures that in wall time;
    this captures it in a machine-independent unit when the agent publishes one.
    """
    for attr in ("planning_sim_steps", "plan_sim_steps", "model_calls",
                 "n_planning_steps"):
        v = getattr(agent, attr, None)
        if v is not None:
            with contextlib.suppress(TypeError, ValueError):
                return float(v)
    for k in ("plan_sim_steps", "planning_sim_steps", "model_calls"):
        if k in stats:
            return float(stats[k])
    return 0.0


def make_agent(spec: ExperimentSpec) -> Agent:
    """Instantiate the agent for a cell and seed everything it owns."""
    agent = AGENT.make(
        spec.agent, obs_dim=OBS_DIM, action_dim=CANONICAL_ACTION_DIM,
        **dict(spec.agent_kwargs),
    )
    with contextlib.suppress(Exception):
        agent.set_seed(_AGENT_SEED_OFFSET + int(spec.seed))
    return agent


def seed_all(replicate: int) -> None:
    """Seed every global RNG identically for a given replicate, in every cell."""
    random.seed(_AGENT_SEED_OFFSET + replicate)
    np.random.seed((_AGENT_SEED_OFFSET + replicate) % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(_TORCH_SEED_OFFSET + replicate)
        if hasattr(torch, "use_deterministic_algorithms"):
            with contextlib.suppress(Exception):
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    except Exception:  # torch is optional at runtime for scripted-only sweeps
        pass


# --- checkpoints -------------------------------------------------------------
def save_checkpoint(
    path: Path, agent: Agent, env: Any, replicate: int, step: int
) -> None:
    """Persist agent parameters, normalisation statistics and RNG state.

    All three are needed to resume: parameters alone give a policy that sees
    differently normalised observations than the one that was trained, and
    without the RNG state a resumed run is a different experiment.
    """
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        agent.save(path / "agent")
    state = get_norm_state(env)
    if state is not None:
        with contextlib.suppress(Exception):
            (path / "norm_stats.json").write_text(
                json.dumps(_jsonable(state), indent=2), encoding="utf-8"
            )
    rng: dict[str, Any] = {
        "replicate": replicate, "step": step,
        "python": random.getstate(), "numpy": np.random.get_state(),
    }
    with contextlib.suppress(Exception):
        import torch

        rng["torch"] = torch.get_rng_state()
    with contextlib.suppress(Exception):
        (path / "rng_state.pkl").write_bytes(pickle.dumps(rng))


def load_checkpoint(path: Path, agent: Agent, env: Any | None = None) -> bool:
    """Restore what :func:`save_checkpoint` wrote. Returns success."""
    ok = False
    try:
        agent.load(path / "agent")
        ok = True
    except Exception:
        logger.debug("no agent parameters restored from %s", path)
    ns = path / "norm_stats.json"
    if env is not None and ns.exists():
        with contextlib.suppress(Exception):
            set_norm_state(env, json.loads(ns.read_text(encoding="utf-8")))
    return ok


# --- per-cell output ---------------------------------------------------------
def _write_cell_outputs(
    cell: Path,
    spec: ExperimentSpec,
    cfg: RunnerConfig,
    row: Mapping[str, Any],
    curve: Sequence[Mapping[str, Any]],
    reported: EvalSummary,
    final_val: Mapping[str, Any] | None,
) -> None:
    import pandas as pd

    cell.mkdir(parents=True, exist_ok=True)
    (cell / "row.json").write_text(json.dumps(_jsonable(row), indent=2), "utf-8")
    (cell / "spec.json").write_text(
        json.dumps(_jsonable(spec.content() | {"slug": spec.slug}), indent=2), "utf-8"
    )
    if curve:
        pd.DataFrame(list(curve)).to_csv(cell / "curve.csv", index=False)
    if reported.episodes:
        pd.DataFrame([e.as_row() for e in reported.episodes]).to_csv(
            cell / "episodes.csv", index=False
        )
    # Telemetry for every episode blows up disk fast (10^4 steps x 10^3 cells);
    # the best episode is the one anyone actually opens.
    if cfg.save_telemetry:
        best = reported.best_episode(cfg.selection_metric)
        if best is not None and best.telemetry:
            df = pd.DataFrame(best.telemetry)
            df.insert(0, "episode_seed", best.seed)
            df.to_csv(cell / "telemetry_best.csv", index=False)
    (cell / "status.json").write_text(
        json.dumps({"status": "ok", "run_id": spec.run_id,
                    "finished": time.time(),
                    "final_val": _jsonable(final_val)}, indent=2), "utf-8"
    )


def _failed_row(spec: ExperimentSpec, exc: BaseException, wall_s: float) -> dict[str, Any]:
    row = spec.as_row()
    row.update({
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "wall_total_s": wall_s,
        "success_rate": float("nan"),
        "return": float("nan"),
    })
    return row


def _run_cell_safe(args: tuple[ExperimentSpec, RunnerConfig]) -> dict[str, Any]:
    """Worker entry point. Never raises: one bad cell must not end the sweep."""
    spec, cfg = args
    t0 = time.perf_counter()
    try:
        return run_cell(spec, cfg)
    except BaseException as exc:  # noqa: BLE001 -- deliberate: isolate the cell
        tb = traceback.format_exc()
        logger.error("cell %s (%s) FAILED: %s\n%s", spec.run_id, spec.cell_key,
                     exc, tb)
        cell = cfg.cell_dir(spec)
        with contextlib.suppress(Exception):
            cell.mkdir(parents=True, exist_ok=True)
            (cell / "status.json").write_text(
                json.dumps({"status": "failed", "run_id": spec.run_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": tb, "finished": time.time()}, indent=2),
                encoding="utf-8",
            )
        return _failed_row(spec, exc, time.perf_counter() - t0)


# --- resume ------------------------------------------------------------------
def load_cell_row(spec: ExperimentSpec, cfg: RunnerConfig) -> dict[str, Any] | None:
    """Previously completed row for this exact content hash, or None."""
    p = cfg.cell_dir(spec) / "row.json"
    if not p.exists():
        return None
    try:
        row = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("unreadable cached row at %s; will re-run", p)
        return None
    if row.get("status") != "ok":
        return None
    row["cached"] = True
    return row


# --- the sweep ---------------------------------------------------------------
def run_sweep(
    matrix: ExperimentMatrix,
    cfg: RunnerConfig,
    *,
    on_row: Any = None,
):  # -> pandas.DataFrame
    """Execute every cell, writing results incrementally.

    Returns the tidy DataFrame. Also writes ``results.csv`` (rewritten after
    every completion, so it is always valid), ``results.jsonl`` (append-only,
    the crash-proof record) and ``results.parquet`` at the end when a parquet
    engine is available.
    """
    import pandas as pd

    sweep_dir = cfg.sweep_dir
    (sweep_dir / "cells").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_plan(matrix, cfg)

    todo: list[ExperimentSpec] = []
    rows: list[dict[str, Any]] = []
    for spec in matrix:
        cached = None if cfg.force else load_cell_row(spec, cfg)
        if cached is not None:
            rows.append(cached)
        else:
            todo.append(spec)
    n_cached = len(rows)
    logger.info("sweep '%s': %d cells, %d cached, %d to run, %d worker(s)",
                cfg.sweep_name, len(matrix), n_cached, len(todo), cfg.workers)

    jsonl = sweep_dir / "results.jsonl"
    if cfg.force and jsonl.exists():
        jsonl.unlink()

    t0 = time.perf_counter()
    n_done = 0
    n_failed = 0

    def _accept(row: dict[str, Any]) -> None:
        nonlocal n_done, n_failed
        rows.append(row)
        n_done += 1
        n_failed += int(row.get("status") != "ok")
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(row)) + "\n")
        # Rewrite the CSV every time: a sweep killed at any moment leaves a
        # readable, complete-so-far table behind.
        pd.DataFrame(rows).to_csv(sweep_dir / "results.csv", index=False)
        if on_row is not None:
            with contextlib.suppress(Exception):
                on_row(row, n_done, len(todo))

    if todo:
        if cfg.workers > 1:
            for row in _run_parallel(todo, cfg):
                _accept(row)
        else:
            for spec in todo:
                logger.info("[%d/%d] %s", n_done + 1, len(todo), spec.cell_key
                            + f" seed={spec.seed}")
                _accept(_run_cell_safe((spec, cfg)))

    df = pd.DataFrame(rows)
    _write_manifest(matrix, cfg, df, wall_s=time.perf_counter() - t0,
                    n_cached=n_cached, n_failed=n_failed)
    with contextlib.suppress(Exception):
        df.to_parquet(sweep_dir / "results.parquet", index=False)
    if not (sweep_dir / "results.parquet").exists():
        logger.info("no parquet engine available; results.csv is the artefact")
    logger.info(
        "sweep '%s' finished: %d rows (%d cached, %d run, %d failed) in %.1fs",
        cfg.sweep_name, len(df), n_cached, n_done, n_failed,
        time.perf_counter() - t0,
    )
    return df


def _run_parallel(
    todo: Sequence[ExperimentSpec], cfg: RunnerConfig
) -> Iterable[dict[str, Any]]:
    """Fan cells out over processes, one torch thread each.

    Torch defaults to one thread per core *per process*; with 8 workers that is
    64 threads fighting over 8 cores and everything gets slower than serial. The
    environment variables must be set before the children import numpy/torch,
    which with the spawn context means setting them here, in the parent.
    """
    import multiprocessing as mp

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    ctx = mp.get_context("spawn")
    workers = max(1, min(int(cfg.workers), len(todo)))
    logger.info("dispatching %d cells over %d spawn workers", len(todo), workers)
    with ctx.Pool(
        processes=workers, initializer=_worker_init, initargs=(cfg.log_level,)
    ) as pool:
        for row in pool.imap_unordered(
            _run_cell_safe, [(s, cfg) for s in todo], chunksize=1
        ):
            yield row


def _worker_init(log_level: str) -> None:
    """Runs in every spawned worker before any cell does."""
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(levelname)s %(processName)s %(name)s: %(message)s",
    )
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with contextlib.suppress(Exception):
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


# --- sweep-level artefacts ---------------------------------------------------
def _write_plan(matrix: ExperimentMatrix, cfg: RunnerConfig) -> None:
    d = cfg.sweep_dir
    with contextlib.suppress(Exception):
        matrix.to_frame().to_csv(d / "matrix.csv", index=False)
    if matrix.exclusions:
        with contextlib.suppress(Exception):
            matrix.exclusions_frame().to_csv(d / "excluded.csv", index=False)
    if matrix.config is not None:
        with contextlib.suppress(Exception):
            (d / "config.json").write_text(
                json.dumps(_jsonable(asdict(matrix.config)), indent=2), "utf-8"
            )
    (d / "plan.txt").write_text(matrix.summary(), encoding="utf-8")


def _write_manifest(
    matrix: ExperimentMatrix, cfg: RunnerConfig, df: Any, *,
    wall_s: float, n_cached: int, n_failed: int,
) -> None:
    manifest = {
        "sweep": cfg.sweep_name,
        "protocol_version": PROTOCOL_VERSION,
        "finished": time.time(),
        "wall_s": wall_s,
        "n_cells": len(matrix),
        "n_rows": int(len(df)),
        "n_cached": n_cached,
        "n_failed": n_failed,
        "workers": cfg.workers,
        "report_policy": cfg.report_policy,
        "selection_metric": cfg.selection_metric,
        "seed_bands": {
            "train": [TRAIN_SEED_LO, TRAIN_SEED_HI],
            "val": [VAL_SEED_LO, VAL_SEED_HI],
            "test": [TEST_SEED_LO, TEST_SEED_HI],
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "excluded_pairings": [
            {"propulsion": p, "mission": m, "agent": a, "rule": r, "reason": why}
            for (p, m, a), (r, why) in matrix.excluded_pairings().items()
        ],
    }
    with contextlib.suppress(Exception):
        import torch

        manifest["platform"]["torch"] = torch.__version__
    (cfg.sweep_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )


# --- standalone evaluation (the `eval` subcommand) ---------------------------
def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    episodes: int | None = None,
    seed: int | None = None,
    cfg: RunnerConfig | None = None,
    collect_telemetry: bool = True,
) -> dict[str, Any]:
    """Re-evaluate a saved policy under the identical held-out test protocol.

    ``checkpoint`` is a cell directory (``results/<sweep>/cells/<slug>``) or one
    of its ``final``/``best`` subdirectories.
    """
    path = Path(checkpoint)
    ckpt_dir = path if (path / "agent").exists() or (path.name in ("final", "best")) \
        else path / "best"
    if not ckpt_dir.exists():
        ckpt_dir = path / "final"
    cell_dir = ckpt_dir.parent if ckpt_dir.name in ("final", "best") else ckpt_dir
    spec_file = cell_dir / "spec.json"
    if not spec_file.exists():
        raise FileNotFoundError(f"no spec.json under {cell_dir}; not a cell directory")
    content = json.loads(spec_file.read_text(encoding="utf-8"))
    spec = ExperimentSpec(
        propulsion=content["propulsion"], mission=content["mission"],
        agent=content["agent"], cost_model=content["cost_model"],
        seed=int(content["seed"]), train_steps=int(content.get("train_steps", 0)),
        eval_episodes=int(episodes or content.get("eval_episodes", 10)),
        val_episodes=int(content.get("val_episodes", 5)),
        eval_interval=int(content.get("eval_interval", 0)),
        agent_kwargs=content.get("agent_kwargs", {}) or {},
        env_config=content.get("env_config", {}) or {},
    )
    cfg = cfg or RunnerConfig()
    replicate = int(seed if seed is not None else spec.seed)
    seed_all(replicate)
    agent = make_agent(spec)
    seeds = test_episode_seeds(replicate, spec.eval_episodes)
    env = build_env(spec, seed=seeds[0], training=False, cfg=cfg)
    load_checkpoint(ckpt_dir, agent, env)
    set_norm_training(env, False)
    max_ep = int(cfg.max_episode_steps or 1_000_000)
    summary = evaluate(env, agent, seeds, collect_telemetry=collect_telemetry,
                       max_steps=max_ep)
    with contextlib.suppress(Exception):
        env.close()
    out = spec.as_row()
    out.update({"checkpoint": str(ckpt_dir), "status": "ok"})
    out.update(summary.metrics(""))
    return out


# --- small numeric helpers ---------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if _finite(x)]
    return float(np.mean(vals)) if vals else 0.0


def _std(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if _finite(x)]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _mode(xs: Sequence[str]) -> str:
    if not xs:
        return "unknown"
    counts: dict[str, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)


def _jsonable(obj: Any) -> Any:
    """Recursively coerce to something ``json.dumps`` accepts."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    return str(obj)


__all__ = [
    "EpisodeRecord",
    "EvalSummary",
    "RunnerConfig",
    "build_env",
    "evaluate",
    "evaluate_checkpoint",
    "load_cell_row",
    "make_agent",
    "run_cell",
    "run_episode",
    "run_sweep",
    "test_episode_seeds",
    "train_episode_seeds",
    "val_episode_seeds",
]
