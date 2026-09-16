"""The experiment matrix: which cells exist, in what order, and under what id.

A "cell" is one fully-specified experiment: a propulsion system, a mission, an
agent, a cost model and a seed. :class:`ExperimentMatrix` turns a YAML sweep
config into the deterministic, de-duplicated list of cells the runner executes.

Two properties matter enough to be design constraints rather than conveniences:

**Deterministic, stable ordering.** The same config always expands to the same
sequence of :class:`ExperimentSpec` objects, in the same order, regardless of
dict iteration order or registry insertion order. A sweep that reorders itself
between invocations cannot be resumed or diffed.

**Content-addressed ``run_id``.** Every cell hashes its own semantic content --
including the evaluation protocol version -- into a short stable id. That id is
the cache key: a cell whose results already exist is skipped, and changing
anything that would change the numbers (train budget, agent kwargs, eval
protocol) changes the id, so stale results can never be silently reused.

**Implausible pairings are excluded explicitly, never silently.** A 10 kWe
Kilopower-class NEP stage on a fast crewed Mars run is not a hard cell, it is a
meaningless one: the pairing cannot express the trade the mission exists to
test, and leaving it in drags down every aggregate that averages over missions.
Such pairings are removed by named rules in :data:`PLAUSIBILITY_RULES`, each
carrying a written justification, and every drop is recorded in
:attr:`ExperimentMatrix.exclusions` so the reader can audit -- and disagree
with -- the judgement calls. ``allow_implausible: true`` disables them all.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.registry import AGENT, COST_MODEL, MISSION, PROPULSION, Registry

logger = logging.getLogger(__name__)

#: Bumped whenever the evaluation protocol changes in a way that makes old
#: numbers incomparable to new ones. It is hashed into every ``run_id``, so a
#: bump invalidates every cached cell in every sweep. That is the point.
PROTOCOL_VERSION = 3

#: Length of the hex digest kept in ``run_id``. 12 hex chars = 48 bits; at the
#: ~10^4 cells a large sweep reaches, collision probability is ~1e-9.
_RUN_ID_CHARS = 12


# --- one cell ----------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ExperimentSpec:
    """One cell of the matrix: everything needed to run and to identify it.

    Identity is content-based: two specs are equal iff their :attr:`run_id`
    matches, which happens iff every field that could change the resulting
    numbers matches. ``sweep`` is deliberately excluded from the hash so the
    same cell run under two sweep names is recognisably the same experiment.
    """

    propulsion: str
    mission: str
    agent: str
    cost_model: str
    seed: int
    train_steps: int = 0
    eval_episodes: int = 10
    #: Episodes per periodic evaluation, used only to pick the best checkpoint.
    #: Disjoint from both the training seeds and the reported test seeds.
    val_episodes: int = 5
    #: Environment steps between periodic evaluations. 0 disables them.
    eval_interval: int = 0
    agent_kwargs: Mapping[str, Any] = field(default_factory=dict)
    env_config: Mapping[str, Any] = field(default_factory=dict)
    sweep: str = "default"
    #: Free-form provenance, not hashed (e.g. which config file produced this).
    tags: Mapping[str, Any] = field(default_factory=dict)

    # --- identity ------------------------------------------------------------
    def content(self) -> dict[str, Any]:
        """The semantic payload that :attr:`run_id` hashes. Order-independent."""
        return {
            "protocol": PROTOCOL_VERSION,
            "propulsion": self.propulsion,
            "mission": self.mission,
            "agent": self.agent,
            "cost_model": self.cost_model,
            "seed": int(self.seed),
            "train_steps": int(self.train_steps),
            "eval_episodes": int(self.eval_episodes),
            "val_episodes": int(self.val_episodes),
            "eval_interval": int(self.eval_interval),
            "agent_kwargs": _canonical(self.agent_kwargs),
            "env_config": _canonical(self.env_config),
        }

    @property
    def run_id(self) -> str:
        blob = json.dumps(
            self.content(), sort_keys=True, separators=(",", ":"), default=repr
        )
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()[
            :_RUN_ID_CHARS
        ]

    @property
    def pairing(self) -> str:
        """``agent@propulsion`` -- the unit the study is actually ranking."""
        return f"{self.agent}@{self.propulsion}"

    @property
    def cell_key(self) -> str:
        """Identity of the cell ignoring the seed, i.e. what gets averaged."""
        return f"{self.agent}|{self.propulsion}|{self.mission}|{self.cost_model}"

    @property
    def slug(self) -> str:
        """Human-readable, filesystem-safe directory name."""
        parts = [self.agent, self.propulsion, self.mission, self.cost_model]
        stem = "-".join(_slugify(p) for p in parts)
        return f"{stem}-s{self.seed:03d}-{self.run_id}"

    @property
    def learns(self) -> bool:
        """Whether this agent is expected to consume a training budget.

        Read from registry metadata when the agent module publishes it, so the
        matrix does not have to instantiate an agent to plan a sweep. Falls back
        to ``True``; the runner re-checks against the real ``agent.learns`` and
        zeroes the budget there, which is the authoritative test.
        """
        meta = AGENT.meta(self.agent)
        return bool(meta.get("learns", True))

    def with_(self, **changes: Any) -> "ExperimentSpec":
        """Copy with overrides. Changes the ``run_id`` if it changes content."""
        data = {f: getattr(self, f) for f in _SPEC_FIELDS}
        data.update(changes)
        return ExperimentSpec(**data)

    def as_row(self) -> dict[str, Any]:
        """Flat identity columns, merged into every result row."""
        return {
            "run_id": self.run_id,
            "sweep": self.sweep,
            "agent": self.agent,
            "propulsion": self.propulsion,
            "mission": self.mission,
            "cost_model": self.cost_model,
            "seed": int(self.seed),
            "pairing": self.pairing,
            "family": propulsion_family(self.propulsion),
            "train_steps_requested": int(self.train_steps),
            "eval_episodes": int(self.eval_episodes),
        }

    # Identity is the content hash, not the tuple of fields: dict-valued fields
    # are not hashable, and the hash is the thing we actually mean by "same".
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExperimentSpec) and other.run_id == self.run_id

    def __hash__(self) -> int:
        return hash(self.run_id)

    def __repr__(self) -> str:
        return (
            f"<ExperimentSpec {self.agent}@{self.propulsion} on {self.mission} "
            f"seed={self.seed} steps={self.train_steps} id={self.run_id}>"
        )


_SPEC_FIELDS = (
    "propulsion",
    "mission",
    "agent",
    "cost_model",
    "seed",
    "train_steps",
    "eval_episodes",
    "val_episodes",
    "eval_interval",
    "agent_kwargs",
    "env_config",
    "sweep",
    "tags",
)


# --- plausibility rules ------------------------------------------------------
@dataclass(frozen=True)
class PlausibilityRule:
    """A named, justified reason a (propulsion, mission, agent) cell is dropped.

    ``predicate`` returns True when the cell should be excluded. Keep the
    reasoning in ``reason``: this table is the study's statement about which
    comparisons it considers meaningful, and a reader must be able to disagree
    with a specific sentence rather than with an unexplained gap in a heatmap.
    """

    name: str
    reason: str
    predicate: Callable[[str, str, str], bool]

    def applies(self, propulsion: str, mission: str, agent: str) -> bool:
        try:
            return bool(self.predicate(propulsion, mission, agent))
        except Exception:  # a broken rule must not take the sweep with it
            logger.exception("plausibility rule %s raised; treating as pass", self.name)
            return False


@dataclass(frozen=True)
class Exclusion:
    """Record of a cell that was expanded and then dropped."""

    propulsion: str
    mission: str
    agent: str
    cost_model: str
    rule: str
    reason: str

    def as_row(self) -> dict[str, str]:
        return {
            "propulsion": self.propulsion,
            "mission": self.mission,
            "agent": self.agent,
            "cost_model": self.cost_model,
            "rule": self.rule,
            "reason": self.reason,
        }


#: Nuclear-electric and nuclear-thermal system names, by convention of the
#: registry naming scheme. Checked against registry metadata first.
_NUCLEAR_PREFIXES = ("ntp_", "nep_")
_LOW_POWER_EP = ("hall_spt100", "ion_nstar")
#: Fallback only. The frame is registry metadata that each mission owns, so
#: :func:`is_planetocentric` reads that first; this list catches a mission that
#: neglects to declare one.
_PLANETOCENTRIC_MISSIONS = ("leo_geo_transfer", "geo_station_keeping")


def propulsion_family(name: str) -> str:
    """Family of a propulsion system: ``electric``, ``nuclear`` or ``unknown``.

    Prefers registry metadata (``family=``), which the propulsion modules own,
    and falls back to the naming convention so the matrix can still be planned
    against a registry that has not landed yet.
    """
    meta = PROPULSION.meta(name)
    fam = meta.get("family")
    if fam is not None:
        return str(getattr(fam, "value", fam))
    if name.startswith(_NUCLEAR_PREFIXES):
        return "nuclear"
    if name.startswith(("hall_", "ion_", "ppt_", "fee")):
        return "electric"
    return "unknown"


def is_planetocentric(mission: str) -> bool:
    """Whether *mission* is flown about a planet rather than about the Sun.

    Reads the ``frame`` the mission registered, falling back to the name list
    only when a mission declares none. A hand-maintained list silently
    mis-classifies every mission added after it was written -- which for the
    Edelbaum rule below means excluding the one baseline that transfer has.
    """
    frame = MISSION.meta(mission).get("frame")
    if frame is not None:
        return str(frame).lower() == "planetocentric"
    return mission in _PLANETOCENTRIC_MISSIONS


def _is_nuclear(propulsion: str) -> bool:
    return propulsion_family(propulsion) == "nuclear"


def _power_floor_rule(propulsion: str, mission: str, agent: str) -> bool:
    """Metadata-driven floor: fires only when both sides publish the numbers.

    If a propulsion module advertises ``power_w`` and a mission advertises
    ``min_power_w``, an under-powered pairing is excluded arithmetically rather
    than by a hand-maintained name list. Silent no-op when either is absent.
    """
    p_meta = PROPULSION.meta(propulsion)
    m_meta = MISSION.meta(mission)
    have = p_meta.get("power_w"), m_meta.get("min_power_w")
    if have[0] is None or have[1] is None:
        return False
    return float(have[0]) < float(have[1])


@functools.lru_cache(maxsize=None)
def _mass_floor_rule(propulsion: str, mission: str, agent: str = "") -> bool:
    """Exclude a stage whose engine will not fit inside its own wet mass.

    Every mission fixes the launched wet mass and lets propellant be the
    remainder, so charging the propulsion system's dry mass to that budget can
    leave nothing to burn: an 18 t Brayton NEP module does not go into a 5 t
    comsat, and an NTP stage whose engine alone outweighs the vehicle is not a
    hard control problem but a mis-specified one.

    Asks the objects rather than a name list, so a newly registered thruster is
    covered the moment it publishes a bill of materials. Cached because the
    matrix evaluates every rule across the full cross product, and construction
    is the expensive part. A pairing that cannot be built at all is left to the
    conformance suite to fail loudly; it is not excluded quietly here.
    """
    try:
        system = PROPULSION.make(propulsion)
        task = MISSION.make(mission)
        task.account_for_propulsion(float(system.bom().dry_mass_kg))
        return not task.mass_budget_closes()
    except Exception:  # noqa: BLE001 - a build failure is not an exclusion
        return False


PLAUSIBILITY_RULES: tuple[PlausibilityRule, ...] = (
    PlausibilityRule(
        name="no_reactor_for_stationkeeping",
        reason=(
            "GEO station keeping is ~50 m/s per year delivered in milli-newton-second "
            "impulse bits across a 15-year residence. A fission stage cannot throttle "
            "to that impulse bit, its restart budget is order 1e2 rather than 1e5, and "
            "reactor disposal practice precludes parking a core in the GEO belt. The "
            "cell is a category error, not a hard problem."
        ),
        predicate=lambda p, m, a: _is_nuclear(p) and m == "geo_station_keeping",
    ),
    PlausibilityRule(
        name="underpowered_nep_for_fast_crew",
        reason=(
            "A Kilopower-class NEP stage is ~10 kWe, roughly 0.4 N of thrust. A crewed "
            "fast-transfer stack is >=50 t, so acceleration is ~8e-6 m/s^2 and the "
            "transfer is a multi-year spiral against a mission defined by a <=180-day "
            "constraint. Every seed fails for the same trivial reason, which is not a "
            "measurement of the controller."
        ),
        predicate=lambda p, m, a: p == "nep_kilopower" and m == "mars_crew_fast",
    ),
    PlausibilityRule(
        name="underpowered_ep_for_fast_crew",
        reason=(
            "SPT-100 (1.35 kW, 83 mN) and NSTAR (2.3 kW, 92 mN) are single-string "
            "smallsat-class thrusters. On a crewed stack they are an order of magnitude "
            "worse than the Kilopower case above. The high-power variants of the same "
            "technologies (hall_hermes, ion_next) are kept, because there the pairing is "
            "at least arguable and that argument is the experiment."
        ),
        predicate=lambda p, m, a: p in _LOW_POWER_EP and m == "mars_crew_fast",
    ),
    PlausibilityRule(
        name="edelbaum_is_planetocentric",
        reason=(
            "Edelbaum's law is a closed-form circle-to-circle combined plane-change "
            "steering law for a planetocentric low-thrust spiral. Applied to a "
            "heliocentric transfer it is not a weak baseline, it is the wrong equation, "
            "and a broken baseline inflates every RL win measured against it."
        ),
        predicate=lambda p, m, a: a == "edelbaum" and not is_planetocentric(m),
    ),
    PlausibilityRule(
        name="power_floor_from_metadata",
        reason=(
            "The propulsion system's advertised electrical power is below the mission's "
            "advertised minimum. Arithmetic exclusion from registry metadata; inactive "
            "whenever either side does not publish the figure."
        ),
        predicate=_power_floor_rule,
    ),
    PlausibilityRule(
        name="mass_budget_does_not_close",
        reason=(
            "The propulsion system's own dry mass, charged against the mission's fixed "
            "launched wet mass, leaves no useful propellant. The vehicle cannot be "
            "built, so no controller can fly it; excluding the cell is the honest "
            "alternative to reporting a failure that is structural rather than "
            "behavioural. Arithmetic exclusion from each system's bill of materials."
        ),
        predicate=_mass_floor_rule,
    ),
)


# --- selectors ---------------------------------------------------------------
Selector = str | Sequence[str] | Mapping[str, Any] | None


def resolve_names(sel: Selector, registry: Registry, what: str) -> tuple[str, ...]:
    """Turn a YAML selector into a validated, ordered tuple of registry names.

    Accepted forms::

        propulsion: all
        propulsion: [hall_spt100, ion_nstar]
        propulsion: {family: electric}
        propulsion: {include: all, exclude: [ntp_nerva]}

    Ordering is the declared order for explicit lists (so a config author can
    control the order of a report), and sorted order for ``all`` / metadata
    queries (so it does not depend on import order).
    """
    if sel is None or sel == "all" or sel == ["all"]:
        names: list[str] = list(registry.names())
    elif isinstance(sel, str):
        names = [sel]
    elif isinstance(sel, Mapping):
        include = sel.get("include", "all")
        filters = {
            k: v
            for k, v in sel.items()
            if k not in ("include", "exclude") and v is not None
        }
        if include in (None, "all", ["all"]):
            base = list(registry.names(**filters)) if filters else list(registry.names())
        else:
            base = list(resolve_names(include, registry, what))
            if filters:
                base = [n for n in base if all(registry.meta(n).get(k) == v
                                               for k, v in filters.items())]
        drop = set(resolve_names(sel.get("exclude"), registry, what)) if sel.get(
            "exclude"
        ) else set()
        names = [n for n in base if n not in drop]
    elif isinstance(sel, Iterable):
        names = [str(s) for s in sel]
    else:
        raise TypeError(f"cannot interpret {what} selector: {sel!r}")

    out: list[str] = []
    for n in names:
        key = str(n).lower()
        if key not in registry:
            raise KeyError(
                f"unknown {what} '{n}'. Registered: {sorted(registry.names())}"
            )
        if key not in out:  # de-duplicate, keep first occurrence
            out.append(key)
    if not out:
        raise ValueError(
            f"{what} selector {sel!r} resolved to nothing. "
            f"Registered: {sorted(registry.names())}"
        )
    return tuple(out)


# --- sweep config ------------------------------------------------------------
@dataclass
class SweepConfig:
    """Parsed YAML sweep definition. Plain data; no I/O beyond :meth:`from_yaml`."""

    name: str = "unnamed"
    description: str = ""
    propulsion: Selector = None
    missions: Selector = None
    agents: Selector = None
    cost_models: Selector = None
    seeds: int | Sequence[int] = 5
    train_steps: int = 20_000
    eval_episodes: int = 10
    val_episodes: int = 5
    eval_interval: int = 0
    #: Per-agent overrides of any scalar spec field, e.g.
    #: ``per_agent: {cem_mpc: {eval_episodes: 5}}``.
    per_agent: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Constructor kwargs forwarded to ``AGENT.make``, per agent name.
    agent_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Forwarded to ``EnvConfig``; reserved keys ``propulsion_kwargs`` and
    #: ``mission_kwargs`` are peeled off by the runner.
    env: Mapping[str, Any] = field(default_factory=dict)
    #: Extra hand-written exclusions, each a partial match dict.
    exclude: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    #: Whitelist; when non-empty only cells matching one entry survive.
    include_only: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    allow_implausible: bool = False
    #: Which stored policy the headline row reports: "best" (chosen on the
    #: validation seeds) or "final".
    report_policy: str = "best"
    #: Metric used to pick the best checkpoint during training.
    selection_metric: str = "return"
    results_dir: str = "results"
    workers: int = 1
    save_telemetry: bool = True
    source: str = ""

    @staticmethod
    def from_dict(data: Mapping[str, Any], source: str = "") -> "SweepConfig":
        known = {f for f in SweepConfig.__dataclass_fields__}
        # Tolerate documentation-only keys so configs can carry prose.
        unknown = {k for k in data if k not in known and not k.startswith("_")}
        if unknown:
            logger.warning(
                "sweep config %s: ignoring unknown keys %s", source or "<dict>",
                sorted(unknown),
            )
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("source", source)
        cfg = SweepConfig(**kwargs)
        cfg.validate()
        return cfg

    @staticmethod
    def from_yaml(path: str | Path) -> "SweepConfig":
        import yaml

        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{p} must contain a YAML mapping, got {type(data).__name__}")
        cfg = SweepConfig.from_dict(data, source=str(p))
        if cfg.name == "unnamed":
            cfg.name = p.stem
        return cfg

    def validate(self) -> None:
        if self.report_policy not in ("best", "final"):
            raise ValueError(
                f"report_policy must be 'best' or 'final', got {self.report_policy!r}"
            )
        if self.train_steps < 0:
            raise ValueError("train_steps must be >= 0")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")
        if isinstance(self.seeds, int) and self.seeds < 1:
            raise ValueError("seeds must be >= 1")

    def seed_list(self) -> tuple[int, ...]:
        """The seeds every cell is repeated over. Identical across cells: that
        is what makes the seed-level comparisons paired rather than independent."""
        if isinstance(self.seeds, int):
            return tuple(range(self.seeds))
        return tuple(int(s) for s in self.seeds)


# --- the matrix --------------------------------------------------------------
class ExperimentMatrix:
    """The expanded, filtered, deterministically ordered set of cells."""

    def __init__(
        self,
        specs: Sequence[ExperimentSpec],
        exclusions: Sequence[Exclusion] = (),
        config: SweepConfig | None = None,
    ) -> None:
        self.specs: tuple[ExperimentSpec, ...] = tuple(specs)
        self.exclusions: tuple[Exclusion, ...] = tuple(exclusions)
        self.config = config
        dupes = len(self.specs) - len({s.run_id for s in self.specs})
        if dupes:
            raise ValueError(f"matrix contains {dupes} duplicate run_ids")

    # --- construction --------------------------------------------------------
    @staticmethod
    def from_config(cfg: SweepConfig) -> "ExperimentMatrix":
        """Expand the full cross product, then filter it.

        Iteration order is (mission, propulsion, agent, cost_model, seed) using
        each axis's resolved order. That groups a report by mission -- the way a
        reader consumes it -- and is a pure function of the config.
        """
        missions = resolve_names(cfg.missions, MISSION, "mission")
        props = resolve_names(cfg.propulsion, PROPULSION, "propulsion system")
        agents = resolve_names(cfg.agents, AGENT, "agent")
        costs = resolve_names(cfg.cost_models, COST_MODEL, "cost model")
        seeds = cfg.seed_list()

        env_cfg = _canonical(cfg.env)
        specs: list[ExperimentSpec] = []
        exclusions: list[Exclusion] = []

        for mission in missions:
            for prop in props:
                for agent in agents:
                    rule = _first_blocking_rule(cfg, prop, mission, agent)
                    if rule is not None:
                        for cost in costs:
                            exclusions.append(
                                Exclusion(prop, mission, agent, cost, *rule)
                            )
                        continue
                    for cost in costs:
                        if not _passes_include_only(cfg, prop, mission, agent, cost):
                            exclusions.append(
                                Exclusion(
                                    prop, mission, agent, cost, "include_only",
                                    "not matched by the config's include_only whitelist",
                                )
                            )
                            continue
                        over = dict(cfg.per_agent.get(agent, {}))
                        akw = _canonical(cfg.agent_kwargs.get(agent, {}))
                        train_steps = int(over.get("train_steps", cfg.train_steps))
                        for seed in seeds:
                            specs.append(
                                ExperimentSpec(
                                    propulsion=prop,
                                    mission=mission,
                                    agent=agent,
                                    cost_model=cost,
                                    seed=int(seed),
                                    train_steps=train_steps,
                                    eval_episodes=int(
                                        over.get("eval_episodes", cfg.eval_episodes)
                                    ),
                                    val_episodes=int(
                                        over.get("val_episodes", cfg.val_episodes)
                                    ),
                                    eval_interval=int(
                                        over.get("eval_interval", cfg.eval_interval)
                                    ),
                                    agent_kwargs=akw,
                                    env_config=env_cfg,
                                    sweep=cfg.name,
                                    tags={"source": cfg.source},
                                )
                            )
        matrix = ExperimentMatrix(specs, exclusions, cfg)
        logger.info(
            "matrix '%s': %d cells (%d unique configurations x %d seeds), "
            "%d pairings excluded",
            cfg.name, len(matrix), len(matrix.cell_keys()), len(seeds),
            len({(e.propulsion, e.mission, e.agent) for e in exclusions}),
        )
        return matrix

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentMatrix":
        return ExperimentMatrix.from_config(SweepConfig.from_yaml(path))

    # --- views ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[ExperimentSpec]:
        return iter(self.specs)

    def __getitem__(self, i: int) -> ExperimentSpec:
        return self.specs[i]

    def cell_keys(self) -> tuple[str, ...]:
        """Distinct (agent, propulsion, mission, cost) groups, in first-seen order."""
        seen: list[str] = []
        for s in self.specs:
            if s.cell_key not in seen:
                seen.append(s.cell_key)
        return tuple(seen)

    def groups(self) -> dict[str, tuple[ExperimentSpec, ...]]:
        """Cells grouped by :attr:`ExperimentSpec.cell_key`, i.e. by what gets
        averaged over seeds."""
        out: dict[str, list[ExperimentSpec]] = {}
        for s in self.specs:
            out.setdefault(s.cell_key, []).append(s)
        return {k: tuple(v) for k, v in out.items()}

    def filter(self, **match: Any) -> "ExperimentMatrix":
        """Sub-matrix whose cells match every given field value."""
        keep = [
            s for s in self.specs
            if all(getattr(s, k, None) == v for k, v in match.items())
        ]
        return ExperimentMatrix(keep, self.exclusions, self.config)

    def learning_cells(self) -> tuple[ExperimentSpec, ...]:
        return tuple(s for s in self.specs if s.learns and s.train_steps > 0)

    def total_train_steps(self) -> int:
        """Total environment steps of training the sweep will spend."""
        return sum(s.train_steps for s in self.specs if s.learns)

    def total_eval_episodes(self) -> int:
        n_evals = 0
        for s in self.specs:
            periodic = (
                (s.train_steps // s.eval_interval) if (s.eval_interval and s.learns)
                else 0
            )
            n_evals += s.eval_episodes + s.val_episodes * (periodic + 1)
        return n_evals

    def summary(self) -> str:
        """Human-readable plan, printed by ``run --dry-run``."""
        axes = {
            "missions": sorted({s.mission for s in self.specs}),
            "propulsion": sorted({s.propulsion for s in self.specs}),
            "agents": sorted({s.agent for s in self.specs}),
            "cost_models": sorted({s.cost_model for s in self.specs}),
            "seeds": sorted({s.seed for s in self.specs}),
        }
        lines = [
            f"sweep: {self.config.name if self.config else '?'}",
            f"cells: {len(self)}  ({len(self.cell_keys())} configurations "
            f"x {len(axes['seeds'])} seeds)",
            f"training budget: {self.total_train_steps():,} env steps",
            f"evaluation episodes: {self.total_eval_episodes():,}",
        ]
        for k, v in axes.items():
            lines.append(f"  {k:<12} ({len(v)}): {', '.join(str(x) for x in v)}")
        if self.exclusions:
            lines.append(f"excluded pairings ({len(self.excluded_pairings())}):")
            for (p, m, a), (rule, reason) in self.excluded_pairings().items():
                who = f"{a}@{p} on {m}" if a != "*" else f"{p} on {m}"
                lines.append(f"  - {who}  [{rule}]")
                lines.append(f"      {_wrap(reason, 74, '      ')}")
        return "\n".join(lines)

    def excluded_pairings(self) -> dict[tuple[str, str, str], tuple[str, str]]:
        """Excluded (propulsion, mission, agent) triples -> (rule, reason).

        Collapses the cost-model axis, which never participates in a
        plausibility decision, so the report reads as one line per judgement.
        """
        out: dict[tuple[str, str, str], tuple[str, str]] = {}
        for e in self.exclusions:
            out.setdefault((e.propulsion, e.mission, e.agent), (e.rule, e.reason))
        return out

    def to_frame(self):  # -> pandas.DataFrame
        """The plan as a DataFrame, for inspection and for joining onto results."""
        import pandas as pd

        return pd.DataFrame([s.as_row() for s in self.specs])

    def exclusions_frame(self):  # -> pandas.DataFrame
        import pandas as pd

        return pd.DataFrame([e.as_row() for e in self.exclusions])


# --- helpers -----------------------------------------------------------------
def _first_blocking_rule(
    cfg: SweepConfig, prop: str, mission: str, agent: str
) -> tuple[str, str] | None:
    """(rule name, reason) for the first rule that drops this triple, else None."""
    for entry in cfg.exclude:
        if _matches(entry, prop, mission, agent, None):
            return (
                str(entry.get("rule", "config_exclude")),
                str(entry.get("reason", "excluded by the sweep config")),
            )
    if cfg.allow_implausible:
        return None
    for rule in PLAUSIBILITY_RULES:
        if rule.applies(prop, mission, agent):
            return rule.name, rule.reason
    return None


def _passes_include_only(
    cfg: SweepConfig, prop: str, mission: str, agent: str, cost: str
) -> bool:
    if not cfg.include_only:
        return True
    return any(
        _matches(entry, prop, mission, agent, cost) for entry in cfg.include_only
    )


def _matches(
    entry: Mapping[str, Any], prop: str, mission: str, agent: str, cost: str | None
) -> bool:
    """Partial match: keys absent from ``entry`` are wildcards."""
    fields = {"propulsion": prop, "mission": mission, "agent": agent}
    if cost is not None:
        fields["cost_model"] = cost
    for key, value in entry.items():
        if key in ("rule", "reason"):
            continue
        if key not in fields:
            continue
        wanted = value if isinstance(value, (list, tuple, set)) else [value]
        if fields[key] not in {str(w).lower() for w in wanted}:
            return False
    return True


def _canonical(obj: Any) -> Any:
    """Recursively convert to plain, sorted, JSON-stable containers.

    Guarantees that two configs differing only in key order or in
    list-vs-tuple hash identically.
    """
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _slugify(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(text, width))


__all__ = [
    "PROTOCOL_VERSION",
    "PLAUSIBILITY_RULES",
    "Exclusion",
    "ExperimentMatrix",
    "ExperimentSpec",
    "PlausibilityRule",
    "SweepConfig",
    "propulsion_family",
    "resolve_names",
]
