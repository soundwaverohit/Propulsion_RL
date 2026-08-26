"""Shared fixtures and registry-driven parametrisation.

The suite has two layers.

*Contract* tests pin down ``core`` and the abstract base classes. They import
their subjects directly and must pass unconditionally -- if they fail, the
shared vocabulary of the whole package has moved.

*Conformance* tests are generated from the registries at collection time. Any
test that asks for the ``propulsion_name``, ``mission_name``, ``agent_name``,
``cost_model_name`` or ``pairing`` fixture is automatically parametrised over
everything currently registered, so a thruster added by another module is
covered by the entire conformance suite the moment it registers, with no edit
here or in the test modules.

When a registry is empty (or the package will not import at all) the
parametrisation degrades to a single explicitly skipped case carrying the real
reason, rather than silently collecting nothing.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import itertools
import pkgutil
from typing import Any, Callable

import numpy as np
import pytest

# --- package import ----------------------------------------------------------
# The import is guarded so that a half-written implementation module elsewhere
# in the tree produces one clear skip reason per generated test instead of a
# collection crash. The exception text is carried into every skip message, so
# nothing is swallowed.
_IMPORT_ERROR: str | None = None
try:
    import propulsion_rl as _pkg
except Exception as exc:  # noqa: BLE001 - re-surfaced verbatim in skip reasons
    _pkg = None  # type: ignore[assignment]
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

if _pkg is not None:
    from propulsion_rl.core.constants import AU
    from propulsion_rl.core.types import (
        CANONICAL_ACTION_DIM,
        BillOfMaterials,
        CanonicalCommand,
        HealthReport,
        PropulsionFamily,
        StepContext,
        TerminationReason,
        ThrusterOutput,
        VehicleState,
    )
    from propulsion_rl.missions.base import MissionResult


#: How far outside [-1, 1] a "normalised" observation may stray before we call
#: it un-normalised. Generous enough for a transient overshoot, tight enough to
#: catch a raw temperature in kelvin or a raw power in watts.
OBS_ABS_MAX = 1.5

_MISSING = "<none-registered>"


def require_propulsion_rl() -> Any:
    """Return the imported package, or skip the calling module with the reason.

    Guards modules whose subject matter only exists once ``propulsion_rl``
    imports cleanly; contract tests deliberately do *not* use this, because a
    broken package must fail them loudly.
    """
    if _pkg is None:
        pytest.skip(
            f"propulsion_rl does not import yet ({_IMPORT_ERROR})",
            allow_module_level=True,
        )
    return _pkg


# --- registry access ---------------------------------------------------------
_REGISTRY_FIXTURES: dict[str, tuple[str, str]] = {
    "propulsion_name": ("PROPULSION", "propulsion system"),
    "mission_name": ("MISSION", "mission"),
    "agent_name": ("AGENT", "agent"),
    "cost_model_name": ("COST_MODEL", "cost model"),
}


def registry(attr: str) -> Any:
    """The named global registry, or ``None`` when the package is unimportable."""
    return None if _pkg is None else getattr(_pkg, attr, None)


def registered(attr: str) -> list[str]:
    """Sorted names currently in the named registry (empty if unavailable)."""
    reg = registry(attr)
    return list(reg.names()) if reg is not None else []


def _empty_reason(attr: str, kind: str) -> str:
    if _pkg is None:
        return f"propulsion_rl does not import yet ({_IMPORT_ERROR})"
    if registry(attr) is None:
        return f"propulsion_rl exposes no {attr} registry"
    return f"no {kind} registered yet - nothing for the conformance suite to cover"


def _registry_params(attr: str, kind: str) -> list[Any]:
    names = registered(attr)
    if names:
        return [pytest.param(n, id=n) for n in names]
    return [
        pytest.param(
            _MISSING,
            id=_MISSING,
            marks=pytest.mark.skip(reason=_empty_reason(attr, kind)),
        )
    ]


@functools.lru_cache(maxsize=None)
def family_of(name: str) -> str | None:
    """Best-effort propulsion family for *name*, for building readable ids.

    Registration metadata is optional, so fall back to constructing the system
    and reading its attribute. A construction failure is not hidden: it is
    asserted directly by ``test_propulsion_conformance`` and would fail there.
    """
    reg = registry("PROPULSION")
    if reg is None:
        return None
    fam = reg.meta(name).get("family")
    if fam is None:
        try:
            fam = getattr(reg.make(name), "family", None)
        except Exception:  # noqa: BLE001 - id-building only, asserted elsewhere
            return None
    if fam is None:
        return None
    return str(getattr(fam, "value", fam)).lower()


def propulsion_names_in_family(family: str) -> list[str]:
    """Registered propulsion names belonging to *family* ("electric"/"nuclear")."""
    return [n for n in registered("PROPULSION") if family_of(n) == family]


def _pairing_params() -> list[Any]:
    """A representative (propulsion, mission) cross-section, not the full matrix.

    One system per family times the first couple of missions keeps the default
    run fast while still exercising both physics branches end to end.
    """
    props = registered("PROPULSION")
    missions = registered("MISSION")
    if not props or not missions:
        missing = "propulsion systems" if not props else "missions"
        reason = _empty_reason(
            "PROPULSION" if not props else "MISSION", missing
        )
        return [pytest.param((_MISSING, _MISSING), id=_MISSING,
                             marks=pytest.mark.skip(reason=reason))]
    chosen: list[str] = []
    for fam in ("electric", "nuclear"):
        chosen.extend(propulsion_names_in_family(fam)[:1])
    if not chosen:
        chosen = props[:2]
    return [
        pytest.param((p, m), id=f"{p}+{m}")
        for p in chosen
        for m in missions[:2]
    ]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise every registry-driven fixture from the live registries."""
    for fixture, (attr, kind) in _REGISTRY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            metafunc.parametrize(fixture, _registry_params(attr, kind))
    if "pairing" in metafunc.fixturenames:
        metafunc.parametrize("pairing", _pairing_params())


# --- construction helpers ----------------------------------------------------
def _build(attr: str, kind: str, name: str, **kwargs: Any) -> Any:
    reg = registry(attr)
    if reg is None or name == _MISSING:
        pytest.skip(_empty_reason(attr, kind))
    try:
        return reg.make(name, **kwargs)
    except TypeError as exc:
        pytest.fail(
            f"{kind} {name!r} cannot be built through the registry with "
            f"{kwargs or 'no arguments'}: {type(exc).__name__}: {exc}. "
            "Sweeps are configured by name, so a registered factory must accept "
            "the canonical construction call."
        )


def make_propulsion(name: str, **kwargs: Any) -> Any:
    """Construct the named propulsion system through the registry."""
    return _build("PROPULSION", "propulsion system", name, **kwargs)


def make_mission(name: str, **kwargs: Any) -> Any:
    """Construct the named mission through the registry."""
    return _build("MISSION", "mission", name, **kwargs)


def make_agent(name: str, obs_dim: int | None = None,
               action_dim: int | None = None, **kwargs: Any) -> Any:
    """Construct the named agent for the canonical interface widths."""
    from propulsion_rl.core.types import CANONICAL_ACTION_DIM, OBS_DIM

    return _build(
        "AGENT",
        "agent",
        name,
        obs_dim=OBS_DIM if obs_dim is None else obs_dim,
        action_dim=CANONICAL_ACTION_DIM if action_dim is None else action_dim,
        **kwargs,
    )


def make_cost_model(name: str, **kwargs: Any) -> Any:
    """Construct the named cost model through the registry."""
    return _build("COST_MODEL", "cost model", name, **kwargs)


# --- synthetic contract objects ----------------------------------------------
def synthetic_step_context(**over: Any) -> "StepContext":
    """A plausible one-hour cruise step for a few-kilowatt vehicle at 1 AU."""
    kw: dict[str, Any] = dict(
        t_s=0.0,
        dt_s=3600.0,
        vehicle_mass_kg=2000.0,
        available_power_w=10_000.0,
        heliocentric_radius_m=AU,
        sink_temperature_k=3.0,
        eclipse=False,
        rng=np.random.default_rng(0),
    )
    kw.update(over)
    return StepContext(**kw)


def context_for(system: Any, **over: Any) -> "StepContext":
    """A ``StepContext`` scaled to *system*'s own envelope.

    A 5 kW Hall thruster and a 500 MW nuclear stage cannot share a hard-coded
    power budget, so the bus is sized from ``limits()``; every conformance test
    therefore probes each system inside its designed operating range.
    """
    limits = system.limits()
    power = float(limits.max_power_w)
    if not np.isfinite(power) or power <= 0.0:
        power = 10_000.0
    kw: dict[str, Any] = dict(available_power_w=power)
    kw.update(over)
    return synthetic_step_context(**kw)


def synthetic_vehicle_state(**over: Any) -> "VehicleState":
    """A 400 km circular LEO state with a realistic mass split."""
    from propulsion_rl.core.constants import LEO_RADIUS, MU_EARTH

    speed = float(np.sqrt(MU_EARTH / LEO_RADIUS))
    kw: dict[str, Any] = dict(
        position_m=np.array([LEO_RADIUS, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, speed, 0.0]),
        dry_mass_kg=800.0,
        propellant_kg=400.0,
        payload_kg=300.0,
        t_s=0.0,
        power_generated_w=12_000.0,
        power_available_w=10_000.0,
        delta_v_applied_m_s=0.0,
        propellant_used_kg=0.0,
    )
    kw.update(over)
    return VehicleState(**kw)


def synthetic_mission_result(**over: Any) -> "MissionResult":
    """A successful, fully-delivered mission -- the happy path for economics."""
    kw: dict[str, Any] = dict(
        reason=TerminationReason.SUCCESS,
        success=True,
        progress=1.0,
        elapsed_s=180.0 * 86400.0,
        delta_v_m_s=5_000.0,
        propellant_used_kg=350.0,
        payload_delivered_kg=300.0,
        terminal_error=1.0e3,
        constraint_violations=0,
        total_constraint_cost=0.0,
    )
    kw.update(over)
    return MissionResult(**kw)


def synthetic_bom(**over: Any) -> "BillOfMaterials":
    """A mid-size solar-electric tug bill of materials."""
    kw: dict[str, Any] = dict(
        system_name="synthetic",
        family=PropulsionFamily.ELECTRIC,
        thruster_units=2,
        rated_power_w=12_500.0,
        power_source_w=30_000.0,
        reactor_thermal_w=0.0,
        radiator_area_m2=20.0,
        dry_mass_kg=800.0,
        propellant_type="xenon",
        tank_capacity_kg=450.0,
        qualified_life_s=5.0e7,
    )
    kw.update(over)
    return BillOfMaterials(**kw)


def synthetic_health(**over: Any) -> "HealthReport":
    """A part-worn but healthy thruster."""
    kw: dict[str, Any] = dict(
        wear_fraction=0.35,
        remaining_life_s=3.0e7,
        throughput_kg=350.0,
        burn_time_s=1.5e7,
        restarts=42,
        degraded_efficiency=0.95,
        failed=False,
    )
    kw.update(over)
    return HealthReport(**kw)


# --- commands ----------------------------------------------------------------
def command_grid() -> list["CanonicalCommand"]:
    """Corners of the canonical action box plus the two throttle extremes.

    The corners are where saturation logic, sign errors and divide-by-zero
    guards live; a model that only ever sees mid-range commands is untested.
    """
    cmds = [
        CanonicalCommand.from_array(np.array(c, dtype=np.float64))
        for c in itertools.product((-1.0, 1.0), repeat=CANONICAL_ACTION_DIM)
    ]
    cmds.append(CanonicalCommand.from_array(np.zeros(CANONICAL_ACTION_DIM)))
    cmds.append(nominal_command(throttle=0.0))
    cmds.append(nominal_command(throttle=1.0))
    return cmds


def nominal_command(
    throttle: float = 1.0,
    operating_point: float = 0.5,
    thermal_margin: float = 0.5,
) -> "CanonicalCommand":
    """A prograde burn at the requested throttle, mid Isp/thrust trade."""
    return CanonicalCommand(
        throttle=throttle,
        operating_point=operating_point,
        thrust_yaw=0.0,
        thrust_pitch=0.0,
        thermal_margin=thermal_margin,
    )


def output_signature(out: "ThrusterOutput") -> tuple:
    """Hashable, bit-exact snapshot of a ``ThrusterOutput`` for equality checks."""
    return (
        out.thrust_n,
        out.mdot_kg_s,
        out.isp_s,
        out.power_draw_w,
        out.thermal_power_w,
        out.heat_reject_w,
        out.efficiency,
        out.throttled_by,
        tuple((e.name, e.severity, e.value) for e in out.events),
    )


def float_attributes(obj: Any) -> dict[str, float]:
    """Public scalar attributes of *obj*, for invariants over internal state."""
    out: dict[str, float] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float, np.floating)):
            continue
        out[attr] = float(val)
    return out


# --- environment-layer discovery ---------------------------------------------
def _iter_submodules(package_name: str) -> list[str]:
    pkg = importlib.import_module(package_name)
    paths = getattr(pkg, "__path__", [])
    return [f"{package_name}.{m.name}" for m in pkgutil.iter_modules(paths)]


def find_symbol(package_name: str, symbol: str) -> Any:
    """Find *symbol* on a package or any of its immediate submodules.

    The environment layer is written by another module and its file layout is
    not part of any contract, so the suite locates ``make_env`` and friends by
    name instead of guessing an import path.
    """
    try:
        pkg = importlib.import_module(package_name)
    except Exception as exc:  # noqa: BLE001 - reported through the skip below
        pytest.skip(f"{package_name} does not import yet ({exc!r})")
    found = getattr(pkg, symbol, None)
    if found is not None:
        return found
    problems: list[str] = []
    for mod_name in _iter_submodules(package_name):
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - collected into the skip reason
            problems.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue
        found = getattr(mod, symbol, None)
        if found is not None:
            return found
    detail = ("; import failures: " + " | ".join(problems)) if problems else ""
    pytest.skip(f"{symbol!r} not found in {package_name} yet{detail}")


def _match_param(params: Any, needles: tuple[str, ...]) -> str | None:
    for pname in params:
        low = pname.lower()
        if any(n in low for n in needles):
            return pname
    return None


def call_env_factory(factory: Callable[..., Any], propulsion: str, mission: str,
                     **extra: Any) -> Any:
    """Call ``make_env(propulsion, mission, ...)`` under either naming style.

    Binds by parameter name where possible so the suite survives the factory
    being spelled ``make_env(propulsion_name=..., mission_name=...)``.
    """
    params = inspect.signature(factory).parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    p_key = _match_param(params, ("propulsion", "thruster"))
    m_key = _match_param(params, ("mission",))
    kwargs: dict[str, Any] = {}
    for key, value in extra.items():
        if key in params or accepts_kwargs:
            kwargs[key] = value
    if p_key and m_key:
        return factory(**{p_key: propulsion, m_key: mission}, **kwargs)
    return factory(propulsion, mission, **kwargs)


def flatten_info(info: dict[str, Any]) -> dict[str, Any]:
    """Flatten one step's ``info``, expanding telemetry records one level deep."""
    flat: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            flat.update(value)
        elif hasattr(value, "as_row") and callable(value.as_row):
            flat.update(value.as_row())
        else:
            flat[key] = value
    flat.update(info)
    return flat


def find_instance(info: dict[str, Any], cls: type) -> Any:
    """First value in *info* (or one level down) that is an instance of *cls*."""
    for value in info.values():
        if isinstance(value, cls):
            return value
    for value in info.values():
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, cls):
                    return inner
    return None


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def rng() -> np.random.Generator:
    """A generator on a fixed seed, so a failure reproduces on the next run."""
    return np.random.default_rng(20260825)


@pytest.fixture
def make_rng() -> Callable[[int], np.random.Generator]:
    """Factory for independently seeded generators."""
    return lambda seed=0: np.random.default_rng(seed)


@pytest.fixture
def make_step_context() -> Callable[..., Any]:
    """Factory for ``StepContext`` values with keyword overrides."""
    require_propulsion_rl()
    return synthetic_step_context


@pytest.fixture
def make_vehicle_state() -> Callable[..., Any]:
    """Factory for synthetic ``VehicleState`` values with keyword overrides."""
    require_propulsion_rl()
    return synthetic_vehicle_state


@pytest.fixture
def make_mission_result() -> Callable[..., Any]:
    """Factory for synthetic ``MissionResult`` values with keyword overrides."""
    require_propulsion_rl()
    return synthetic_mission_result


@pytest.fixture
def make_bom() -> Callable[..., Any]:
    """Factory for synthetic ``BillOfMaterials`` values with keyword overrides."""
    require_propulsion_rl()
    return synthetic_bom


@pytest.fixture
def make_health() -> Callable[..., Any]:
    """Factory for synthetic ``HealthReport`` values with keyword overrides."""
    require_propulsion_rl()
    return synthetic_health


@pytest.fixture
def propulsion(propulsion_name: str) -> Any:
    """A freshly constructed, not-yet-reset propulsion system."""
    return make_propulsion(propulsion_name)


@pytest.fixture
def mission(mission_name: str) -> Any:
    """A freshly constructed mission."""
    return make_mission(mission_name)


@pytest.fixture
def agent(agent_name: str) -> Any:
    """An agent built for the canonical (36, 5) interface."""
    return make_agent(agent_name)


@pytest.fixture
def cost_model(cost_model_name: str) -> Any:
    """A freshly constructed cost model."""
    return make_cost_model(cost_model_name)
