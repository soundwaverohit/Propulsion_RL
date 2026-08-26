"""Contract tests for the name -> factory registries.

Sweeps are configured by strings in YAML, so the registry is the seam between
configuration and code. Every test here builds its own throwaway ``Registry``:
mutating the four global ones would leak between test modules and corrupt the
conformance parametrisation.
"""

from __future__ import annotations

import pytest

from propulsion_rl.core.registry import AGENT, COST_MODEL, MISSION, PROPULSION, Registry


class _Widget:
    def __init__(self, size: int = 1) -> None:
        self.size = size


def test_decorator_registers_and_returns_the_factory_unchanged() -> None:
    """``@register`` must be transparent: the decorated class stays importable
    and usable directly, not replaced by a wrapper."""
    reg = Registry("widget")

    @reg.register("alpha", family="electric")
    class Alpha(_Widget):
        pass

    assert Alpha is not None
    assert Alpha(size=3).size == 3
    assert "alpha" in reg
    assert len(reg) == 1
    assert isinstance(reg.make("alpha"), Alpha)


def test_make_forwards_keyword_arguments() -> None:
    """Config files supply per-entry keyword arguments; they must reach the
    factory untouched."""
    reg = Registry("widget")
    reg.add("beta", _Widget)
    assert reg.make("beta", size=7).size == 7


def test_names_are_case_insensitive() -> None:
    """YAML written by hand mixes case; lookup must not care."""
    reg = Registry("widget")
    reg.add("MixedCase", _Widget)
    assert "mixedcase" in reg
    assert "MIXEDCASE" in reg
    assert reg.make("MiXeDcAsE") is not None
    assert reg.names() == ["mixedcase"]


def test_duplicate_registration_is_rejected() -> None:
    """Two modules claiming one name would make the sweep silently depend on
    import order, so the second registration is a hard error."""
    reg = Registry("propulsion system")
    reg.add("hall", _Widget)
    with pytest.raises(KeyError, match="already registered"):
        reg.add("hall", _Widget)
    with pytest.raises(KeyError, match="already registered"):
        reg.register("HALL")(_Widget)


def test_duplicate_error_names_the_kind_and_the_key() -> None:
    reg = Registry("cost model")
    reg.add("optimistic", _Widget)
    with pytest.raises(KeyError) as excinfo:
        reg.add("optimistic", _Widget)
    message = str(excinfo.value)
    assert "cost model" in message
    assert "optimistic" in message


def test_unknown_name_error_lists_what_is_available() -> None:
    """A typo in a config should be diagnosable from the traceback alone, so
    the error names the kind, the bad key and every registered alternative."""
    reg = Registry("mission")
    reg.add("mars_transfer", _Widget)
    reg.add("geo_raise", _Widget)
    with pytest.raises(KeyError) as excinfo:
        reg.make("mars_transfr")
    message = str(excinfo.value)
    assert "mission" in message
    assert "mars_transfr" in message
    assert "mars_transfer" in message
    assert "geo_raise" in message


def test_metadata_filtering() -> None:
    """The experiment matrix selects whole families by metadata; filtering must
    be exact-match on every supplied key and must ignore unrelated entries."""
    reg = Registry("propulsion system")
    reg.add("hall", _Widget, family="electric", power_class="high")
    reg.add("gridded_ion", _Widget, family="electric", power_class="low")
    reg.add("ntp", _Widget, family="nuclear", power_class="high")

    assert reg.names() == ["gridded_ion", "hall", "ntp"]
    assert reg.names(family="electric") == ["gridded_ion", "hall"]
    assert reg.names(family="nuclear") == ["ntp"]
    assert reg.names(family="electric", power_class="high") == ["hall"]
    assert reg.names(family="antimatter") == []


def test_metadata_filtering_skips_entries_without_the_key() -> None:
    """An entry registered with no metadata must not accidentally match a
    filter, and must still be listed by an unfiltered query."""
    reg = Registry("agent")
    reg.add("scripted", _Widget)
    reg.add("ppo", _Widget, learns=True)
    assert reg.names() == ["ppo", "scripted"]
    assert reg.names(learns=True) == ["ppo"]


def test_meta_returns_a_defensive_copy() -> None:
    """Callers annotate metadata for reports; that must not rewrite the
    registry's own record."""
    reg = Registry("propulsion system")
    reg.add("hall", _Widget, family="electric")
    meta = reg.meta("hall")
    meta["family"] = "tampered"
    assert reg.meta("hall")["family"] == "electric"
    assert reg.names(family="electric") == ["hall"]


def test_meta_of_unknown_name_is_empty() -> None:
    assert Registry("agent").meta("nope") == {}


def test_iteration_is_sorted_and_len_counts_entries() -> None:
    """Report ordering must be stable across runs, so iteration is sorted."""
    reg = Registry("agent")
    for name in ("zeta", "alpha", "mu"):
        reg.add(name, _Widget)
    assert list(reg) == ["alpha", "mu", "zeta"]
    assert len(reg) == 3


def test_registries_are_independent() -> None:
    """The four global tables share a class but never share state."""
    a, b = Registry("agent"), Registry("mission")
    a.add("shared_name", _Widget)
    assert "shared_name" not in b
    assert len(b) == 0


def test_the_four_global_registries_exist_and_are_distinct() -> None:
    """Every sweep dimension has exactly one table, each self-describing for
    error messages."""
    globals_ = [PROPULSION, MISSION, AGENT, COST_MODEL]
    assert len({id(r) for r in globals_}) == 4
    assert [r.kind for r in globals_] == [
        "propulsion system",
        "mission",
        "agent",
        "cost model",
    ]


def test_global_registry_names_are_lowercase_and_unique() -> None:
    """Whatever has registered so far must obey the key convention, otherwise
    a config file that spells a name correctly still misses."""
    for reg in (PROPULSION, MISSION, AGENT, COST_MODEL):
        names = reg.names()
        assert names == sorted(names)
        assert len(names) == len(set(names))
        for name in names:
            assert name == name.lower()
            assert name.strip() == name
            assert name in reg
