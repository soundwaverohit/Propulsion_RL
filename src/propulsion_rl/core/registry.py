"""Name -> factory registries.

Everything the experiment matrix iterates over -- propulsion systems, missions,
agents, cost models -- is registered here, so a sweep is configured by strings
in a YAML file rather than by imports. Registration happens as a side effect of
importing the owning subpackage; see ``propulsion_rl/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")


class Registry:
    """A tiny string-keyed factory table with helpful failure messages."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., Any]] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def register(
        self, name: str, **meta: Any
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: ``@REGISTRY.register("hall_thruster", family="electric")``."""

        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._factories:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._factories[key] = factory
            self._meta[key] = meta
            return factory

        return deco

    def add(self, name: str, factory: Callable[..., Any], **meta: Any) -> None:
        """Imperative form of :meth:`register`, for dynamically built entries."""
        key = name.lower()
        if key in self._factories:
            raise KeyError(f"{self.kind} '{name}' is already registered")
        self._factories[key] = factory
        self._meta[key] = meta

    def make(self, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in self._factories:
            raise KeyError(
                f"unknown {self.kind} '{name}'. Registered: {sorted(self._factories)}"
            )
        return self._factories[key](**kwargs)

    def meta(self, name: str) -> dict[str, Any]:
        return dict(self._meta.get(name.lower(), {}))

    def names(self, **filters: Any) -> list[str]:
        """Registered names, optionally filtered on metadata equality."""
        out = []
        for key, meta in self._meta.items():
            if all(meta.get(k) == v for k, v in filters.items()):
                out.append(key)
        return sorted(out)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._factories))

    def __len__(self) -> int:
        return len(self._factories)


PROPULSION = Registry("propulsion system")
MISSION = Registry("mission")
AGENT = Registry("agent")
COST_MODEL = Registry("cost model")
