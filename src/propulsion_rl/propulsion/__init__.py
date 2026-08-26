"""Propulsion-system models and registry population.

Importing :mod:`propulsion_rl.propulsion` loads both technology families.  The
family packages own their preset registrations; this module only provides the
stable public surface shared by users and by :mod:`propulsion_rl.__init__`.

Concrete classes declared public by either family are re-exported here.  Doing
that after both packages have finished importing keeps the dependency direction
simple: implementations depend on :mod:`.base`, while :mod:`.base` never
depends on an implementation or on the registry-population side effects.
"""

from __future__ import annotations

from types import ModuleType

from .base import PropulsionSystem

# These imports are intentionally eager: ``import propulsion_rl`` promises a
# fully populated PROPULSION registry, and its _register_all() function reaches
# only this package.  Each family package imports its own concrete presets.
from . import electric as electric  # noqa: E402
from . import nuclear as nuclear  # noqa: E402


def _reexport_concrete(module: ModuleType) -> list[str]:
    """Expose the family's public concrete ``PropulsionSystem`` classes.

    Family modules remain free to publish design records and physics helpers in
    their own ``__all__``.  The top-level propulsion namespace stays narrower:
    the common contract, family namespaces, and concrete systems a caller can
    instantiate.  A duplicate class name across families is rejected loudly
    rather than silently changing which implementation an import resolves to.
    """

    exported: list[str] = []
    for name in getattr(module, "__all__", ()):
        value = getattr(module, name, None)
        if not (
            isinstance(value, type)
            and value is not PropulsionSystem
            and issubclass(value, PropulsionSystem)
        ):
            continue
        existing = globals().get(name)
        if existing is not None and existing is not value:
            raise ImportError(
                f"propulsion class name {name!r} is public in more than one family"
            )
        globals()[name] = value
        exported.append(name)
    return exported


_CONCRETE_EXPORTS = _reexport_concrete(electric) + _reexport_concrete(nuclear)

__all__ = ["PropulsionSystem", "electric", "nuclear", *_CONCRETE_EXPORTS]

del ModuleType, _CONCRETE_EXPORTS
