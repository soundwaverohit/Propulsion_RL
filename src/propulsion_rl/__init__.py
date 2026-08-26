"""Propulsion_RL -- a benchmark for pairing RL methods with propulsion systems.

Importing this package populates the registries in
:mod:`propulsion_rl.core.registry`, so ``PROPULSION.names()`` and friends are
usable immediately.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import constants, types  # noqa: F401
from .core.registry import AGENT, COST_MODEL, MISSION, PROPULSION  # noqa: F401


def _register_all() -> None:
    """Import the subpackages whose import side effect is registration.

    Wrapped in a function and called once at import time so a partially built
    tree (during development) fails loudly on the missing module rather than
    silently registering nothing.
    """
    from . import agents, economics, missions, propulsion  # noqa: F401


_register_all()

__all__ = ["PROPULSION", "MISSION", "AGENT", "COST_MODEL", "__version__"]
