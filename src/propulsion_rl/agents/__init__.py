"""Every controller in the benchmark, registered by importing this package.

``import propulsion_rl`` reaches here, and reaching here is what populates
:data:`propulsion_rl.core.registry.AGENT`. After that a sweep can name any of
these in a YAML file and never import an agent module directly::

    from propulsion_rl import AGENT
    agent = AGENT.make("edelbaum", obs_dim=36, action_dim=5)
    AGENT.names(kind="scripted")     # the reference controllers
    AGENT.names(constrained=True)    # the constrained-RL entries

Registry metadata carried by every entry
----------------------------------------
``kind``
    ``"scripted"`` (hand-written reference), ``"on_policy"``, ``"off_policy"``,
    ``"planning"`` or ``"evolution"``. The comparison groups on this.
``learns``
    Whether the runner should spend a training budget on it.
``constrained``
    Whether it consumes ``Transition.cost`` rather than a reward-folded penalty.

The modules split by owner: ``networks``, ``buffers``, ``ppo``, ``sac`` and
``td3`` are shared infrastructure and the model-free learners; ``baselines``,
``lagrangian_ppo``, ``cem_mpc`` and ``cmaes`` are the reference controllers and
the three algorithms built on top of that infrastructure. A tree missing any of
them fails here with a message naming the module, rather than with an
``ImportError`` from four frames deep inside whichever agent happened to import
it first.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from ..core.registry import AGENT
from .base import Agent, ScriptedAgent, TrainStats, Transition

logger = logging.getLogger(__name__)

#: Shared infrastructure plus the model-free learners.
_CORE_MODULES = ("networks", "buffers", "ppo", "sac", "td3")
#: Reference controllers and the algorithms layered on the core.
_EXTRA_MODULES = ("baselines", "lagrangian_ppo", "cem_mpc", "cmaes")


def _import_module(name: str) -> ModuleType:
    """Import one agent module, or fail with a message that names it."""
    dotted = f"{__name__}.{name}"
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        if exc.name in (dotted, name):
            raise ImportError(
                f"propulsion_rl.agents: required module '{dotted}' is missing. "
                f"The agent package expects {', '.join(_CORE_MODULES)} "
                f"(shared networks, buffers and the model-free learners) and "
                f"{', '.join(_EXTRA_MODULES)} (reference controllers, "
                f"constrained PPO, CEM-MPC, CMA-ES). Add the file or drop the "
                f"name from _CORE_MODULES/_EXTRA_MODULES in "
                f"propulsion_rl/agents/__init__.py."
            ) from exc
        raise ImportError(
            f"propulsion_rl.agents: '{dotted}' exists but could not be "
            f"imported because it needs '{exc.name}', which is missing."
        ) from exc


for _name in (*_CORE_MODULES, *_EXTRA_MODULES):
    _import_module(_name)

from .baselines import (  # noqa: E402
    BangBangAgent,
    EdelbaumAgent,
    LifeAwareAgent,
    MaxIspAgent,
    MaxThrustAgent,
    ObsLayout,
    ObsReader,
    PIDAgent,
    ProgradeAgent,
    RandomAgent,
)
from .buffers import ReplayBuffer, RolloutBuffer  # noqa: E402
from .cem_mpc import CEMMPCAgent, EnsembleDynamics  # noqa: E402
from .cmaes import CMAES, CMAESAgent, FlatMLPPolicy  # noqa: E402
from .lagrangian_ppo import LagrangianPPOAgent  # noqa: E402
from .networks import (  # noqa: E402
    MLP,
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
    RunningMeanStd,
    SquashedGaussianPolicy,
    ValueNetwork,
)
from .ppo import PPOAgent  # noqa: E402
from .sac import SACAgent  # noqa: E402
from .td3 import TD3Agent  # noqa: E402

# ``ppo``/``sac``/``td3`` define their agents but leave registration to this
# module, so the whole registry is described in one readable place. The
# decorated agents in baselines/lagrangian_ppo/cem_mpc/cmaes registered
# themselves on import above.
_LEARNER_ENTRIES: tuple[tuple[str, type[Agent], dict[str, object]], ...] = (
    ("ppo", PPOAgent, {"kind": "on_policy", "learns": True, "constrained": False}),
    ("sac", SACAgent, {"kind": "off_policy", "learns": True, "constrained": False}),
    ("td3", TD3Agent, {"kind": "off_policy", "learns": True, "constrained": False}),
)

for _key, _cls, _meta in _LEARNER_ENTRIES:
    if _key in AGENT:
        # Another module got there first (an agent decorating itself later, or
        # a re-import under a different package path). Its registration wins;
        # duplicating it would raise out of Registry.register.
        logger.debug("agent %r already registered; leaving it alone", _key)
    else:
        AGENT.add(_key, _cls, **_meta)

del _key, _cls, _meta, _name

__all__ = [
    # contract
    "Agent",
    "ScriptedAgent",
    "TrainStats",
    "Transition",
    # scripted reference controllers
    "BangBangAgent",
    "EdelbaumAgent",
    "LifeAwareAgent",
    "MaxIspAgent",
    "MaxThrustAgent",
    "PIDAgent",
    "ProgradeAgent",
    "RandomAgent",
    "ObsLayout",
    "ObsReader",
    # learners
    "CEMMPCAgent",
    "CMAESAgent",
    "LagrangianPPOAgent",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    # shared components
    "CMAES",
    "EnsembleDynamics",
    "FlatMLPPolicy",
    "MLP",
    "DeterministicPolicy",
    "GaussianPolicy",
    "QNetwork",
    "ReplayBuffer",
    "RolloutBuffer",
    "RunningMeanStd",
    "SquashedGaussianPolicy",
    "ValueNetwork",
]
