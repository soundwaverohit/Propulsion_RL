"""Gymnasium interop, without a Gymnasium dependency.

``gymnasium`` is imported lazily inside the functions that need it, so this
module imports cleanly on a machine that has never heard of it -- which is the
default for this project. Only calling :func:`to_gymnasium` or
:func:`register_gym_envs` requires the package to be installed.

The native :class:`~propulsion_rl.envs.propulsion_env.PropulsionEnv` already
speaks the five-tuple API, so the adapter is a thin shell whose real job is
converting the local :class:`~propulsion_rl.core.spaces.Box` into the real
``gymnasium.spaces.Box`` and inheriting from ``gymnasium.Env`` so that
``isinstance`` checks inside stable-baselines3 and CleanRL pass.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from ..core.spaces import Box

LOGGER = logging.getLogger(__name__)

_INSTALL_HINT = (
    "gymnasium is not installed. It is an optional dependency of propulsion_rl; "
    "install it with `pip install gymnasium` to use the Gymnasium adapter. The "
    "native PropulsionEnv needs no such thing."
)

#: Built once, on first use, so repeated calls return the same class and
#: ``isinstance`` comparisons between two adapted envs behave.
_GYM_ENV_CLASS: Any = None


def _require_gymnasium() -> Any:
    """Import gymnasium or explain what is missing."""
    try:
        import gymnasium  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return gymnasium


def _to_gym_space(space: Box, gymnasium: Any) -> Any:
    """Local Box -> ``gymnasium.spaces.Box``, preserving bounds and dtype."""
    return gymnasium.spaces.Box(
        low=np.asarray(space.low),
        high=np.asarray(space.high),
        shape=tuple(space.shape),
        dtype=space.dtype.type,
    )


def _gym_env_class(gymnasium: Any) -> Any:
    """Define (once) the ``gymnasium.Env`` subclass that wraps a native env."""
    global _GYM_ENV_CLASS
    if _GYM_ENV_CLASS is not None:
        return _GYM_ENV_CLASS

    class GymPropulsionEnv(gymnasium.Env):  # type: ignore[misc, name-defined]
        """A real ``gymnasium.Env`` delegating to a native ``PropulsionEnv``.

        ``info`` is passed through untouched, so ``info["constraint_cost"]``
        and the terminal ``info["mission_result"]`` survive the trip.
        """

        metadata = {"render_modes": []}

        def __init__(self, env: Any) -> None:
            self.env = env
            self.observation_space = _to_gym_space(env.observation_space, gymnasium)
            self.action_space = _to_gym_space(env.action_space, gymnasium)
            self.render_mode = None

        def __getattr__(self, name: str) -> Any:
            # Guard the delegate itself: __getattr__ fires before __init__ runs
            # during unpickling, and forwarding "env" would recurse forever.
            if name.startswith("_") or name == "env":
                raise AttributeError(name)
            return getattr(self.env, name)

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            return self.env.reset(seed=seed, options=options)

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            return self.env.step(action)

        def render(self) -> None:
            return None

        def close(self) -> None:
            self.env.close()

        def __repr__(self) -> str:
            return f"<GymPropulsionEnv {self.env!r}>"

    _GYM_ENV_CLASS = GymPropulsionEnv
    return _GYM_ENV_CLASS


def to_gymnasium(env: Any) -> Any:
    """Wrap a native environment in a real ``gymnasium.Env``.

    Raises
    ------
    ImportError
        If gymnasium is not installed, with an actionable message.
    """
    gymnasium = _require_gymnasium()
    return _gym_env_class(gymnasium)(env)


def _gym_entry_point(propulsion: str, mission: str, **kwargs: Any) -> Any:
    """Registered entry point. Importable by path, so ``gymnasium.make`` works."""
    from .propulsion_env import make_env  # noqa: PLC0415 - avoids an import cycle

    return to_gymnasium(make_env(propulsion, mission, **kwargs))


def gym_env_id(propulsion: str, mission: str, namespace: str = "PropulsionRL") -> str:
    """``PropulsionRL/<propulsion>-<mission>-v0``."""
    return f"{namespace}/{propulsion}-{mission}-v0"


def register_gym_envs(
    *,
    propulsion: Iterable[str] | None = None,
    missions: Iterable[str] | None = None,
    namespace: str = "PropulsionRL",
    max_episode_steps: int | None = None,
    force: bool = False,
) -> list[str]:
    """Register every (propulsion, mission) pairing with Gymnasium.

    Returns the ids that were newly registered. Already-registered ids are
    skipped unless ``force`` is set, so calling this twice in one process (a
    common accident in notebooks) is harmless.
    """
    gymnasium = _require_gymnasium()

    import propulsion_rl  # noqa: F401, PLC0415 - populates the registries
    from ..core.registry import MISSION, PROPULSION  # noqa: PLC0415

    systems = list(propulsion) if propulsion is not None else list(PROPULSION)
    tasks = list(missions) if missions is not None else list(MISSION)
    if not systems or not tasks:
        LOGGER.warning(
            "nothing to register: %d propulsion systems, %d missions",
            len(systems),
            len(tasks),
        )
        return []

    registered: list[str] = []
    for system in systems:
        for task in tasks:
            env_id = gym_env_id(system, task, namespace)
            if env_id in gymnasium.registry:
                if not force:
                    continue
                del gymnasium.registry[env_id]
            gymnasium.register(
                id=env_id,
                entry_point="propulsion_rl.envs.gym_adapter:_gym_entry_point",
                kwargs={"propulsion": system, "mission": task},
                max_episode_steps=max_episode_steps,
                disable_env_checker=True,
            )
            registered.append(env_id)
    LOGGER.info("registered %d Gymnasium environment ids", len(registered))
    return registered


__all__ = ["gym_env_id", "register_gym_envs", "to_gymnasium"]
