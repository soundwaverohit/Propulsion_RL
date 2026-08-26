"""Batched environments.

On-policy algorithms need throughput more than they need anything else, and the
propulsion physics here is Python-heavy enough that a single environment leaves
a lot on the table.

Auto-reset semantics
--------------------
Both classes auto-reset a sub-environment the moment it finishes, and return
the *reset* observation in the batch. The true terminal observation is preserved
in ``info["final_observation"][i]``, with ``info["_final_observation"]`` as the
boolean mask of which entries are real.

This matters more than it looks. Generalised advantage estimation bootstraps the
last step of a truncated episode with ``V(s_terminal)``. If the batch quietly
contains the reset observation instead, the critic bootstraps off the *start of
the next episode* -- values that are typically an order of magnitude different
-- and the advantages for the final step of every episode are wrong. Nothing
crashes; the learning curve is just worse than it should be, which is the
hardest class of bug to find. Hence the explicit test in the self-test.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from typing import Any, Callable, Sequence

import numpy as np

from ..core.spaces import Box

LOGGER = logging.getLogger(__name__)

EnvFn = Callable[[], Any]


def _batch_space(space: Box, num_envs: int) -> Box:
    """Stack a single-env Box into a batched one."""
    low = np.broadcast_to(space.low, (num_envs,) + space.shape)
    high = np.broadcast_to(space.high, (num_envs,) + space.shape)
    return Box(low, high, (num_envs,) + space.shape, space.dtype)


def _seed_list(seed: int | Sequence[int | None] | None, num_envs: int) -> list[int | None]:
    """One seed per sub-environment; an int seeds them as ``seed + i``."""
    if seed is None:
        return [None] * num_envs
    if isinstance(seed, (int, np.integer)):
        return [int(seed) + i for i in range(num_envs)]
    seeds = list(seed)
    if len(seeds) != num_envs:
        raise ValueError(f"expected {num_envs} seeds, got {len(seeds)}")
    return seeds


class SyncVectorEnv:
    """``num_envs`` environments stepped in-process, one after another.

    Satisfies :class:`~propulsion_rl.core.protocols.VectorEnvProtocol`.
    ``observation_space`` and ``action_space`` are the *batched* spaces (leading
    ``num_envs`` axis), matching the Gymnasium convention;
    ``single_observation_space`` and ``single_action_space`` are the per-env
    ones.

    ``info`` from :meth:`step` always carries ``"constraint_cost"`` and
    ``"progress"`` as ``(num_envs,)`` float arrays, and ``"env_info"`` as an
    object array of the untouched per-environment dicts.
    """

    def __init__(self, env_fns: Sequence[EnvFn], *, copy: bool = True) -> None:
        self.env_fns = list(env_fns)
        if not self.env_fns:
            raise ValueError("SyncVectorEnv needs at least one env_fn")
        self.envs = [fn() for fn in self.env_fns]
        self.num_envs = len(self.envs)
        self.copy = bool(copy)
        self.closed = False

        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = _batch_space(self.single_observation_space, self.num_envs)
        self.action_space = _batch_space(self.single_action_space, self.num_envs)

        obs_shape = tuple(self.single_observation_space.shape)
        self._observations = np.zeros((self.num_envs,) + obs_shape, dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._terminateds = np.zeros(self.num_envs, dtype=bool)
        self._truncateds = np.zeros(self.num_envs, dtype=bool)
        self._costs = np.zeros(self.num_envs, dtype=np.float64)
        self._progress = np.zeros(self.num_envs, dtype=np.float64)

    def _out(self) -> np.ndarray:
        return self._observations.copy() if self.copy else self._observations

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seeds = _seed_list(seed, self.num_envs)
        env_infos = np.empty(self.num_envs, dtype=object)
        for i, (env, sub_seed) in enumerate(zip(self.envs, seeds)):
            obs, info = env.reset(seed=sub_seed, options=options)
            self._observations[i] = obs
            self._costs[i] = float(info.get("constraint_cost", 0.0))
            self._progress[i] = float(info.get("progress", 0.0))
            env_infos[i] = info
        infos: dict[str, Any] = {
            "constraint_cost": self._costs.copy(),
            "progress": self._progress.copy(),
            "env_info": env_infos,
        }
        return self._out(), infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got shape {actions.shape}"
            )
        env_infos = np.empty(self.num_envs, dtype=object)
        final_obs: np.ndarray | None = None
        final_infos: np.ndarray | None = None
        done_mask: np.ndarray | None = None

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            self._rewards[i] = reward
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated
            self._costs[i] = float(info.get("constraint_cost", 0.0))
            self._progress[i] = float(info.get("progress", 0.0))
            env_infos[i] = info

            if terminated or truncated:
                if final_obs is None:
                    final_obs = np.empty(self.num_envs, dtype=object)
                    final_infos = np.empty(self.num_envs, dtype=object)
                    done_mask = np.zeros(self.num_envs, dtype=bool)
                # Copy before the reset below can touch anything.
                final_obs[i] = np.array(obs, dtype=np.float32, copy=True)
                final_infos[i] = info
                done_mask[i] = True
                obs, _reset_info = env.reset()
            self._observations[i] = obs

        infos: dict[str, Any] = {
            "constraint_cost": self._costs.copy(),
            "progress": self._progress.copy(),
            "env_info": env_infos,
        }
        if final_obs is not None:
            infos["final_observation"] = final_obs
            infos["_final_observation"] = done_mask
            infos["final_info"] = final_infos
            infos["_final_info"] = done_mask
        return (
            self._out(),
            self._rewards.copy(),
            self._terminateds.copy(),
            self._truncateds.copy(),
            infos,
        )

    # --- sub-environment access ---------------------------------------------
    def call(self, name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method (or read an attribute) on every sub-environment."""
        out = []
        for env in self.envs:
            attr = getattr(env, name)
            out.append(attr(*args, **kwargs) if callable(attr) else attr)
        return out

    def get_attr(self, name: str) -> list[Any]:
        return [getattr(env, name) for env in self.envs]

    def set_attr(self, name: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} values, got {len(values)}")
        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close(self) -> None:
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:
        return f"<SyncVectorEnv num_envs={self.num_envs}>"


# --- multiprocessing ---------------------------------------------------------
_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CALL = "call"
_CMD_GET = "get_attr"
_CMD_SET = "set_attr"
_CMD_CLOSE = "close"


def _worker(remote: Any, parent_remote: Any, env_fn: EnvFn) -> None:
    """Child-process loop. Never raises out; errors are shipped to the parent."""
    parent_remote.close()
    env = None
    try:
        env = env_fn()
        while True:
            try:
                command, payload = remote.recv()
            except EOFError:
                break
            if command == _CMD_STEP:
                obs, reward, terminated, truncated, info = env.step(payload)
                final_obs = None
                final_info = None
                if terminated or truncated:
                    final_obs = np.array(obs, dtype=np.float32, copy=True)
                    final_info = info
                    obs, _ = env.reset()
                remote.send(
                    ("ok", (obs, reward, terminated, truncated, info, final_obs, final_info))
                )
            elif command == _CMD_RESET:
                seed, options = payload
                remote.send(("ok", env.reset(seed=seed, options=options)))
            elif command == _CMD_CALL:
                name, args, kwargs = payload
                attr = getattr(env, name)
                remote.send(("ok", attr(*args, **kwargs) if callable(attr) else attr))
            elif command == _CMD_GET:
                remote.send(("ok", getattr(env, payload)))
            elif command == _CMD_SET:
                setattr(env, payload[0], payload[1])
                remote.send(("ok", None))
            elif command == _CMD_CLOSE:
                remote.send(("ok", None))
                break
            else:
                remote.send(("error", f"unknown command {command!r}"))
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        pass
    except Exception:  # pragma: no cover - reported to the parent instead
        try:
            remote.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except Exception:
            pass


class AsyncVectorEnv:
    """``num_envs`` environments in worker processes, stepped in parallel.

    Same observation, action and ``info`` contract as :class:`SyncVectorEnv`,
    including the ``final_observation`` handling (done inside the worker, so the
    terminal observation crosses the pipe rather than being reconstructed).

    Start method
    ------------
    Defaults to ``fork`` where the platform has it, because the natural
    ``env_fns`` are ``lambda: make_env(...)`` closures and those are not
    picklable, which is what ``spawn`` requires. Pass ``start_method="spawn"``
    together with module-level, picklable factory callables if you need it --
    e.g. ``functools.partial(make_env, "ion_nstar", "earth_mars_cargo")``.

    Worker exceptions are re-raised in the parent with the child's traceback
    attached. :meth:`close` is idempotent and is also called from ``__del__``.
    """

    def __init__(
        self,
        env_fns: Sequence[EnvFn],
        *,
        start_method: str | None = None,
        timeout: float | None = 120.0,
    ) -> None:
        self.env_fns = list(env_fns)
        if not self.env_fns:
            raise ValueError("AsyncVectorEnv needs at least one env_fn")
        self.num_envs = len(self.env_fns)
        self.timeout = timeout
        self.closed = False
        self._waiting = False

        if start_method is None:
            available = mp.get_all_start_methods()
            start_method = "fork" if "fork" in available else available[0]
        self._start_method = start_method
        ctx = mp.get_context(start_method)

        # Probe the spaces in-process so the parent knows them without a
        # round-trip, and so a broken env_fn fails here rather than in a child.
        probe = self.env_fns[0]()
        self.single_observation_space = probe.observation_space
        self.single_action_space = probe.action_space
        probe.close()

        self.observation_space = _batch_space(self.single_observation_space, self.num_envs)
        self.action_space = _batch_space(self.single_action_space, self.num_envs)
        obs_shape = tuple(self.single_observation_space.shape)
        self._observations = np.zeros((self.num_envs,) + obs_shape, dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._terminateds = np.zeros(self.num_envs, dtype=bool)
        self._truncateds = np.zeros(self.num_envs, dtype=bool)
        self._costs = np.zeros(self.num_envs, dtype=np.float64)
        self._progress = np.zeros(self.num_envs, dtype=np.float64)

        self.remotes: list[Any] = []
        self.processes: list[Any] = []
        for index, env_fn in enumerate(self.env_fns):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                args=(child_conn, parent_conn, env_fn),
                name=f"AsyncVectorEnv-worker-{index}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.remotes.append(parent_conn)
            self.processes.append(process)

    # --- plumbing ------------------------------------------------------------
    def _assert_open(self) -> None:
        if self.closed:
            raise RuntimeError("AsyncVectorEnv has been closed")

    def _recv(self, remote: Any, index: int) -> Any:
        if self.timeout is not None and not remote.poll(self.timeout):
            self.close(terminate=True)
            raise TimeoutError(
                f"worker {index} did not answer within {self.timeout}s"
            )
        try:
            status, payload = remote.recv()
        except EOFError as exc:
            self.close(terminate=True)
            raise RuntimeError(f"worker {index} died unexpectedly") from exc
        if status == "error":
            self.close(terminate=True)
            raise RuntimeError(f"worker {index} raised:\n{payload}")
        return payload

    def _broadcast(self, command: str, payloads: Sequence[Any]) -> list[Any]:
        self._assert_open()
        for remote, payload in zip(self.remotes, payloads):
            remote.send((command, payload))
        return [self._recv(remote, i) for i, remote in enumerate(self.remotes)]

    # --- api -----------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        seeds = _seed_list(seed, self.num_envs)
        results = self._broadcast(_CMD_RESET, [(s, options) for s in seeds])
        env_infos = np.empty(self.num_envs, dtype=object)
        for i, (obs, info) in enumerate(results):
            self._observations[i] = obs
            self._costs[i] = float(info.get("constraint_cost", 0.0))
            self._progress[i] = float(info.get("progress", 0.0))
            env_infos[i] = info
        infos: dict[str, Any] = {
            "constraint_cost": self._costs.copy(),
            "progress": self._progress.copy(),
            "env_info": env_infos,
        }
        return self._observations.copy(), infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self._assert_open()
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got shape {actions.shape}"
            )
        for remote, action in zip(self.remotes, actions):
            remote.send((_CMD_STEP, action))

        env_infos = np.empty(self.num_envs, dtype=object)
        final_obs: np.ndarray | None = None
        final_infos: np.ndarray | None = None
        done_mask: np.ndarray | None = None

        for i, remote in enumerate(self.remotes):
            obs, reward, terminated, truncated, info, f_obs, f_info = self._recv(remote, i)
            self._observations[i] = obs
            self._rewards[i] = reward
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated
            self._costs[i] = float(info.get("constraint_cost", 0.0))
            self._progress[i] = float(info.get("progress", 0.0))
            env_infos[i] = info
            if f_obs is not None:
                if final_obs is None:
                    final_obs = np.empty(self.num_envs, dtype=object)
                    final_infos = np.empty(self.num_envs, dtype=object)
                    done_mask = np.zeros(self.num_envs, dtype=bool)
                final_obs[i] = f_obs
                final_infos[i] = f_info
                done_mask[i] = True

        infos: dict[str, Any] = {
            "constraint_cost": self._costs.copy(),
            "progress": self._progress.copy(),
            "env_info": env_infos,
        }
        if final_obs is not None:
            infos["final_observation"] = final_obs
            infos["_final_observation"] = done_mask
            infos["final_info"] = final_infos
            infos["_final_info"] = done_mask
        return (
            self._observations.copy(),
            self._rewards.copy(),
            self._terminateds.copy(),
            self._truncateds.copy(),
            infos,
        )

    def call(self, name: str, *args: Any, **kwargs: Any) -> list[Any]:
        return self._broadcast(_CMD_CALL, [(name, args, kwargs)] * self.num_envs)

    def get_attr(self, name: str) -> list[Any]:
        return self._broadcast(_CMD_GET, [name] * self.num_envs)

    def set_attr(self, name: str, values: Any) -> None:
        if not isinstance(values, (list, tuple)):
            values = [values] * self.num_envs
        self._broadcast(_CMD_SET, [(name, v) for v in values])

    def close(self, *, terminate: bool = False) -> None:
        """Shut the workers down. Safe to call twice, and from ``__del__``."""
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                if not terminate:
                    remote.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            if terminate:
                if process.is_alive():
                    process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():  # pragma: no cover - stubborn child
                process.terminate()
                process.join(timeout=1.0)
        for remote in self.remotes:
            try:
                remote.close()
            except (BrokenPipeError, OSError):
                pass

    def __len__(self) -> int:
        return self.num_envs

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close(terminate=True)
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"<AsyncVectorEnv num_envs={self.num_envs} "
            f"start_method={self._start_method!r}>"
        )


def make_vector_env(
    env_fns: Sequence[EnvFn], *, asynchronous: bool = False, **kwargs: Any
) -> SyncVectorEnv | AsyncVectorEnv:
    """Pick a vector env. Falls back to sync if processes cannot be started."""
    if not asynchronous:
        return SyncVectorEnv(env_fns, **kwargs)
    try:
        return AsyncVectorEnv(env_fns, **kwargs)
    except Exception:  # pragma: no cover - platform dependent
        LOGGER.exception("AsyncVectorEnv failed to start; falling back to SyncVectorEnv")
        return SyncVectorEnv(env_fns)


__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "make_vector_env"]
