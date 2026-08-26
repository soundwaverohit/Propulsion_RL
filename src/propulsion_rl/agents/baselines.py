"""Hand-written reference controllers.

These are the line the whole study is measured against. An RL method that
cannot beat a *well-tuned* scripted controller has not earned its complexity,
and half the value of this benchmark is being able to say that honestly -- so
none of these is a strawman. They are tuned, they read the same observation the
learners read, and where an analytical optimum exists (Edelbaum) they implement
it rather than a caricature of it.

Reading the observation
-----------------------
Baselines are the only agents that need to know what the observation *means*
rather than just its width. The canonical vector is three fixed-width blocks,
contracted in :mod:`propulsion_rl.core.types`::

    MISSION_OBS_DIM    = 12  ->  indices  0 .. 11
    VEHICLE_OBS_DIM    =  8  ->  indices 12 .. 19
    PROPULSION_OBS_DIM = 16  ->  indices 20 .. 35

Blocks are right-padded with zeros and never re-ordered. Rather than scatter
magic indices through the controllers, every channel this module reads has a
named constant below and every read goes through :class:`ObsReader`, which
resolves channels in three descending tiers:

1. **Labels.** Pass ``obs_labels=env.observation_labels`` and channels are
   matched by name (the environment emits ``"mission/<name>"``,
   ``"vehicle/<name>"``, ``"prop/<name>"`` and ``"<block>/padN"`` for padding).
   This is exact and survives any mission or thruster re-ordering its author
   chooses, so *use it* -- it is one kwarg.
2. **Default indices.** Without labels the ``IX_*`` constants below are the
   assumed layout.
3. **Fallbacks.** A channel that has read as exactly ``0.0`` on every step of
   the current episode is treated as zero-padding and the controller uses a
   documented default instead of steering on a constant. A channel that is ever
   non-zero is trusted from then on, so a genuinely-zero-at-t0 signal (an
   inclination error already at target, say) is not permanently written off.

The point of tier 3 is that a controller handed a propulsion system that does
not report wear must degrade to "ignore wear", not to "believe wear is zero and
then divide by it".
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Sequence

import numpy as np

from ..core.registry import AGENT
from ..core.types import (
    CANONICAL_ACTION_DIM,
    MISSION_OBS_DIM,
    OBS_DIM,
    PROPULSION_OBS_DIM,
    VEHICLE_OBS_DIM,
    CanonicalCommand,
)
from .base import ScriptedAgent

try:  # pragma: no cover - exercised only when the astrodynamics module exists
    from ..spacecraft.orbit import edelbaum_delta_v as _edelbaum_delta_v
except ImportError:  # pragma: no cover
    _edelbaum_delta_v = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "BangBangAgent",
    "EdelbaumAgent",
    "LifeAwareAgent",
    "MaxIspAgent",
    "MaxThrustAgent",
    "ObsLayout",
    "ObsReader",
    "PIDAgent",
    "ProgradeAgent",
    "RandomAgent",
    "edelbaum_yaw_pitch",
]

# --- Block boundaries (mirrors core/types.py; see the module docstring) -------
MISSION_BLOCK = slice(0, MISSION_OBS_DIM)
VEHICLE_BLOCK = slice(MISSION_OBS_DIM, MISSION_OBS_DIM + VEHICLE_OBS_DIM)
PROPULSION_BLOCK = slice(MISSION_OBS_DIM + VEHICLE_OBS_DIM, OBS_DIM)

# --- Mission block, indices 0..11 --------------------------------------------
IX_PROGRESS = 0            # [0, 1] fraction of the goal achieved
IX_TIME_FRAC = 1           # [0, 1] fraction of the mission clock spent
IX_RADIUS_ERROR = 2        # (r - r_target) / scale; > 0 means "too high"
IX_SMA_ERROR = 3           # (a - a_target) / scale
IX_ECCENTRICITY = 4        # current eccentricity
IX_INC_ERROR = 5           # (i_target - i) normalised; > 0 means "raise i"
IX_INC_CURRENT = 6
IX_INC_TARGET = 7
IX_SPEED_RATIO = 8         # v / v_target - 1
IX_RADIAL_VELOCITY = 9
IX_TRANSVERSE_VELOCITY = 10
IX_PHASE = 11              # cos(argument of latitude); sets the node sign

# --- Vehicle block, indices 12..19 -------------------------------------------
IX_MASS_FRACTION = 12
IX_PROPELLANT_FRACTION = 13
IX_POWER_AVAILABLE = 14
IX_POWER_MARGIN = 15
IX_DELTA_V = 16
IX_ECLIPSE = 17
IX_SUN_DISTANCE = 18

# --- Propulsion block, indices 20..35 ----------------------------------------
IX_THRUST = 20
IX_ISP = 21
IX_MDOT = 22
IX_POWER_DRAW = 23
IX_EFFICIENCY = 24
IX_TEMPERATURE = 25
IX_TEMP_MARGIN = 26
IX_WEAR = 27               # [0, 1]; 1.0 == end of qualified life
IX_REMAINING_LIFE = 28
IX_THROUGHPUT = 29
IX_BURN_TIME = 30
IX_WORST_MARGIN = 31       # min over constraints; < 0 == violated
IX_OPERATING_POINT = 32
IX_THERMAL_STATE = 33

#: Canonical channel name -> default index. Tier 2 of the resolution above.
DEFAULT_CHANNELS: dict[str, int] = {
    "progress": IX_PROGRESS,
    "time_frac": IX_TIME_FRAC,
    "radius_error": IX_RADIUS_ERROR,
    "sma_error": IX_SMA_ERROR,
    "eccentricity": IX_ECCENTRICITY,
    "inc_error": IX_INC_ERROR,
    "inc_current": IX_INC_CURRENT,
    "inc_target": IX_INC_TARGET,
    "speed_ratio": IX_SPEED_RATIO,
    "radial_velocity": IX_RADIAL_VELOCITY,
    "transverse_velocity": IX_TRANSVERSE_VELOCITY,
    "phase": IX_PHASE,
    "mass_fraction": IX_MASS_FRACTION,
    "propellant_fraction": IX_PROPELLANT_FRACTION,
    "power_available": IX_POWER_AVAILABLE,
    "power_margin": IX_POWER_MARGIN,
    "delta_v": IX_DELTA_V,
    "eclipse": IX_ECLIPSE,
    "sun_distance": IX_SUN_DISTANCE,
    "thrust": IX_THRUST,
    "isp": IX_ISP,
    "mdot": IX_MDOT,
    "power_draw": IX_POWER_DRAW,
    "efficiency": IX_EFFICIENCY,
    "temperature": IX_TEMPERATURE,
    "temp_margin": IX_TEMP_MARGIN,
    "wear": IX_WEAR,
    "remaining_life": IX_REMAINING_LIFE,
    "throughput": IX_THROUGHPUT,
    "burn_time": IX_BURN_TIME,
    "worst_margin": IX_WORST_MARGIN,
    "operating_point": IX_OPERATING_POINT,
    "thermal_state": IX_THERMAL_STATE,
}

#: Which block each channel must be found in when matching by label. Prevents a
#: propulsion channel called "temp_margin" from being claimed by the mission
#: block just because it sorted first.
_CHANNEL_BLOCK: dict[str, slice] = {
    **{k: MISSION_BLOCK for k in ("progress", "time_frac", "radius_error",
                                  "sma_error", "eccentricity", "inc_error",
                                  "inc_current", "inc_target", "speed_ratio",
                                  "radial_velocity", "transverse_velocity",
                                  "phase")},
    **{k: VEHICLE_BLOCK for k in ("mass_fraction", "propellant_fraction",
                                  "power_available", "power_margin", "delta_v",
                                  "eclipse", "sun_distance")},
    **{k: PROPULSION_BLOCK for k in ("thrust", "isp", "mdot", "power_draw",
                                     "efficiency", "temperature", "temp_margin",
                                     "wear", "remaining_life", "throughput",
                                     "burn_time", "worst_margin",
                                     "operating_point", "thermal_state")},
}

#: Substring patterns per channel, most specific first. The resolution order of
#: this dict is load-bearing: "inc_error" must get first refusal on a label
#: called ``mission/inc_error`` before the looser "inc_current" patterns run.
_CHANNEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "progress": ("progress", "completion", "frac_done"),
    "time_frac": ("time_frac", "t_frac", "elapsed", "time_left", "time_remain"),
    "inc_error": ("inc_err", "incl_err", "inclination_err", "delta_i", "di_",
                  "plane_err", "d_inc"),
    "inc_target": ("inc_target", "target_inc", "i_target"),
    "inc_current": ("inc", "incl"),
    "sma_error": ("sma_err", "a_err", "semi_major", "sma", "energy_err"),
    "radius_error": ("radius_err", "r_err", "alt_err", "radius_ratio",
                     "range_err", "radius", "altitude"),
    "eccentricity": ("ecc",),
    "speed_ratio": ("speed_ratio", "v_ratio", "speed_err", "vel_err", "speed"),
    "radial_velocity": ("v_radial", "radial_vel", "v_r", "vr"),
    "transverse_velocity": ("v_transverse", "transverse", "v_t", "vt"),
    "phase": ("cos_u", "u_cos", "arg_lat", "phase", "true_anom", "longitude",
              "lon_err"),
    "mass_fraction": ("mass_frac", "mass"),
    "propellant_fraction": ("prop_frac", "propellant", "fuel"),
    "power_margin": ("power_margin", "power_frac"),
    "power_available": ("power_avail", "p_avail", "power"),
    "delta_v": ("delta_v", "dv"),
    "eclipse": ("eclipse", "shadow"),
    "sun_distance": ("sun", "helio"),
    "temp_margin": ("temp_margin", "thermal_margin", "t_margin", "temp_head"),
    "temperature": ("temp", "t_wall", "t_fuel", "t_chamber"),
    "wear": ("wear", "degrad", "erosion"),
    "remaining_life": ("remaining_life", "life_frac", "life"),
    "throughput": ("throughput",),
    "burn_time": ("burn",),
    "worst_margin": ("worst_margin", "min_margin", "margin", "constraint"),
    "operating_point": ("operating_point", "op_point", "voltage", "chamber"),
    "thermal_state": ("coolant", "radiator", "thermal"),
    "thrust": ("thrust",),
    "isp": ("isp",),
    "mdot": ("mdot", "flow"),
    "power_draw": ("power_draw", "draw"),
    "efficiency": ("eff",),
}


class ObsLayout:
    """Channel name -> observation index, resolved from labels when available.

    Constructed once per agent. ``index(name)`` returns ``None`` for a channel
    this observation demonstrably does not carry (a label matching ``/padN``,
    or no label match at all when labels were supplied), which is what lets the
    controllers fall back deliberately instead of reading a padding zero.
    """

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        self.labels: tuple[str, ...] | None = None
        self._map: dict[str, int | None] = dict(DEFAULT_CHANNELS)
        if labels is not None:
            self.labels = tuple(str(x) for x in labels)
            if len(self.labels) != OBS_DIM:
                logger.warning(
                    "obs_labels has %d entries, expected %d; ignoring them and "
                    "falling back to the default index layout",
                    len(self.labels),
                    OBS_DIM,
                )
                self.labels = None
            else:
                self._map = self._resolve(self.labels)

    @staticmethod
    def _resolve(labels: tuple[str, ...]) -> dict[str, int | None]:
        lowered = [s.lower() for s in labels]
        taken: set[int] = set()
        out: dict[str, int | None] = {}
        for channel, patterns in _CHANNEL_PATTERNS.items():
            block = _CHANNEL_BLOCK[channel]
            found: int | None = None
            for pattern in patterns:
                for i in range(block.start, block.stop):
                    if i in taken or "/pad" in lowered[i]:
                        continue
                    # Compare against the part after the block prefix so a
                    # prefix like "prop/" cannot accidentally match "progress".
                    tail = lowered[i].split("/", 1)[-1]
                    if pattern in tail:
                        found = i
                        break
                if found is not None:
                    break
            if found is not None:
                taken.add(found)
            out[channel] = found
        missing = [c for c, i in out.items() if i is None]
        if missing:
            logger.debug("ObsLayout: no label matched channels %s", sorted(missing))
        return out

    def index(self, channel: str) -> int | None:
        try:
            return self._map[channel]
        except KeyError:  # pragma: no cover - programming error
            raise KeyError(f"unknown observation channel {channel!r}") from None

    def __repr__(self) -> str:
        source = "labels" if self.labels is not None else "defaults"
        return f"<ObsLayout source={source} resolved={sum(v is not None for v in self._map.values())}>"


class ObsReader:
    """Named, fallback-safe access to the current observation.

    Call :meth:`update` once per ``act``; then :meth:`get` reads channels by
    name. A channel that has been exactly ``0.0`` for every step of the episode
    so far is reported as not live and :meth:`get` returns the caller's default
    -- see the module docstring for why that beats trusting a padding zero.
    """

    __slots__ = ("layout", "_obs", "_seen")

    def __init__(self, layout: ObsLayout) -> None:
        self.layout = layout
        self._obs = np.zeros(OBS_DIM, dtype=np.float64)
        self._seen = np.zeros(OBS_DIM, dtype=bool)

    def update(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        if arr.size < OBS_DIM:
            padded = np.zeros(OBS_DIM, dtype=np.float64)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > OBS_DIM:
            arr = arr[:OBS_DIM]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._obs = arr
        self._seen |= arr != 0.0
        return arr

    def live(self, channel: str) -> bool:
        i = self.layout.index(channel)
        return i is not None and bool(self._seen[i])

    def get(self, channel: str, default: float = 0.0) -> float:
        i = self.layout.index(channel)
        if i is None or not self._seen[i]:
            return float(default)
        return float(self._obs[i])

    def raw(self, index: int, default: float = 0.0) -> float:
        """Positional escape hatch, for a `signal_index` kwarg the user set."""
        if not 0 <= index < OBS_DIM or not self._seen[index]:
            return float(default)
        return float(self._obs[index])

    def reset(self) -> None:
        self._obs[:] = 0.0
        self._seen[:] = False


# --- shared helpers -----------------------------------------------------------
def _action(
    throttle: float,
    operating_point: float,
    thrust_yaw: float,
    thrust_pitch: float,
    thermal_margin: float,
) -> np.ndarray:
    """Build a canonical ``[-1, 1]^5`` action from physical command units."""
    cmd = CanonicalCommand(
        throttle=float(np.clip(throttle, 0.0, 1.0)),
        operating_point=float(np.clip(operating_point, 0.0, 1.0)),
        thrust_yaw=float(np.clip(thrust_yaw, -math.pi, math.pi)),
        thrust_pitch=float(np.clip(thrust_pitch, -0.5 * math.pi, 0.5 * math.pi)),
        thermal_margin=float(np.clip(thermal_margin, 0.0, 1.0)),
    )
    return np.clip(cmd.to_array(), -1.0, 1.0).astype(np.float32)


def edelbaum_yaw_pitch(
    inc_error_rad: float, speed_ratio: float, node_sign: float
) -> tuple[float, float]:
    """Edelbaum's optimal steering, expressed as ``(thrust_yaw, thrust_pitch)``.

    Edelbaum's 1961 solution for combined circular orbit raising and plane
    change puts the thrust entirely in the transverse/normal plane (no radial
    component) at an out-of-plane angle ``beta``. Written in *feedback* form --
    legitimate by Bellman's principle, because the remaining leg of an optimal
    Edelbaum transfer is itself an optimal Edelbaum transfer -- the law needs
    only two dimensionless quantities, the plane change still owed and the
    ratio of current to target circular speed::

        x    = (pi / 2) * |di_remaining|
        beta = atan2(sin(x), v / v_target - cos(x))

    The out-of-plane sign must flip at the nodes (``di/dt`` goes as
    ``cos(argument of latitude)``) or the plane change averages to nothing over
    an orbit, which is what ``node_sign`` carries.

    Both limits are the right ones: with no plane change owed the thrust is
    pure prograde, and with the orbit already at target speed the thrust goes
    fully out-of-plane. The transverse component is allowed to go negative
    (orbit *lowering*), which shows up here as a yaw flip to retrograde rather
    than as a pitch outside its contracted ``[-pi/2, pi/2]`` range.
    """
    x = 0.5 * math.pi * abs(float(inc_error_rad))
    transverse = float(speed_ratio) - math.cos(x)
    normal = math.sin(x) * math.copysign(1.0, inc_error_rad or 1.0) * node_sign
    if abs(transverse) < 1e-12 and abs(normal) < 1e-12:
        return 0.0, 0.0
    yaw = 0.0 if transverse >= 0.0 else math.pi
    pitch = math.atan2(normal, abs(transverse))
    return yaw, pitch


class _BaselineAgent(ScriptedAgent):
    """Shared plumbing: the observation reader and the action builder.

    Every scripted controller accepts ``obs_labels`` (pass
    ``env.observation_labels``) to resolve channels by name instead of by the
    default index layout.
    """

    name = "baseline"

    #: These controllers are defined *by* the canonical command semantics --
    #: index 0 is throttle, 1 the operating point, 2 yaw, 3 pitch, 4 thermal
    #: margin -- so "prograde" or "max_isp" has no meaning in some other action
    #: space. Such a controller refuses a width it cannot serve at construction
    #: rather than quietly emitting five components into a three-wide space,
    #: which would only surface as a shape error deep inside the rollout loop.
    #: :class:`RandomAgent` sets this False: uniform noise is well defined at
    #: any width, and it is useful as a floor for reduced-actuator ablations.
    _requires_canonical_actions = True

    def __init__(self, obs_dim: int, action_dim: int, **kwargs: Any) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        if self._requires_canonical_actions and action_dim != CANONICAL_ACTION_DIM:
            raise ValueError(
                f"{type(self).__name__} is a scripted controller written "
                f"against the canonical command layout (throttle, "
                f"operating_point, thrust_yaw, thrust_pitch, thermal_margin), "
                f"so it only has meaning at action_dim="
                f"{CANONICAL_ACTION_DIM}; got {action_dim}. Use RandomAgent "
                f"for a floor at another width."
            )
        if obs_dim != OBS_DIM:
            logger.warning(
                "%s: obs_dim=%d but the channel layout is written for %d; "
                "out-of-range channels will fall back to defaults",
                type(self).__name__,
                obs_dim,
                OBS_DIM,
            )
        self.layout = ObsLayout(kwargs.get("obs_labels"))
        self.reader = ObsReader(self.layout)
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.info(msg, *args)

    def reset(self) -> None:
        self.reader.reset()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.reader.update(obs)
        return self._control(deterministic)

    def _control(self, deterministic: bool) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # --- signals several controllers share -----------------------------------
    def _node_sign(self) -> float:
        """+/-1 from the orbital phase channel; +1 (with a note) without one."""
        if self.reader.live("phase"):
            return 1.0 if self.reader.get("phase") >= 0.0 else -1.0
        self._warn_once(
            "phase",
            "%s: no orbital-phase channel; out-of-plane thrust cannot be "
            "switched at the nodes, so the plane-change term is only correct "
            "for an environment that models the secular rate",
            type(self).__name__,
        )
        return 1.0

    def _speed_ratio(self, initial: float) -> float:
        """Current circular speed / target circular speed, dimensionless.

        Tried in order: an explicit speed-ratio channel, the radius channel via
        ``v/v_t = sqrt(r_t/r)``, then linear interpolation from ``initial`` to
        1.0 on mission progress. All three are scale-free in the mission's own
        normalisation as long as it is a plain division, which is why the ratio
        form of the steering law is used rather than the absolute one.
        """
        if self.reader.live("speed_ratio"):
            return max(1e-3, 1.0 + self.reader.get("speed_ratio"))
        if self.reader.live("radius_error"):
            r_ratio = max(1e-3, 1.0 + self.reader.get("radius_error"))
            return math.sqrt(1.0 / r_ratio)
        if self.reader.live("progress"):
            p = float(np.clip(self.reader.get("progress"), 0.0, 1.0))
            return max(1e-3, initial + (1.0 - initial) * p)
        self._warn_once(
            "speed_ratio",
            "%s: no speed, radius or progress channel; holding the assumed "
            "initial speed ratio %.3f",
            type(self).__name__,
            initial,
        )
        return initial

    def _inc_error_rad(self, scale_rad: float) -> float:
        """Signed remaining plane change in radians (0 if unobservable)."""
        if self.reader.live("inc_error"):
            return self.reader.get("inc_error") * scale_rad
        if self.reader.live("inc_current") and self.reader.live("inc_target"):
            delta = self.reader.get("inc_target") - self.reader.get("inc_current")
            return delta * scale_rad
        return 0.0

    def _thermal_backoff(self, base: float, threshold: float) -> float:
        """Raise the coolant/flow margin as the worst constraint margin closes."""
        if threshold <= 0.0 or not self.reader.live("worst_margin"):
            return base
        worst = self.reader.get("worst_margin", threshold)
        urgency = float(np.clip((threshold - worst) / threshold, 0.0, 1.0))
        return base + (1.0 - base) * urgency

    def _power_starved(self, floor: float) -> bool:
        """True when the bus says there is no useful power to thrust with."""
        if self.reader.live("eclipse") and self.reader.get("eclipse") > 0.5:
            return True
        if self.reader.live("power_available"):
            return self.reader.get("power_available") < floor
        return False


# --- the controllers ----------------------------------------------------------
@AGENT.register("random", kind="scripted", learns=False)
class RandomAgent(_BaselineAgent):
    """Uniform in ``[-1, 1]^5``. The floor every other number is measured over.

    ``deterministic=True`` still samples uniformly -- returning the mean action
    would measure "half throttle prograde", not the random floor, and that is
    not the reference this benchmark wants. It is made reproducible instead:
    the sample is drawn from a stream seeded by the agent seed and a hash of
    the observation, so the same state always yields the same action, across
    processes and independent of ``PYTHONHASHSEED``.
    """

    name = "random"

    def __init__(self, obs_dim: int, action_dim: int, *, seed: int = 0,
                 **kwargs: Any) -> None:
        super().__init__(obs_dim, action_dim, seed=seed, **kwargs)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)
        self.config["seed"] = self.seed
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        arr = self.reader.update(obs)
        if deterministic:
            digest = hashlib.blake2b(
                arr.tobytes() + self.seed.to_bytes(8, "little", signed=True),
                digest_size=8,
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
        else:
            rng = self._rng
        return rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def _control(self, deterministic: bool) -> np.ndarray:  # pragma: no cover
        raise AssertionError("RandomAgent overrides act()")


@AGENT.register("prograde", kind="scripted", learns=False)
class ProgradeAgent(_BaselineAgent):
    """Full throttle, pure prograde, mid operating point. The trivial policy.

    Deliberately *not* adaptive: no thermal guard, no coast, no plane change.
    It answers "what does simply pushing get you", which is the question the
    other baselines are differences against.
    """

    name = "prograde"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        throttle: float = 1.0,
        operating_point: float = 0.5,
        thermal_margin: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.throttle = float(throttle)
        self.operating_point = float(operating_point)
        self.thermal_margin = float(thermal_margin)

    def _control(self, deterministic: bool) -> np.ndarray:
        return _action(
            self.throttle, self.operating_point, 0.0, 0.0, self.thermal_margin
        )


class _FixedOperatingPoint(_BaselineAgent):
    """Full throttle pinned to one end of the Isp/thrust trade.

    :class:`MaxThrustAgent` and :class:`MaxIspAgent` differ only in that pin,
    which is the point: the gap between their scores is the value an adaptive
    policy has to earn on the operating-point axis, so nothing else may vary
    between them. The one concession is the coolant margin, which tracks the
    worst constraint margin -- a fixed operating point that destroys the
    thruster on step ten is a strawman in the other direction. Steering,
    throttle and operating point stay constant.
    """

    _operating_point = 0.5

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        throttle: float = 1.0,
        operating_point: float | None = None,
        thermal_margin: float = 0.6,
        thermal_guard: bool = True,
        guard_threshold: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.throttle = float(throttle)
        self.operating_point = float(
            self._operating_point if operating_point is None else operating_point
        )
        self.thermal_margin = float(thermal_margin)
        self.thermal_guard = bool(thermal_guard)
        self.guard_threshold = float(guard_threshold)

    def _control(self, deterministic: bool) -> np.ndarray:
        margin = self.thermal_margin
        if self.thermal_guard:
            margin = self._thermal_backoff(margin, self.guard_threshold)
        return _action(self.throttle, self.operating_point, 0.0, 0.0, margin)


@AGENT.register("max_thrust", kind="scripted", learns=False)
class MaxThrustAgent(_FixedOperatingPoint):
    """Max throttle at the high-thrust / low-Isp end (``operating_point = 0``).

    Fastest transfer, worst propellant mass. On a thermal system this is the
    low chamber-temperature end; on an electric one, low discharge voltage.
    """

    name = "max_thrust"
    _operating_point = 0.0


@AGENT.register("max_isp", kind="scripted", learns=False)
class MaxIspAgent(_FixedOperatingPoint):
    """Max throttle at the high-Isp / low-thrust end (``operating_point = 1``).

    Least propellant, longest transfer, and the harsher end thermally on both
    families (peak chamber temperature, peak discharge voltage) -- which is why
    the coolant guard matters more here than it looks.
    """

    name = "max_isp"
    _operating_point = 1.0


@AGENT.register("edelbaum", kind="scripted", learns=False)
class EdelbaumAgent(_BaselineAgent):
    """Edelbaum optimal low-thrust steering. The strong baseline for transfers.

    Continuous thrust at the analytically optimal out-of-plane angle for a
    combined circular orbit raise and plane change (see
    :func:`edelbaum_yaw_pitch`). For ``leo_geo_transfer`` this should be hard to
    beat: the law *is* the optimum of the averaged problem, so an RL agent only
    wins by exploiting what averaging discards -- eclipse scheduling, the
    thrust/Isp trade over the transfer, and hardware wear.

    Parameters
    ----------
    inc_scale_rad:
        Radians per unit of the inclination-error channel. The mission owns
        that normalisation; set this to match it. The default assumes the
        channel is scaled by ``pi/2``.
    initial_speed_ratio:
        ``v0 / v_target`` used only when neither a speed nor a radius channel
        is live (2.5 is LEO(400 km) -> GEO).
    operating_point:
        Where to sit on the Isp/thrust trade. A long spiral is propellant-bound
        rather than time-bound, so the tuned default leans towards Isp.
    stop_progress:
        Coast above this progress instead of burning propellant past the goal.
    """

    name = "edelbaum"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        inc_scale_rad: float = 0.5 * math.pi,
        initial_speed_ratio: float = 2.5,
        operating_point: float = 0.65,
        thermal_margin: float = 0.5,
        throttle: float = 1.0,
        stop_progress: float = 0.999,
        guard_threshold: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.inc_scale_rad = float(inc_scale_rad)
        self.initial_speed_ratio = float(initial_speed_ratio)
        self.operating_point = float(operating_point)
        self.thermal_margin = float(thermal_margin)
        self.throttle = float(throttle)
        self.stop_progress = float(stop_progress)
        self.guard_threshold = float(guard_threshold)

    @staticmethod
    def ideal_delta_v(a0: float, a1: float, i0: float, i1: float, mu: float) -> float:
        """Edelbaum's ideal cost for the transfer, m/s.

        Delegates to :func:`propulsion_rl.spacecraft.orbit.edelbaum_delta_v`
        when the astrodynamics module is present, and reproduces it locally
        otherwise so this controller stays importable in a partial tree.
        """
        if _edelbaum_delta_v is not None:
            return float(_edelbaum_delta_v(a0, a1, i0, i1, mu))
        v0, v1 = math.sqrt(mu / a0), math.sqrt(mu / a1)
        di = abs(i1 - i0)
        inner = v0 * v0 + v1 * v1 - 2.0 * v0 * v1 * math.cos(0.5 * math.pi * di)
        return math.sqrt(max(inner, 0.0))

    def steering(self) -> tuple[float, float]:
        """``(yaw, pitch)`` for the current observation. Reused by other agents."""
        di = self._inc_error_rad(self.inc_scale_rad)
        ratio = self._speed_ratio(self.initial_speed_ratio)
        return edelbaum_yaw_pitch(di, ratio, self._node_sign())

    def _control(self, deterministic: bool) -> np.ndarray:
        yaw, pitch = self.steering()
        throttle = self.throttle
        if self.reader.live("progress") and self.reader.get("progress") >= self.stop_progress:
            throttle = 0.0
        margin = self._thermal_backoff(self.thermal_margin, self.guard_threshold)
        return _action(throttle, self.operating_point, yaw, pitch, margin)


@AGENT.register("pid", kind="scripted", learns=False)
class PIDAgent(_BaselineAgent):
    """PID on the target-error channels. The natural ``geo_station_keeping`` baseline.

    A proportional-integral-derivative loop on the radius/semi-major-axis error
    sets the throttle, with the sign of the loop output choosing prograde
    (raise) or retrograde (lower); a second, proportional-derivative loop on the
    inclination error sets the out-of-plane angle. Station keeping is exactly
    the regime PID was invented for -- a small, persistent, sign-flipping error
    about a setpoint -- so this is the baseline to beat there, not a token one.

    The integrator uses conditional integration for anti-windup: when the
    command is already saturated and the error would push it further into
    saturation, the integral is frozen rather than accumulated. Without that,
    a long thrust-limited stretch (an eclipse, a power-starved arc) winds the
    integral up and the loop overshoots badly on the way back.
    """

    name = "pid"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        kp: float = 4.0,
        ki: float = 0.05,
        kd: float = 1.5,
        kp_inc: float = 3.0,
        kd_inc: float = 1.0,
        dt: float = 1.0,
        integral_limit: float = 10.0,
        deadband: float = 1e-4,
        error_sign: float = 1.0,
        operating_point: float = 0.8,
        thermal_margin: float = 0.5,
        guard_threshold: float = 0.15,
        inc_scale_rad: float = 0.5 * math.pi,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.kp_inc, self.kd_inc = float(kp_inc), float(kd_inc)
        self.dt = float(dt)
        self.integral_limit = float(integral_limit)
        self.deadband = float(deadband)
        self.error_sign = float(error_sign)
        self.operating_point = float(operating_point)
        self.thermal_margin = float(thermal_margin)
        self.guard_threshold = float(guard_threshold)
        self.inc_scale_rad = float(inc_scale_rad)
        self._integral = 0.0
        self._prev_error: float | None = None
        self._prev_inc: float | None = None

    def reset(self) -> None:
        super().reset()
        self._integral = 0.0
        self._prev_error = None
        self._prev_inc = None

    def _primary_error(self) -> float:
        """Signed "how much energy is owed", positive meaning "raise".

        Prefers the radius error, then the semi-major-axis error, then
        ``1 - progress``. The last is unsigned and can only ever ask for more
        energy, which is the correct degradation for a raising task and is
        flagged once for a task where lowering is possible.
        """
        for channel in ("radius_error", "sma_error"):
            if self.reader.live(channel):
                return -self.error_sign * self.reader.get(channel)
        if self.reader.live("progress"):
            self._warn_once(
                "pid_error",
                "%s: no radius or semi-major-axis error channel; driving on "
                "(1 - progress), which cannot command a retrograde burn",
                type(self).__name__,
            )
            return float(np.clip(1.0 - self.reader.get("progress"), 0.0, 1.0))
        return 0.0

    def _control(self, deterministic: bool) -> np.ndarray:
        error = self._primary_error()
        if abs(error) < self.deadband:
            error = 0.0
        derivative = 0.0 if self._prev_error is None else (
            (error - self._prev_error) / max(self.dt, 1e-9)
        )
        self._prev_error = error

        candidate = self._integral + error * self.dt
        raw = self.kp * error + self.ki * candidate + self.kd * derivative
        if abs(raw) > 1.0 and math.copysign(1.0, raw) == math.copysign(1.0, error or 1.0):
            # Saturated and the error is pushing further out: freeze the
            # integrator instead of winding it up.
            raw = self.kp * error + self.ki * self._integral + self.kd * derivative
        else:
            self._integral = float(
                np.clip(candidate, -self.integral_limit, self.integral_limit)
            )

        throttle = float(np.clip(abs(raw), 0.0, 1.0))
        yaw = 0.0 if raw >= 0.0 else math.pi

        inc_error = self._inc_error_rad(self.inc_scale_rad)
        d_inc = 0.0 if self._prev_inc is None else inc_error - self._prev_inc
        self._prev_inc = inc_error
        pitch_cmd = self.kp_inc * inc_error + self.kd_inc * d_inc
        pitch = float(
            np.clip(pitch_cmd, -0.5 * math.pi, 0.5 * math.pi) * self._node_sign()
        ) if inc_error != 0.0 else 0.0

        margin = self._thermal_backoff(self.thermal_margin, self.guard_threshold)
        return _action(throttle, self.operating_point, yaw, pitch, margin)


@AGENT.register("bang_bang", kind="scripted", learns=False)
class BangBangAgent(_BaselineAgent):
    """Full throttle or nothing, switched on an observed threshold.

    Bang-bang is the structure the Pontryagin maximum principle predicts for a
    thrust-limited, propellant-optimal transfer: the switching function is
    linear in the throttle, so the optimum is at a bound almost everywhere and
    interior arcs are singular. That makes this the *structurally* right
    baseline for the low-thrust problems even where the switching surface it
    uses here is a proxy for the true costate.

    A hysteresis band separates the on and off thresholds, because a bare
    threshold on a noisy channel chatters, and chatter on a real thruster is
    restart cycles -- a life-limited resource in :class:`Limits`. Thrust is also
    suppressed when the bus reports no power, which is free performance on a
    solar-electric mission and costs nothing on a self-powered one.
    """

    name = "bang_bang"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        on_threshold: float = 0.02,
        off_threshold: float = 0.005,
        signal: str = "auto",
        signal_index: int | None = None,
        invert: bool = False,
        operating_point: float = 0.5,
        thermal_margin: float = 0.5,
        guard_threshold: float = 0.15,
        power_floor: float = 0.02,
        steering: str = "edelbaum",
        inc_scale_rad: float = 0.5 * math.pi,
        initial_speed_ratio: float = 2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        if off_threshold > on_threshold:
            raise ValueError("off_threshold must not exceed on_threshold")
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.signal = str(signal)
        self.signal_index = None if signal_index is None else int(signal_index)
        self.invert = bool(invert)
        self.operating_point = float(operating_point)
        self.thermal_margin = float(thermal_margin)
        self.guard_threshold = float(guard_threshold)
        self.power_floor = float(power_floor)
        self.steering = str(steering)
        self.inc_scale_rad = float(inc_scale_rad)
        self.initial_speed_ratio = float(initial_speed_ratio)
        self._on = True

    def reset(self) -> None:
        super().reset()
        self._on = True

    def _switching_signal(self) -> float:
        """Non-negative "distance still to go" driving the switch."""
        if self.signal_index is not None:
            return abs(self.reader.raw(self.signal_index))
        if self.signal != "auto":
            return abs(self.reader.get(self.signal))
        for channel in ("radius_error", "sma_error", "inc_error"):
            if self.reader.live(channel):
                return abs(self.reader.get(channel))
        if self.reader.live("progress"):
            return float(np.clip(1.0 - self.reader.get("progress"), 0.0, 1.0))
        self._warn_once(
            "bang_signal",
            "%s: no switching channel is live; holding thrust on",
            type(self).__name__,
        )
        return 1.0

    def _control(self, deterministic: bool) -> np.ndarray:
        signal = self._switching_signal()
        if self.invert:
            signal = -signal
        if self._on:
            self._on = signal > self.off_threshold
        else:
            self._on = signal > self.on_threshold
        if self._power_starved(self.power_floor):
            self._on = False

        if self.steering == "edelbaum":
            di = self._inc_error_rad(self.inc_scale_rad)
            ratio = self._speed_ratio(self.initial_speed_ratio)
            yaw, pitch = edelbaum_yaw_pitch(di, ratio, self._node_sign())
        else:
            yaw, pitch = 0.0, 0.0

        margin = self._thermal_backoff(self.thermal_margin, self.guard_threshold)
        return _action(1.0 if self._on else 0.0, self.operating_point, yaw, pitch, margin)


@AGENT.register("life_aware", kind="scripted", learns=False)
class LifeAwareAgent(_BaselineAgent):
    """Backs the operating point off as wear accumulates. The economics baseline.

    This is the hand-written statement of the exact trade the RL agents are
    supposed to discover on their own, so it is the number the economics
    comparison cares about most: performance now against hardware left later.
    Three levers move together with wear --

    * **throttle** falls, cutting power density, thermal cycling and burn time;
    * **operating point** slides towards the gentle end of the envelope (low
      chamber temperature on a thermal system, low discharge voltage on an
      electric one), buying life at the cost of Isp;
    * **coolant margin** rises, spending flow budget on protecting hardware.

    The interesting part is the *schedule*. Wear on its own is the wrong
    trigger: an engine 60% worn at 20% of the mission is in trouble, the same
    engine at 90% of the mission is fine. So the controller compares wear to
    what the mission clock says it should have spent by now and reacts hard to
    the excess, on top of a gentle always-on derating. That keeps it monotone in
    wear (more wear is always more caution) while staying aggressive early,
    when caution is worth nothing.
    """

    name = "life_aware"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        wear_budget: float = 0.9,
        k_wear: float = 0.30,
        k_excess: float = 1.60,
        throttle_min: float = 0.15,
        operating_point: float = 0.75,
        k_operating_point: float = 0.60,
        operating_point_min: float = 0.15,
        thermal_margin: float = 0.45,
        k_thermal: float = 0.50,
        wear_limit: float = 0.98,
        guard_threshold: float = 0.15,
        steering: str = "edelbaum",
        inc_scale_rad: float = 0.5 * math.pi,
        initial_speed_ratio: float = 2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, action_dim, **kwargs)
        self.wear_budget = float(wear_budget)
        self.k_wear = float(k_wear)
        self.k_excess = float(k_excess)
        self.throttle_min = float(throttle_min)
        self.operating_point = float(operating_point)
        self.k_operating_point = float(k_operating_point)
        self.operating_point_min = float(operating_point_min)
        self.thermal_margin = float(thermal_margin)
        self.k_thermal = float(k_thermal)
        self.wear_limit = float(wear_limit)
        self.guard_threshold = float(guard_threshold)
        self.steering = str(steering)
        self.inc_scale_rad = float(inc_scale_rad)
        self.initial_speed_ratio = float(initial_speed_ratio)

    def _wear(self) -> float:
        """Observed wear in [0, 1], from either the wear or the life channel."""
        if self.reader.live("wear"):
            return float(np.clip(self.reader.get("wear"), 0.0, 1.0))
        if self.reader.live("remaining_life"):
            return float(np.clip(1.0 - self.reader.get("remaining_life"), 0.0, 1.0))
        self._warn_once(
            "wear",
            "%s: no wear or remaining-life channel; the life/performance trade "
            "is unobservable and this controller degrades to a fixed point",
            type(self).__name__,
        )
        return 0.0

    def _control(self, deterministic: bool) -> np.ndarray:
        wear = self._wear()
        time_frac = float(np.clip(self.reader.get("time_frac", 0.0), 0.0, 1.0))
        allowance = self.wear_budget * time_frac
        excess = max(0.0, wear - allowance)

        throttle = 1.0 - self.k_wear * wear - self.k_excess * excess
        if wear >= self.wear_limit:
            # Past the qualified life, keep the asset alive and crawl.
            throttle = min(throttle, self.throttle_min)
        if self.reader.live("worst_margin") and self.reader.get("worst_margin") < 0.0:
            throttle *= 0.5
        throttle = float(np.clip(throttle, self.throttle_min, 1.0))

        op = float(
            np.clip(
                self.operating_point - self.k_operating_point * wear,
                self.operating_point_min,
                1.0,
            )
        )
        margin = float(np.clip(self.thermal_margin + self.k_thermal * wear, 0.0, 1.0))
        margin = self._thermal_backoff(margin, self.guard_threshold)

        if self.steering == "edelbaum":
            di = self._inc_error_rad(self.inc_scale_rad)
            ratio = self._speed_ratio(self.initial_speed_ratio)
            yaw, pitch = edelbaum_yaw_pitch(di, ratio, self._node_sign())
        else:
            yaw, pitch = 0.0, 0.0

        return _action(throttle, op, yaw, pitch, margin)
