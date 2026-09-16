"""The environment: where mission, vehicle, power bus and propulsion meet.

This is the only place in the package that knows about all four. Everything
else talks through :mod:`propulsion_rl.core.types`. The step loop here is the
hot path of the whole study -- it runs tens of millions of times per sweep --
so it allocates as little as possible and localises attribute lookups.

Step ordering (fixed; downstream results depend on it)
-----------------------------------------------------
1. decode the raw action into a :class:`CanonicalCommand`
2. assemble the :class:`StepContext` for this step (time, mass, offered power,
   heliocentric radius, eclipse, episode rng)
3. ``propulsion.step(command, ctx)``
4. ``power_bus.step(...)`` with the *actual* draw that step produced
5. ``dynamics.propagate(...)`` with the achieved thrust and flow
6. read ``propulsion.constraints()`` and ``propulsion.health()``
7. ``mission.reward(...)``, then fold in the constraint penalty
8. ``mission.terminated(...)``, then the environment's own step-limit truncation
9. build the observation, telemetry record and info dict

Doing 4 before 3 would offer the bus draw before it exists; doing 5 before 3
would fly on last step's thrust; doing 6 before 3 would report pre-firing
margins. All three are silent-corruption bugs, which is why the order is
spelled out rather than left to reading the code.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core.registry import COST_MODEL, MISSION, PROPULSION
from ..core.spaces import Box
from ..core.types import (
    CANONICAL_ACTION_DIM,
    MISSION_OBS_DIM,
    OBS_DIM,
    PROPULSION_OBS_DIM,
    VEHICLE_OBS_DIM,
    CanonicalCommand,
    StepContext,
    Telemetry,
    TerminationReason,
    VehicleState,
    pad_to,
)
from ..economics.base import CostModel, EconomicResult
from ..missions.base import Mission, MissionResult
from ..propulsion.base import PropulsionSystem
from ..spacecraft import dynamics
from ..spacecraft.power import FixedPower, PowerBus, SolarArray
from ..spacecraft.vehicle import (
    VehicleConfig,
    build_initial_state,
    vehicle_observation,
    vehicle_observation_labels,
)

LOGGER = logging.getLogger(__name__)

#: Index of the first entry of each observation block in the 36-wide vector.
MISSION_SLICE = slice(0, MISSION_OBS_DIM)
VEHICLE_SLICE = slice(MISSION_OBS_DIM, MISSION_OBS_DIM + VEHICLE_OBS_DIM)
PROPULSION_SLICE = slice(MISSION_OBS_DIM + VEHICLE_OBS_DIM, OBS_DIM)


@dataclass
class EnvConfig:
    """Knobs that change how the environment runs, not what it models.

    Attributes
    ----------
    max_steps:
        Episode length cap in macro-steps. ``None`` derives it from
        ``mission.max_duration_s / mission.step_dt_s`` so a mission's wall-clock
        limit and the RL horizon can never drift apart.
    normalize_obs:
        Clip the assembled observation into ``observation_space``. The blocks
        are contractually pre-normalised to ~[-1, 1]; this is the backstop that
        stops one misbehaving block from handing a learner a 1e9.
    constraint_penalty_weight:
        Multiplier on ``ConstraintReport.cost`` when the environment folds the
        penalty into the reward. Ignored when the mission already priced it.
    terminate_on_violation:
        End the episode the first time any margin goes negative. Off by default
        -- most constraints here are recoverable, and hard termination makes
        credit assignment much harder.
    record_telemetry:
        Append a :class:`Telemetry` row per step. Turn off for large sweeps
        where only the terminal result matters.
    substeps:
        Floor on the integrator substeps handed to :func:`dynamics.propagate`.
        The actual count is derived per step from the local orbital period (see
        ``substeps_per_orbit``) and is never below this.
    substeps_per_orbit:
        Target RK4 sub-intervals per orbital revolution. RK4 is not symplectic,
        so a step that is coarse relative to the period bleeds orbital energy
        and deorbits the vehicle on numerics alone; in LEO, where one macro-step
        is most of a revolution, a fixed low count swamps the thrust effect the
        benchmark is trying to measure. Set to 0 to disable the derivation and
        use ``substeps`` verbatim.
    seed:
        Seed used by the first ``reset()`` that is not given one explicitly.
    obs_clip:
        Symmetric bound of ``observation_space`` and the clip applied when
        ``normalize_obs`` is set.
    """

    max_steps: int | None = None
    normalize_obs: bool = True
    constraint_penalty_weight: float = 1.0
    terminate_on_violation: bool = False
    record_telemetry: bool = True
    substeps: int = 10
    substeps_per_orbit: int = dynamics.DEFAULT_SUBSTEPS_PER_ORBIT
    seed: int | None = None
    obs_clip: float = 10.0


def _derive_vehicle_config(
    state: VehicleState, propulsion: PropulsionSystem, mission: Mission
) -> VehicleConfig:
    """Build a :class:`VehicleConfig` that matches the mission's own vehicle.

    The config exists to *normalise* the vehicle observation block, so its
    reference values have to describe the vehicle the mission actually handed
    over. A stock config would report a 5 t Mars tug as 900% of nominal mass
    and hand the policy a saturated observation for the whole episode.

    ``structure_mass_fraction`` is zeroed because the mission has already
    accounted for tankage inside ``state.dry_mass_kg``; letting the config add
    its own would double count it and make ``mass_fraction`` drift.
    """
    isp_lo, isp_hi = propulsion.limits().isp_range_s
    isp_ref = 0.5 * (float(isp_lo) + float(isp_hi))
    rated_w = float(propulsion.limits().max_power_w)
    if not math.isfinite(rated_w) or rated_w <= 0.0:
        rated_w = max(float(propulsion.housekeeping_power_w()), 1.0) * 10.0
    duration_s = float(mission.max_duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        duration_s = max(mission.step_dt_s, 1.0) * 1000.0
    return VehicleConfig(
        dry_mass_kg=max(float(state.dry_mass_kg), 1e-6),
        payload_kg=max(float(state.payload_kg), 0.0),
        propellant_capacity_kg=max(float(state.propellant_kg), 0.0),
        structure_mass_fraction=0.0,
        name=f"{mission.name}-derived",
        reference_power_w=rated_w,
        reference_duration_s=duration_s,
        reference_isp_s=isp_ref if isp_ref > 0.0 else 2000.0,
    )


def _default_power_bus(
    propulsion: PropulsionSystem, vehicle_housekeeping_w: float
) -> PowerBus:
    """Size a bus that can actually run this propulsion system.

    A self-powered system (a reactor) needs the bus only for its housekeeping
    load, so it gets a fixed source rather than an array that eclipses and
    degrades. Everything else gets a solar array rated 15% above what the
    thruster and its housekeeping ask for at 1 AU, plus an hour of battery so
    an eclipse does not immediately starve the avionics.

    The bus's ``housekeeping_w`` is the *vehicle's* standing load. The
    propulsion system's own housekeeping is a separate figure passed per step
    as ``extra_load_w``; configuring it in both places would double count it.
    """
    limits = propulsion.limits()
    rated_w = float(limits.max_power_w)
    if not math.isfinite(rated_w) or rated_w <= 0.0:
        rated_w = 0.0
    propulsion_housekeeping_w = float(propulsion.housekeeping_power_w())
    standing_w = max(vehicle_housekeeping_w + propulsion_housekeeping_w, 50.0)

    if propulsion.self_powered:
        source: Any = FixedPower(max(rated_w, 2.0 * standing_w, 100.0))
    else:
        source = SolarArray(1.15 * max(rated_w + propulsion_housekeeping_w, 100.0))
    return PowerBus(
        source,
        housekeeping_w=vehicle_housekeeping_w,
        battery_capacity_wh=standing_w,
    )


def _padded_labels(labels: tuple[str, ...], dim: int, prefix: str) -> tuple[str, ...]:
    """Prefix a block's labels and pad them out to the contracted width."""
    if len(labels) > dim:
        raise ValueError(
            f"{prefix} emitted {len(labels)} labels, contract allows {dim}"
        )
    out = [f"{prefix}/{name}" for name in labels]
    out.extend(f"{prefix}/pad{i}" for i in range(len(labels), dim))
    return tuple(out)


def _has_diverged(state: VehicleState) -> bool:
    """Whether this state must not be propagated any further.

    Defers to :func:`dynamics.is_diverged`, which already covers NaN, overflow
    and the collapse-into-the-central-body case, and which recognises the
    ``DivergedState`` tag the integrator returns instead of NaNs. The extra mass
    check catches a propulsion model that reported a nonsense flow rate.
    """
    return bool(dynamics.is_diverged(state)) or not (
        math.isfinite(state.propellant_kg) and math.isfinite(state.dry_mass_kg)
    )


class PropulsionEnv:
    """A single-agent, five-tuple environment satisfying ``EnvProtocol``.

    Observation is always 36 wide: ``mission.observe`` (12) then
    ``vehicle_observation`` (8) then ``propulsion.observe`` (16), zero-padded
    per block. Action is always 5 wide in [-1, 1]. Both widths are fixed across
    every propulsion system and mission so a policy trained on one pairing can
    be evaluated zero-shot on another.

    Notes
    -----
    A single :class:`StepContext` instance is reused across steps and mutated
    in place -- it is a ``slots`` dataclass and allocating one per step shows up
    in a 10-million-step sweep. Propulsion systems must therefore treat the
    context as valid only for the duration of the call they receive it in, and
    must not retain a reference to it. Reading it inside ``step``, ``observe``
    and ``constraints`` (as the contract describes) is fine.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None

    def __init__(
        self,
        propulsion: PropulsionSystem,
        mission: Mission,
        vehicle: VehicleConfig | None = None,
        power_bus: PowerBus | None = None,
        cost_model: CostModel | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        self.config = config if config is not None else EnvConfig()
        self.propulsion = propulsion
        self.mission = mission
        self.cost_model = cost_model

        # A caller-supplied vehicle overrides the mission's mass budget. With
        # none supplied, one is derived from the mission's own initial state at
        # the first reset -- see _derive_vehicle_config for why that matters.
        self.vehicle: VehicleConfig | None = vehicle
        self._vehicle_overrides_state = vehicle is not None

        vehicle_housekeeping_w = (
            float(getattr(vehicle, "housekeeping_power_w", 0.0))
            if vehicle is not None
            else 0.0
        )
        self.power_bus = (
            power_bus
            if power_bus is not None
            else _default_power_bus(propulsion, vehicle_housekeeping_w)
        )

        dt_s = float(mission.step_dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(
                f"mission {mission.name!r} has step_dt_s={mission.step_dt_s!r}; "
                "it must be a positive number of seconds"
            )
        self._dt_s = dt_s
        # The bus converts stored energy into offered power over a step, and
        # documents that the environment owns this figure.
        self.power_bus.reference_dt_s = dt_s

        if self.config.max_steps is not None:
            max_steps = int(self.config.max_steps)
            if max_steps <= 0:
                raise ValueError(f"max_steps must be positive, got {max_steps}")
        elif mission.max_duration_s and math.isfinite(mission.max_duration_s):
            max_steps = max(1, int(math.ceil(float(mission.max_duration_s) / dt_s)))
        else:
            LOGGER.warning(
                "mission %r has no max_duration_s; falling back to a 1e6 step cap",
                mission.name,
            )
            max_steps = 1_000_000
        self._max_steps = max_steps

        clip = float(self.config.obs_clip)
        self.observation_space = Box(-clip, clip, (OBS_DIM,), np.float32)
        self.action_space = Box(-1.0, 1.0, (CANONICAL_ACTION_DIM,), np.float32)

        limits = propulsion.limits()
        rated_w = float(limits.max_power_w)
        self._rated_power_w = rated_w if math.isfinite(rated_w) and rated_w > 0 else 0.0
        self._self_powered = bool(propulsion.self_powered)
        self._warn_if_solar_distance_unset()
        # The mission fixes the launched wet mass; the propulsion system's own
        # dry mass has to come out of the propellant, or an 18 t reactor flies
        # as if it weighed nothing and the cross-family comparison is a fiction.
        with contextlib.suppress(Exception):
            mission.account_for_propulsion(float(propulsion.bom().dry_mass_kg))

        self._substeps = int(self.config.substeps)
        self._substeps_per_orbit = int(self.config.substeps_per_orbit)
        self._mu = float(getattr(mission, "mu", 0.0) or 0.0)
        self._clip = clip
        self._do_clip = bool(self.config.normalize_obs)
        self._penalty_weight = float(self.config.constraint_penalty_weight)
        self._terminate_on_violation = bool(self.config.terminate_on_violation)
        self._record_telemetry = bool(self.config.record_telemetry)

        # Reused per-step scratch. See the class docstring.
        self._ctx = StepContext(
            t_s=0.0,
            dt_s=dt_s,
            vehicle_mass_kg=0.0,
            available_power_w=0.0,
            heliocentric_radius_m=1.0,
            sink_temperature_k=float(getattr(mission, "sink_temperature_k", 3.0)),
            eclipse=False,
            rng=None,
        )

        self._rng: np.random.Generator | None = None
        self._state: VehicleState | None = None
        self._last_obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)
        self._labels: tuple[str, ...] | None = None
        self._telemetry: list[Telemetry] = []
        self._mission_result: MissionResult | None = None
        self._economics: EconomicResult | None = None
        self._step_count = 0
        self._episode_return = 0.0
        self._total_constraint_cost = 0.0
        self._violation_steps = 0
        self._last_progress = 0.0
        self._done = False
        self._closed = False
        self._warned_diverged = False

    # --- introspection -------------------------------------------------------
    @property
    def unwrapped(self) -> "PropulsionEnv":
        return self

    @property
    def max_steps(self) -> int:
        """Step-limit actually in force, derived or configured."""
        return self._max_steps

    @property
    def dt_s(self) -> float:
        """Simulated seconds per macro-step."""
        return self._dt_s

    @property
    def telemetry(self) -> list[Telemetry]:
        """Per-step records for the episode in progress (or the last one)."""
        return self._telemetry

    @property
    def mission_result(self) -> MissionResult | None:
        """Terminal summary, ``None`` until the episode ends."""
        return self._mission_result

    @property
    def economics(self) -> EconomicResult | None:
        """Costed result, ``None`` without a cost model or before the end."""
        return self._economics

    @property
    def observation_labels(self) -> tuple[str, ...]:
        """Names for all 36 observation entries, block-prefixed. Cached."""
        if self._labels is None:
            self._labels = (
                _padded_labels(
                    tuple(self.mission.observation_labels()),
                    MISSION_OBS_DIM,
                    "mission",
                )
                + _padded_labels(
                    tuple(vehicle_observation_labels()), VEHICLE_OBS_DIM, "vehicle"
                )
                + _padded_labels(
                    tuple(self.propulsion.observation_labels()),
                    PROPULSION_OBS_DIM,
                    "prop",
                )
            )
        return self._labels

    def __repr__(self) -> str:
        return (
            f"<PropulsionEnv propulsion={self.propulsion.name!r} "
            f"mission={self.mission.name!r} max_steps={self._max_steps}>"
        )

    # --- lifecycle -----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Passing ``seed`` restarts the episode rng; omitting it continues the
        existing stream, which is what you want when rolling many episodes off
        one seeded environment. Every stochastic component -- mission
        initialisation, propulsion unit-to-unit variation, bus noise -- draws
        from that one generator, in that fixed order, so a seed reproduces a
        trajectory bit for bit.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self.config.seed)
        rng = self._rng

        state = self.mission.reset(rng)
        if self._vehicle_overrides_state:
            state = build_initial_state(
                self.vehicle,
                state.position_m,
                state.velocity_m_s,
                t0=state.t_s,
            )
        elif self.vehicle is None:
            # Derived once, from the first episode, so the observation
            # normalisation stays stationary across episodes.
            self.vehicle = _derive_vehicle_config(state, self.propulsion, self.mission)
        self.propulsion.reset(rng)
        self.power_bus.reset(rng)

        self._state = state
        self._step_count = 0
        self._episode_return = 0.0
        self._total_constraint_cost = 0.0
        self._violation_steps = 0
        self._telemetry = []
        self._mission_result = None
        self._economics = None
        self._done = False

        eclipse = self.mission.eclipse(state)
        distance_m = self.mission.heliocentric_radius_m(state)
        housekeeping_w = float(self.propulsion.housekeeping_power_w())
        available_w = self._offered_power_w(
            state, eclipse, housekeeping_w, distance_m
        )
        state.power_available_w = available_w
        state.power_generated_w = float(self.power_bus.last_generated_w)

        ctx = self._ctx
        ctx.t_s = state.t_s
        ctx.dt_s = self._dt_s
        ctx.vehicle_mass_kg = state.total_mass_kg
        ctx.available_power_w = available_w
        ctx.heliocentric_radius_m = distance_m
        ctx.eclipse = eclipse
        ctx.rng = rng

        obs = self._build_obs(state, ctx)
        self._last_progress = float(self.mission.progress(state))
        info: dict[str, Any] = {
            "constraint_cost": 0.0,
            "progress": self._last_progress,
            "reward_terms": {},
            "throttled_by": "none",
            "worst_margin": float("inf"),
            "t_s": float(state.t_s),
            "step": 0,
        }
        if options:
            info["options"] = dict(options)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one macro-step. See the module docstring for the ordering."""
        state = self._state
        if state is None:
            raise RuntimeError("step() called before reset()")

        mission = self.mission
        propulsion = self.propulsion
        bus = self.power_bus
        dt_s = self._dt_s

        # 1 -- decode the action.
        command = CanonicalCommand.from_array(action)

        # 2 -- assemble the context this step runs in.
        eclipse = mission.eclipse(state)
        # The mission is the only thing that knows how far from the Sun we are
        # in a planetocentric frame; the solar array is useless without it.
        distance_m = mission.heliocentric_radius_m(state)
        housekeeping_w = float(propulsion.housekeeping_power_w())
        available_w = self._offered_power_w(
            state, eclipse, housekeeping_w, distance_m
        )

        ctx = self._ctx
        ctx.t_s = state.t_s
        ctx.dt_s = dt_s
        ctx.vehicle_mass_kg = state.total_mass_kg
        ctx.available_power_w = available_w
        ctx.heliocentric_radius_m = distance_m
        ctx.eclipse = eclipse
        ctx.rng = self._rng

        # 3 -- fire.
        output = propulsion.step(command, ctx)

        # 4 -- charge the bus for what was actually drawn. A self-powered system
        #      feeds its own thrusters, so the vehicle bus only sees housekeeping.
        draw_w = (
            housekeeping_w
            if self._self_powered
            else float(output.power_draw_w) + housekeeping_w
        )
        bus.step(dt_s, draw_w, state, eclipse, distance_m)

        # 5 -- fly.
        # The substep count follows the orbit, not the wall clock: one macro-step
        # is a large fraction of a LEO revolution and a negligible fraction of a
        # heliocentric one, and RK4's dissipative truncation error at a coarse
        # step deorbits the vehicle on numerics alone.
        substeps = (
            dynamics.substeps_for(
                state.radius_m,
                self._mu,
                dt_s,
                minimum=self._substeps,
                per_orbit=self._substeps_per_orbit,
            )
            if self._substeps_per_orbit > 0
            else self._substeps
        )
        next_state = dynamics.propagate(
            state,
            output.thrust_n,
            command.direction_rtn(),
            output.mdot_kg_s,
            mission,
            dt_s,
            substeps=substeps,
        )
        next_state.power_available_w = available_w
        next_state.power_generated_w = float(bus.last_generated_w)

        step_index = self._step_count + 1
        self._step_count = step_index

        # 6 -- read the post-firing margins and wear.
        constraints = propulsion.constraints()
        health = propulsion.health()
        constraint_cost = constraints.cost
        violated = constraints.violated

        diverged = _has_diverged(next_state)

        if diverged:
            # Do not hand a NaN state to mission code; it would either raise or
            # quietly return NaN rewards that poison the learner's gradients.
            reward = 0.0
            reward_terms: dict[str, float] = {}
            reason = TerminationReason.DIVERGED
            progress = self._last_progress
            obs = self._last_obs
        else:
            # 7 -- reward, then fold in the constraint penalty if the mission
            #      has not already priced it (a non-zero constraint_penalty term
            #      is the mission saying "I handled this").
            terms = mission.reward(state, next_state, output, constraints, health)
            if (
                constraint_cost > 0.0
                and terms.constraint_penalty == 0.0
                and self._penalty_weight != 0.0
            ):
                terms.constraint_penalty = -self._penalty_weight * constraint_cost
            # Coerced: a mission that assembles RewardTerms out of numpy scalars
            # would otherwise leak a 0-d array into every replay buffer.
            reward = float(terms.total)
            reward_terms = terms.as_dict()

            # 8 -- termination, then our own step-limit truncation.
            reason = mission.terminated(next_state, health, constraints)
            if reason is TerminationReason.RUNNING:
                if self._terminate_on_violation and violated:
                    reason = TerminationReason.SAFETY_VIOLATION
                elif step_index >= self._max_steps:
                    reason = TerminationReason.TIMEOUT

            # 9 -- observation.
            ctx.t_s = next_state.t_s
            ctx.vehicle_mass_kg = next_state.total_mass_kg
            ctx.heliocentric_radius_m = mission.heliocentric_radius_m(next_state)
            obs = self._build_obs(next_state, ctx)
            progress = float(mission.progress(next_state))

            if not math.isfinite(reward):
                reward = 0.0
                reason = TerminationReason.DIVERGED
            elif not np.isfinite(obs).all():
                np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                reason = TerminationReason.DIVERGED
            if reason is TerminationReason.DIVERGED:
                diverged = True

        if diverged and not self._warned_diverged:
            self._warned_diverged = True
            LOGGER.warning(
                "%s/%s diverged at step %d (t=%.3es); ending the episode as DIVERGED",
                self.propulsion.name,
                mission.name,
                step_index,
                state.t_s,
            )

        self._state = next_state
        self._last_obs = obs
        self._last_progress = progress
        self._episode_return += reward
        self._total_constraint_cost += constraint_cost
        if violated:
            self._violation_steps += 1

        truncated = reason is TerminationReason.TIMEOUT
        terminated = (reason is not TerminationReason.RUNNING) and not truncated

        info: dict[str, Any] = {
            "constraint_cost": constraint_cost,
            "progress": progress,
            "reward_terms": reward_terms,
            "throttled_by": output.throttled_by,
            "worst_margin": constraints.worst,
            "t_s": float(next_state.t_s),
            "step": step_index,
        }

        if self._record_telemetry:
            record = Telemetry(
                t_s=float(next_state.t_s),
                step=step_index,
                thrust_n=float(output.thrust_n),
                isp_s=float(output.isp_s),
                mdot_kg_s=float(output.mdot_kg_s),
                power_draw_w=float(output.power_draw_w),
                power_available_w=available_w,
                efficiency=float(output.efficiency),
                mass_kg=float(next_state.total_mass_kg),
                propellant_kg=float(next_state.propellant_kg),
                radius_m=float(next_state.radius_m),
                speed_m_s=float(next_state.speed_m_s),
                delta_v_m_s=float(next_state.delta_v_applied_m_s),
                wear_fraction=float(health.wear_fraction),
                constraint_cost=constraint_cost,
                worst_margin=constraints.worst,
                reward=reward,
                progress=progress,
                throttled_by=output.throttled_by,
            )
            extras = propulsion.info()
            if extras:
                record.extras.update(extras)
            self._telemetry.append(record)

        if terminated or truncated:
            self._finalize(next_state, reason, health, info)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Release resources. Nothing external is held; kept for the protocol."""
        self._closed = True

    # --- internals -----------------------------------------------------------
    def _warn_if_solar_distance_unset(self) -> None:
        """Catch the silent-power-loss trap in planetocentric missions.

        A solar array is offered ``mission.heliocentric_radius_m(state)`` as its
        distance to the Sun. The base implementation returns ``state.radius_m``,
        which in a planetocentric frame is the distance to *Earth* -- around
        7e6 m, five orders of magnitude too small. The array clamps that at its
        floor and delivers a fraction of a percent of nameplate power, and the
        run looks like a badly designed spacecraft rather than a bug. Warn once
        at construction instead of losing a sweep to it.
        """
        mission = self.mission
        if mission.frame == "heliocentric" or self._self_powered:
            return
        if type(mission).heliocentric_radius_m is Mission.heliocentric_radius_m:
            LOGGER.warning(
                "mission %r is %s but does not override heliocentric_radius_m; "
                "its solar array will see the distance to the central body "
                "instead of to the Sun and deliver almost no power",
                mission.name,
                mission.frame,
            )

    def _offered_power_w(
        self,
        state: VehicleState,
        eclipse: bool,
        housekeeping_w: float,
        distance_m: float,
    ) -> float:
        """Electrical power the propulsion system may draw this step.

        The bus is asked what is left after the propulsion system's own
        housekeeping load (``extra_load_w``), which is why the load is not also
        configured on the bus. A self-powered system is by contract not limited
        by the bus, so it is offered at least its own rated power -- a finite
        stand-in for "unlimited" that cannot turn into a NaN if a propulsion
        model normalises by it.
        """
        available_w = float(
            self.power_bus.available_w(
                state,
                eclipse,
                extra_load_w=housekeeping_w,
                distance_m=distance_m,
                dt_s=self._dt_s,
            )
        )
        if not math.isfinite(available_w) or available_w < 0.0:
            available_w = 0.0
        if self._self_powered and self._rated_power_w > available_w:
            return self._rated_power_w
        return available_w

    def _build_obs(self, state: VehicleState, ctx: StepContext) -> np.ndarray:
        """Assemble the 36-wide observation into a fresh array.

        Fresh rather than a reused buffer on purpose: replay buffers and
        on-policy rollout stores keep references to what ``step`` returns, and
        handing them a view that the next step overwrites is the classic way to
        silently train on the wrong data.
        """
        out = np.empty(OBS_DIM, dtype=np.float32)
        out[MISSION_SLICE] = self.mission.observe(state)
        out[VEHICLE_SLICE] = pad_to(
            vehicle_observation(state, self.vehicle), VEHICLE_OBS_DIM, "vehicle obs"
        )
        out[PROPULSION_SLICE] = self.propulsion.observe(ctx)
        if self._do_clip:
            np.clip(out, -self._clip, self._clip, out=out)
        return out

    def _finalize(
        self,
        state: VehicleState,
        reason: TerminationReason,
        health: Any,
        info: dict[str, Any],
    ) -> None:
        """Build the terminal records and attach them to ``info``."""
        result = self.mission.summarize(state, reason)
        # The mission owns the physics summary; the environment owns the
        # constraint bookkeeping, so fill those in only if left untouched.
        if result.total_constraint_cost == 0.0:
            result.total_constraint_cost = self._total_constraint_cost
        if result.constraint_violations == 0:
            result.constraint_violations = self._violation_steps
        self._mission_result = result
        self._done = True

        economics: EconomicResult | None = None
        if self.cost_model is not None:
            try:
                economics = self.cost_model.evaluate(
                    self.propulsion.bom(), result, health
                )
            except Exception:  # pragma: no cover - economics must never kill a rollout
                LOGGER.exception(
                    "cost model %r failed on %s/%s; reporting economics=None",
                    getattr(self.cost_model, "name", self.cost_model),
                    self.propulsion.name,
                    self.mission.name,
                )
        self._economics = economics

        info["mission_result"] = result
        info["economics"] = economics
        info["termination_reason"] = reason.value
        info["episode_return"] = self._episode_return
        info["episode_length"] = self._step_count


def make_env(
    propulsion: str,
    mission: str,
    *,
    config: EnvConfig | None = None,
    propulsion_kwargs: dict[str, Any] | None = None,
    mission_kwargs: dict[str, Any] | None = None,
    vehicle: VehicleConfig | None = None,
    cost_model: str | None = None,
    seed: int | None = None,
) -> PropulsionEnv:
    """Build an env from registry names. The one function the matrix calls.

    Parameters
    ----------
    propulsion, mission:
        Registry keys, e.g. ``"hall_spt100"`` and ``"leo_geo_transfer"``.
    config:
        Environment settings. ``seed`` overrides ``config.seed`` when both are
        given, so a sweep can share one config across every seeded replicate.
    cost_model:
        Registry key of a :class:`CostModel`, or ``None`` for no economics.

    Examples
    --------
    >>> env = make_env("hall_spt100", "leo_geo_transfer", seed=0)  # doctest: +SKIP
    >>> obs, info = env.reset()                                    # doctest: +SKIP
    """
    system = PROPULSION.make(propulsion, **(propulsion_kwargs or {}))
    task = MISSION.make(mission, **(mission_kwargs or {}))
    costs = COST_MODEL.make(cost_model) if cost_model else None

    if seed is not None:
        config = replace(config, seed=seed) if config is not None else EnvConfig(seed=seed)

    return PropulsionEnv(
        system,
        task,
        vehicle=vehicle,
        cost_model=costs,
        config=config,
    )


__all__ = ["EnvConfig", "PropulsionEnv", "make_env"]
