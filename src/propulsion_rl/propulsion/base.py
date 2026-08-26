"""The propulsion system contract.

A ``PropulsionSystem`` is a stateful physical model. It owns everything from
the power input terminal to the exhaust plane: conversion efficiency, thermal
state, wear, failure modes, and the actuator mapping from the canonical command
space. It owns nothing outside that boundary -- no orbital mechanics, no
mission logic, no dollars.

Implementations live in ``propulsion/electric/`` and ``propulsion/nuclear/``
and are exposed through ``core.registry.PROPULSION``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.types import (
    PROPULSION_OBS_DIM,
    BillOfMaterials,
    CanonicalCommand,
    ConstraintReport,
    HealthReport,
    Limits,
    PropulsionFamily,
    StepContext,
    ThrusterOutput,
    pad_to,
)


class PropulsionSystem(ABC):
    """Base class for every thruster / reactor model.

    Lifecycle per episode::

        sys.reset(rng)
        for each step:
            out = sys.step(command, ctx)     # advances internal state by ctx.dt_s
            obs = sys.observe(ctx)
            con = sys.constraints()

    Contract notes for implementers
    -------------------------------
    * ``step`` must be the *only* method that mutates state. ``observe``,
      ``constraints``, ``health`` and ``bom`` must be pure reads, because the
      environment and the logger call them repeatedly within a step.
    * ``step`` must respect ``ctx.available_power_w``. Drawing more power than
      the bus can supply is a modelling bug, not a constraint violation: clamp,
      and record it in ``ThrusterOutput.throttled_by``.
    * Never raise from ``step`` for physical reasons. Degrade, set
      ``HealthReport.failed``, and emit an ``Event`` with ``Severity.FATAL``.
      The environment decides whether that ends the episode.
    * Integrate internal ODEs with substeps when ``ctx.dt_s`` is large relative
      to your fastest time constant. The environment may hand you steps of many
      hours; a reactor thermal model must not be Euler-integrated at that rate.
    """

    #: Human-readable identifier, matches the registry key.
    name: str = "abstract"
    #: Which comparison bucket this belongs to.
    family: PropulsionFamily = PropulsionFamily.ELECTRIC
    #: Propellant species key, used by the economics model for pricing.
    propellant: str = "xenon"
    #: True when the system carries its own power source (a reactor) and so is
    #: not limited by ``StepContext.available_power_w``. Solar-electric systems
    #: leave this False and live within the bus budget.
    self_powered: bool = False

    # --- lifecycle -----------------------------------------------------------
    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Restore to a start-of-mission state.

        ``rng`` seeds unit-to-unit manufacturing variation and stochastic wear.
        Two calls with identically seeded generators must produce identical
        trajectories -- reproducibility is load-bearing for the comparison.
        """

    @abstractmethod
    def step(self, command: CanonicalCommand, ctx: StepContext) -> ThrusterOutput:
        """Advance internal state by ``ctx.dt_s`` and return what was produced."""

    # --- observation ---------------------------------------------------------
    @abstractmethod
    def observe_raw(self, ctx: StepContext) -> np.ndarray:
        """Propulsion-specific observation block, already normalised to ~[-1, 1].

        Return at most :data:`PROPULSION_OBS_DIM` values. Order must be stable
        across versions and must match :meth:`observation_labels`.
        """

    def observe(self, ctx: StepContext) -> np.ndarray:
        """Zero-padded observation block of the contracted fixed width."""
        return pad_to(self.observe_raw(ctx), PROPULSION_OBS_DIM, f"{self.name} obs")

    @abstractmethod
    def observation_labels(self) -> tuple[str, ...]:
        """Names for the entries of :meth:`observe_raw`, for interpretability."""

    # --- introspection -------------------------------------------------------
    @abstractmethod
    def limits(self) -> Limits:
        """Static operating envelope. Must not change during an episode."""

    @abstractmethod
    def constraints(self) -> ConstraintReport:
        """Signed, limit-normalised safety margins. Negative == violated."""

    @abstractmethod
    def health(self) -> HealthReport:
        """Current degradation state."""

    @abstractmethod
    def bom(self) -> BillOfMaterials:
        """Physical inventory for the cost model. Never returns dollars."""

    # --- optional hooks ------------------------------------------------------
    def decode_action(self, command: CanonicalCommand) -> dict[str, float]:
        """Native actuator setpoints for this command, for logging and for MPC.

        Default is a passthrough; override to expose e.g. discharge voltage or
        control-drum angle so telemetry is physically interpretable.
        """
        return {
            "throttle": command.throttle,
            "operating_point": command.operating_point,
            "thermal_margin": command.thermal_margin,
        }

    def housekeeping_power_w(self) -> float:
        """Standing electrical load this system imposes even when not firing.

        Cathode keeper, PPU idle draw, reactor instrumentation and coolant pumps
        all belong here. The vehicle subtracts it before offering power.
        """
        return 0.0

    def info(self) -> dict[str, Any]:
        """Free-form extra telemetry merged into ``Telemetry.extras``."""
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} family={self.family.value}>"
