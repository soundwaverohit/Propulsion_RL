"""Nuclear electric propulsion: a reactor, a conversion cycle, and an ion/Hall string.

Two scales:

* ``nep_kilopower`` -- KRUSTY-class ~10 kWe Stirling, one Hall string. The
  underpowered end the plausibility rules keep off a crewed fast-Mars run.
* ``nep_brayton`` -- ~2 MWe closed Brayton, a clustered ion string. The
  high-power NEP cell of the comparison.

The reactor is self-powered, so the solar bus does not bind. The electrostatic
string still sees a finite electric budget -- whatever the conversion cycle
delivers this step -- which is how radiator margin and xenon transients become
control problems rather than decorations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...core.constants import EPS, HOUR, YEAR
from ...core.registry import PROPULSION
from ...core.types import (
    BillOfMaterials,
    CanonicalCommand,
    ConstraintReport,
    Event,
    HealthReport,
    Limits,
    PropulsionFamily,
    Severity,
    StepContext,
    ThrusterOutput,
)
from ..base import PropulsionSystem
from ..electric.electrostatic import ElectrostaticDesign, ElectrostaticThruster
from ..electric.hall import HERMES
from ..electric.ion import NEXT
from .reactor import FissionReactor, ReactorDesign


def _sym(x: float) -> float:
    if x <= 0.0:
        return -1.0
    if x >= 1.0:
        return 1.0
    return 2.0 * x - 1.0


@dataclass(frozen=True, slots=True)
class NEPDesign:
    name: str
    reactor: ReactorDesign
    thruster: ElectrostaticDesign
    rated_electric_w: float
    cycle_carnot_fraction: float = 0.55
    radiator_area_m2: float = 40.0
    dry_mass_kg: float = 2_000.0
    housekeeping_w: float = 200.0
    conversion_mass_kg: float = 400.0


class NuclearElectricStage(PropulsionSystem):
    """Reactor + conversion cycle + electrostatic string. ``self_powered``."""

    family = PropulsionFamily.NUCLEAR
    self_powered = True
    propellant = "xenon"

    _OBS_LABELS: tuple[str, ...] = (
        "throttle_achieved",
        "voltage_frac",
        "fuel_temp_frac",
        "electric_power_frac",
        "cycle_eta",
        "xenon_frac",
        "wear_thruster",
        "burnup",
        "thermal_margin",
        "restart_margin",
        "efficiency",
        "isp_frac",
        "period_frac",
        "dtdt_frac",
    )

    def __init__(self, design: NEPDesign) -> None:
        self.design = design
        self.name = design.name
        self.reactor = FissionReactor(design.reactor)
        self.thruster = ElectrostaticThruster(design.thruster)
        self._failed = False
        self._cycle_eta = 0.0
        self._p_electric = 0.0
        self._limits = self._compute_limits()
        self._bom = self._compute_bom()

    def _compute_limits(self) -> Limits:
        tlim = self.thruster.limits()
        d = self.design
        return Limits(
            max_thrust_n=tlim.max_thrust_n,
            min_thrust_n=0.0,
            max_power_w=d.rated_electric_w,
            min_power_w=0.0,
            isp_range_s=tlim.isp_range_s,
            max_temperature_k=d.reactor.max_fuel_temp_k,
            qualified_life_s=min(tlim.qualified_life_s, 15.0 * YEAR),
            max_throughput_kg=tlim.max_throughput_kg,
            max_restarts=tlim.max_restarts,
            min_off_time_s=0.0,
        )

    def _compute_bom(self) -> BillOfMaterials:
        d = self.design
        tb = self.thruster.bom()
        return BillOfMaterials(
            system_name=d.name,
            family=PropulsionFamily.NUCLEAR,
            thruster_units=tb.thruster_units,
            rated_power_w=d.thruster.rated_power_w,
            power_source_w=d.rated_electric_w,
            reactor_thermal_w=d.reactor.rated_thermal_w,
            radiator_area_m2=d.radiator_area_m2,
            dry_mass_kg=d.dry_mass_kg,
            propellant_type="xenon",
            tank_capacity_kg=tb.tank_capacity_kg,
            qualified_life_s=self._limits.qualified_life_s,
            extras={
                "core_mass_kg": d.reactor.core_mass_kg,
                "shield_mass_kg": d.reactor.shield_mass_kg,
                "conversion_mass_kg": d.conversion_mass_kg,
            },
        )

    def reset(self, rng: np.random.Generator) -> None:
        self.reactor.reset(rng)
        self.thruster.reset(rng)
        self._failed = False
        self._cycle_eta = 0.0
        self._p_electric = 0.0

    def _cycle_power(self, thermal_margin: float, sink_k: float) -> tuple[float, float]:
        d = self.design
        t_hot = max(self.reactor.fuel_temperature_k, sink_k + 1.0)
        t_cold = max(sink_k, 250.0)
        carnot = 1.0 - t_cold / t_hot
        eta = d.cycle_carnot_fraction * max(carnot, 0.0)
        eta *= 1.0 - 0.35 * thermal_margin
        eta = min(max(eta, 0.0), 0.45)
        p_e = eta * self.reactor.thermal_power_w
        p_e = min(p_e, d.rated_electric_w)
        return p_e, eta

    def step(self, command: CanonicalCommand, ctx: StepContext) -> ThrusterOutput:
        d = self.design
        dt = float(max(ctx.dt_s, 0.0))
        tm = float(min(max(command.thermal_margin, 0.0), 1.0))
        throttle = float(min(max(command.throttle, 0.0), 1.0))
        events: list[Event] = []

        if self._failed or self.reactor.destroyed or self.thruster.health().failed:
            self._failed = True
            self.reactor.step(
                -self.reactor.drum_worth_pcm,
                coolant_flow=max(tm, 0.1),
                heat_removal_w=0.2 * d.reactor.rated_thermal_w,
                dt_s=dt,
                scram=True,
            )
            off = CanonicalCommand(0.0, command.operating_point, 0.0, 0.0, tm)
            dead_ctx = StepContext(
                t_s=ctx.t_s,
                dt_s=dt,
                vehicle_mass_kg=ctx.vehicle_mass_kg,
                available_power_w=0.0,
                heliocentric_radius_m=ctx.heliocentric_radius_m,
                sink_temperature_k=ctx.sink_temperature_k,
                eclipse=ctx.eclipse,
                rng=ctx.rng,
            )
            out = self.thruster.step(off, dead_ctx)
            return ThrusterOutput(
                thermal_power_w=self.reactor.thermal_power_w,
                heat_reject_w=self.reactor.decay_heat_w,
                throttled_by="failed",
                events=list(self.reactor.events) + list(out.events),
            )

        firing = throttle > 0.02
        if firing:
            reactivity_cmd = self.reactor.drum_for_critical() + 120.0 * throttle
            flow = 0.3 + 0.7 * throttle
            heat_cap = (0.35 + 0.65 * throttle) * d.reactor.rated_thermal_w
        else:
            reactivity_cmd = self.reactor.drum_for_critical() - 400.0
            flow = max(0.15, 0.5 * tm)
            heat_cap = flow * 0.25 * d.reactor.rated_thermal_w

        tel = self.reactor.step(
            reactivity_cmd,
            coolant_flow=flow,
            heat_removal_w=heat_cap,
            dt_s=dt,
            power_fraction=throttle if firing else 0.05,
        )
        events.extend(self.reactor.events)

        p_e, eta_cycle = self._cycle_power(tm, ctx.sink_temperature_k)
        self._cycle_eta = eta_cycle
        self._p_electric = p_e

        inner_ctx = StepContext(
            t_s=ctx.t_s,
            dt_s=dt,
            vehicle_mass_kg=ctx.vehicle_mass_kg,
            available_power_w=p_e,
            heliocentric_radius_m=ctx.heliocentric_radius_m,
            sink_temperature_k=ctx.sink_temperature_k,
            eclipse=ctx.eclipse,
            rng=ctx.rng,
        )
        out = self.thruster.step(command, inner_ctx)
        events.extend(out.events)

        thermal_w = float(tel["thermal_power_w"])
        heat_reject = max(thermal_w - out.power_draw_w, 0.0)
        efficiency = out.efficiency
        if self.reactor.destroyed or self.thruster.health().failed:
            self._failed = True
            events.append(
                Event("nep_failed", Severity.FATAL, f"{d.name} failed", 1.0)
            )

        throttled = out.throttled_by
        if firing and p_e < 0.5 * throttle * d.rated_electric_w:
            throttled = "power"

        return ThrusterOutput(
            thrust_n=out.thrust_n,
            mdot_kg_s=out.mdot_kg_s,
            isp_s=out.isp_s,
            power_draw_w=out.power_draw_w,
            thermal_power_w=thermal_w,
            heat_reject_w=heat_reject,
            efficiency=efficiency,
            throttled_by=throttled,
            events=events,
        )

    def observe_raw(self, ctx: StepContext) -> np.ndarray:
        d = self.design
        t_obs = self.thruster.observe_raw(ctx)
        xe = abs(self.reactor.xenon_reactivity_pcm) / max(
            abs(d.reactor.xenon_equilibrium_pcm), 1.0
        )
        restart = self.reactor.restart_margin_pcm() / max(self.reactor.drum_worth_pcm, 1.0)
        period = self.reactor.period_s
        period_frac = 1.0 if not math.isfinite(period) else min(
            period / max(d.reactor.min_period_s * 20.0, 1.0), 1.0
        )
        health = self.thruster.health()
        return np.array(
            [
                float(t_obs[0]),
                float(t_obs[1]),
                _sym(self.reactor.fuel_temperature_k / d.reactor.max_fuel_temp_k),
                _sym(self._p_electric / max(d.rated_electric_w, EPS)),
                _sym(self._cycle_eta / 0.45),
                _sym(min(xe, 1.0)),
                _sym(health.wear_fraction),
                _sym(self.reactor.burnup_fraction / max(d.reactor.burnup_limit_fima, EPS)),
                float(t_obs[8]),
                float(np.clip(restart, -1.0, 1.0)),
                float(t_obs[5]),
                float(t_obs[7]),
                _sym(period_frac),
                _sym(
                    abs(self.reactor.fuel_dtdt_k_s) / max(d.reactor.max_fuel_dtdt_k_s, 1.0)
                ),
            ],
            dtype=np.float32,
        )

    def observation_labels(self) -> tuple[str, ...]:
        return self._OBS_LABELS

    def limits(self) -> Limits:
        return self._limits

    def constraints(self) -> ConstraintReport:
        core = self.reactor.constraint_margins()
        thr = self.thruster.constraints()
        merged = dict(core)
        for name, margin in zip(thr.names, thr.margins):
            merged[f"thruster_{name}"] = float(margin)
        names = tuple(merged)
        margins = np.fromiter(merged.values(), dtype=np.float64, count=len(merged))
        return ConstraintReport(names=names, margins=margins)

    def health(self) -> HealthReport:
        th = self.thruster.health()
        burnup_frac = self.reactor.burnup_fraction / max(
            self.design.reactor.burnup_limit_fima, EPS
        )
        wear = max(th.wear_fraction, min(burnup_frac, 1.0))
        return HealthReport(
            wear_fraction=wear,
            remaining_life_s=th.remaining_life_s,
            throughput_kg=th.throughput_kg,
            burn_time_s=th.burn_time_s,
            restarts=th.restarts,
            degraded_efficiency=th.degraded_efficiency,
            failed=self._failed or self.reactor.destroyed or th.failed,
        )

    def bom(self) -> BillOfMaterials:
        return replace(self._bom, extras=dict(self._bom.extras))

    def decode_action(self, command: CanonicalCommand) -> dict[str, float]:
        decoded = self.thruster.decode_action(command)
        decoded["cycle_eta"] = self._cycle_eta
        decoded["electric_power_w"] = self._p_electric
        return decoded

    def housekeeping_power_w(self) -> float:
        return float(self.design.housekeeping_w + self.thruster.housekeeping_power_w())

    def info(self) -> dict[str, Any]:
        info = self.thruster.info()
        info.update(
            {
                "fuel_temperature_k": self.reactor.fuel_temperature_k,
                "cycle_eta": self._cycle_eta,
                "electric_power_w": self._p_electric,
                "xenon_pcm": self.reactor.xenon_reactivity_pcm,
            }
        )
        return info


_KILO_CORE = ReactorDesign(
    name="kilopower_core",
    rated_thermal_w=43_000.0,
    fuel_mass_kg=28.0,
    struct_mass_kg=80.0,
    fuel_to_struct_ua_w_k=800.0,
    coolant_ua_w_k=1_200.0,
    coolant_inlet_k=400.0,
    initial_temperature_k=800.0,
    max_fuel_temp_k=1100.0,
    fuel_melt_temp_k=1400.0,
    max_fuel_dtdt_k_s=15.0,
    drum_worth_pcm=2_500.0,
    max_reactivity_rate_pcm_s=20.0,
    doppler_pcm_per_k_at_300k=-1.8,
    moderator_pcm_per_k=-0.8,
    min_period_s=10.0,
    design_flux_n_cm2_s=5.0e12,
    xenon_equilibrium_pcm=-400.0,
    fissile_mass_kg=28.0,
    burnup_limit_fima=0.05,
    burnup_swing_pcm=-1_500.0,
    core_mass_kg=400.0,
    shield_mass_kg=800.0,
    initial_power_fraction=0.15,
    fuel_cp_j_kg_k=200.0,
    struct_cp_j_kg_k=500.0,
)

_KILO_THRUSTER = replace(
    HERMES,
    name="nep_kilopower_hall",
    rated_power_w=10_000.0,
    mdot_anode_max_kg_s=1.7e-5,
    voltage_min_v=250.0,
    voltage_max_v=600.0,
    voltage_nominal_v=400.0,
    dry_mass_kg=90.0,
    housekeeping_w=40.0,
    qualified_life_s=50_000.0 * HOUR,
    max_throughput_kg=1_200.0,
)

KILOPOWER = NEPDesign(
    name="nep_kilopower",
    reactor=_KILO_CORE,
    thruster=_KILO_THRUSTER,
    rated_electric_w=10_000.0,
    cycle_carnot_fraction=0.45,
    radiator_area_m2=30.0,
    dry_mass_kg=1_800.0,
    housekeeping_w=80.0,
    conversion_mass_kg=250.0,
)

_BRAYTON_CORE = ReactorDesign(
    name="brayton_core",
    rated_thermal_w=10.0e6,
    fuel_mass_kg=350.0,
    struct_mass_kg=600.0,
    fuel_to_struct_ua_w_k=8.0e3,
    coolant_ua_w_k=1.2e4,
    coolant_inlet_k=500.0,
    initial_temperature_k=1100.0,
    max_fuel_temp_k=1500.0,
    fuel_melt_temp_k=1800.0,
    max_fuel_dtdt_k_s=20.0,
    drum_worth_pcm=5_000.0,
    max_reactivity_rate_pcm_s=30.0,
    doppler_pcm_per_k_at_300k=-2.0,
    min_period_s=8.0,
    design_flux_n_cm2_s=2.0e13,
    xenon_equilibrium_pcm=-1_200.0,
    fissile_mass_kg=80.0,
    burnup_limit_fima=0.08,
    burnup_swing_pcm=-4_000.0,
    core_mass_kg=4_500.0,
    shield_mass_kg=6_000.0,
    initial_power_fraction=0.20,
)

_BRAYTON_THRUSTER = replace(
    NEXT,
    name="nep_brayton_ion",
    rated_power_w=2.0e6,
    mdot_anode_max_kg_s=1.7e-3,
    voltage_min_v=600.0,
    voltage_max_v=2_000.0,
    voltage_nominal_v=1_800.0,
    open_area_m2=12.0,
    dry_mass_kg=2_000.0,
    thruster_units=48,
    housekeeping_w=2_000.0,
    qualified_life_s=40_000.0 * HOUR,
    max_throughput_kg=20_000.0,
    thermal_mass_kg=2_500.0,
    thermal_area_m2=180.0,
    max_temperature_k=600.0,
)

BRAYTON = NEPDesign(
    name="nep_brayton",
    reactor=_BRAYTON_CORE,
    thruster=_BRAYTON_THRUSTER,
    rated_electric_w=2.0e6,
    cycle_carnot_fraction=0.55,
    radiator_area_m2=4_500.0,
    dry_mass_kg=18_000.0,
    housekeeping_w=1_500.0,
    conversion_mass_kg=3_500.0,
)


def _factory(design: NEPDesign):
    def make(**_kwargs: Any) -> NuclearElectricStage:
        return NuclearElectricStage(design)

    make.__name__ = design.name
    make.__qualname__ = design.name
    return make


PROPULSION.add(
    KILOPOWER.name,
    _factory(KILOPOWER),
    family=PropulsionFamily.NUCLEAR,
    power_w=KILOPOWER.rated_electric_w,
    kind="nep",
)
PROPULSION.add(
    BRAYTON.name,
    _factory(BRAYTON),
    family=PropulsionFamily.NUCLEAR,
    power_w=BRAYTON.rated_electric_w,
    kind="nep",
)

__all__ = [
    "NuclearElectricStage",
    "NEPDesign",
    "KILOPOWER",
    "BRAYTON",
]
