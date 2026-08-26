"""Nuclear thermal propulsion: a fission core heating hydrogen through a nozzle.

Pewee (LASL 1968, 503 MW(t), ~845 s) and NERVA XE-Prime (~1.1 GW(t)) are the
two ground-test anchors. The agent commands throttle (hydrogen flow / power
demand), operating point (chamber temperature / Isp) and thermal margin
(radiator bypass vs. putting the heat into the propellant).

Energy bookkeeping is forced: jet power cannot exceed the heat the core
actually delivered into the hydrogen this step. Specific impulse is a
calibrated sqrt(T) hydrogen expansion, so the Isp/thrust trade at fixed reactor
power is the one the comparison is about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...core.constants import EPS, G0, HOUR
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
from .reactor import FissionReactor, ReactorDesign


def _sym(x: float) -> float:
    if x <= 0.0:
        return -1.0
    if x >= 1.0:
        return 1.0
    return 2.0 * x - 1.0


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


@dataclass(frozen=True, slots=True)
class NTPDesign:
    """Stage wrapping a :class:`ReactorDesign` with a hydrogen expander nozzle."""

    name: str
    reactor: ReactorDesign
    isp_ref_s: float
    t_ref_k: float
    t_chamber_min_k: float
    t_chamber_max_k: float
    mdot_max_kg_s: float
    nozzle_efficiency: float = 0.88
    cp_j_kg_k: float = 1.4e4  # effective, hot H2 including some dissociation
    coolant_inlet_k: float = 30.0
    dry_mass_kg: float = 6_000.0
    radiator_area_m2: float = 20.0
    tank_capacity_kg: float = 0.0
    housekeeping_w: float = 2_000.0
    pump_w_per_kg_s: float = 1.2e5
    qualified_life_s: float = 2.0 * HOUR  # campaign burn time
    run_duration_s: float = 1.5 * HOUR  # one start, NERVA/Pewee class
    max_restarts: int = 60
    max_throughput_kg: float = 80_000.0


def ntp_isp(t_gas_k: float, design: NTPDesign) -> float:
    """Frozen-flow hydrogen Isp, calibrated to the design's reference point."""
    t = max(float(t_gas_k), 50.0)
    return design.isp_ref_s * math.sqrt(t / max(design.t_ref_k, 1.0))


class NuclearThermalRocket(PropulsionSystem):
    """Hydrogen-cooled NTP stage. ``self_powered`` -- the core is the power source."""

    family = PropulsionFamily.NUCLEAR
    self_powered = True
    propellant = "lh2"

    _OBS_LABELS: tuple[str, ...] = (
        "throttle_achieved",
        "chamber_temp_frac",
        "fuel_temp_frac",
        "power_frac",
        "reactivity_dollar",
        "xenon_frac",
        "period_frac",
        "burnup",
        "thermal_margin",
        "dtdt_frac",
        "restart_margin",
        "isp_frac",
        "drum_frac",
        "efficiency",
    )

    def __init__(self, design: NTPDesign) -> None:
        self.design = design
        self.name = design.name
        self.reactor = FissionReactor(design.reactor)
        self._failed = False
        self._was_firing = False
        self._throughput_kg = 0.0
        self._burn_time_s = 0.0
        self._restarts = 0
        self._last: dict[str, float] = {}
        self._limits = self._compute_limits()
        self._bom = self._compute_bom()
        self._zero_last()

    def _compute_limits(self) -> Limits:
        d = self.design
        isp_lo = ntp_isp(d.t_chamber_min_k, d) * 0.92
        isp_hi = ntp_isp(d.t_chamber_max_k, d) * 1.05
        t_flow = d.mdot_max_kg_s * G0 * isp_lo
        t_power = (
            2.0 * d.nozzle_efficiency * d.reactor.rated_thermal_w / (G0 * isp_lo)
        )
        return Limits(
            max_thrust_n=min(t_flow, t_power),
            min_thrust_n=0.0,
            max_power_w=d.reactor.rated_thermal_w,
            min_power_w=0.0,
            isp_range_s=(max(isp_lo, 200.0), isp_hi),
            max_temperature_k=d.reactor.max_fuel_temp_k,
            qualified_life_s=d.qualified_life_s,
            max_throughput_kg=d.max_throughput_kg,
            max_restarts=d.max_restarts,
            min_off_time_s=0.0,
        )

    def _compute_bom(self) -> BillOfMaterials:
        d = self.design
        rd = d.reactor
        return BillOfMaterials(
            system_name=d.name,
            family=PropulsionFamily.NUCLEAR,
            thruster_units=1,
            rated_power_w=0.0,
            power_source_w=0.0,
            reactor_thermal_w=rd.rated_thermal_w,
            radiator_area_m2=d.radiator_area_m2,
            dry_mass_kg=d.dry_mass_kg,
            propellant_type="lh2",
            tank_capacity_kg=d.tank_capacity_kg,
            qualified_life_s=d.qualified_life_s,
            extras={
                "core_mass_kg": rd.core_mass_kg,
                "shield_mass_kg": rd.shield_mass_kg,
            },
        )

    def _zero_last(self) -> None:
        self._last = {
            "throttle": 0.0,
            "t_gas_k": self.design.coolant_inlet_k,
            "isp_s": 0.0,
            "efficiency": 0.0,
            "thermal_margin": 0.5,
            "power_frac": 0.0,
            "mdot": 0.0,
        }

    def reset(self, rng: np.random.Generator) -> None:
        self.reactor.reset(rng)
        self._failed = False
        self._was_firing = False
        self._throughput_kg = 0.0
        self._burn_time_s = 0.0
        self._restarts = 0
        self._zero_last()

    def step(self, command: CanonicalCommand, ctx: StepContext) -> ThrusterOutput:
        d = self.design
        dt = float(max(ctx.dt_s, 0.0))
        throttle = float(min(max(command.throttle, 0.0), 1.0))
        tm = float(min(max(command.thermal_margin, 0.0), 1.0))
        t_cmd = _lerp(d.t_chamber_min_k, d.t_chamber_max_k, command.operating_point)
        events: list[Event] = []

        if self._failed or self.reactor.destroyed:
            self._failed = True
            self.reactor.step(
                -self.reactor.drum_worth_pcm,
                coolant_flow=max(tm, 0.05),
                heat_removal_w=0.05 * d.reactor.rated_thermal_w,
                dt_s=dt,
                scram=True,
            )
            self._was_firing = False
            self._zero_last()
            return ThrusterOutput(
                thermal_power_w=self.reactor.thermal_power_w,
                heat_reject_w=self.reactor.decay_heat_w,
                throttled_by="failed",
                events=list(self.reactor.events),
            )

        remaining_life = max(d.qualified_life_s - self._burn_time_s, 0.0)
        # A day-long environment step is many heritage NTP run times. Throttle
        # is the power level of one pulse; the pulse is capped at run_duration.
        on_s = dt
        firing = throttle > 0.02 and remaining_life > 1.0
        if firing:
            on_s = min(dt, d.run_duration_s, remaining_life)
            flow = min(max(throttle, 0.05), 1.5)
            max_rate = d.reactor.max_reactivity_rate_pcm_s * (0.4 + 0.6 * (1.0 - tm))
            heat_needed = throttle * d.reactor.rated_thermal_w * (1.0 - 0.25 * tm)
        else:
            flow = max(0.08, 0.4 * tm)
            max_rate = d.reactor.max_reactivity_rate_pcm_s
            heat_needed = 0.0

        tel = self.reactor.step(
            0.0,
            coolant_flow=flow,
            heat_removal_w=heat_needed if firing else flow * 0.15 * d.reactor.rated_thermal_w,
            dt_s=on_s if firing else dt,
            max_rate_pcm_s=max_rate,
            coolant_inlet_k=d.coolant_inlet_k,
            power_fraction=throttle if firing else 0.0,
        )
        events.extend(self.reactor.events)
        if firing and dt > on_s + 1.0:
            self.reactor.step(
                0.0,
                coolant_flow=max(0.08, 0.4 * tm),
                heat_removal_w=0.1 * d.reactor.rated_thermal_w,
                dt_s=dt - on_s,
                max_rate_pcm_s=d.reactor.max_reactivity_rate_pcm_s,
                coolant_inlet_k=d.coolant_inlet_k,
                power_fraction=0.0,
            )
            events.extend(self.reactor.events)

        thermal_w = float(tel["thermal_power_w"])
        heat_h2 = float(tel["heat_removed_w"]) if firing else 0.0
        heat_reject = max(thermal_w - heat_h2, 0.0)

        if firing and heat_h2 > EPS:
            dT = max(t_cmd - d.coolant_inlet_k, 1.0)
            mdot = heat_h2 / (d.cp_j_kg_k * dT)
            mdot = min(max(mdot, 0.0), d.mdot_max_kg_s)
            t_gas = d.coolant_inlet_k + heat_h2 / max(mdot * d.cp_j_kg_k, EPS)
            t_gas = min(t_gas, self.reactor.fuel_temperature_k, d.t_chamber_max_k)
            t_gas = max(t_gas, d.coolant_inlet_k + 1.0)
            isp = ntp_isp(t_gas, d)
            jet = 0.5 * mdot * (isp * G0) ** 2
            jet_cap = d.nozzle_efficiency * max(heat_h2, 0.0)
            if jet > jet_cap and mdot > EPS:
                ve = math.sqrt(2.0 * max(jet_cap, 0.0) / mdot)
                isp = ve / G0
                jet = 0.5 * mdot * ve * ve
            thrust = mdot * G0 * isp
            efficiency = jet / thermal_w if thermal_w > EPS else 0.0
            efficiency = min(max(efficiency, 0.0), 1.0)
            pump_w = d.pump_w_per_kg_s * mdot
            if not self._was_firing:
                self._restarts += 1
            self._was_firing = True
            self._throughput_kg += mdot * on_s
            self._burn_time_s += on_s
            if dt > on_s > 0.0:
                # Time-average the pulse over the macro-step so the integrator
                # sees the right impulse from a minutes-long burn in an hour-scale step.
                scale = on_s / dt
                thrust *= scale
                mdot *= scale
            throttled_by = "none"
            if heat_h2 + 1.0 < heat_needed * 0.85:
                throttled_by = "thermal"
        else:
            t_gas = d.coolant_inlet_k
            isp = 0.0
            thrust = 0.0
            mdot = 0.0
            efficiency = 0.0
            pump_w = 0.0
            self._was_firing = False
            throttled_by = "none"

        if self.reactor.destroyed or self._wear_fraction() >= 1.0:
            if not self._failed:
                events.append(
                    Event("ntp_failed", Severity.FATAL, f"{d.name} failed", 1.0)
                )
            self._failed = True

        self._last = {
            "throttle": throttle if firing else 0.0,
            "t_gas_k": t_gas,
            "isp_s": isp,
            "efficiency": efficiency,
            "thermal_margin": tm,
            "power_frac": self.reactor.power_fraction,
            "mdot": mdot,
        }
        return ThrusterOutput(
            thrust_n=thrust,
            mdot_kg_s=mdot,
            isp_s=isp,
            power_draw_w=pump_w,
            thermal_power_w=thermal_w,
            heat_reject_w=heat_reject,
            efficiency=efficiency,
            throttled_by=throttled_by,
            events=events,
        )

    def _wear_fraction(self) -> float:
        d = self.design
        burnup = self.reactor.burnup_fraction / max(d.reactor.burnup_limit_fima, EPS)
        time_f = self._burn_time_s / max(d.qualified_life_s, EPS)
        thru = self._throughput_kg / max(d.max_throughput_kg, EPS)
        rest = self._restarts / max(d.max_restarts, 1)
        return float(min(max(burnup, time_f, thru, rest), 1.0))

    def observe_raw(self, ctx: StepContext) -> np.ndarray:
        d = self.design
        isp_lo, isp_hi = self._limits.isp_range_s
        isp_frac = 0.5
        if self._last["isp_s"] > 0.0 and isp_hi > isp_lo:
            isp_frac = (self._last["isp_s"] - isp_lo) / (isp_hi - isp_lo)
        period = self.reactor.period_s
        if not math.isfinite(period):
            period_frac = 1.0
        else:
            period_frac = min(period / max(d.reactor.min_period_s * 20.0, 1.0), 1.0)
        xe = abs(self.reactor.xenon_reactivity_pcm) / max(
            abs(d.reactor.xenon_equilibrium_pcm), 1.0
        )
        restart = self.reactor.restart_margin_pcm() / max(self.reactor.drum_worth_pcm, 1.0)
        rho_dollar = self.reactor.reactivity_pcm / 650.0
        drum = 0.5 + 0.5 * self.reactor.drum_reactivity_pcm / max(
            self.reactor.drum_worth_pcm, 1.0
        )
        return np.array(
            [
                _sym(self._last["throttle"]),
                _sym(self._last["t_gas_k"] / d.t_chamber_max_k),
                _sym(self.reactor.fuel_temperature_k / d.reactor.max_fuel_temp_k),
                _sym(min(self.reactor.power_fraction, 1.5) / 1.5),
                float(np.clip(rho_dollar, -1.0, 1.0)),
                _sym(min(xe, 1.0)),
                _sym(period_frac),
                _sym(self.reactor.burnup_fraction / max(d.reactor.burnup_limit_fima, EPS)),
                _sym(self._last["thermal_margin"]),
                _sym(
                    abs(self.reactor.fuel_dtdt_k_s) / max(d.reactor.max_fuel_dtdt_k_s, 1.0)
                ),
                float(np.clip(restart, -1.0, 1.0)),
                _sym(isp_frac),
                _sym(min(max(drum, 0.0), 1.0)),
                _sym(self._last["efficiency"]),
            ],
            dtype=np.float32,
        )

    def observation_labels(self) -> tuple[str, ...]:
        return self._OBS_LABELS

    def limits(self) -> Limits:
        return self._limits

    def constraints(self) -> ConstraintReport:
        core = self.reactor.constraint_margins()
        d = self.design
        core["chamber_temperature"] = (
            d.t_chamber_max_k - self._last["t_gas_k"]
        ) / d.t_chamber_max_k
        core["wear"] = 1.0 - self._wear_fraction()
        names = tuple(core)
        margins = np.fromiter(core.values(), dtype=np.float64, count=len(core))
        return ConstraintReport(names=names, margins=margins)

    def health(self) -> HealthReport:
        d = self.design
        remaining = max(d.qualified_life_s - self._burn_time_s, 0.0)
        return HealthReport(
            wear_fraction=self._wear_fraction(),
            remaining_life_s=remaining,
            throughput_kg=self._throughput_kg,
            burn_time_s=self._burn_time_s,
            restarts=self._restarts,
            degraded_efficiency=max(0.0, 1.0 - 0.2 * self._wear_fraction()),
            failed=self._failed or self.reactor.destroyed,
        )

    def bom(self) -> BillOfMaterials:
        return replace(self._bom, extras=dict(self._bom.extras))

    def decode_action(self, command: CanonicalCommand) -> dict[str, float]:
        d = self.design
        return {
            "throttle": command.throttle,
            "operating_point": command.operating_point,
            "thermal_margin": command.thermal_margin,
            "chamber_temperature_k": _lerp(
                d.t_chamber_min_k, d.t_chamber_max_k, command.operating_point
            ),
            "mdot_demand_kg_s": command.throttle * d.mdot_max_kg_s,
        }

    def housekeeping_power_w(self) -> float:
        return float(self.design.housekeeping_w)

    def info(self) -> dict[str, Any]:
        return {
            "fuel_temperature_k": self.reactor.fuel_temperature_k,
            "reactivity_pcm": self.reactor.reactivity_pcm,
            "xenon_pcm": self.reactor.xenon_reactivity_pcm,
            "burnup_fraction": self.reactor.burnup_fraction,
        }


_PEWEE_CORE = ReactorDesign(
    name="pewee_core",
    rated_thermal_w=503e6,
    fuel_mass_kg=450.0,
    struct_mass_kg=800.0,
    fuel_to_struct_ua_w_k=2.5e5,
    coolant_ua_w_k=4.0e5,
    coolant_inlet_k=30.0,
    initial_temperature_k=300.0,
    max_fuel_temp_k=2750.0,
    fuel_melt_temp_k=3100.0,
    max_fuel_dtdt_k_s=80.0,
    drum_worth_pcm=8_000.0,
    max_reactivity_rate_pcm_s=80.0,
    doppler_pcm_per_k_at_300k=-2.4,
    min_period_s=5.0,
    design_flux_n_cm2_s=1.0e14,
    xenon_equilibrium_pcm=-2_200.0,
    fissile_mass_kg=28.0,
    burnup_limit_fima=0.02,
    burnup_swing_pcm=-3_000.0,
    core_mass_kg=1_800.0,
    shield_mass_kg=2_200.0,
    initial_power_fraction=1.0e-3,
)

PEWEE = NTPDesign(
    name="ntp_pewee",
    reactor=_PEWEE_CORE,
    isp_ref_s=845.0,
    t_ref_k=2550.0,
    t_chamber_min_k=1800.0,
    t_chamber_max_k=2550.0,
    mdot_max_kg_s=14.0,
    dry_mass_kg=6_500.0,
    radiator_area_m2=25.0,
    housekeeping_w=1_500.0,
    qualified_life_s=90.0 * HOUR,
    run_duration_s=1.5 * HOUR,
    max_throughput_kg=4.5e6,
)

_NERVA_CORE = ReactorDesign(
    name="nerva_core",
    rated_thermal_w=1_140e6,
    fuel_mass_kg=900.0,
    struct_mass_kg=1_400.0,
    fuel_to_struct_ua_w_k=4.0e5,
    coolant_ua_w_k=7.0e5,
    coolant_inlet_k=30.0,
    initial_temperature_k=300.0,
    max_fuel_temp_k=2700.0,
    fuel_melt_temp_k=3100.0,
    max_fuel_dtdt_k_s=60.0,
    drum_worth_pcm=9_000.0,
    max_reactivity_rate_pcm_s=70.0,
    doppler_pcm_per_k_at_300k=-2.2,
    min_period_s=5.0,
    design_flux_n_cm2_s=8.0e13,
    xenon_equilibrium_pcm=-2_400.0,
    fissile_mass_kg=55.0,
    burnup_limit_fima=0.025,
    burnup_swing_pcm=-3_500.0,
    core_mass_kg=3_400.0,
    shield_mass_kg=4_000.0,
    initial_power_fraction=1.0e-3,
)

NERVA = NTPDesign(
    name="ntp_nerva",
    reactor=_NERVA_CORE,
    isp_ref_s=825.0,
    t_ref_k=2500.0,
    t_chamber_min_k=1700.0,
    t_chamber_max_k=2500.0,
    mdot_max_kg_s=32.0,
    dry_mass_kg=12_000.0,
    radiator_area_m2=40.0,
    housekeeping_w=3_000.0,
    qualified_life_s=100.0 * HOUR,
    run_duration_s=2.5 * HOUR,
    max_restarts=40,
    max_throughput_kg=1.15e7,
)


def _factory(design: NTPDesign):
    def make(**_kwargs: Any) -> NuclearThermalRocket:
        return NuclearThermalRocket(design)

    make.__name__ = design.name
    make.__qualname__ = design.name
    return make


PROPULSION.add(
    PEWEE.name,
    _factory(PEWEE),
    family=PropulsionFamily.NUCLEAR,
    power_w=PEWEE.reactor.rated_thermal_w,
    kind="ntp",
)
PROPULSION.add(
    NERVA.name,
    _factory(NERVA),
    family=PropulsionFamily.NUCLEAR,
    power_w=NERVA.reactor.rated_thermal_w,
    kind="ntp",
)

__all__ = [
    "NuclearThermalRocket",
    "NTPDesign",
    "ntp_isp",
    "PEWEE",
    "NERVA",
]
