"""Generic electrostatic thruster: Hall-effect and gridded-ion engines.

Both families accelerate a xenon (or krypton) ion beam through a potential
and are retired by erosion. The differences that matter for the comparison --
the voltage/Isp envelope, whether a Child-Langmuir grid set binds, and which
surface the ions chew through -- live on :class:`ElectrostaticDesign`. This
module owns the shared step: map the canonical command onto (voltage, mass
flow), close the jet-power identity, respect the bus, and accumulate wear.

The power-first accounting is load-bearing for the conformance suite::

    P_bus = throttle * min(rated, available)
    T     = 2 eta P_bus / (g0 Isp)
    mdot  = T / (g0 Isp)
    jet   = eta P_bus

so ``T = mdot g0 Isp`` and ``jet <= P_bus`` hold by construction rather than
by a post-hoc clamp. A model that picked mdot first and then invented a
voltage would have to fudge one of those two identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS, G0, HOUR, M_XENON
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
from .common import (
    ThermalNode,
    WearAccumulator,
    isp_from_beam,
    mass_correction_beta,
    perveance_limited_current,
    sputter_yield_factor,
    thrust_from_power,
    total_efficiency,
)


def _sym(x: float) -> float:
    """Map a [0, 1] quantity to [-1, 1], saturating outside the unit interval."""
    if x <= 0.0:
        return -1.0
    if x >= 1.0:
        return 1.0
    return 2.0 * x - 1.0


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


@dataclass(frozen=True, slots=True)
class ElectrostaticDesign:
    """Static description of one Hall or gridded-ion string."""

    name: str
    kind: str  # "hall" or "ion"
    voltage_min_v: float
    voltage_max_v: float
    voltage_nominal_v: float
    mdot_anode_max_kg_s: float
    rated_power_w: float
    mass_utilisation: float  # beam-ion / anode flow at nominal
    eta_beam: float
    eta_ppu: float
    divergence_half_angle_rad: float
    doubles_ratio: float
    cathode_fraction: float  # cathode/anode at thermal_margin=0
    cathode_fraction_hi: float  # cathode/anode at thermal_margin=1
    # Optics -- used for gridded engines; Hall designs leave area at 0 to skip.
    accel_voltage_v: float = 180.0
    grid_gap_m: float = 6.0e-4
    screen_hole_diameter_m: float = 1.9e-3
    open_area_m2: float = 0.0
    perveance_fraction: float = 0.30
    # Thermal
    thermal_mass_kg: float = 8.0
    thermal_cp_j_kg_k: float = 700.0
    thermal_area_m2: float = 0.12
    max_temperature_k: float = 850.0
    # Life
    qualified_life_s: float = 9_000.0 * HOUR
    max_throughput_kg: float = 150.0
    erosion_depth_limit_m: float = 2.0e-3
    reference_erosion_rate_m_s: float = 1.0e-11
    sputter_threshold_ev: float = 40.0
    # BOM / vehicle
    dry_mass_kg: float = 40.0
    thruster_units: int = 1
    radiator_area_m2: float = 4.0
    tank_capacity_kg: float = 0.0
    housekeeping_w: float = 40.0
    ion_mass_kg: float = M_XENON
    max_restarts: int = 10_000
    propellant: str = "xenon"

    def __post_init__(self) -> None:
        if self.kind not in ("hall", "ion"):
            raise ValueError(f"kind must be 'hall' or 'ion', got {self.kind!r}")
        if self.voltage_max_v <= self.voltage_min_v:
            raise ValueError("voltage_max_v must exceed voltage_min_v")
        if self.rated_power_w <= 0.0:
            raise ValueError("rated_power_w must be positive")


class ElectrostaticThruster(PropulsionSystem):
    """One Hall or gridded-ion string, configured by an :class:`ElectrostaticDesign`.

    Subclasses only exist so the public names ``HallThruster`` and
    ``GriddedIonEngine`` show up in ``propulsion_rl.propulsion.__all__``.
    """

    family = PropulsionFamily.ELECTRIC
    self_powered = False

    _OBS_LABELS: tuple[str, ...] = (
        "throttle_achieved",
        "voltage_frac",
        "temperature_frac",
        "wear",
        "remaining_life",
        "efficiency",
        "power_frac",
        "isp_frac",
        "thermal_margin",
        "erosion",
        "cathode_frac",
        "throttled",
    )

    def __init__(self, design: ElectrostaticDesign) -> None:
        self.design = design
        self.name = design.name
        self.propellant = design.propellant
        self._thermal = ThermalNode(
            mass_kg=design.thermal_mass_kg,
            cp_j_kg_k=design.thermal_cp_j_kg_k,
            area_m2=design.thermal_area_m2,
            temperature_k=293.0,
        )
        self._wear = WearAccumulator(
            qualified_life_s=design.qualified_life_s,
            max_throughput_kg=design.max_throughput_kg,
            erosion_depth_limit_m=design.erosion_depth_limit_m,
        )
        self._failed = False
        self._was_firing = False
        self._eta_scale = 1.0
        self._last: dict[str, float] = {
            "throttle": 0.0,
            "voltage_v": design.voltage_nominal_v,
            "isp_s": 0.0,
            "power_w": 0.0,
            "efficiency": 0.0,
            "thermal_margin": 0.5,
            "cathode_frac": design.cathode_fraction,
            "throttled": 0.0,
        }
        self._events: list[Event] = []
        self._limits = self._compute_limits()
        self._bom = self._compute_bom()

    # --- envelope ------------------------------------------------------------
    def _isp_at(self, voltage_v: float, cathode_frac: float, eta_m_anode: float) -> float:
        mdot_a = 1.0
        mdot_c = cathode_frac * mdot_a
        eta_m = eta_m_anode * mdot_a / max(mdot_a + mdot_c, EPS)
        return isp_from_beam(
            voltage_v,
            self.design.ion_mass_kg,
            eta_m,
            self.design.divergence_half_angle_rad,
            self.design.doubles_ratio,
        )

    def _breakdown(self, eta_m: float, eta_beam: float):
        return total_efficiency(
            eta_m,
            eta_beam,
            self.design.divergence_half_angle_rad,
            self.design.doubles_ratio,
            self.design.eta_ppu,
        )

    def _compute_limits(self) -> Limits:
        d = self.design
        isp_lo = self._isp_at(d.voltage_min_v, d.cathode_fraction_hi, d.mass_utilisation)
        isp_hi = self._isp_at(d.voltage_max_v, d.cathode_fraction, d.mass_utilisation)
        if isp_hi < isp_lo:
            isp_lo, isp_hi = isp_hi, isp_lo
        # Pad a little so a worn, hot, high-cathode point cannot fall outside
        # the envelope the observation normalises against.
        isp_lo = max(isp_lo * 0.92, 200.0)
        isp_hi = isp_hi * 1.08
        br = self._breakdown(
            d.mass_utilisation / (1.0 + d.cathode_fraction), d.eta_beam
        )
        # Highest thrust is low-Isp, full power (and not more than the anode
        # feed can supply at that Isp).
        t_power = thrust_from_power(br.total, d.rated_power_w, isp_lo)
        mdot_max = d.mdot_anode_max_kg_s * (1.0 + d.cathode_fraction_hi)
        t_flow = mdot_max * G0 * isp_lo
        max_thrust = min(t_power, t_flow) if t_flow > 0.0 else t_power
        return Limits(
            max_thrust_n=max(max_thrust, 1e-6),
            min_thrust_n=0.0,
            max_power_w=d.rated_power_w,
            min_power_w=0.0,
            isp_range_s=(isp_lo, isp_hi),
            max_temperature_k=d.max_temperature_k,
            qualified_life_s=d.qualified_life_s,
            max_throughput_kg=d.max_throughput_kg,
            max_restarts=d.max_restarts,
            min_off_time_s=0.0,
        )

    def _compute_bom(self) -> BillOfMaterials:
        d = self.design
        return BillOfMaterials(
            system_name=d.name,
            family=PropulsionFamily.ELECTRIC,
            thruster_units=d.thruster_units,
            rated_power_w=d.rated_power_w,
            power_source_w=0.0,
            reactor_thermal_w=0.0,
            radiator_area_m2=d.radiator_area_m2,
            dry_mass_kg=d.dry_mass_kg,
            propellant_type=d.propellant,
            tank_capacity_kg=d.tank_capacity_kg,
            qualified_life_s=d.qualified_life_s,
        )

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        d = self.design
        self._eta_scale = float(min(max(rng.normal(1.0, 0.02), 0.94), 1.06))
        self._thermal.temperature_k = 293.0
        self._wear.reset()
        self._failed = False
        self._was_firing = False
        self._events = []
        self._last = {
            "throttle": 0.0,
            "voltage_v": d.voltage_nominal_v,
            "isp_s": 0.0,
            "power_w": 0.0,
            "efficiency": 0.0,
            "thermal_margin": 0.5,
            "cathode_frac": d.cathode_fraction,
            "throttled": 0.0,
        }

    def step(self, command: CanonicalCommand, ctx: StepContext) -> ThrusterOutput:
        self._events = []
        d = self.design
        dt = float(max(ctx.dt_s, 0.0))
        available = max(float(ctx.available_power_w), 0.0)
        tm = float(min(max(command.thermal_margin, 0.0), 1.0))
        throttle = float(min(max(command.throttle, 0.0), 1.0))
        voltage = _lerp(d.voltage_min_v, d.voltage_max_v, command.operating_point)
        cathode_frac = _lerp(d.cathode_fraction, d.cathode_fraction_hi, tm)

        if self._failed or throttle <= 0.0 or available <= 0.0:
            reason = "failed" if self._failed else "none"
            return self._idle(ctx, tm, voltage, cathode_frac, reason)

        wear = self._wear.fraction
        degrade = max(0.0, 1.0 - 0.25 * wear) * self._eta_scale
        t_frac = (self._thermal.temperature_k - 293.0) / max(
            d.max_temperature_k - 293.0, 1.0
        )
        thermal_derate = max(0.35, 1.0 - 0.45 * max(t_frac - 0.85, 0.0) / 0.15)
        eta_beam = d.eta_beam * degrade * thermal_derate
        eta_m_anode = d.mass_utilisation * degrade
        eta_m = eta_m_anode / (1.0 + cathode_frac)
        br = self._breakdown(eta_m, eta_beam)
        eta = br.total
        isp = self._isp_at(voltage, cathode_frac, eta_m_anode)
        isp_lo, isp_hi = self._limits.isp_range_s
        isp = min(max(isp, isp_lo), isp_hi)

        p_bus = throttle * min(d.rated_power_w, available)
        throttled_by = "none"
        if throttle * d.rated_power_w > available + 1e-9:
            throttled_by = "power"

        # Perveance (gridded only): a low-voltage / high-flow command cannot
        # extract more beam current than the optics will pass.
        if d.kind == "ion" and d.open_area_m2 > 0.0 and eta > EPS and isp > EPS:
            gap = math.hypot(d.grid_gap_m, 0.5 * d.screen_hole_diameter_m)
            i_max = perveance_limited_current(
                voltage + d.accel_voltage_v,
                gap,
                d.ion_mass_kg,
                d.open_area_m2,
                d.perveance_fraction,
            )
            beta = mass_correction_beta(d.doubles_ratio)
            mdot_beam_max = i_max * (d.ion_mass_kg / ELEMENTARY_CHARGE) * beta
            mdot_total_max = (
                mdot_beam_max * (1.0 + cathode_frac) / max(eta_m_anode, EPS)
            )
            p_cap = 0.5 * mdot_total_max * (isp * G0) ** 2 / max(eta, EPS)
            if p_bus > p_cap:
                p_bus = max(p_cap, 0.0)
                throttled_by = "perveance"

        if self._thermal.temperature_k > 0.98 * d.max_temperature_k:
            hot = (self._thermal.temperature_k / d.max_temperature_k - 0.98) / 0.08
            p_bus *= max(0.0, 1.0 - min(hot, 1.0))
            throttled_by = "thermal"

        if eta <= EPS or isp <= EPS or p_bus <= 0.0:
            return self._idle(ctx, tm, voltage, cathode_frac, throttled_by)

        thrust = thrust_from_power(eta, p_bus, isp)
        mdot = thrust / (G0 * isp) if isp > EPS else 0.0
        jet = 0.5 * mdot * (isp * G0) ** 2

        # Thermal node: waste heat into the body, extra cooling from the
        # thermal-margin actuator (cathode flow / dedicated radiator loop).
        q_in = max(p_bus - jet, 0.0)
        q_cool = tm * 0.35 * d.rated_power_w
        self._thermal.step(dt, q_in, q_cool, ctx.sink_temperature_k)

        # Wear: Hall channel walls see discharge-ion energy ~ e V_d; ion
        # accelerator grids see charge-exchange ions at ~e V_accel.
        energy_ev = voltage if d.kind == "hall" else d.accel_voltage_v
        y = sputter_yield_factor(energy_ev, d.sputter_threshold_ev)
        y_ref = sputter_yield_factor(
            d.voltage_nominal_v if d.kind == "hall" else d.accel_voltage_v,
            d.sputter_threshold_ev,
        )
        flow_frac = mdot / max(d.mdot_anode_max_kg_s * (1.0 + d.cathode_fraction), EPS)
        ero = d.reference_erosion_rate_m_s * max(flow_frac, 0.0) * (
            y / y_ref if y_ref > EPS else 0.0
        )
        self._wear.accumulate(dt, mdot, ero)
        if not self._was_firing:
            self._wear.restarts += 1
        self._was_firing = True

        if self._wear.fraction >= 1.0 and not self._failed:
            self._failed = True
            self._events.append(
                Event(
                    "end_of_life",
                    Severity.FATAL,
                    f"{d.name} reached end of life",
                    1.0,
                )
            )

        if self._thermal.temperature_k > d.max_temperature_k:
            self._events.append(
                Event(
                    "overtemp",
                    Severity.CRITICAL,
                    f"{d.name} body {self._thermal.temperature_k:.0f} K",
                    self._thermal.temperature_k,
                )
            )

        self._last = {
            "throttle": 0.0 if d.rated_power_w <= EPS else p_bus / d.rated_power_w,
            "voltage_v": voltage,
            "isp_s": isp,
            "power_w": p_bus,
            "efficiency": eta,
            "thermal_margin": tm,
            "cathode_frac": cathode_frac,
            "throttled": 0.0 if throttled_by == "none" else 1.0,
        }
        return ThrusterOutput(
            thrust_n=thrust,
            mdot_kg_s=mdot,
            isp_s=isp,
            power_draw_w=p_bus,
            thermal_power_w=0.0,
            heat_reject_w=q_in,
            efficiency=eta,
            throttled_by=throttled_by,
            events=list(self._events),
        )

    def _idle(
        self,
        ctx: StepContext,
        tm: float,
        voltage: float,
        cathode_frac: float,
        throttled_by: str,
    ) -> ThrusterOutput:
        d = self.design
        self._thermal.step(
            max(ctx.dt_s, 0.0),
            0.0,
            tm * 0.1 * d.rated_power_w,
            ctx.sink_temperature_k,
        )
        self._was_firing = False
        self._last = {
            "throttle": 0.0,
            "voltage_v": voltage,
            "isp_s": 0.0,
            "power_w": 0.0,
            "efficiency": 0.0,
            "thermal_margin": tm,
            "cathode_frac": cathode_frac,
            "throttled": 0.0 if throttled_by == "none" else 1.0,
        }
        return ThrusterOutput(throttled_by=throttled_by, events=list(self._events))

    # --- observation ---------------------------------------------------------
    def observe_raw(self, ctx: StepContext) -> np.ndarray:
        d = self.design
        life = self._wear.remaining_life_s(1e-6)
        if not math.isfinite(life):
            life_frac = 1.0 - self._wear.fraction
        else:
            life_frac = min(life / max(d.qualified_life_s, EPS), 1.0)
        isp_lo, isp_hi = self._limits.isp_range_s
        isp_frac = 0.5
        if self._last["isp_s"] > 0.0 and isp_hi > isp_lo:
            isp_frac = (self._last["isp_s"] - isp_lo) / (isp_hi - isp_lo)
        v_span = d.voltage_max_v - d.voltage_min_v
        return np.array(
            [
                _sym(self._last["throttle"]),
                _sym((self._last["voltage_v"] - d.voltage_min_v) / max(v_span, EPS)),
                _sym(self._thermal.temperature_k / d.max_temperature_k),
                _sym(self._wear.fraction),
                _sym(max(life_frac, 0.0)),
                _sym(self._last["efficiency"]),
                _sym(self._last["power_w"] / max(d.rated_power_w, EPS)),
                _sym(isp_frac),
                _sym(self._last["thermal_margin"]),
                _sym(self._wear.erosion_fraction),
                _sym(self._last["cathode_frac"] / max(d.cathode_fraction_hi, EPS)),
                _sym(self._last["throttled"]),
            ],
            dtype=np.float32,
        )

    def observation_labels(self) -> tuple[str, ...]:
        return self._OBS_LABELS

    # --- introspection -------------------------------------------------------
    def limits(self) -> Limits:
        return self._limits

    def constraints(self) -> ConstraintReport:
        d = self.design
        t_max = d.max_temperature_k
        names = ("temperature", "wear", "restarts")
        margins = np.array(
            [
                max((t_max - self._thermal.temperature_k) / t_max, -1.0),
                1.0 - self._wear.fraction,
                1.0 - self._wear.restarts / max(d.max_restarts, 1),
            ],
            dtype=np.float64,
        )
        return ConstraintReport(names=names, margins=margins)

    def health(self) -> HealthReport:
        d = self.design
        mdot_ref = d.mdot_anode_max_kg_s * (1.0 + d.cathode_fraction)
        remaining = self._wear.remaining_life_s(mdot_ref)
        if not math.isfinite(remaining):
            remaining = max(d.qualified_life_s - self._wear.burn_time_s, 0.0)
        return HealthReport(
            wear_fraction=self._wear.fraction,
            remaining_life_s=max(remaining, 0.0),
            throughput_kg=self._wear.throughput_kg,
            burn_time_s=self._wear.burn_time_s,
            restarts=self._wear.restarts,
            degraded_efficiency=max(0.0, 1.0 - 0.25 * self._wear.fraction),
            failed=self._failed,
        )

    def bom(self) -> BillOfMaterials:
        return replace(self._bom)

    def decode_action(self, command: CanonicalCommand) -> dict[str, float]:
        d = self.design
        voltage = _lerp(d.voltage_min_v, d.voltage_max_v, command.operating_point)
        cathode = _lerp(d.cathode_fraction, d.cathode_fraction_hi, command.thermal_margin)
        return {
            "throttle": command.throttle,
            "operating_point": command.operating_point,
            "thermal_margin": command.thermal_margin,
            "discharge_voltage_v": voltage,
            "cathode_fraction": cathode,
        }

    def housekeeping_power_w(self) -> float:
        return float(self.design.housekeeping_w)

    def info(self) -> dict[str, Any]:
        return {
            "voltage_v": self._last["voltage_v"],
            "body_temperature_k": self._thermal.temperature_k,
            "wear_mechanism": self._wear.limiting_mechanism,
            "restarts": float(self._wear.restarts),
        }
