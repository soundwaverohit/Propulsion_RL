"""The spacecraft electrical bus: generation, storage and distribution.

Solar-electric propulsion lives or dies on this module. A thruster's rated
power is a nameplate; what it actually gets is whatever the array delivers at
the current heliocentric distance, minus housekeeping, minus whatever the
battery is trying to claw back after the last eclipse.

The single most consequential number here is the inverse-square falloff:

    solar_flux(1.00 AU) / solar_flux(1.52 AU) = 1.52^2 = 2.31

An array sized for a comfortable 10 kW at Earth delivers 4.3 kW at Mars. That
one ratio is why nuclear electric propulsion keeps getting proposed for the
outer solar system, and the benchmark has to show it rather than assert it.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from ..core.constants import AU, HOUR, SOLAR_CONSTANT_1AU, YEAR, solar_flux
from ..core.types import VehicleState

logger = logging.getLogger(__name__)

__all__ = ["PowerSource", "SolarArray", "FixedPower", "RTG", "PowerBus"]


class PowerSource(ABC):
    """Anything that puts electrical power onto the bus."""

    #: Registry-friendly identifier, echoed into telemetry.
    name: str = "abstract"

    @abstractmethod
    def generated_w(self, state: VehicleState, eclipse: bool) -> float:
        """Electrical power produced right now, W."""

    @abstractmethod
    def bom_power_w(self) -> float:
        """Nameplate (beginning-of-life) rating, W. What the cost model prices."""

    # --- optional hooks ------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        """Restore to start-of-mission. ``rng`` seeds unit-to-unit variation."""

    def advance(self, dt_s: float, state: VehicleState, eclipse: bool) -> None:
        """Age the source by ``dt_s``. Called once per step by :class:`PowerBus`."""

    def info(self) -> dict[str, float]:
        """Extra telemetry, merged into the step record."""
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} bol={self.bom_power_w():.0f}W>"


class SolarArray(PowerSource):
    """Photovoltaic array: inverse-square falloff plus end-of-life degradation.

    Two degradation mechanisms, because they dominate in different regimes:

    * **Radiation.** Displacement damage from trapped protons and solar
      particle events, roughly exponential in exposure time. 2%/year is a
      reasonable triple-junction GaAs figure for an interplanetary cruise; a
      GEO transfer that grinds through the proton belts is far worse.
    * **Thermal cycling.** Every eclipse entry/exit cracks interconnects a
      little. Negligible interplanetary (a handful of cycles), dominant in LEO
      (~5500 cycles/year), which is exactly where solar-electric orbit raising
      wants to operate.

    The delivered power is deliberately *pure* inverse-square by default
    (``temperature_coefficient=0``) so the Earth/Mars ratio is exactly 1.52^2.
    Real cells run cooler and slightly more efficiently further out; set a
    positive coefficient to model that, at the cost of muddying the headline
    comparison.
    """

    name = "solar_array"

    def __init__(
        self,
        power_1au_w: float,
        degradation_per_year: float = 0.02,
        efficiency: float = 0.30,
        *,
        pointing_efficiency: float = 1.0,
        cycling_degradation_per_cycle: float = 2.0e-6,
        temperature_coefficient: float = 0.0,
        fixed_distance_m: float | None = None,
        bol_dispersion: float = 0.0,
        min_distance_m: float = 0.1 * AU,
    ) -> None:
        if power_1au_w < 0.0:
            raise ValueError("power_1au_w must be non-negative")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1]")
        if not 0.0 <= degradation_per_year < 1.0:
            raise ValueError("degradation_per_year must be in [0, 1)")

        self.power_1au_w = float(power_1au_w)
        self.degradation_per_year = float(degradation_per_year)
        self.efficiency = float(efficiency)
        self.pointing_efficiency = float(pointing_efficiency)
        self.cycling_degradation_per_cycle = float(cycling_degradation_per_cycle)
        self.temperature_coefficient = float(temperature_coefficient)
        self.fixed_distance_m = fixed_distance_m
        self.bol_dispersion = float(bol_dispersion)
        self.min_distance_m = float(min_distance_m)

        #: Physical cell area implied by the rating and the cell efficiency.
        self.area_m2 = self.power_1au_w / (self.efficiency * SOLAR_CONSTANT_1AU)

        self._quality = 1.0          # as-built dispersion, sampled at reset
        self._cycles = 0             # eclipse entries seen
        self._was_eclipsed = False
        self._age_s = 0.0

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        self._cycles = 0
        self._was_eclipsed = False
        self._age_s = 0.0
        if self.bol_dispersion > 0.0:
            self._quality = float(
                np.clip(1.0 + rng.normal(0.0, self.bol_dispersion), 0.5, 1.2)
            )
        else:
            self._quality = 1.0

    def advance(self, dt_s: float, state: VehicleState, eclipse: bool) -> None:
        self._age_s += dt_s
        if eclipse and not self._was_eclipsed:
            self._cycles += 1
        self._was_eclipsed = eclipse

    # --- physics -------------------------------------------------------------
    def distance_m(self, state: VehicleState, override: float | None = None) -> float:
        """Heliocentric distance to use for the flux, m.

        In a heliocentric frame ``state.radius_m`` is already right. In a
        planetocentric frame it is the distance to the *planet*, so either pass
        ``override`` (the environment has it in ``StepContext``) or construct
        the array with ``fixed_distance_m``.
        """
        if override is not None:
            d = float(override)
        elif self.fixed_distance_m is not None:
            d = float(self.fixed_distance_m)
        else:
            d = state.radius_m
        return max(d, self.min_distance_m)

    def degradation_factor(self, state: VehicleState) -> float:
        """Surviving fraction of beginning-of-life output, [0, 1]."""
        years = max(state.t_s, 0.0) / YEAR
        rad = (1.0 - self.degradation_per_year) ** years
        cyc = (1.0 - self.cycling_degradation_per_cycle) ** self._cycles
        f = rad * cyc * self._quality
        return f if f < 1.0 else 1.0

    def generated_w(
        self, state: VehicleState, eclipse: bool, distance_m: float | None = None
    ) -> float:
        if eclipse:
            return 0.0
        d = self.distance_m(state, distance_m)
        flux = solar_flux(d)
        eff = self.efficiency
        if self.temperature_coefficient != 0.0:
            # Cells run cooler further out, so efficiency creeps up.
            eff *= 1.0 + self.temperature_coefficient * (d / AU - 1.0)
            eff = max(eff, 0.0)
        p = (
            self.area_m2
            * eff
            * flux
            * self.pointing_efficiency
            * self.degradation_factor(state)
        )
        return p if p > 0.0 else 0.0

    def bom_power_w(self) -> float:
        return self.power_1au_w

    def info(self) -> dict[str, float]:
        return {
            "array_area_m2": self.area_m2,
            "array_cycles": float(self._cycles),
            "array_quality": self._quality,
        }


class FixedPower(PowerSource):
    """Constant electrical output, independent of distance and eclipse.

    For reactor-fed vehicles whose propulsion system is ``self_powered``: the
    reactor's own thermal/electrical model lives in the propulsion module, and
    the bus just sees a steady rail. Also useful as a control case -- run the
    same mission on ``SolarArray`` and on ``FixedPower`` of equal BOL rating and
    the difference is entirely the inverse-square penalty.
    """

    name = "fixed_power"

    def __init__(self, power_w: float, *, degradation_per_year: float = 0.0) -> None:
        if power_w < 0.0:
            raise ValueError("power_w must be non-negative")
        self.power_w = float(power_w)
        self.degradation_per_year = float(degradation_per_year)

    def generated_w(
        self, state: VehicleState, eclipse: bool, distance_m: float | None = None
    ) -> float:
        if self.degradation_per_year <= 0.0:
            return self.power_w
        years = max(state.t_s, 0.0) / YEAR
        return self.power_w * (1.0 - self.degradation_per_year) ** years

    def bom_power_w(self) -> float:
        return self.power_w


class RTG(FixedPower):
    """Radioisotope generator: fixed output decaying with the isotope half-life.

    Pu-238's 87.7-year half-life plus thermocouple degradation gives the ~0.8%
    per year that MMRTG missions budget for.
    """

    name = "rtg"

    def __init__(self, power_w: float, half_life_years: float = 87.7,
                 thermocouple_loss_per_year: float = 0.008) -> None:
        super().__init__(power_w)
        self.half_life_years = float(half_life_years)
        self.thermocouple_loss_per_year = float(thermocouple_loss_per_year)

    def generated_w(
        self, state: VehicleState, eclipse: bool, distance_m: float | None = None
    ) -> float:
        years = max(state.t_s, 0.0) / YEAR
        decay = 0.5 ** (years / self.half_life_years)
        tc = (1.0 - self.thermocouple_loss_per_year) ** years
        return self.power_w * decay * tc


class PowerBus:
    """Generation + storage + distribution, and the arbiter of what propulsion gets.

    Per-step contract::

        avail = bus.available_w(state, eclipse)      # offered to propulsion
        out   = thruster.step(cmd, ctx)              # draws <= avail
        bus.step(dt_s, out.power_draw_w, state, eclipse)

    ``available_w`` is a pure read so the environment can call it while
    assembling the :class:`~propulsion_rl.core.types.StepContext`; ``step`` is
    the only mutator.
    """

    def __init__(
        self,
        source: PowerSource,
        housekeeping_w: float = 0.0,
        battery_capacity_wh: float = 0.0,
        *,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        distribution_efficiency: float = 0.95,
        min_state_of_charge: float = 0.2,
        initial_state_of_charge: float = 1.0,
        max_discharge_c_rate: float = 1.0,
        max_charge_c_rate: float = 0.5,
        reference_dt_s: float = HOUR,
    ) -> None:
        if housekeeping_w < 0.0 or battery_capacity_wh < 0.0:
            raise ValueError("housekeeping and battery capacity must be non-negative")
        for nm, val in (
            ("charge_efficiency", charge_efficiency),
            ("discharge_efficiency", discharge_efficiency),
            ("distribution_efficiency", distribution_efficiency),
        ):
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{nm} must be in (0, 1]")

        self.source = source
        self.housekeeping_w = float(housekeeping_w)
        self.battery_capacity_wh = float(battery_capacity_wh)
        self.capacity_j = self.battery_capacity_wh * 3600.0
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.distribution_efficiency = float(distribution_efficiency)
        self.min_state_of_charge = float(min_state_of_charge)
        self.initial_state_of_charge = float(initial_state_of_charge)
        self.max_discharge_w = self.battery_capacity_wh * float(max_discharge_c_rate)
        self.max_charge_w = self.battery_capacity_wh * float(max_charge_c_rate)
        #: Step length assumed when converting stored energy into offered power.
        #: The environment should set this to ``mission.step_dt_s``.
        self.reference_dt_s = float(reference_dt_s)

        self._energy_j = self.capacity_j * self.initial_state_of_charge
        # Telemetry, all last-step values.
        self.last_generated_w = 0.0
        self.last_available_w = 0.0
        self.last_draw_w = 0.0
        self.last_battery_w = 0.0     # >0 discharging, <0 charging
        self.energy_deficit_j = 0.0   # unmet demand, accumulated over the episode
        self.time_in_deficit_s = 0.0
        self.shunted_j = 0.0          # surplus dumped with a full battery

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        self.source.reset(rng)
        self._energy_j = self.capacity_j * self.initial_state_of_charge
        self.last_generated_w = 0.0
        self.last_available_w = 0.0
        self.last_draw_w = 0.0
        self.last_battery_w = 0.0
        self.energy_deficit_j = 0.0
        self.time_in_deficit_s = 0.0
        self.shunted_j = 0.0

    # --- reads ---------------------------------------------------------------
    @property
    def state_of_charge(self) -> float:
        """Stored energy as a fraction of capacity, [0, 1]. Zero with no battery."""
        if self.capacity_j <= 0.0:
            return 0.0
        return self._energy_j / self.capacity_j

    @property
    def stored_wh(self) -> float:
        return self._energy_j / 3600.0

    def bom_power_w(self) -> float:
        return self.source.bom_power_w()

    def generated_w(
        self, state: VehicleState, eclipse: bool, distance_m: float | None = None
    ) -> float:
        """Raw source output before distribution losses, W."""
        return _source_power(self.source, state, eclipse, distance_m)

    def discharge_capability_w(self, dt_s: float | None = None) -> float:
        """Power the battery can sustain for ``dt_s`` without going below the floor."""
        if self.capacity_j <= 0.0:
            return 0.0
        usable_j = self._energy_j - self.min_state_of_charge * self.capacity_j
        if usable_j <= 0.0:
            return 0.0
        dt = self.reference_dt_s if dt_s is None else max(float(dt_s), 1.0e-9)
        return min(self.max_discharge_w, usable_j * self.discharge_efficiency / dt)

    def available_w(
        self,
        state: VehicleState,
        eclipse: bool,
        extra_load_w: float = 0.0,
        distance_m: float | None = None,
        dt_s: float | None = None,
    ) -> float:
        """Power offered to propulsion after housekeeping and other loads.

        Housekeeping has priority: in eclipse the battery covers it first, and
        only the remaining discharge capability is offered onward. Never
        negative -- a starved bus offers zero, it does not owe power.
        """
        gen = _source_power(self.source, state, eclipse, distance_m)
        self.last_generated_w = gen
        net = gen * self.distribution_efficiency - self.housekeeping_w - extra_load_w
        if net >= 0.0:
            avail = net
        else:
            avail = self.discharge_capability_w(dt_s) + net
            if avail < 0.0:
                avail = 0.0
        self.last_available_w = avail
        return avail

    # --- mutation ------------------------------------------------------------
    def step(
        self,
        dt_s: float,
        draw_w: float,
        state: VehicleState,
        eclipse: bool,
        distance_m: float | None = None,
    ) -> None:
        """Advance battery state of charge and array degradation."""
        dt = float(dt_s)
        if dt <= 0.0:
            return
        gen = _source_power(self.source, state, eclipse, distance_m)
        self.last_generated_w = gen
        self.last_draw_w = float(draw_w)

        net_w = gen * self.distribution_efficiency - self.housekeeping_w - float(draw_w)
        if net_w >= 0.0:
            # Charge with whatever is left over, respecting the charge rate limit.
            charge_w = min(net_w, self.max_charge_w) if self.capacity_j > 0.0 else 0.0
            room_j = self.capacity_j - self._energy_j
            added_j = min(charge_w * dt * self.charge_efficiency, max(room_j, 0.0))
            self._energy_j += added_j
            self.shunted_j += net_w * dt - added_j / max(self.charge_efficiency, 1e-12)
            self.last_battery_w = -added_j / dt
        else:
            demand_j = -net_w * dt
            drawn_j = demand_j / self.discharge_efficiency
            if drawn_j > self._energy_j:
                unmet_j = (drawn_j - self._energy_j) * self.discharge_efficiency
                self.energy_deficit_j += unmet_j
                self.time_in_deficit_s += dt
                drawn_j = self._energy_j
                logger.debug(
                    "power bus deficit: %.1f J unmet at t=%.0f s", unmet_j, state.t_s
                )
            self._energy_j -= drawn_j
            self.last_battery_w = drawn_j / dt

        if self._energy_j < 0.0:
            self._energy_j = 0.0
        elif self._energy_j > self.capacity_j:
            self._energy_j = self.capacity_j

        self.source.advance(dt, state, eclipse)

    def info(self) -> dict[str, float]:
        d = {
            "power_generated_w": self.last_generated_w,
            "power_available_w": self.last_available_w,
            "battery_soc": self.state_of_charge,
            "battery_w": self.last_battery_w,
            "energy_deficit_j": self.energy_deficit_j,
        }
        d.update(self.source.info())
        return d

    def __repr__(self) -> str:
        return (
            f"<PowerBus source={self.source.name!r} "
            f"housekeeping={self.housekeeping_w:.0f}W "
            f"battery={self.battery_capacity_wh:.0f}Wh soc={self.state_of_charge:.2f}>"
        )


def _source_power(
    source: PowerSource,
    state: VehicleState,
    eclipse: bool,
    distance_m: float | None,
) -> float:
    """Call ``generated_w`` with the optional distance override when supported."""
    if distance_m is None:
        p = source.generated_w(state, eclipse)
    else:
        try:
            p = source.generated_w(state, eclipse, distance_m)  # type: ignore[call-arg]
        except TypeError:
            p = source.generated_w(state, eclipse)
    return p if math.isfinite(p) and p > 0.0 else 0.0
