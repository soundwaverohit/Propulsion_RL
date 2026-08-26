"""Cost models: turn a finished mission plus a bill of materials into dollars.

Three models are registered, and the study should report all of them:

``reference``
    Expendable hardware, single use, baseline prices. The straightforward
    answer, and the one to quote if you quote only one.
``reusable``
    Amortises capital over the number of further missions the hardware can
    still support, inferred from :class:`~propulsion_rl.core.types.HealthReport`.
    This is where propulsion *life* becomes economically load-bearing: an agent
    that preserves thruster life by running at lower discharge voltage buys more
    remaining uses, and so delivers a cheaper $/kg even if it is slower.
``conservative``
    Pessimistic prices, no reuse credit, full insurance on a hardened market.

Modelling choices that are judgement calls, stated once here
------------------------------------------------------------
* **Capex fields hold this flight's allocated share.** For the reusable model
  the ``CostBreakdown`` capex entries are the amortised per-flight share, so the
  contract identity ``cost_per_kg_delivered == breakdown.total / payload`` holds
  exactly. The full acquisition cost is preserved in ``breakdown.extras`` and in
  ``EconomicResult.notes``.
* **A failed mission is costed honestly.** Capital and launch are spent whatever
  happens; ``payload_delivered_kg`` is whatever the mission reports, which for a
  failure is zero. Zero delivered mass gives ``inf`` dollars per kilogram, never
  ``nan``, and the result is flagged in ``notes``.
* **Insurance is charged on the full value at risk every flight**, including for
  the reusable model. Reuse does not reduce what a launch failure destroys.
* **``time_value_cost`` is the carrying cost of capital during the transfer**,
  ``(capex + launch) * ((1 + r)**T - 1)``. It uses only numbers already in the
  price book. It deliberately does NOT use an assumed market value for delivered
  payload -- that assumption lives in ``npv`` alone, where it can be argued with
  separately.
* **Trip time reaches $/kg through two channels**: operations, which are charged
  per day, and the capital carrying cost above. Both are real; neither requires
  believing any particular number for what a kilogram on Mars is worth.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from ..core.constants import DAY, EPS, YEAR
from ..core.registry import COST_MODEL
from ..core.types import BillOfMaterials, HealthReport, PropulsionFamily
from ..missions.base import MissionResult
from .base import CostBreakdown, CostModel, EconomicResult
from .prices import BASELINE, CONSERVATIVE, PriceBook, get_price_book

LOGGER = logging.getLogger(__name__)

INF = float("inf")


@dataclass(frozen=True, slots=True)
class FigureOfMeritWeights:
    """Weights for the single ranking scalar. **This is a value judgement.**

    There is no physically correct way to trade a 5% higher success rate against
    a 30% cheaper kilogram against six months of extra transit. The figure of
    merit picks one trade and states it out loud so a reader can disagree with a
    specific number rather than with a vibe.

    The form is a weighted geometric mean of three dimensionless ratios::

        FoM = technical**w_tech
              * (reference_cost_per_kg / cost_per_kg)**w_cost
              * (reference_trip_time / trip_time)**w_time

    * ``technical`` is 1.0 for a successful mission, and ``MissionResult.progress``
      for a partial one when ``partial_credit`` is set (0.0 otherwise).
    * The two reference scales are normalisers only: they set FoM ~= 1.0 for a
      reference mission and have no effect on rank ordering within a fixed set of
      weights. The *exponents* are the value judgement.
    * Defaults put technical outcome first (w=1.0), cost second (w=0.5) and trip
      time third (w=0.25), matching the study's stated ranking rule of technical
      first and economics second. Trip time is weighted separately from cost
      because it already enters cost through operations and capital carrying
      cost; the extra 0.25 represents schedule value beyond dollars.
    * Geometric rather than arithmetic so that a zero in any factor zeroes the
      score: a mission that delivers nothing is not rescued by being cheap.
    """

    technical: float = 1.0
    cost: float = 0.5
    time: float = 0.25
    reference_cost_per_kg: float = 50_000.0
    reference_trip_time_s: float = YEAR
    partial_credit: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "w_technical": self.technical,
            "w_cost": self.cost,
            "w_time": self.time,
            "reference_cost_per_kg": self.reference_cost_per_kg,
            "reference_trip_time_s": self.reference_trip_time_s,
            "partial_credit": self.partial_credit,
        }


# --- small helpers -----------------------------------------------------------
def _thruster_kind(bom: BillOfMaterials) -> str:
    """Best guess at thruster technology, for pricing.

    Prefers an explicit ``bom.extras['thruster_kind']`` (as a float flag it is
    useless, so the convention is a key in ``system_name``); otherwise sniffs the
    system name. Falls back to Hall, which is the cheaper of the two, so an
    unrecognised electric system is not silently over-priced.
    """
    name = bom.system_name.lower()
    if any(k in name for k in ("grid", "ion", "nstar", "next", "ntp_ion")):
        return "gridded_ion"
    return "hall"


def _intended_payload_kg(
    bom: BillOfMaterials, result: MissionResult, override: float | None
) -> float:
    """Payload mass that was *launched*, which is not what was delivered.

    A failed mission delivers zero but still paid to lift the payload, so the
    launch bill must use the intended mass. Resolution order: explicit kwarg,
    ``bom.extras['payload_kg']``, then the delivered mass (correct for a success,
    and the only available value otherwise).
    """
    if override is not None:
        return max(float(override), 0.0)
    if "payload_kg" in bom.extras:
        return max(float(bom.extras["payload_kg"]), 0.0)
    return max(float(result.payload_delivered_kg), 0.0)


def _propellant_loaded_kg(
    bom: BillOfMaterials,
    result: MissionResult,
    prices: PriceBook,
    override: float | None,
) -> tuple[float, float]:
    """(loaded propellant, boil-off allowance) in kg.

    You buy and launch what you load, not what you burn. Tank capacity is the
    honest load figure when the BOM gives one; otherwise fall back to what was
    actually consumed. Cryogens additionally need a boil-off allowance sized by
    the mission duration -- another reason a fast trip is cheaper than a slow one
    on a hydrogen stage.
    """
    if override is not None:
        base = max(float(override), 0.0)
    elif bom.tank_capacity_kg > 0.0:
        base = float(bom.tank_capacity_kg)
    else:
        base = max(float(result.propellant_used_kg), 0.0)

    boiloff = 0.0
    if prices.is_cryogenic(bom.propellant_type) and base > 0.0:
        days = max(result.elapsed_s, 0.0) / DAY
        # Simple exponential loss over the trip, expressed as extra mass that
        # must be loaded at departure to still have `base` available.
        retained = math.exp(-prices.lh2_boiloff_frac_per_day * days)
        retained = max(retained, 1e-3)
        boiloff = base * (1.0 / retained - 1.0)
        if boiloff > base:
            # More than doubling the load to cover losses means the architecture
            # does not close on this trip time. The number is reported honestly
            # rather than clamped -- that is the point -- and surfaced as
            # ``notes['boiloff_exceeds_load']`` so a sweep can filter on it.
            # Logged at debug, not warning: inside a Monte Carlo this fires
            # thousands of times and the flag in the result is the useful signal.
            LOGGER.debug(
                "cryogenic boil-off allowance (%.0f kg) exceeds the tank load "
                "(%.0f kg) over %.0f days at %.3f%%/day",
                boiloff,
                base,
                days,
                100.0 * prices.lh2_boiloff_frac_per_day,
            )
    return base, boiloff


class _CostModelBase(CostModel):
    """Shared costing machinery. Subclasses set prices and the reuse policy.

    Accepted ``evaluate`` keyword arguments (all optional):

    ``payload_kg``
        Payload mass launched, overriding the BOM/result inference.
    ``destination``
        Launch drop-off: ``leo`` (default), ``gto`` or ``escape``.
    ``bus_mass_kg``
        Spacecraft bus mass outside the propulsion BOM. The BOM reports the
        propulsion system only, so avionics, structure and comms belong here.
    ``propellant_loaded_kg``
        Overrides the tank-capacity-derived propellant load.
    ``cumulative_units``
        Production index for the Wright's-law thruster cost.
    """

    name = "abstract"
    #: Whether this model gives credit for remaining hardware life.
    reuse_credit: bool = False

    def __init__(
        self,
        prices: PriceBook | str = BASELINE,
        *,
        weights: FigureOfMeritWeights | None = None,
        refurbishment_fraction: float = 0.0,
        max_uses: float = 20.0,
        default_destination: str = "leo",
        include_time_value: bool = True,
    ) -> None:
        self.prices = get_price_book(prices) if isinstance(prices, str) else prices
        self.weights = weights or FigureOfMeritWeights()
        # Refurbishment between flights, as a fraction of acquisition capex.
        # 0 for expendable models. 10% is the working assumption for a tug that
        # is inspected remotely and has no hardware touched between flights;
        # anything requiring physical access implies a return trip and is far
        # more expensive.
        self.refurbishment_fraction = float(refurbishment_fraction)
        # Cap on inferred reuses. Without it, a mission that produced no
        # measurable wear would amortise capital to zero, which is not a
        # finding, it is a divide-by-small-number.
        self.max_uses = float(max_uses)
        self.default_destination = default_destination
        self.include_time_value = bool(include_time_value)

    # --- capital -------------------------------------------------------------
    def _hardware_capex(
        self, bom: BillOfMaterials, cumulative_units: float | None
    ) -> dict[str, float]:
        """Acquisition cost of the propulsion stage, by contract category."""
        p = self.prices
        rated_kw = max(bom.rated_power_w, 0.0) / 1e3

        # Thrusters. Power is per-string, so divide the rated total by the unit
        # count before applying the per-kW term.
        #
        # A nuclear-thermal stage has no electric thruster: its nozzle, turbopump
        # and feed system are part of the engine and are priced inside the
        # reactor line below, so charging an electric-thruster price here would
        # double count. Detected by `rated_power_w == 0` on a NUCLEAR system,
        # which is what a thermal (as opposed to nuclear-electric) system reports.
        units = max(int(bom.thruster_units), 0)
        thermal_only = bom.family is PropulsionFamily.NUCLEAR and rated_kw <= 0.0
        if thermal_only:
            thruster = 0.0
        else:
            per_unit_kw = rated_kw / units if units else 0.0
            kind = _thruster_kind(bom)
            thruster = units * p.thruster_unit_cost(kind, per_unit_kw, cumulative_units)

        # Power source and conditioning.
        power_system = 0.0
        reactor = 0.0
        if bom.family is PropulsionFamily.NUCLEAR:
            mw_thermal = max(bom.reactor_thermal_w, 0.0) / 1e6
            if mw_thermal > 0.0:
                # The fixed term carries the non-scaling engine content: nozzle,
                # turbopump, control drums, instrumentation and qualification.
                reactor += p.reactor_fixed_usd
                reactor += p.reactor_usd_per_kw_thermal * mw_thermal * 1e3
                reactor += p.shield_kg_per_mw_thermal * mw_thermal * p.shield_usd_per_kg
            # Nuclear-electric only: converting reactor heat into bus power.
            # Zero for nuclear-thermal, where power_source_w is housekeeping.
            kwe = max(bom.power_source_w, 0.0) / 1e3
            power_system += p.power_conversion_usd_per_kw_electric * kwe
        else:
            # Solar array, priced at beginning-of-life watts.
            power_system += p.solar_array_usd_per_w * max(bom.power_source_w, 0.0)

        # PPU sized on thruster throughput power, for anything electric.
        if rated_kw > 0.0 and (
            bom.family is PropulsionFamily.ELECTRIC or bom.power_source_w > 0.0
        ):
            power_system += p.ppu_usd_per_w * max(bom.rated_power_w, 0.0)

        # Radiators.
        power_system += p.radiator_usd_per_m2() * max(bom.radiator_area_m2, 0.0)

        # Tankage, sized from capacity and propellant class.
        tank_mass = max(bom.tank_capacity_kg, 0.0) * p.tank_mass_fraction(
            bom.propellant_type
        )
        tankage = tank_mass * p.tank_usd_per_kg

        hardware = thruster + power_system + reactor + tankage
        integration = hardware * p.integration_markup

        return {
            "thruster_capex": thruster,
            "power_system_capex": power_system,
            "reactor_capex": reactor,
            "tankage_capex": tankage,
            "integration_capex": integration,
        }

    # --- reuse ---------------------------------------------------------------
    def _uses_remaining(self, bom: BillOfMaterials, health: HealthReport) -> float:
        """How many more missions like this one the hardware can support.

        Four independent estimators are formed where the data exists and the
        binding (smallest) one wins, because life is limited by whichever margin
        runs out first:

        1. Wear fraction: ``(1 - wear) / wear`` further missions of this size.
        2. Qualified life: ``remaining_life_s / burn_time_s``.
        3. BOM qualified life against accumulated burn time.
        4. Throughput: ``(max_throughput - throughput) / throughput``, the real
           life metric for a Hall thruster, whose channel walls erode in
           proportion to processed propellant.

        A failed unit has zero remaining uses regardless of what the counters
        say. A mission that produced no measurable wear is capped at
        ``max_uses`` rather than treated as immortal.
        """
        if health.failed:
            return 0.0

        estimates: list[float] = []
        wear = float(health.wear_fraction)
        if wear > EPS:
            estimates.append(max(1.0 - wear, 0.0) / wear)
        burn = float(health.burn_time_s)
        if burn > EPS and math.isfinite(float(health.remaining_life_s)):
            estimates.append(max(float(health.remaining_life_s), 0.0) / burn)
        if burn > EPS and bom.qualified_life_s > 0.0:
            estimates.append(max(bom.qualified_life_s - burn, 0.0) / burn)
        max_throughput = bom.extras.get("max_throughput_kg", 0.0)
        thr = float(health.throughput_kg)
        if max_throughput > 0.0 and thr > EPS:
            estimates.append(max(max_throughput - thr, 0.0) / thr)

        if not estimates:
            # Nothing measured. Treat as expendable rather than invent life.
            return 0.0
        return float(min(min(estimates), self.max_uses))

    # --- the main entry point ------------------------------------------------
    def evaluate(
        self,
        bom: BillOfMaterials,
        result: MissionResult,
        health: HealthReport,
        **kwargs: Any,
    ) -> EconomicResult:
        p = self.prices
        notes: dict[str, Any] = {
            "cost_model": self.name,
            "price_book": p.label,
            "currency_year": p.currency_year,
            "launch_price_is_speculative": p.launch_price_is_speculative,
        }

        destination = str(kwargs.get("destination", self.default_destination))
        payload_kg = _intended_payload_kg(bom, result, kwargs.get("payload_kg"))
        bus_mass_kg = max(float(kwargs.get("bus_mass_kg", 0.0)), 0.0)
        loaded_kg, boiloff_kg = _propellant_loaded_kg(
            bom, result, p, kwargs.get("propellant_loaded_kg")
        )
        cumulative_units = kwargs.get("cumulative_units")

        # --- capital ---------------------------------------------------------
        acquisition = self._hardware_capex(bom, cumulative_units)
        acquisition_total = sum(acquisition.values())

        uses_remaining = self._uses_remaining(bom, health) if self.reuse_credit else 0.0
        total_uses = 1.0 + uses_remaining
        divisor = total_uses if self.reuse_credit else 1.0
        allocated = {k: v / divisor for k, v in acquisition.items()}
        allocated_capex = acquisition_total / divisor

        # --- recurring -------------------------------------------------------
        propellant_kg_bought = loaded_kg + boiloff_kg
        propellant_cost = propellant_kg_bought * p.propellant_usd_per_kg(
            bom.propellant_type
        )

        launch_rate = p.launch_usd_per_kg(destination)
        hardware_launch_mass = max(bom.dry_mass_kg, 0.0) + bus_mass_kg
        consumable_launch_mass = propellant_kg_bought + payload_kg
        launch_hardware = hardware_launch_mass * launch_rate
        launch_consumables = consumable_launch_mass * launch_rate
        # Lifting the stage itself is a capital-like cost for reusable hardware
        # that stays in space; propellant and payload are lifted every flight.
        launch_cost = launch_hardware / divisor + launch_consumables

        days = max(result.elapsed_s, 0.0) / DAY
        ops_flight_team = days * p.ops_usd_per_day
        ops_tracking = days * p.dsn_usd_per_day()
        operations_cost = ops_flight_team + ops_tracking

        refurbishment_cost = (
            acquisition_total * self.refurbishment_fraction
            if (self.reuse_credit and uses_remaining > 0.0)
            else 0.0
        )

        # --- risk and finance -------------------------------------------------
        # Value at risk is the full replacement cost plus the ride, every flight.
        # Reuse does not reduce what a launch failure destroys.
        value_at_risk = acquisition_total + launch_hardware + launch_consumables
        insurance_cost = value_at_risk * p.insurance_rate_of_vehicle_value

        years = max(result.elapsed_s, 0.0) / YEAR
        r = p.discount_rate_annual
        carry_base = allocated_capex + launch_cost
        time_value_cost = (
            carry_base * ((1.0 + r) ** years - 1.0) if self.include_time_value else 0.0
        )

        breakdown = CostBreakdown(
            currency_year=p.currency_year,
            thruster_capex=allocated["thruster_capex"],
            power_system_capex=allocated["power_system_capex"],
            reactor_capex=allocated["reactor_capex"],
            tankage_capex=allocated["tankage_capex"],
            integration_capex=allocated["integration_capex"],
            propellant_cost=propellant_cost,
            launch_cost=launch_cost,
            operations_cost=operations_cost,
            refurbishment_cost=refurbishment_cost,
            insurance_cost=insurance_cost,
            time_value_cost=time_value_cost,
            extras={
                "acquisition_capex_usd": acquisition_total,
                "launch_hardware_usd": launch_hardware / divisor,
                "launch_consumables_usd": launch_consumables,
                "ops_flight_team_usd": ops_flight_team,
                "ops_tracking_usd": ops_tracking,
                "propellant_bought_kg": propellant_kg_bought,
                "propellant_boiloff_kg": boiloff_kg,
                "launch_mass_kg": hardware_launch_mass + consumable_launch_mass,
                "payload_launched_kg": payload_kg,
            },
        )

        total = breakdown.total
        if not math.isfinite(total):
            LOGGER.error(
                "non-finite total cost for %s / %s; check the bill of materials",
                self.name,
                bom.system_name,
            )
            total = INF

        # --- headline ratios --------------------------------------------------
        delivered = float(result.payload_delivered_kg)
        if delivered > EPS and math.isfinite(total):
            cost_per_kg = total / delivered
        else:
            cost_per_kg = INF
            notes["zero_payload_delivered"] = True
            notes["termination_reason"] = getattr(
                result.reason, "value", str(result.reason)
            )

        dv = float(result.delta_v_m_s)
        if delivered > EPS and dv > EPS and math.isfinite(total):
            cost_per_delta_v = total / (delivered * dv)
        else:
            cost_per_delta_v = INF

        # --- NPV --------------------------------------------------------------
        # Costs at t=0 for capital, launch and propellant; operations spread
        # uniformly over the transfer; revenue realised on arrival. Computed from
        # components rather than from `total` so the carrying-cost term is not
        # double counted against the discounting done here.
        annuity = _uniform_annuity_factor(years, r)
        pv_cost = (
            allocated_capex
            + launch_cost
            + propellant_cost
            + refurbishment_cost
            + insurance_cost
            + operations_cost * annuity
        )
        revenue = delivered * p.payload_value_usd_per_kg
        pv_revenue = revenue / (1.0 + r) ** years if years > 0.0 else revenue
        npv = pv_revenue - pv_cost

        # --- figure of merit --------------------------------------------------
        fom = self._figure_of_merit(result, cost_per_kg)

        notes.update(
            {
                "acquisition_capex_usd": acquisition_total,
                "capex_amortization_divisor": divisor,
                "destination": destination,
                "launch_usd_per_kg": launch_rate,
                "propellant_type": bom.propellant_type,
                "propellant_usd_per_kg": p.propellant_usd_per_kg(bom.propellant_type),
                "trip_time_days": days,
                "trip_time_years": years,
                "pv_revenue_usd": pv_revenue,
                "pv_cost_usd": pv_cost,
                "success": bool(result.success),
                "progress": float(result.progress),
                "hardware_failed": bool(health.failed),
                "wear_fraction": float(health.wear_fraction),
                # True when the cryogenic boil-off allowance exceeds the tank
                # load, i.e. the architecture does not close on this trip time.
                "boiloff_exceeds_load": boiloff_kg > loaded_kg > 0.0,
                # $/kg with the capital carrying cost stripped out, for readers
                # who want a pure transport price with no finance in it.
                "cost_per_kg_ex_time_value": (
                    (total - time_value_cost) / delivered if delivered > EPS else INF
                ),
            }
        )

        return EconomicResult(
            breakdown=breakdown,
            cost_per_kg_delivered=cost_per_kg,
            cost_per_delta_v=cost_per_delta_v,
            amortized_cost=total,
            uses_remaining=uses_remaining,
            npv=npv,
            figure_of_merit=fom,
            notes=notes,
        )

    def _figure_of_merit(self, result: MissionResult, cost_per_kg: float) -> float:
        """Single higher-is-better ranking scalar. See :class:`FigureOfMeritWeights`."""
        w = self.weights
        if result.success:
            technical = 1.0
        elif w.partial_credit:
            technical = max(min(float(result.progress), 1.0), 0.0)
        else:
            technical = 0.0
        if technical <= 0.0:
            return 0.0
        if not math.isfinite(cost_per_kg) or cost_per_kg <= 0.0:
            return 0.0

        cost_factor = w.reference_cost_per_kg / cost_per_kg
        trip = max(float(result.elapsed_s), 0.0)
        time_factor = (
            w.reference_trip_time_s / trip if trip > EPS else 1.0
        )
        try:
            fom = (
                technical**w.technical
                * cost_factor**w.cost
                * time_factor**w.time
            )
        except (OverflowError, ValueError):  # pragma: no cover - defensive
            return 0.0
        return float(fom) if math.isfinite(fom) else 0.0

    def assumptions(self) -> dict[str, Any]:
        d = self.prices.as_dict()
        d.update(
            cost_model=self.name,
            reuse_credit=self.reuse_credit,
            refurbishment_fraction=self.refurbishment_fraction,
            max_uses=self.max_uses,
            default_destination=self.default_destination,
            include_time_value=self.include_time_value,
        )
        d.update(self.weights.as_dict())
        return d


def _uniform_annuity_factor(years: float, rate: float) -> float:
    """PV of one dollar spent uniformly over ``years`` at continuous discounting.

    ``(1 - (1+r)**-T) / (T * ln(1+r))``, which tends to 1 as T -> 0. Used to
    discount operations, which are spent throughout the transfer rather than up
    front.
    """
    if years <= EPS or rate <= EPS:
        return 1.0
    k = math.log1p(rate)
    return (1.0 - (1.0 + rate) ** (-years)) / (years * k)


@COST_MODEL.register("reference", reuse=False, prices="baseline")
class ReferenceCostModel(_CostModelBase):
    """Expendable hardware, single use, baseline prices.

    The default answer. Every dollar of capital is charged to one flight, which
    is what actually happens to an interplanetary stage today: nothing that has
    ever gone to Mars has come back to be used again. If you quote one number,
    quote this one.
    """

    name = "reference"
    reuse_credit = False

    def __init__(self, prices: PriceBook | str = BASELINE, **kwargs: Any) -> None:
        kwargs.setdefault("refurbishment_fraction", 0.0)
        super().__init__(prices, **kwargs)


@COST_MODEL.register("reusable", reuse=True, prices="baseline")
class ReusableCostModel(_CostModelBase):
    """Amortises capital over the remaining life of the hardware.

    This is the model in which propulsion life is economically load-bearing, and
    therefore the one in which a control policy can pay for itself.

    The mechanism: a Hall thruster's channel erodes in proportion to processed
    propellant, and erosion rate rises steeply with discharge voltage. An agent
    that accepts a lower operating point trades Isp for wear. It arrives later
    and burns more propellant, both of which cost money -- but it leaves the
    stage with more qualified life, so a larger fraction of the capital is
    charged to future flights. Whether that trade wins is an empirical question
    this model exists to answer, and the answer depends entirely on the ratio of
    capital cost to propellant-plus-time cost, which is exactly why the price
    book is swept.

    ``uses_remaining`` is inferred from the health report (see
    :meth:`_CostModelBase._uses_remaining`); capital and the hardware's share of
    the launch bill are divided by ``1 + uses_remaining``. Refurbishment between
    flights is charged in full on every flight that leaves usable hardware.
    """

    name = "reusable"
    reuse_credit = True

    def __init__(self, prices: PriceBook | str = BASELINE, **kwargs: Any) -> None:
        # 10% of acquisition cost per flight. There is no flight experience with
        # refurbishing an in-space stage, so this is an assumption; it is
        # deliberately non-trivial so that reuse is not modelled as free.
        kwargs.setdefault("refurbishment_fraction", 0.10)
        super().__init__(prices, **kwargs)


@COST_MODEL.register("conservative", reuse=False, prices="conservative")
class ConservativeCostModel(_CostModelBase):
    """Pessimistic prices, no reuse credit, full insurance.

    Uses :data:`~propulsion_rl.economics.prices.CONSERVATIVE` by default: Falcon
    9 list prices, xenon at the top of its 2022 spike, first-article hardware
    with a 95% learning slope, a 12% cost of capital and an 18% insurance rate
    reflecting the hardened post-2023 space insurance market.

    A caveat this model cannot capture: a nuclear stage would in practice either
    be government-indemnified (so the insurance line is a transfer, not a cost)
    or would face a bespoke rate that no public market quotes. 18% of a
    half-billion-dollar stage is a placeholder for a number that does not exist.
    """

    name = "conservative"
    reuse_credit = False

    def __init__(self, prices: PriceBook | str = CONSERVATIVE, **kwargs: Any) -> None:
        kwargs.setdefault("refurbishment_fraction", 0.0)
        super().__init__(prices, **kwargs)


__all__ = [
    "FigureOfMeritWeights",
    "ReferenceCostModel",
    "ReusableCostModel",
    "ConservativeCostModel",
]
