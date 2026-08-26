"""The audited price book: every dollar figure in the study, in one place.

Design rules for this module
----------------------------
1. **Every number is cited and dated.** An uncited price is worthless -- it is
   the first thing a reviewer will attack, and rightly so. Each field carries a
   comment naming the source and the year the figure applies to.
2. **Constant 2026 USD.** Where a source is older, the escalation applied is
   stated. No implicit inflation adjustment anywhere else in the package.
3. **Flat scalars only.** :class:`PriceBook` is a frozen dataclass of plain
   floats so that :func:`dataclasses.replace` makes a one-at-a-time sensitivity
   sweep a one-liner. Nested dicts would break that, so lookups by species or
   destination go through methods instead.
4. **Honesty about ignorance.** Some numbers are quoted commercial prices
   (Falcon Heavy, liquid hydrogen). Some are order-of-magnitude estimates from
   programme totals (thruster recurring cost). One -- the cost of a flight
   nuclear reactor -- is genuinely unknowable today. Those are marked, given an
   explicit wide range in :data:`UNCERTAIN_PARAMETERS`, and swept rather than
   asserted.

The three reference instances :data:`BASELINE`, :data:`OPTIMISTIC` and
:data:`CONSERVATIVE` bracket the answer. The study should report all three
rather than pretend one set of assumptions is true.

Currency-year note
------------------
US GDP deflator / BLS CPI escalation from 2020 to 2026 is roughly 1.25x, and
from 2015 to 2026 roughly 1.35x. Where a cited figure predates 2022 it has been
escalated by those factors and the raw figure is given in the comment so the
arithmetic is checkable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields, replace
from typing import Any

LOGGER = logging.getLogger(__name__)

CURRENCY_YEAR = 2026

# --- Documented anchors that are not themselves swept parameters -------------
# Kept as module constants so the derivations in the field comments below can be
# re-checked without leaving the file.

#: SpaceX published list prices and expendable-mode performance, as advertised
#: on the SpaceX capabilities & services page over 2022-2024. Falcon 9 list
#: price rose from $62M (2020) -> $67M (2022) -> $69.75M (2024).
FALCON_9_PRICE_USD = 69.75e6
FALCON_9_LEO_KG = 22_800.0
FALCON_9_GTO_KG = 8_300.0
FALCON_9_MARS_KG = 4_020.0

#: Falcon Heavy list price $97M (2022 advertised, recovered-booster mode);
#: fully expendable missions have been contracted at ~$150M.
FALCON_HEAVY_PRICE_USD = 97e6
FALCON_HEAVY_EXPENDABLE_PRICE_USD = 150e6
FALCON_HEAVY_LEO_KG = 63_800.0
FALCON_HEAVY_GTO_KG = 26_700.0
FALCON_HEAVY_MARS_KG = 16_800.0

#: NASA OIG report IG-22-003 (Nov 2021): ~$4.1B per Artemis launch for
#: SLS + Orion + ground systems; ~$2.2B of that is the SLS vehicle itself.
#: SLS Block 1 lifts ~95 t to LEO. Included as the documented upper bound on
#: what a government-launched nuclear stage could plausibly be charged.
SLS_LAUNCH_USD = 2.2e9
SLS_LEO_KG = 95_000.0

#: DARPA DRACO: Lockheed Martin selected as prime July 2023, ~$499M through the
#: 2027 flight demonstration, with BWXT building the reactor. The DRACO engine
#: is ~25,000 lbf at ~900 s Isp, i.e. roughly 500 MW thermal. This is the single
#: best public anchor for "what does one flight NTP reactor cost", and it is a
#: development contract, not a recurring unit price.
DRACO_PROGRAMME_USD = 499e6
DRACO_REACTOR_MW_THERMAL = 500.0

#: NASA/Aerojet Rocketdyne AEPS (12.5 kW Hall thruster string for the Gateway
#: PPE): ~$67M development award (2016), options exercised in 2019 to a total
#: contract value of ~$297M covering flight units. Development-loaded; the
#: recurring cost of the n-th unit is far below this. Used only to bound the
#: first-article end of the thruster cost range.
AEPS_CONTRACT_USD = 297e6
AEPS_STRING_KW = 12.5

#: Mars Science Laboratory: ~$2.5B (2012) for an 899 kg rover, i.e. ~$2.8M/kg
#: landed. Mars 2020/Perseverance: ~$2.7B for 1,025 kg. These bound the value
#: of delivered mass at the expensive, science-payload end.
MSL_USD_PER_KG_LANDED = 2.8e6


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Every price the cost models are allowed to use, in constant 2026 USD.

    Frozen and flat by design: a sensitivity sweep is
    ``dataclasses.replace(book, xenon_usd_per_kg=5000.0)``, and
    :func:`propulsion_rl.economics.sensitivity.tornado` walks
    :data:`UNCERTAIN_PARAMETERS` mechanically. Adding a field means adding a
    citation comment; there is no other way in.
    """

    label: str = "baseline"
    currency_year: int = CURRENCY_YEAR

    # --- LAUNCH --------------------------------------------------------------
    # Marginal price of a ride, not the cost of building the rocket. All three
    # destinations are quoted because a solar-electric tug is dropped in LEO and
    # spirals out on its own, while a nuclear stage's launch bill is set by a
    # much heavier stack that may want a higher drop-off orbit.
    #
    # BASELINE derivation: Falcon Heavy recovered, $97M / 63,800 kg = $1,520/kg
    # to LEO; rounded up to $2,000/kg to allow for the fact that a real rideshare
    # or dedicated contract rarely fills the vehicle to its expendable maximum
    # and that FH recovered-mode LEO capacity is below the 63.8 t expendable
    # figure. Falcon 9 for comparison is $69.75M / 22,800 kg = $3,060/kg (the
    # widely quoted "$2,700/kg" uses the older $62M price).
    launch_leo_usd_per_kg: float = 2_000.0
    # $97M / 26,700 kg = $3,633/kg (FH, GTO, expendable-max). Rounded to
    # $5,500/kg for the same fill-factor reason.
    launch_gto_usd_per_kg: float = 5_500.0
    # C3 = 0 / Earth escape. FH "to Mars" is 16,800 kg for $97M = $5,770/kg;
    # note SpaceX's Mars number is a C3 ~ 8-16 km^2/s^2 injection, so true
    # C3 = 0 capacity is slightly higher and the $/kg slightly lower. Falcon 9
    # to Mars is $69.75M / 4,020 kg = $17,350/kg -- the spread between vehicles
    # is larger at escape than at LEO, which is why this is a separate field.
    launch_escape_usd_per_kg: float = 8_000.0
    #: Which vehicle class the three numbers above were derived from.
    launch_vehicle: str = "falcon_heavy_recovered"
    #: True when the launch prices are a projection for a vehicle that has not
    #: flown a commercial payload at that price. Starship's $200-1,000/kg is a
    #: projection; treat any conclusion that depends on it as provisional.
    launch_price_is_speculative: bool = False

    # --- PROPELLANT ----------------------------------------------------------
    # Xenon. The volatile one, and the one most likely to move the answer for a
    # solar-electric mission.
    #   * Global production is only ~50-70 t/yr. Xenon is present in air at
    #     0.087 ppm and is recovered as a by-product of cryogenic air separation
    #     for steel and industrial-gas plants, so supply is inelastic on any
    #     timescale an EP programme cares about.
    #   * A single large EP mission loading 5-10 t of xenon is therefore a
    #     visible fraction of a year's world production. This is a real
    #     constraint, not a footnote: EP demand competes with lighting, medical
    #     imaging and semiconductor etch for the same few tens of tonnes.
    #   * Pre-2021 contract prices were ~$1,200/kg. The 2022 Russia/Ukraine
    #     supply disruption (Ukraine and Russia supplied a large share of the
    #     world's neon and a significant share of krypton/xenon capacity) drove
    #     spot prices to several thousand $/kg before partially retreating.
    # BASELINE $3,000/kg reflects a post-2022 contract price with no assumption
    # that the spike fully unwinds. Sources: industrial-gas market reporting
    # 2022-2024 and the electric-propulsion literature's standing estimate of
    # world xenon output.
    xenon_usd_per_kg: float = 3_000.0
    # Krypton. 1.14 ppm in air, so roughly an order of magnitude more abundant
    # than xenon and correspondingly cheaper; production is in the hundreds of
    # t/yr. Pre-2021 ~$300/kg, 2022 spike to well over $1,000/kg. Krypton buys
    # ~10% lower thrust efficiency at a given power for a large cost saving,
    # which is why Starlink v1.0 flew krypton Hall thrusters.
    krypton_usd_per_kg: float = 600.0
    # Argon. 0.93% of the atmosphere -- effectively unlimited, and the commodity
    # price is $1-2/kg in bulk liquid. The $10/kg here is the *delivered,
    # spaceflight-purity, loaded* price: ultra-high-purity gas in flight-rated
    # bottles with certification, not bulk liquid off a tanker. SpaceX moved
    # Starlink v2 to argon Hall thrusters (publicly confirmed 2023), with cost
    # and supply security as stated drivers.
    argon_usd_per_kg: float = 10.0
    # Iodine. Stored as a solid, so tankage is far lighter, but it is corrosive
    # and flight heritage is thin (ThrustMe NPT30-I2 flew 2020). Price is a
    # rough estimate from high-purity iodine commodity pricing plus handling.
    iodine_usd_per_kg: float = 100.0
    # Liquid hydrogen commodity price. DOE Hydrogen Program cost records put
    # delivered liquid hydrogen at roughly $6-8/kg in the early 2020s; NASA's
    # historical bulk purchases are consistent with the low end of that.
    # Liquefaction alone costs ~10-13 kWh/kg. $10/kg for 2026 delivered.
    lh2_usd_per_kg: float = 10.0
    # The commodity price is NOT the cost of using LH2. Ground storage, tanker
    # delivery, transfer losses, densification, pad boiloff during a long launch
    # campaign, MLI, and (for a stage that must loiter) active zero-boil-off
    # cryocoolers dominate. This multiplier is applied to the commodity price.
    # 6x is an engineering estimate, not a quoted figure -- it is the weakest
    # number in the propellant block and is swept.
    cryo_handling_multiplier: float = 6.0
    # In-space boiloff for a passively insulated LH2 tank: 0.1-1 %/day is the
    # usual quoted band (NASA CPST / eCryo cryogenic propellant storage work).
    # Extra propellant must be loaded to cover it, which is a real reason fast
    # trips favour hydrogen stages. 0.15 %/day assumes good MLI and a vapour
    # cooled shield but no active cooling.
    lh2_boiloff_frac_per_day: float = 0.0015

    # --- POWER ---------------------------------------------------------------
    # Space-qualified solar array, $/W at beginning of life, 1 AU, array level
    # (cells + substrate + deployment mechanism + harness), not cell level.
    # Historical rigid-panel arrays are widely quoted at $300-1,000/W.
    #
    # Best public checkable anchor: NASA's 2021 award to Boeing of ~$103M for six
    # iROSA roll-out arrays for the ISS, each ~20 kW, i.e. ~120 kW for $103M or
    # ~$860/W -- and that is for ROSA-class hardware, in a six-unit buy, from an
    # established supplier. It sits at the TOP of the historical band, not the
    # bottom, which is a useful corrective to the assumption that roll-out
    # arrays are automatically cheap.
    #
    # $250/W is a 2026 baseline for a large SEP array ordered in quantity: below
    # iROSA on the argument that a 75 kW single array has better economies than
    # six 20 kW ISS-qualified units, and far above Starlink-scale production.
    # NOTE: the tornado shows this is usually the single largest driver of the
    # solar-electric answer, ahead of both launch price and xenon. It deserves
    # more scrutiny than it normally gets.
    solar_array_usd_per_w: float = 250.0
    # Power processing unit, $/W of throughput. The PPU is historically a large
    # fraction of EP system recurring cost -- often comparable to the thruster
    # itself -- because it is high-voltage, radiation-tolerant, and built in
    # small numbers. Direct-drive (array voltage matched to discharge voltage)
    # can cut this substantially but has limited flight heritage.
    ppu_usd_per_w: float = 100.0
    # Deployable space radiator. There is no published $/m^2; this is built up
    # from areal density x a structure-level $/kg. 5 kg/m^2 is typical for a
    # deployable single-sided panel with heat pipes.
    radiator_areal_density_kg_m2: float = 5.0
    # $/kg for radiator structure. Spacecraft *structure* runs far cheaper than
    # whole-spacecraft $/kg (which is $100k-500k/kg for a science mission);
    # $5,000/kg is a structure-and-thermal-hardware estimate. Estimate, not a
    # quote.
    radiator_usd_per_kg: float = 5_000.0

    # --- THRUSTERS -----------------------------------------------------------
    # Recurring cost of one flight thruster, modelled as a fixed per-unit charge
    # plus a $/kW term. The fixed term is the qualification, acceptance test and
    # cathode/feed-system content that does not scale with power.
    #
    # Anchors: the AEPS 12.5 kW string contract totals ~$297M including
    # development, which if naively divided over ~6 flight units is ~$50M/unit
    # or $4,000/W -- an upper bound dominated by non-recurring cost. At the other
    # end, Starlink builds Hall thrusters in the thousands per year and the
    # implied recurring cost is plausibly in the tens of thousands of dollars per
    # unit, i.e. a few $/W. The baseline sits deliberately near the low-rate
    # government end because that is what a first cargo tug would actually pay.
    hall_thruster_fixed_usd: float = 250_000.0
    hall_thruster_usd_per_kw: float = 150_000.0
    # Gridded ion is more expensive per kW than Hall: more parts, tighter
    # tolerances, and grid life qualification is long and costly. NEXT-C flight
    # units (7 kW) came in around $10M each including PPU in the 2010s.
    gridded_ion_fixed_usd: float = 500_000.0
    gridded_ion_usd_per_kw: float = 250_000.0
    # Wright's-law slope for thruster recurring cost: the cost of the N-th unit
    # is C1 * N^(log2(slope)). 85% is the classic aerospace airframe slope
    # (Wright 1936); low-rate space hardware is usually assumed at 90-95%.
    # This is why "recurring cost falls steeply with production rate" is a real
    # effect and not hand-waving: at a 90% slope the 100th unit costs 47% of the
    # first, and the 1000th costs 35%.
    thruster_learning_slope: float = 0.90
    #: Cumulative production index of the units being priced. 1 = first article.
    thruster_production_units: float = 1.0

    # --- NUCLEAR -------------------------------------------------------------
    # THIS IS THE HEADLINE SWEPT PARAMETER AND THE LEAST KNOWABLE NUMBER IN THE
    # STUDY. No flight nuclear propulsion reactor has ever been sold, so there
    # is no price. What exists:
    #   * DRACO: ~$499M for one reactor plus a flight demonstration (~500 MWt
    #     class). Charging the whole programme to one reactor gives ~$1,000/kWt;
    #     that number is almost entirely non-recurring.
    #   * NASA/DOE NTP programme estimates through the 2010s put a ground-test
    #     and flight-demonstration campaign in the low billions.
    #   * Kilopower/KRUSTY (1 kWe, demonstrated 2018) and the 40 kWe Fission
    #     Surface Power programme (three ~$5M Phase 1 design contracts, 2022)
    #     are the only other public data, and neither has a unit price.
    # A defensible recurring unit cost for a 500 MWt NTP core after development
    # is somewhere in $100M-$1B. Expressed as $/kWt over a 500 MWt core that is
    # $200-2,000/kWt. BASELINE takes $700/kWt (i.e. ~$350M for a 500 MWt core)
    # plus a $100M fixed charge for the non-scaling content. Anyone who tells
    # you they know this number more precisely is guessing with more confidence.
    reactor_usd_per_kw_thermal: float = 700.0
    reactor_fixed_usd: float = 100e6
    # Radiation shield. Mass scales with thermal power and with how close the
    # crew/avionics sit; a shadow shield for an uncrewed NTP stage is a few
    # tonnes. 4 kg/MWt is an engineering estimate for a shadow shield sized to
    # protect avionics and the LH2 tank, not a crew.
    shield_kg_per_mw_thermal: float = 4.0
    # $/kg for shield material (tungsten / lithium hydride / borated composite)
    # fabricated and qualified. Estimate.
    shield_usd_per_kg: float = 8_000.0
    # Power conversion for a *nuclear-electric* system: Brayton or Stirling
    # converters turning reactor heat into bus power. Zero for nuclear-thermal,
    # which uses the reactor heat directly. Terrestrial Brayton is ~$1,000/kWe;
    # space-qualified, low-rate, radiation-tolerant conversion is far more.
    power_conversion_usd_per_kw_electric: float = 20_000.0

    # --- TANKAGE / INTEGRATION -----------------------------------------------
    # Tank dry mass as a fraction of propellant capacity. Xenon is stored
    # supercritical at 100-180 bar in composite-overwrapped pressure vessels:
    # 3-5% of propellant mass is typical. Cryogenic LH2 tanks are volume-driven
    # (70.85 kg/m^3) and carry MLI, so 10-15% is typical for a stage that must
    # hold propellant for months.
    tank_mass_fraction_stored_gas: float = 0.04
    tank_mass_fraction_cryogenic: float = 0.12
    # $/kg of tank dry mass, fabricated and qualified. COPVs and cryo tanks are
    # both in the low thousands of $/kg at spacecraft scale. Estimate.
    tank_usd_per_kg: float = 5_000.0
    # Integration, assembly and test as a fraction of hardware capex. 15-30% is
    # the standard range in space-systems cost estimating handbooks; 20% is the
    # usual working number for an integrated stage.
    integration_markup: float = 0.20

    # --- OPERATIONS ----------------------------------------------------------
    # Mission operations, $/day, for a robotic deep-space cargo mission in
    # cruise. Anchors from NASA Planetary Science Division senior-review budgets:
    # MRO extended operations ~$25-30M/yr (~$75k/day), Dawn extended mission
    # ~$12M/yr (~$33k/day), New Horizons extended ~$14M/yr (~$38k/day). A cargo
    # tug is simpler than a science mission but still needs a flight team,
    # navigation, and a thruster that is on essentially continuously.
    #
    # THIS IS THE TERM PEOPLE FORGET. It is proportional to trip time, so a
    # 3-year electric transfer pays 6x the operations bill of a 6-month nuclear
    # one. On a mission where the transport hardware is cheap, ops can be a
    # quarter of the total, and it is most of the economic case for going fast.
    ops_usd_per_day: float = 40_000.0
    # Deep Space Network tracking. NASA charges an aperture fee per hour of
    # antenna time; the published rates for a 34 m beam-waveguide antenna are in
    # the $1,000-2,500/hr band, with 70 m antennas several times higher. This
    # figure is approximate -- the DSN cost model is a tiered structure that
    # depends on mission class and aperture, not a single posted price.
    dsn_usd_per_hour: float = 1_200.0
    dsn_pass_hours: float = 8.0
    #: Cruise-phase tracking cadence. Continuous-thrust EP needs more navigation
    #: than a coasting stage, but 3/week is a reasonable common baseline.
    dsn_passes_per_week: float = 3.0

    # --- FINANCE -------------------------------------------------------------
    # Real discount rate. OMB Circular A-94 used a 7% real rate as the default
    # for public-investment cost-effectiveness analysis for decades; the 2023
    # revision lowered the general rate to ~2% real. Commercial space ventures
    # discount at 8-15%. 7% is kept as the baseline because it is the number
    # most readers of a NASA-adjacent study will expect, and because it sits in
    # the middle of the public/private spread.
    discount_rate_annual: float = 0.07
    # Insurance as a fraction of insured vehicle value (launch + first year in
    # orbit). Historic rates were 5-8%. The space insurance market took ~$995M
    # of claims against ~$557M of premium in 2023, and rates hardened sharply
    # into 2024, with some risks quoted in the mid-teens. 8% is a post-hardening
    # baseline. A nuclear stage would in reality face a bespoke and probably
    # much worse rate, or would be government-indemnified; see the note in
    # ConservativeCostModel.
    insurance_rate_of_vehicle_value: float = 0.08
    # Campaign-level learning-curve slope for the whole vehicle, used by the
    # levelised cost calculation. Same Wright's-law convention as the thruster
    # slope. 90% is the usual assumption for low-rate space systems; 85% is
    # aggressive, 95% conservative.
    learning_curve_slope: float = 0.90
    # Value of one kilogram delivered to the destination, used ONLY for the NPV
    # calculation, never for $/kg. This is the most arbitrary number in the book
    # and it is a policy input, not a market price -- nobody sells cargo delivery
    # to Mars. Anchors: MSL delivered mass to the Mars surface cost ~$2.8M/kg
    # all-in; bulk cargo to Mars orbit should be far cheaper than a flagship
    # rover. $300k/kg is a deliberately conservative stand-in. Sweep it, or set
    # it to your own programme's willingness to pay.
    payload_value_usd_per_kg: float = 300_000.0

    # --- derived lookups -----------------------------------------------------
    def propellant_usd_per_kg(self, species: str) -> float:
        """Delivered, loaded price of one kilogram of propellant.

        Cryogens get :attr:`cryo_handling_multiplier` applied on top of the
        commodity price, because for hydrogen the commodity price is a small
        part of what it actually costs to put propellant in a tank.

        Unknown species fall back to the xenon price with a warning rather than
        raising, so a new propulsion model does not crash the whole sweep.
        """
        key = species.strip().lower()
        if key in ("xenon", "xe"):
            return self.xenon_usd_per_kg
        if key in ("krypton", "kr"):
            return self.krypton_usd_per_kg
        if key in ("argon", "ar"):
            return self.argon_usd_per_kg
        if key in ("iodine", "i2", "i"):
            return self.iodine_usd_per_kg
        if key in ("hydrogen", "h2", "lh2", "liquid_hydrogen"):
            return self.lh2_usd_per_kg * self.cryo_handling_multiplier
        LOGGER.warning(
            "no price for propellant %r; falling back to the xenon price (%.0f $/kg)",
            species,
            self.xenon_usd_per_kg,
        )
        return self.xenon_usd_per_kg

    def is_cryogenic(self, species: str) -> bool:
        """Whether this propellant boils off and needs cryogenic handling."""
        return species.strip().lower() in (
            "hydrogen",
            "h2",
            "lh2",
            "liquid_hydrogen",
            "methane",
            "ch4",
            "oxygen",
            "lox",
        )

    def tank_mass_fraction(self, species: str) -> float:
        """Tank dry mass as a fraction of propellant capacity."""
        return (
            self.tank_mass_fraction_cryogenic
            if self.is_cryogenic(species)
            else self.tank_mass_fraction_stored_gas
        )

    def launch_usd_per_kg(self, destination: str = "leo") -> float:
        """Launch price to a drop-off orbit.

        ``destination`` is one of ``leo``, ``gto``, ``escape`` (C3 = 0). A
        solar-electric tug is normally dropped in LEO and spirals; a nuclear
        stage may be bought a higher drop-off, which is a real and large cost
        difference, hence the separate prices.
        """
        key = destination.strip().lower()
        if key in ("leo", "low_earth_orbit"):
            return self.launch_leo_usd_per_kg
        if key in ("gto", "geo_transfer"):
            return self.launch_gto_usd_per_kg
        if key in ("escape", "c3_0", "c3=0", "tli", "tmi", "interplanetary"):
            return self.launch_escape_usd_per_kg
        LOGGER.warning(
            "unknown launch destination %r; using the LEO price", destination
        )
        return self.launch_leo_usd_per_kg

    def thruster_unit_cost(
        self, kind: str, power_kw: float, cumulative_units: float | None = None
    ) -> float:
        """Recurring cost of one flight thruster at a given production index.

        Wright's law: the cost of the N-th unit is ``C1 * N ** log2(slope)``.
        With the default 90% slope the 100th unit costs 47% of the first. This
        is the mechanism behind "recurring cost falls steeply with production
        rate" and it is why a Starlink-scale Hall thruster and a one-off
        government Hall thruster differ by two orders of magnitude despite being
        the same device.
        """
        key = kind.strip().lower()
        if "grid" in key or "ion" in key or "nstar" in key or "next" in key:
            first_unit = self.gridded_ion_fixed_usd + self.gridded_ion_usd_per_kw * max(
                power_kw, 0.0
            )
        else:
            first_unit = self.hall_thruster_fixed_usd + self.hall_thruster_usd_per_kw * max(
                power_kw, 0.0
            )
        n = self.thruster_production_units if cumulative_units is None else cumulative_units
        n = max(float(n), 1.0)
        exponent = math.log2(max(self.thruster_learning_slope, 1e-6))
        return first_unit * n**exponent

    def radiator_usd_per_m2(self) -> float:
        """Deployable radiator cost per square metre of radiating area."""
        return self.radiator_areal_density_kg_m2 * self.radiator_usd_per_kg

    def dsn_usd_per_day(self) -> float:
        """Averaged DSN tracking cost per day of cruise."""
        return (
            self.dsn_usd_per_hour * self.dsn_pass_hours * self.dsn_passes_per_week / 7.0
        )

    # --- plumbing ------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """Flat dict of every field, for logging and for ``assumptions()``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_values(self, **overrides: Any) -> "PriceBook":
        """Copy with fields overridden. Unknown field names raise."""
        known = {f.name for f in fields(self)}
        bad = set(overrides) - known
        if bad:
            raise KeyError(
                f"unknown PriceBook field(s) {sorted(bad)}. Known: {sorted(known)}"
            )
        return replace(self, **overrides)

    @staticmethod
    def numeric_field_names() -> tuple[str, ...]:
        """Names of the sweepable (float) fields, in declaration order."""
        return tuple(
            f.name
            for f in fields(PriceBook)
            if f.type in ("float", float) and f.name != "currency_year"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<PriceBook {self.label!r} {self.currency_year} USD "
            f"launch_leo={self.launch_leo_usd_per_kg:.0f}$/kg "
            f"xenon={self.xenon_usd_per_kg:.0f}$/kg "
            f"reactor={self.reactor_usd_per_kw_thermal:.0f}$/kWt>"
        )


# --- The three reference books ------------------------------------------------

#: Best current estimate. Falcon-Heavy-class launch, post-2022 xenon prices,
#: a low-rate but not first-article production run, and a nuclear reactor at the
#: geometric middle of a range spanning an order of magnitude.
BASELINE = PriceBook(label="baseline")

#: What a mature, high-flight-rate 2035 launch and production environment could
#: plausibly look like. The dominant assumption here is Starship-class launch at
#: $500/kg to LEO, which is a PROJECTION -- no vehicle has flown a commercial
#: payload at that price. ``launch_price_is_speculative`` is set accordingly and
#: any conclusion that flips between BASELINE and OPTIMISTIC purely on launch
#: price should be reported as launch-price-dependent, not as a result.
OPTIMISTIC = PriceBook(
    label="optimistic",
    # Starship projections span $200-1,000/kg to LEO depending on who is talking
    # and how much of the development cost is amortised. $500/kg is the middle
    # of the credible band, and is still ~4x above the most aggressive claims.
    launch_leo_usd_per_kg=500.0,
    launch_gto_usd_per_kg=1_500.0,
    # Escape with orbital refuelling; without refuelling a Starship's escape
    # performance is poor, so this number embeds a whole architecture assumption.
    launch_escape_usd_per_kg=2_500.0,
    launch_vehicle="starship_projected",
    launch_price_is_speculative=True,
    # Xenon retreating to pre-2021 contract prices, escalated to 2026.
    xenon_usd_per_kg=1_500.0,
    krypton_usd_per_kg=300.0,
    argon_usd_per_kg=3.0,
    iodine_usd_per_kg=60.0,
    lh2_usd_per_kg=6.0,
    cryo_handling_multiplier=3.0,
    # Active or near-zero boil-off cryo storage, demonstrated at scale.
    lh2_boiloff_frac_per_day=0.0003,
    # ROSA-class arrays at Starlink-adjacent production volume.
    solar_array_usd_per_w=80.0,
    ppu_usd_per_w=30.0,
    radiator_usd_per_kg=2_500.0,
    hall_thruster_fixed_usd=60_000.0,
    hall_thruster_usd_per_kw=30_000.0,
    gridded_ion_fixed_usd=150_000.0,
    gridded_ion_usd_per_kw=80_000.0,
    thruster_learning_slope=0.85,
    thruster_production_units=20.0,
    # The low end of the reactor range: a production line exists and the
    # non-recurring cost has been paid off by an earlier programme.
    reactor_usd_per_kw_thermal=200.0,
    reactor_fixed_usd=30e6,
    shield_usd_per_kg=5_000.0,
    power_conversion_usd_per_kw_electric=8_000.0,
    tank_usd_per_kg=2_500.0,
    integration_markup=0.12,
    # A lean commercial flight team running several vehicles from one console.
    ops_usd_per_day=10_000.0,
    dsn_usd_per_hour=800.0,
    dsn_passes_per_week=2.0,
    # OMB Circular A-94 (2023 revision) general real rate, ~2%.
    discount_rate_annual=0.03,
    insurance_rate_of_vehicle_value=0.04,
    learning_curve_slope=0.85,
    payload_value_usd_per_kg=300_000.0,
)

#: The defensible pessimistic bracket. Falcon-9-class launch prices, xenon at
#: the top of its 2022 spike, first-article hardware, a hard insurance market,
#: and a reactor at the top of the credible range. If a conclusion survives
#: CONSERVATIVE it is robust; if it only holds under OPTIMISTIC it is a claim
#: about the future price of launch, not about propulsion or control.
CONSERVATIVE = PriceBook(
    label="conservative",
    # Falcon 9 at list: $69.75M / 22,800 kg = $3,060/kg. Rounded up for fill
    # factor. For reference, an SLS-launched nuclear stage would be
    # $2.2B / 95 t = $23,000/kg, which is off this scale entirely.
    launch_leo_usd_per_kg=3_500.0,
    launch_gto_usd_per_kg=9_000.0,
    # Falcon 9 to Mars: $69.75M / 4,020 kg = $17,350/kg.
    launch_escape_usd_per_kg=18_000.0,
    launch_vehicle="falcon_9_expendable",
    launch_price_is_speculative=False,
    # Top of the 2022 xenon spike, held.
    xenon_usd_per_kg=5_000.0,
    krypton_usd_per_kg=1_000.0,
    argon_usd_per_kg=30.0,
    iodine_usd_per_kg=200.0,
    lh2_usd_per_kg=12.0,
    cryo_handling_multiplier=10.0,
    lh2_boiloff_frac_per_day=0.005,
    solar_array_usd_per_w=600.0,
    ppu_usd_per_w=250.0,
    radiator_usd_per_kg=12_000.0,
    hall_thruster_fixed_usd=600_000.0,
    hall_thruster_usd_per_kw=400_000.0,
    gridded_ion_fixed_usd=1_500_000.0,
    gridded_ion_usd_per_kw=700_000.0,
    thruster_learning_slope=0.95,
    thruster_production_units=1.0,
    # Top of the credible reactor band: ~$1B for a 500 MWt core.
    reactor_usd_per_kw_thermal=2_000.0,
    reactor_fixed_usd=250e6,
    shield_kg_per_mw_thermal=8.0,
    shield_usd_per_kg=20_000.0,
    power_conversion_usd_per_kw_electric=60_000.0,
    tank_mass_fraction_cryogenic=0.18,
    tank_usd_per_kg=12_000.0,
    integration_markup=0.30,
    ops_usd_per_day=90_000.0,
    dsn_usd_per_hour=2_500.0,
    dsn_passes_per_week=5.0,
    # Commercial cost of capital for a high-risk venture.
    discount_rate_annual=0.12,
    # Post-2023 hardened space insurance market, novel vehicle, no heritage.
    insurance_rate_of_vehicle_value=0.18,
    learning_curve_slope=0.95,
    payload_value_usd_per_kg=300_000.0,
)

PRICE_BOOKS: dict[str, PriceBook] = {
    "baseline": BASELINE,
    "optimistic": OPTIMISTIC,
    "conservative": CONSERVATIVE,
}


def get_price_book(name: str) -> PriceBook:
    """Look up one of the reference books by name."""
    key = name.strip().lower()
    if key not in PRICE_BOOKS:
        raise KeyError(
            f"unknown price book {name!r}. Available: {sorted(PRICE_BOOKS)}"
        )
    return PRICE_BOOKS[key]


# --- Documented uncertainty ---------------------------------------------------
#: ``field name -> (low, high)`` plausible range in constant 2026 USD.
#:
#: These are the ranges the tornado and the Monte Carlo walk. They are elicited
#: from the citations in the field comments above, NOT fitted to data -- there is
#: no dataset of flight nuclear reactor prices to fit to. Read them as "an
#: informed person would be surprised to be outside this", i.e. roughly a 90%
#: interval.
#:
#: ``reactor_usd_per_kw_thermal`` spans a factor of ten. That is not sloppiness;
#: it is the honest state of knowledge, and it is why it is the headline swept
#: parameter of the whole economics layer.
UNCERTAIN_PARAMETERS: dict[str, tuple[float, float]] = {
    # Launch: $500/kg (Starship, projected) to $18,000/kg (F9 to escape).
    "launch_leo_usd_per_kg": (500.0, 3_500.0),
    "launch_gto_usd_per_kg": (1_500.0, 9_000.0),
    "launch_escape_usd_per_kg": (2_500.0, 18_000.0),
    # Xenon: pre-2021 contract price to the top of the 2022 spike.
    "xenon_usd_per_kg": (1_200.0, 5_000.0),
    "krypton_usd_per_kg": (250.0, 1_200.0),
    "argon_usd_per_kg": (2.0, 30.0),
    "lh2_usd_per_kg": (6.0, 12.0),
    "cryo_handling_multiplier": (3.0, 10.0),
    "lh2_boiloff_frac_per_day": (0.0003, 0.005),
    "solar_array_usd_per_w": (80.0, 1_000.0),
    "ppu_usd_per_w": (30.0, 250.0),
    "radiator_usd_per_kg": (2_500.0, 12_000.0),
    "hall_thruster_usd_per_kw": (30_000.0, 400_000.0),
    "gridded_ion_usd_per_kw": (80_000.0, 700_000.0),
    # THE headline uncertainty: a factor of ten, and no way to narrow it
    # without someone actually building and selling one.
    "reactor_usd_per_kw_thermal": (200.0, 2_000.0),
    "reactor_fixed_usd": (30e6, 250e6),
    "shield_usd_per_kg": (5_000.0, 20_000.0),
    "power_conversion_usd_per_kw_electric": (8_000.0, 60_000.0),
    "tank_usd_per_kg": (2_500.0, 12_000.0),
    "integration_markup": (0.12, 0.30),
    "ops_usd_per_day": (10_000.0, 90_000.0),
    "dsn_usd_per_hour": (800.0, 2_500.0),
    "discount_rate_annual": (0.03, 0.12),
    "insurance_rate_of_vehicle_value": (0.04, 0.18),
}


def uncertainty_range(parameter: str) -> tuple[float, float]:
    """Documented (low, high) range for one swept price parameter."""
    if parameter not in UNCERTAIN_PARAMETERS:
        raise KeyError(
            f"{parameter!r} has no documented uncertainty range. "
            f"Documented: {sorted(UNCERTAIN_PARAMETERS)}"
        )
    return UNCERTAIN_PARAMETERS[parameter]


__all__ = [
    "CURRENCY_YEAR",
    "PriceBook",
    "BASELINE",
    "OPTIMISTIC",
    "CONSERVATIVE",
    "PRICE_BOOKS",
    "UNCERTAIN_PARAMETERS",
    "get_price_book",
    "uncertainty_range",
]
