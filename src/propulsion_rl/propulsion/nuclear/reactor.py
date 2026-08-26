"""Reusable fission core model shared by the NTP and NEP stages.

This is *not* a :class:`~propulsion_rl.propulsion.base.PropulsionSystem`. It is a
component that a propulsion system owns: the thing that makes heat, has a
reactivity budget, poisons itself with xenon, keeps making decay heat after you
scram it, and eventually burns up. ``ntp.py`` and ``nep.py`` each instantiate one
and wrap it with a different heat-to-thrust conversion chain.

Physics content
---------------
**Point kinetics.** Six delayed-neutron groups with the U-235 data in
:mod:`propulsion_rl.core.constants`. The prompt neutron generation time is
``NEUTRON_GEN_TIME = 1e-4 s`` while the environment hands out steps of minutes to
hours, so the prompt mode is ~10^7 times faster than the macro step. Integrating
that explicitly is hopeless.

*Chosen scheme: the prompt-jump (prompt-drop) approximation with analytic
exponential precursor integration.* Per substep:

1. Reactivity is updated (drum slew + Doppler + moderator + xenon + burnup).
2. The prompt jump is applied algebraically,
   ``n = Lambda * (sum_i lambda_i C_i + S) / (beta_eff - rho)``, which is the
   ``dn/dt -> 0`` limit of the prompt equation. This reproduces the instantaneous
   power step that follows a reactivity step exactly, with no time-step limit.
3. The stable inhour root ``omega(rho)`` is found by a bracketed Newton solve of
   ``rho = Lambda*omega + sum_i beta_i*omega/(omega+lambda_i)``. The amplitude is
   advanced as ``n <- n*exp(omega*dt)`` and the precursors are advanced with the
   *exact* integral of ``dC_i/dt = beta_i n/Lambda - lambda_i C_i`` under that
   exponential flux shape,
   ``C_i' = C_i e^{-lambda_i dt} + (beta_i/Lambda)(n' - n e^{-lambda_i dt})/(omega+lambda_i)``.
4. The prompt-jump relation is re-applied as a corrector, which also supplies the
   source-driven subcritical multiplication floor.

Because the fundamental inhour root always satisfies ``omega > -lambda_1``, every
exponential in step 3 is bounded and every precursor stays non-negative. The
scheme is therefore unconditionally stable and positivity-preserving; ``dt = 1 s``,
``60 s`` and ``3600 s`` all give the same steady state and the same asymptotic
period. Substepping is used only when the transient is genuinely fast (adaptive on
``|omega| dt`` and on fuel temperature rise), so the common case -- a reactor
sitting at its setpoint for an hour -- costs one substep.

**Reactivity budget.** Control-drum/reflector worth is the agent-side actuator,
rate-limited in pcm/s. Feedbacks are Doppler broadening of the U-238/U-235
resonances in the fuel (negative, entered as pcm/K at 300 K and scaled as
``1/sqrt(T)``, the textbook form, so the *integral* temperature defect from cold
to 2500-2900 K comes out at the few-dollars level that NERVA/Pewee actually
measured) plus a moderator/reflector coefficient on the structure node. Xenon and
burnup swing are subtracted as well.

**Prompt-critical safety.** ``rho`` approaching ``BETA_EFF`` is a hard constraint;
so is the asymptotic reactor period ``T = 1/omega`` falling below
``min_period_s`` (the classic 10-second-period scram setpoint). Insertion rate is
limited, and breaching either limit emits a CRITICAL event.

**Xenon-135.** Coupled I-135/Xe-135 balance using ``I135_YIELD``,
``XE135_YIELD``, the two decay constants and ``XE135_SIGMA_A``. Note that
``XE135_SIGMA_A = 2.65e-18`` is 2.65e6 barns expressed in **cm^2** (1 b = 1e-24
cm^2), despite the comment in ``constants.py``; it is therefore paired here with a
flux in n/(cm^2 s), which is the unit the design fluxes below are quoted in. The
2x2 system is advanced with its closed-form solution, so it is exact at any dt.
Absolute number densities are never needed: the reactivity worth is calibrated so
that the equilibrium xenon at rated power equals a specified
``xenon_equilibrium_pcm``, and all dynamics then follow from the real constants.
Burnout (``sigma_a * phi``) dominates decay by ~10x at the design fluxes used
here, which is exactly the regime that produces a large post-shutdown xenon peak
around 9 h -- the "iodine pit" / xenon deadtime that gates restart.

**Decay heat.** Way-Wigner, ``P_d/P_0 = F [t^-0.2 - (t+T_op)^-0.2]`` with
``F = DECAY_HEAT_FRACTION``. Implemented as a bank of exponential groups obtained
by the Laplace representation ``tau^-1.2 = Gamma(1.2)^-1 * int alpha^0.2
e^{-alpha tau} d alpha``, discretised on a log grid. That form handles an
*arbitrary* power history (throttling, restarts) instead of only a single clean
shutdown, integrates exactly per group, and reproduces the closed-form Way-Wigner
answer to ~1% (checked in the self-test). Decay heat still has to be rejected
after a scram; the caller cannot simply stop cooling.

**Thermal.** Two lumped nodes (fuel, structure/moderator) with a
Dittus-Boelter-style ``flow^0.8`` coolant conductance, solved by backward Euler as
a 2x2 linear system -- unconditionally stable. Fuel ``dT/dt`` is a constraint:
fast power ramps crack graphite/CERMET elements, which is the real reason NERVA
startups were minutes and not seconds. Burnup accumulates from the fission rate
and limits core life.

References
----------
* Pewee-1, LASL/NRDS 1968: 503 MW(t), peak fuel exit gas 2550 K, Isp ~845 s.
* NERVA NRX-A6 / XE-Prime, 1966-69: ~1100-1140 MW(t), ~10^3 s full-power runs.
* Kilopower / KRUSTY, LANL/NASA 2018: 4 kW(t) UMo fast core, ~800 K hot end,
  demonstrated load-following and a self-regulating negative temperature
  coefficient with no operator action.
* Duderstadt & Hamilton, *Nuclear Reactor Analysis*, ch. 6-7 (point kinetics,
  prompt jump, inhour equation, xenon transients).
* Way & Wigner, Phys. Rev. 73, 1318 (1948) (fission-product decay heat).
* El-Wakil, *Nuclear Heat Transport* (lumped-node core thermal models).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ...core.constants import (
    BETA_EFF,
    DECAY_HEAT_FRACTION,
    DELAYED_BETA,
    DELAYED_LAMBDA,
    EPS,
    I135_DECAY_CONST,
    I135_YIELD,
    NEUTRON_GEN_TIME,
    U235_FISSION_ENERGY_J,
    XE135_DECAY_CONST,
    XE135_SIGMA_A,
    XE135_YIELD,
)
from ...core.types import ConstraintReport, Event, Severity

logger = logging.getLogger(__name__)

# --- Precomputed delayed-neutron data ----------------------------------------
# Kept as plain tuples: the hot loop touches them six elements at a time, where
# Python floats beat numpy's per-call overhead by an order of magnitude.
_BETA_I: tuple[float, ...] = tuple(DELAYED_BETA)
_LAMBDA_I: tuple[float, ...] = tuple(DELAYED_LAMBDA)
_N_GROUPS = len(_BETA_I)
_BETA_SUM = float(sum(_BETA_I))
_LAMBDA_MIN = float(min(_LAMBDA_I))
#: sum(beta_i / lambda_i): the delayed-neutron "effective lifetime", ~0.085 s.
_DELAYED_LIFETIME = float(sum(b / lam for b, lam in zip(_BETA_I, _LAMBDA_I)))
#: beta_i / Lambda, the precursor production coefficients.
_BETA_OVER_GEN: tuple[float, ...] = tuple(b / NEUTRON_GEN_TIME for b in _BETA_I)

#: The kinetics model is only valid below prompt critical. Above this fraction of
#: beta the prompt-jump assumption fails and we clamp, flag, and let the thermal
#: model record the resulting excursion.
_RHO_CLAMP_FRAC = 0.98

# --- Decay-heat group expansion ----------------------------------------------
# tau^-1.2 = (1/Gamma(1.2)) * int_0^inf alpha^0.2 e^{-alpha tau} d(ln alpha),
# discretised midpoint on a log grid spanning 0.01 s .. ~30 yr of decay.
_DH_ALPHA_MIN = 1.0e-9
_DH_ALPHA_MAX = 1.0e2
_DH_N = 28
_DH_DLN = math.log(_DH_ALPHA_MAX / _DH_ALPHA_MIN) / _DH_N
_DH_ALPHA = np.array(
    [_DH_ALPHA_MIN * math.exp((j + 0.5) * _DH_DLN) for j in range(_DH_N)],
    dtype=np.float64,
)
#: production coefficient c_j so that sum_j alpha_j H_j reproduces Way-Wigner.
_DH_C = (0.2 * DECAY_HEAT_FRACTION * _DH_DLN / math.gamma(1.2)) * _DH_ALPHA**0.2


def way_wigner_fraction(t_since_s: float, operating_s: float) -> float:
    """Closed-form Way-Wigner decay heat fraction, for validation only.

    ``P_decay / P_full = F [t^-0.2 - (t + T_op)^-0.2]`` with both times in
    seconds. The runtime model uses the exponential-group expansion instead so it
    can handle arbitrary power histories; this function exists so the self-test
    can check one against the other.
    """
    t = max(t_since_s, 1e-3)
    return DECAY_HEAT_FRACTION * (t**-0.2 - (t + max(operating_s, 0.0)) ** -0.2)


# --- Design ------------------------------------------------------------------
@dataclass(slots=True)
class ReactorDesign:
    """Static description of a core. One instance per propulsion preset.

    Defaults describe a compact, highly enriched, epithermal NTP-style core; the
    NEP presets override the long-life / low-power fields.
    """

    name: str = "generic_core"
    rated_thermal_w: float = 500e6

    # --- thermal nodes -------------------------------------------------------
    fuel_mass_kg: float = 600.0
    fuel_cp_j_kg_k: float = 1400.0          # graphite / UC-ZrC composite
    struct_mass_kg: float = 900.0
    struct_cp_j_kg_k: float = 500.0         # Be reflector + Inconel structure
    fuel_to_struct_ua_w_k: float = 2.0e5    # conduction/radiation fuel -> structure
    coolant_ua_w_k: float = 3.0e5           # structure -> coolant at full flow
    fuel_power_fraction: float = 0.94       # fission energy deposited in the fuel
    coolant_inlet_k: float = 100.0
    initial_temperature_k: float = 300.0
    max_fuel_temp_k: float = 2900.0
    fuel_melt_temp_k: float = 3200.0
    max_fuel_dtdt_k_s: float = 60.0         # thermal-shock limit on the elements

    # --- reactivity ----------------------------------------------------------
    drum_worth_pcm: float = 7000.0          # full-in to full-out control drum span
    max_reactivity_rate_pcm_s: float = 60.0
    doppler_pcm_per_k_at_300k: float = -2.4     # negative; scaled as 1/sqrt(T)
    moderator_pcm_per_k: float = -0.35          # reflector/structure coefficient
    reference_temperature_k: float = 300.0
    min_period_s: float = 5.0
    excess_reactivity_pcm: float = 0.0      # cold-clean excess held down by drums
    source_power_fraction_per_s: float = 1.0e-8   # Sb-Be / spontaneous-fission source
    initial_power_fraction: float = 1.0e-3   # >0 starts at hot standby, 0 is cold

    # --- xenon ---------------------------------------------------------------
    design_flux_n_cm2_s: float = 1.0e14
    xenon_equilibrium_pcm: float = -2600.0  # worth of equilibrium Xe at rated power

    # --- burnup / life -------------------------------------------------------
    fissile_mass_kg: float = 60.0           # U-235 loading
    burnup_limit_fima: float = 0.06         # fissions per initial metal atom
    burnup_swing_pcm: float = -4000.0       # reactivity lost at the burnup limit

    # --- mass bookkeeping (fed to the BOM by the owning propulsion system) ----
    core_mass_kg: float = 1500.0
    shield_mass_kg: float = 1500.0

    def __post_init__(self) -> None:
        if self.rated_thermal_w <= 0.0:
            raise ValueError("rated_thermal_w must be positive")
        if self.doppler_pcm_per_k_at_300k > 0.0:
            raise ValueError("Doppler coefficient must be negative")


# --- Inhour solve ------------------------------------------------------------
def _inhour_residual(omega: float, rho: float) -> float:
    s = NEUTRON_GEN_TIME * omega
    for b, lam in zip(_BETA_I, _LAMBDA_I):
        s += b * omega / (omega + lam)
    return s - rho


def solve_inhour(rho: float) -> float:
    """Fundamental root ``omega`` of the inhour equation for reactivity ``rho``.

    The fundamental root is the largest one and always lies in
    ``(-lambda_1, +inf)``, which is what makes the exponential precursor
    integration in :meth:`FissionReactor.step` unconditionally well behaved.
    Solved by a bisection-safeguarded Newton iteration.
    """
    if abs(rho) < 1e-14:
        return 0.0
    if rho < 0.0:
        lo = -_LAMBDA_MIN * (1.0 - 1e-9)
        hi = 0.0
    else:
        lo = 0.0
        # f(rho/Lambda) = sum_i beta_i*omega/(omega+lambda_i) > 0, so this
        # always brackets the root from above.
        hi = rho / NEUTRON_GEN_TIME
    omega = rho / (NEUTRON_GEN_TIME + _DELAYED_LIFETIME)
    if not (lo < omega < hi):
        omega = 0.5 * (lo + hi)
    tol = 1e-12 + 1e-10 * abs(rho)
    for _ in range(60):
        f = _inhour_residual(omega, rho)
        if abs(f) <= tol:
            break
        if f > 0.0:
            hi = omega
        else:
            lo = omega
        df = NEUTRON_GEN_TIME
        for b, lam in zip(_BETA_I, _LAMBDA_I):
            d = omega + lam
            df += b * lam / (d * d)
        step = f / df if df > EPS else 0.0
        nxt = omega - step
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - omega) <= 1e-15 * max(abs(omega), 1e-12):
            omega = nxt
            break
        omega = nxt
    return omega


@dataclass(slots=True)
class ReactorState:
    """Mutable core state. Split out so it is cheap to snapshot in tests."""

    n: float = 0.0                       # power fraction of rated
    precursors: list[float] = field(default_factory=list)
    fuel_temp_k: float = 300.0
    struct_temp_k: float = 300.0
    drum_pcm: float = 0.0
    iodine: float = 0.0                  # normalised I-135 density
    xenon: float = 0.0                   # normalised Xe-135 density
    decay_groups: np.ndarray = field(default_factory=lambda: np.zeros(_DH_N))
    fissions: float = 0.0                # cumulative fissions
    burn_time_s: float = 0.0
    elapsed_s: float = 0.0
    omega: float = 0.0
    fuel_dtdt_k_s: float = 0.0
    scrammed: bool = False
    destroyed: bool = False


class FissionReactor:
    """Point-kinetics fission core with feedback, xenon, decay heat and burnup.

    Read-only telemetry is exposed as properties (:attr:`power_w`,
    :attr:`fuel_temperature_k`, :attr:`xenon_reactivity_pcm`, :attr:`period_s`,
    :attr:`burnup_fraction`) so the owning propulsion system can build its
    observation vector without reaching into private state.
    """

    #: Adaptive substepping controls; see :meth:`step`.
    #: ``_MAX_OMEGA_DT`` bounds the amplitude change per substep while the core is
    #: on a growing period. ``_TRANSIENT_GROWTH`` makes the substep grow
    #: geometrically out of a reactivity change, which is what resolves the
    #: higher inhour modes that the single-exponential ansatz cannot see; without
    #: it a 300 s step after a 10-cent insertion undershoots by ~15%.
    _MAX_OMEGA_DT = 1.0
    _DECAY_OMEGA_DT = 4.0
    _TRANSIENT_GROWTH = 0.5
    _TRANSIENT_MIN_DT = 0.25
    _TRANSIENT_TRIGGER_PCM = 1.0
    _MAX_DT_PER_SUBSTEP_K = 120.0
    _MAX_SUBSTEPS = 160

    def __init__(self, design: ReactorDesign) -> None:
        self.design = design
        self.state = ReactorState()
        self.events: list[Event] = []
        # Per-unit manufacturing variation, resampled in reset().
        self._doppler_scale = 1.0
        self._drum_scale = 1.0
        self._fuel_cap_scale = 1.0
        # Derived constants, recomputed in reset() because reset() perturbs them.
        self._c_fuel = 1.0
        self._c_struct = 1.0
        self._xenon_worth_per_unit = 0.0
        self._total_fissile_atoms = 1.0
        self._doppler_k = 0.0
        self._sigma_phi_rated = 0.0
        self._dh_buf = np.zeros(_DH_N, dtype=np.float64)
        self._since_rho_change = 1e6
        self._rho_at_last_change = 0.0
        self.reset(np.random.default_rng(0))

    # --- lifecycle -----------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        """Restore to a cold, clean, shutdown core.

        ``rng`` draws the unit-to-unit variation (Doppler coefficient, drum
        worth, fuel heat capacity). Identically seeded generators give identical
        trajectories, which the benchmark relies on.
        """
        d = self.design
        self._doppler_scale = float(rng.normal(1.0, 0.05))
        self._doppler_scale = min(max(self._doppler_scale, 0.85), 1.15)
        self._drum_scale = float(min(max(rng.normal(1.0, 0.03), 0.9), 1.1))
        self._fuel_cap_scale = float(min(max(rng.normal(1.0, 0.02), 0.94), 1.06))

        self._c_fuel = d.fuel_mass_kg * d.fuel_cp_j_kg_k * self._fuel_cap_scale
        self._c_struct = d.struct_mass_kg * d.struct_cp_j_kg_k
        # Doppler: rho_D = K (sqrt(T) - sqrt(T_ref)) so that d(rho)/dT equals the
        # quoted coefficient at T_ref and softens as 1/sqrt(T), the standard
        # resonance-broadening form.
        self._doppler_k = (
            2.0
            * math.sqrt(d.reference_temperature_k)
            * d.doppler_pcm_per_k_at_300k
            * self._doppler_scale
        )
        self._total_fissile_atoms = max(d.fissile_mass_kg / 0.235 * 6.02214076e23, 1.0)
        # Xenon worth calibration: pick the per-unit-density worth that puts the
        # equilibrium xenon at rated power on the quoted reactivity.
        self._sigma_phi_rated = XE135_SIGMA_A * d.design_flux_n_cm2_s
        xe_eq = (XE135_YIELD + I135_YIELD) / (XE135_DECAY_CONST + self._sigma_phi_rated)
        self._xenon_worth_per_unit = d.xenon_equilibrium_pcm / max(xe_eq, EPS)

        n0 = max(d.initial_power_fraction, 0.0)
        self.state = ReactorState(
            n=n0,
            precursors=[
                b * n0 / (NEUTRON_GEN_TIME * lam)
                for b, lam in zip(_BETA_I, _LAMBDA_I)
            ],
            fuel_temp_k=d.initial_temperature_k,
            struct_temp_k=d.initial_temperature_k,
            drum_pcm=-d.drum_worth_pcm * self._drum_scale,
            iodine=0.0,
            xenon=0.0,
            decay_groups=np.zeros(_DH_N, dtype=np.float64),
        )
        if n0 > 0.0:
            # Hot standby: drums parked at the critical position for the current
            # (cold, clean) temperature, which is how a flight stage is held
            # between burns. Cold shutdown is initial_power_fraction = 0.
            self.state.drum_pcm = -(
                d.excess_reactivity_pcm
                + self._doppler_pcm(d.initial_temperature_k)
                + self._moderator_pcm(d.initial_temperature_k)
            )
        self.events = []
        self._since_rho_change = 1e6
        if n0 <= 0.0:
            # Source-driven subcritical multiplication floor; n is never exactly 0.
            self.state.n = self._prompt_jump(self._total_reactivity())
        self._rho_at_last_change = self._total_reactivity()

    # --- reactivity ----------------------------------------------------------
    def _doppler_pcm(self, fuel_temp_k: float) -> float:
        d = self.design
        return self._doppler_k * (
            math.sqrt(max(fuel_temp_k, 1.0)) - math.sqrt(d.reference_temperature_k)
        )

    def _moderator_pcm(self, struct_temp_k: float) -> float:
        d = self.design
        return d.moderator_pcm_per_k * (struct_temp_k - d.reference_temperature_k)

    def _burnup_pcm(self) -> float:
        d = self.design
        frac = self.burnup_fraction / max(d.burnup_limit_fima, EPS)
        return d.burnup_swing_pcm * min(frac, 1.5)

    def _total_reactivity(self) -> float:
        """Net reactivity in absolute units (dk/k, not pcm)."""
        s = self.state
        d = self.design
        pcm = (
            s.drum_pcm
            + d.excess_reactivity_pcm
            + self._doppler_pcm(s.fuel_temp_k)
            + self._moderator_pcm(s.struct_temp_k)
            + self.xenon_reactivity_pcm
            + self._burnup_pcm()
        )
        return pcm * 1e-5

    def _prompt_jump(self, rho: float) -> float:
        """Prompt-jump power level for the current precursor inventory."""
        s = self.state
        acc = self.design.source_power_fraction_per_s
        for lam, c in zip(_LAMBDA_I, s.precursors):
            acc += lam * c
        denom = BETA_EFF - rho
        if denom < BETA_EFF * (1.0 - _RHO_CLAMP_FRAC):
            denom = BETA_EFF * (1.0 - _RHO_CLAMP_FRAC)
        return NEUTRON_GEN_TIME * acc / denom

    # --- step ----------------------------------------------------------------
    def step(
        self,
        reactivity_cmd: float,
        coolant_flow: float,
        heat_removal_w: float,
        dt_s: float,
        *,
        max_rate_pcm_s: float | None = None,
        coolant_inlet_k: float | None = None,
        scram: bool = False,
        power_fraction: float | None = None,
    ) -> dict[str, float]:
        """Advance the core by ``dt_s``.

        Parameters
        ----------
        reactivity_cmd:
            Demanded control-drum reactivity in pcm, an absolute setpoint in
            ``[-drum_worth, +drum_worth]``. The drum slews towards it at
            ``max_rate_pcm_s`` (default: the design rate).
        coolant_flow:
            Normalised coolant mass flow in ``[0, 1]``. The structure-to-coolant
            conductance scales as ``flow^0.8`` (Dittus-Boelter).
        heat_removal_w:
            Hard cap on how much heat the downstream loop can actually accept.
        dt_s:
            Macro step. Internally substepped adaptively; the scheme is stable at
            any value.
        max_rate_pcm_s:
            Override on the drum slew rate, so a caller can expose "ramp
            aggressiveness" to the agent.
        coolant_inlet_k:
            Coolant inlet temperature; defaults to the design value.
        scram:
            Drive the drums fully in at the scram rate (10x the normal limit).
        power_fraction:
            If set, drums are servoed each substep toward this power fraction
            of rated, instead of toward a fixed ``reactivity_cmd``. Needed so a
            multi-hour environment step can follow a throttle without a
            prompt-critical insertion.

        Returns
        -------
        dict
            Flat telemetry: ``fission_power_w``, ``decay_heat_w``,
            ``thermal_power_w``, ``heat_removed_w``, ``fuel_temp_k``,
            ``struct_temp_k``, ``reactivity_pcm``, ``period_s``, ``dtdt_k_s``.
        """
        d = self.design
        s = self.state
        self.events.clear()
        dt_s = float(max(dt_s, 0.0))
        if dt_s <= 0.0:
            return self._telemetry(0.0)

        t_in = d.coolant_inlet_k if coolant_inlet_k is None else float(coolant_inlet_k)
        flow = float(min(max(coolant_flow, 0.0), 1.5))
        rate = d.max_reactivity_rate_pcm_s if max_rate_pcm_s is None else max_rate_pcm_s
        drum_span = d.drum_worth_pcm * self._drum_scale
        if scram or s.scrammed:
            target = -drum_span
            rate = abs(rate) * 10.0
            s.scrammed = True
            power_fraction = None
        else:
            target = float(min(max(reactivity_cmd, -drum_span), drum_span))
        rate = abs(rate)
        track_power = power_fraction is not None and not (scram or s.scrammed)
        n_demand = 0.0 if power_fraction is None else float(min(max(power_fraction, 0.0), 1.2))

        # Coolant conductance: Dittus-Boelter Nu ~ Re^0.8 -> UA ~ mdot^0.8.
        ua_flow = d.coolant_ua_w_k * (flow**0.8) if flow > 0.0 else 0.0

        # Hour-scale environment steps cannot be kinetically resolved: 160
        # substeps of a ~30 s delayed-critical period cover minutes, not a day.
        # Drop to a heat-balanced critical core and advance the slow states.
        if track_power and dt_s > 20.0:
            return self._step_quasi_steady(
                n_demand, flow, heat_removal_w, dt_s, t_in, ua_flow
            )

        remaining = dt_s
        heat_removed_total = 0.0
        energy_j = 0.0
        max_abs_dtdt = 0.0
        min_period_seen = math.inf
        worst_rho = -math.inf
        substeps = 0
        rho_prev = self._total_reactivity()
        # Seed omega for the first substep so the very first h already respects
        # the period; otherwise a step taken straight after a reactivity change
        # would be sized as if the core were at steady state.
        s.omega = solve_inhour(min(rho_prev, BETA_EFF * _RHO_CLAMP_FRAC))
        if abs(rho_prev - self._rho_at_last_change) > 1e-6 or (
            abs(target - s.drum_pcm) > self._TRANSIENT_TRIGGER_PCM
        ):
            self._since_rho_change = 0.0
            self._rho_at_last_change = rho_prev

        while remaining > EPS and substeps < self._MAX_SUBSTEPS:
            substeps += 1
            # --- pick a substep -------------------------------------------
            h = remaining
            w = abs(s.omega)
            if w > EPS:
                cap = self._MAX_OMEGA_DT if s.omega > 0.0 else self._DECAY_OMEGA_DT
                if s.omega < 0.0 and s.n < 1e-6:
                    cap = math.inf      # already parked on the source floor
                if math.isfinite(cap):
                    h = min(h, cap / w)
            h = min(
                h,
                max(
                    self._TRANSIENT_MIN_DT,
                    self._TRANSIENT_GROWTH * self._since_rho_change,
                ),
            )
            p_now = s.n * d.rated_thermal_w + self._decay_heat_w()
            net = p_now * d.fuel_power_fraction - d.fuel_to_struct_ua_w_k * (
                s.fuel_temp_k - s.struct_temp_k
            )
            if abs(net) > EPS:
                h = min(h, self._MAX_DT_PER_SUBSTEP_K * self._c_fuel / abs(net))
            if substeps == self._MAX_SUBSTEPS - 1:
                h = remaining
            h = min(max(h, 1e-4), remaining)

            # --- drum slew -------------------------------------------------
            if track_power:
                # Hold near delayed-critical at the *current* temperature and
                # add a few tens of cents of excess when power is below the
                # setpoint. Adding the error to the current drum position
                # instead would walk the drums onto the stop over a long step.
                err = max(-1.0, min(1.0, n_demand - s.n))
                target = self.drum_for_critical() + 250.0 * err
                target = float(min(max(target, -drum_span), drum_span))
            delta = target - s.drum_pcm
            max_move = rate * h
            s.drum_pcm += math.copysign(min(abs(delta), max_move), delta)

            # --- kinetics --------------------------------------------------
            rho = self._total_reactivity()
            if abs(rho - rho_prev) > 1e-6:
                self._since_rho_change = 0.0
                self._rho_at_last_change = rho
            rho_prev = rho
            worst_rho = max(worst_rho, rho)
            rho_eff = min(rho, BETA_EFF * _RHO_CLAMP_FRAC)
            n_old = self._prompt_jump(rho_eff)          # prompt jump
            omega = solve_inhour(rho_eff)
            s.omega = omega
            if omega > 0.0:
                min_period_seen = min(min_period_seen, 1.0 / omega)
            growth = math.exp(min(omega * h, 60.0))     # e^60 ~ 1e26, then clamped
            n_new = n_old * growth
            for i in range(_N_GROUPS):
                lam = _LAMBDA_I[i]
                decay = math.exp(-lam * h)
                s.precursors[i] = decay * s.precursors[i] + _BETA_OVER_GEN[i] * (
                    n_new - n_old * decay
                ) / (omega + lam)
            s.n = min(self._prompt_jump(rho_eff), 1.0e3)   # corrector + hard cap

            # --- decay heat ------------------------------------------------
            self._advance_decay_heat(s.n, h)
            p_decay = self._decay_heat_w()
            p_fission = s.n * d.rated_thermal_w
            p_total = p_fission + p_decay

            # --- xenon -----------------------------------------------------
            self._advance_xenon(s.n, h)

            # --- thermal (backward Euler, 2x2) -----------------------------
            t_f0, t_s0 = s.fuel_temp_k, s.struct_temp_k
            h_cool = ua_flow
            dts = max(t_s0 - t_in, 1.0)
            if heat_removal_w >= 0.0 and h_cool * dts > heat_removal_w:
                h_cool = heat_removal_w / dts
            p_f = p_total * d.fuel_power_fraction
            p_s = p_total - p_f
            u = d.fuel_to_struct_ua_w_k
            a11 = self._c_fuel / h + u
            a12 = -u
            a21 = -u
            a22 = self._c_struct / h + u + h_cool
            b1 = self._c_fuel / h * t_f0 + p_f
            b2 = self._c_struct / h * t_s0 + p_s + h_cool * t_in
            det = a11 * a22 - a12 * a21
            if abs(det) < EPS:
                t_f1, t_s1 = t_f0, t_s0
            else:
                t_f1 = (b1 * a22 - a12 * b2) / det
                t_s1 = (a11 * b2 - b1 * a21) / det
            s.fuel_temp_k = max(t_f1, 1.0)
            s.struct_temp_k = max(t_s1, 1.0)
            inst_rate = net / self._c_fuel
            avg_rate = (s.fuel_temp_k - t_f0) / h
            s.fuel_dtdt_k_s = avg_rate
            max_abs_dtdt = max(max_abs_dtdt, abs(avg_rate), abs(inst_rate))

            heat_removed = h_cool * max(s.struct_temp_k - t_in, 0.0)
            heat_removed_total += heat_removed * h
            energy_j += p_total * h

            # --- burnup / life ---------------------------------------------
            s.fissions += p_fission * h / U235_FISSION_ENERGY_J
            if s.n > 1e-3:
                s.burn_time_s += h
            s.elapsed_s += h
            self._since_rho_change += h
            remaining -= h

        if remaining > EPS:
            # Substep budget exhausted mid-transient: finish the remainder in one
            # implicit thermal step so the caller still gets a consistent state.
            logger.debug(
                "%s: substep budget exhausted, %.3g s folded into the last step",
                d.name,
                remaining,
            )
            self._advance_decay_heat(self.state.n, remaining)
            self._advance_xenon(self.state.n, remaining)
            self.state.elapsed_s += remaining

        s.fuel_dtdt_k_s = max_abs_dtdt if max_abs_dtdt > 0.0 else 0.0
        self._emit_safety_events(worst_rho, min_period_seen, max_abs_dtdt)
        return self._telemetry(heat_removed_total / max(dt_s, EPS))

    # --- sub-models ----------------------------------------------------------
    def _step_quasi_steady(
        self,
        n_demand: float,
        flow: float,
        heat_removal_w: float,
        dt_s: float,
        t_in: float,
        ua_flow: float,
    ) -> dict[str, float]:
        """Heat-balanced critical core for macro-steps longer than a kinetic period.

        Sets fission power to the demanded fraction (rate-limited by the thermal
        shock constraint), parks the drums at delayed-critical for the new
        temperature, and advances xenon, decay heat and burnup over ``dt_s``.
        """
        d = self.design
        s = self.state
        n_target = min(max(n_demand, 0.0), 1.05)
        if heat_removal_w > 0.0:
            n_target = min(n_target, 1.05 * heat_removal_w / max(d.rated_thermal_w, EPS))
        # Steady-state fuel temperature T_f = t_in + P * (x_f / U_fs + 1 / UA).
        # Cap power so a day-long step cannot walk the core through melt.
        if ua_flow > EPS and d.fuel_to_struct_ua_w_k > EPS:
            r_th = (
                d.fuel_power_fraction / d.fuel_to_struct_ua_w_k + 1.0 / ua_flow
            )
            p_temp = max(d.max_fuel_temp_k - t_in, 1.0) / r_th
            n_target = min(n_target, p_temp / max(d.rated_thermal_w, EPS))
        # Thermal-shock-limited slew of power: dT/dt cap * C / P_rated.
        max_dn = (d.max_fuel_dtdt_k_s * self._c_fuel / max(d.rated_thermal_w, EPS)) * dt_s
        max_dn = max(max_dn, 0.05)  # allow a 5% step even on short-ish dt
        if n_target > s.n:
            s.n = min(n_target, s.n + max_dn)
        else:
            s.n = max(n_target, s.n - max_dn)

        n_steps = int(min(12, max(1, math.ceil(dt_s / 600.0))))
        h = dt_s / n_steps
        heat_removed_total = 0.0
        t_f0 = s.fuel_temp_k
        for _ in range(n_steps):
            self._advance_decay_heat(s.n, h)
            self._advance_xenon(s.n, h)
            p_decay = self._decay_heat_w()
            p_fission = s.n * d.rated_thermal_w
            p_total = p_fission + p_decay
            h_cool = ua_flow
            dts = max(s.struct_temp_k - t_in, 1.0)
            if heat_removal_w >= 0.0 and h_cool * dts > heat_removal_w:
                h_cool = heat_removal_w / dts
            p_f = p_total * d.fuel_power_fraction
            p_s = p_total - p_f
            u = d.fuel_to_struct_ua_w_k
            a11 = self._c_fuel / h + u
            a12 = -u
            a21 = -u
            a22 = self._c_struct / h + u + h_cool
            b1 = self._c_fuel / h * s.fuel_temp_k + p_f
            b2 = self._c_struct / h * s.struct_temp_k + p_s + h_cool * t_in
            det = a11 * a22 - a12 * a21
            if abs(det) > EPS:
                s.fuel_temp_k = min(
                    max((b1 * a22 - a12 * b2) / det, 1.0),
                    d.max_fuel_temp_k,
                )
                s.struct_temp_k = min(
                    max((a11 * b2 - b1 * a21) / det, 1.0),
                    d.max_fuel_temp_k,
                )
            heat_removed_total += h_cool * max(s.struct_temp_k - t_in, 0.0) * h
            s.fissions += p_fission * h / U235_FISSION_ENERGY_J
            if s.n > 1e-3:
                s.burn_time_s += h
            s.elapsed_s += h

        s.fuel_dtdt_k_s = (s.fuel_temp_k - t_f0) / max(dt_s, EPS)
        s.drum_pcm = float(
            min(max(self.drum_for_critical(), -d.drum_worth_pcm * self._drum_scale),
                d.drum_worth_pcm * self._drum_scale)
        )
        s.omega = 0.0
        s.precursors = [
            b * s.n / (NEUTRON_GEN_TIME * lam)
            for b, lam in zip(_BETA_I, _LAMBDA_I)
        ]
        self._emit_safety_events(self._total_reactivity(), math.inf, abs(s.fuel_dtdt_k_s))
        return self._telemetry(heat_removed_total / max(dt_s, EPS))

    def _advance_decay_heat(self, power_fraction: float, dt: float) -> None:
        """Exact per-group integration of ``dH_j/dt = c_j p - alpha_j H_j``."""
        np.multiply(_DH_ALPHA, -dt, out=self._dh_buf)
        np.exp(self._dh_buf, out=self._dh_buf)
        g = self.state.decay_groups
        g *= self._dh_buf
        g += _DH_C * power_fraction * (1.0 - self._dh_buf) / _DH_ALPHA

    def _decay_heat_w(self) -> float:
        return float(
            np.dot(self.state.decay_groups, _DH_ALPHA) * self.design.rated_thermal_w
        )

    def _advance_xenon(self, power_fraction: float, dt: float) -> None:
        """Closed-form advance of the coupled I-135 / Xe-135 balance.

        ``dI/dt = gamma_I R - lambda_I I``
        ``dX/dt = gamma_X R + lambda_I I - (lambda_X + sigma_a phi) X``
        with ``R`` and ``phi`` both proportional to power, so the coefficients are
        constant over the substep and the 2x2 system integrates exactly.
        """
        s = self.state
        p = max(power_fraction, 0.0)
        lam_i = I135_DECAY_CONST
        a = XE135_DECAY_CONST + self._sigma_phi_rated * p
        i_eq = I135_YIELD * p / lam_i
        e_i = math.exp(-lam_i * dt)
        e_a = math.exp(-a * dt)
        i0 = s.iodine
        s.iodine = i_eq + (i0 - i_eq) * e_i
        forcing = XE135_YIELD * p + lam_i * i_eq
        x = s.xenon * e_a + forcing * (1.0 - e_a) / a
        diff = a - lam_i
        if abs(diff) > 1e-9:
            x += lam_i * (i0 - i_eq) * (e_i - e_a) / diff
        else:
            x += lam_i * (i0 - i_eq) * dt * e_a
        s.xenon = max(x, 0.0)

    def _emit_safety_events(
        self, rho: float, min_period_s: float, max_dtdt: float
    ) -> None:
        d = self.design
        s = self.state
        if rho >= BETA_EFF:
            self.events.append(
                Event(
                    "prompt_critical",
                    Severity.CRITICAL,
                    f"{d.name}: rho={rho * 1e5:.0f} pcm >= beta_eff",
                    rho * 1e5,
                )
            )
        if min_period_s < d.min_period_s:
            self.events.append(
                Event(
                    "short_period",
                    Severity.CRITICAL,
                    f"{d.name}: period {min_period_s:.2f} s < {d.min_period_s:.1f} s",
                    min_period_s,
                )
            )
        if max_dtdt > d.max_fuel_dtdt_k_s:
            self.events.append(
                Event(
                    "thermal_shock",
                    Severity.WARNING,
                    f"{d.name}: fuel dT/dt {max_dtdt:.1f} K/s over limit",
                    max_dtdt,
                )
            )
        if s.fuel_temp_k > d.max_fuel_temp_k:
            sev = Severity.CRITICAL
            self.events.append(
                Event(
                    "fuel_overtemp",
                    sev,
                    f"{d.name}: fuel {s.fuel_temp_k:.0f} K > {d.max_fuel_temp_k:.0f} K",
                    s.fuel_temp_k,
                )
            )
        if s.fuel_temp_k > d.fuel_melt_temp_k and not s.destroyed:
            s.destroyed = True
            s.scrammed = True
            self.events.append(
                Event(
                    "core_damage",
                    Severity.FATAL,
                    f"{d.name}: fuel exceeded melt at {s.fuel_temp_k:.0f} K",
                    s.fuel_temp_k,
                )
            )
        if self.burnup_fraction >= d.burnup_limit_fima:
            self.events.append(
                Event(
                    "burnup_limit",
                    Severity.WARNING,
                    f"{d.name}: {self.burnup_fraction * 100:.2f}% FIMA",
                    self.burnup_fraction,
                )
            )

    def _telemetry(self, heat_removed_w: float) -> dict[str, float]:
        s = self.state
        return {
            "fission_power_w": self.power_w,
            "decay_heat_w": self._decay_heat_w(),
            "thermal_power_w": self.thermal_power_w,
            "heat_removed_w": heat_removed_w,
            "fuel_temp_k": s.fuel_temp_k,
            "struct_temp_k": s.struct_temp_k,
            "reactivity_pcm": self.reactivity_pcm,
            "xenon_pcm": self.xenon_reactivity_pcm,
            "period_s": self.period_s,
            "dtdt_k_s": s.fuel_dtdt_k_s,
            "drum_pcm": s.drum_pcm,
            "burnup_fraction": self.burnup_fraction,
        }

    # --- read-only telemetry -------------------------------------------------
    @property
    def power_w(self) -> float:
        """Prompt fission power (excludes decay heat)."""
        return self.state.n * self.design.rated_thermal_w

    @property
    def power_fraction(self) -> float:
        return self.state.n

    @property
    def decay_heat_w(self) -> float:
        return self._decay_heat_w()

    @property
    def thermal_power_w(self) -> float:
        """Total heat the core is making right now: fission plus decay."""
        return self.power_w + self._decay_heat_w()

    @property
    def fuel_temperature_k(self) -> float:
        return self.state.fuel_temp_k

    @property
    def structure_temperature_k(self) -> float:
        return self.state.struct_temp_k

    @property
    def xenon_reactivity_pcm(self) -> float:
        """Negative reactivity currently held by the Xe-135 inventory."""
        return self._xenon_worth_per_unit * self.state.xenon

    @property
    def reactivity_pcm(self) -> float:
        return self._total_reactivity() * 1e5

    @property
    def drum_reactivity_pcm(self) -> float:
        return self.state.drum_pcm

    @property
    def drum_worth_pcm(self) -> float:
        return self.design.drum_worth_pcm * self._drum_scale

    @property
    def period_s(self) -> float:
        """Asymptotic reactor period. ``+inf`` when critical or subcritical."""
        w = self.state.omega
        if w <= EPS:
            return math.inf
        return 1.0 / w

    @property
    def burnup_fraction(self) -> float:
        """Fissions per initial fissile atom (FIMA)."""
        return self.state.fissions / self._total_fissile_atoms

    @property
    def fuel_dtdt_k_s(self) -> float:
        return self.state.fuel_dtdt_k_s

    @property
    def destroyed(self) -> bool:
        return self.state.destroyed

    # --- derived operational quantities --------------------------------------
    def restart_margin_pcm(self) -> float:
        """Drum reactivity left over after paying every held-down poison.

        Positive means the core can be taken critical *right now* from a cold
        start; negative is xenon precluded start -- the operator must wait for the
        Xe-135 to decay. This is what turns the xenon transient into a scheduling
        problem for the agent.
        """
        d = self.design
        cold = d.initial_temperature_k
        return (
            self.drum_worth_pcm
            + d.excess_reactivity_pcm
            + self._doppler_pcm(cold)
            + self._moderator_pcm(cold)
            + self.xenon_reactivity_pcm
            + self._burnup_pcm()
        )

    def drum_for_critical(self, fuel_temp_k: float | None = None) -> float:
        """Drum pcm that would make net reactivity zero at the given fuel temperature.

        Used by the owning propulsion system as a *setpoint*, not as a license to
        command the full drum worth in one macro-step. A day-long environment
        step would otherwise slew the drums onto a prompt-critical insertion.
        """
        d = self.design
        s = self.state
        t_fuel = s.fuel_temp_k if fuel_temp_k is None else float(fuel_temp_k)
        held = (
            d.excess_reactivity_pcm
            + self._doppler_pcm(t_fuel)
            + self._moderator_pcm(s.struct_temp_k)
            + self.xenon_reactivity_pcm
            + self._burnup_pcm()
        )
        return -held

    def can_restart(self) -> bool:
        return self.restart_margin_pcm() > 0.0 and not self.state.destroyed

    def scram(self) -> None:
        """Latch a scram. Cleared only by :meth:`clear_scram`."""
        self.state.scrammed = True

    def clear_scram(self) -> None:
        self.state.scrammed = False

    # --- constraints ---------------------------------------------------------
    def constraint_margins(self) -> dict[str, float]:
        """Signed, limit-normalised margins. ``>= 0`` is safe."""
        d = self.design
        s = self.state
        rho = self._total_reactivity()
        w = s.omega
        period_margin = 1.0 if w <= EPS else 1.0 - d.min_period_s * w
        return {
            "fuel_temperature": max(
                (d.max_fuel_temp_k - s.fuel_temp_k) / d.max_fuel_temp_k, -1.0
            ),
            "prompt_critical": max((BETA_EFF - rho) / BETA_EFF, -1.0),
            "reactor_period": max(period_margin, -1.0),
            "thermal_stress": max(1.0 - abs(s.fuel_dtdt_k_s) / d.max_fuel_dtdt_k_s, -1.0),
            "burnup": 1.0 - self.burnup_fraction / max(d.burnup_limit_fima, EPS),
        }

    def constraints(self) -> ConstraintReport:
        m = self.constraint_margins()
        return ConstraintReport(
            names=tuple(m),
            margins=np.fromiter(m.values(), dtype=np.float64, count=len(m)),
        )


__all__ = [
    "FissionReactor",
    "ReactorDesign",
    "ReactorState",
    "solve_inhour",
    "way_wigner_fraction",
]
