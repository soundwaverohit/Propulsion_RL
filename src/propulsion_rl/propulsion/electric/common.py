"""Shared electrostatic-thruster physics.

Everything in here is common to Hall-effect and gridded-ion engines: the
beam-velocity / specific-impulse relations, the four-factor efficiency
decomposition, the Child-Langmuir space-charge limit, a lumped-capacitance
thermal node that survives multi-hour environment steps, and a wear
accumulator for throughput-limited life.

References
----------
Goebel, D. M. and Katz, I., *Fundamentals of Electric Propulsion: Ion and Hall
    Thrusters*, JPL Space Science and Technology Series, Wiley, 2008.
    Ch. 2 (thrust/Isp/efficiency decomposition), Ch. 5 (ion optics and the
    Child-Langmuir limit).
Brophy, J. R., "Ion Thruster Performance Model", NASA CR-174810, 1984
    (discharge-chamber eV/ion model).
Yamamura, Y. and Tawara, H., "Energy dependence of ion-induced sputtering
    yields from monatomic solids at normal incidence", At. Data Nucl. Data
    Tables 62, 1996 (threshold sputter-yield behaviour).

Sign and unit conventions
-------------------------
* All SI, all magnitudes positive; accelerator-grid voltages are handed in as
  magnitudes, never as the negative bias.
* "beam" always means the extracted ion beam, "discharge" the plasma source.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ...core.constants import ELEMENTARY_CHARGE, EPS, G0, STEFAN_BOLTZMANN

logger = logging.getLogger(__name__)

# ``core.constants`` intentionally carries no vacuum permittivity -- it has no
# other user in the package. The Child-Langmuir law needs one, so it is defined
# here (CODATA 2018) rather than shadowing anything in the contract module.
VACUUM_PERMITTIVITY = 8.8541878128e-12   # F/m

#: Empirical fraction of the ideal one-dimensional Child-Langmuir current a real
#: multi-aperture grid set can extract before direct ion impingement begins.
#: Real optics reach the perveance limit well below the planar-diode value
#: because the upstream sheath is curved and the aspect ratio is finite;
#: Goebel & Katz Ch. 5 puts usable operation at roughly a quarter to a third of
#: the 1-D value for NSTAR/NEXT-class geometry.
DEFAULT_PERVEANCE_FRACTION = 0.30


# --- Beam kinematics ---------------------------------------------------------
def beam_velocity(net_voltage_v: float, ion_mass_kg: float) -> float:
    """Singly-charged ion exhaust velocity from the net accelerating voltage.

    ``v_beam = sqrt(2 q V_net / m_i)`` -- energy conservation for an ion falling
    through ``V_net`` (Goebel & Katz eq. 2.3-4). ``V_net`` is the *net* voltage
    seen by the ion, i.e. the discharge or beam supply voltage less the plasma
    potential and ionisation overhead, not the raw supply setting.

    Returns 0.0 for a non-positive voltage rather than raising: the step loop
    must never blow up on a commanded-off thruster.
    """
    if net_voltage_v <= 0.0 or ion_mass_kg <= 0.0:
        return 0.0
    return math.sqrt(2.0 * ELEMENTARY_CHARGE * net_voltage_v / ion_mass_kg)


def thrust_correction_alpha(doubles_ratio: float) -> float:
    """Thrust correction for doubly-charged ions, ``alpha_t``.

    ``alpha_t = (1 + (I++/I+)/sqrt(2)) / (1 + I++/I+)`` (Goebel & Katz
    eq. 2.3-14). A doubly-charged ion falls through the same potential but
    carries twice the charge, so it leaves at ``sqrt(2) v_beam`` while
    contributing two units of beam current -- less thrust per amp than a single.
    """
    r = max(0.0, doubles_ratio)
    return (1.0 + r / math.sqrt(2.0)) / (1.0 + r)


def mass_correction_beta(doubles_ratio: float) -> float:
    """Mass-flow correction for doubly-charged ions, ``beta_m``.

    ``beta_m = (1 + (I++/I+)/2) / (1 + I++/I+)``. Beam ion mass flow is
    ``(m_i/q) * I_beam * beta_m``: a double carries two charges per atom, so a
    given beam current corresponds to *less* mass than the singly-charged
    estimate. This is why doubles raise Isp even as they cost thrust per amp.
    """
    r = max(0.0, doubles_ratio)
    return (1.0 + 0.5 * r) / (1.0 + r)


def divergence_efficiency(divergence_half_angle_rad: float, doubles_ratio: float) -> float:
    """Beam-vector efficiency: divergence plus charge-state utilisation.

    ``eta_div = cos^2(theta) * alpha_t^2 / beta_m``.

    The ``cos^2`` is the usual momentum-weighted divergence loss. The
    ``alpha_t^2 / beta_m`` group is the charge-utilisation term that falls out
    of writing the jet power with the *mass-corrected* flow; it equals 1 for a
    pure singly-charged beam and is bounded above by 1 for every ``I++/I+``,
    reaching a minimum near ``I++/I+ ~ 0.5``. Folding it in here is what makes
    ``eta_total = eta_mass_util * eta_beam * eta_div * eta_ppu`` exactly
    consistent with ``T = 2 eta_total P / (g0 Isp)``.
    """
    c = math.cos(divergence_half_angle_rad)
    alpha = thrust_correction_alpha(doubles_ratio)
    beta = mass_correction_beta(doubles_ratio)
    return (c * c) * (alpha * alpha) / max(beta, EPS)


def isp_from_beam(
    net_voltage_v: float,
    ion_mass_kg: float,
    mass_utilisation: float,
    divergence_half_angle_rad: float,
    doubles_ratio: float,
) -> float:
    """Specific impulse (s) of an electrostatic beam.

    ``Isp = alpha_t cos(theta) * eta_m * v_beam / (g0 * beta_m)``

    Derivation: thrust is ``alpha_t cos(theta) (m_i/q) v_b I_b`` and total
    propellant flow is ``(m_i/q) I_b beta_m / eta_m``, where ``eta_m`` is the
    beam ion mass flow over the *total* flow (anode + cathode + neutraliser).
    """
    v_b = beam_velocity(net_voltage_v, ion_mass_kg)
    if v_b <= 0.0:
        return 0.0
    gamma = thrust_correction_alpha(doubles_ratio) * math.cos(divergence_half_angle_rad)
    beta = mass_correction_beta(doubles_ratio)
    return gamma * max(0.0, mass_utilisation) * v_b / (G0 * max(beta, EPS))


def thrust_from_power(efficiency: float, power_w: float, isp_s: float) -> float:
    """``T = 2 eta P / (g0 Isp)`` -- the jet-power identity.

    Exact by construction for any consistent (eta, P, Isp) triple, because
    ``eta`` is defined as jet power over input power and jet power is
    ``0.5 mdot ve^2 = 0.5 T ve``.
    """
    denom = G0 * isp_s
    if denom <= EPS or power_w <= 0.0:
        return 0.0
    return 2.0 * efficiency * power_w / denom


# --- Efficiency decomposition ------------------------------------------------
@dataclass(frozen=True, slots=True)
class EfficiencyBreakdown:
    """The four-factor total-efficiency decomposition.

    ``eta_total = eta_mass_util * eta_beam * eta_divergence * eta_ppu``

    * ``eta_mass_util`` -- beam ion mass flow / total propellant flow.
    * ``eta_beam`` -- beam power / thruster input power, i.e. the electrical
      share that ends up in the accelerated ions. For a Hall thruster this is
      the product of voltage utilisation ``V_b/V_d`` and current utilisation
      ``I_b/I_d``; for a gridded ion engine it is ``V_b I_b`` over the sum of
      beam, discharge, accelerator and neutraliser power.
    * ``eta_divergence`` -- :func:`divergence_efficiency`.
    * ``eta_ppu`` -- thruster input power / bus power draw. Absorbs the PPU
      conversion loss *and* the standing keeper/magnet parasitics, so the jet
      identity closes against the number the spacecraft actually pays.
    """

    eta_mass_util: float = 0.0
    eta_beam: float = 0.0
    eta_divergence: float = 0.0
    eta_ppu: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.eta_mass_util
            * self.eta_beam
            * self.eta_divergence
            * self.eta_ppu
        )


def total_efficiency(
    mass_utilisation: float,
    eta_beam: float,
    divergence_half_angle_rad: float,
    doubles_ratio: float,
    eta_ppu: float,
) -> EfficiencyBreakdown:
    """Assemble an :class:`EfficiencyBreakdown` from physical sub-efficiencies."""
    return EfficiencyBreakdown(
        eta_mass_util=max(0.0, mass_utilisation),
        eta_beam=max(0.0, eta_beam),
        eta_divergence=divergence_efficiency(divergence_half_angle_rad, doubles_ratio),
        eta_ppu=max(0.0, eta_ppu),
    )


# --- Ion optics --------------------------------------------------------------
def child_langmuir_current_density(
    total_voltage_v: float, effective_gap_m: float, ion_mass_kg: float
) -> float:
    """Space-charge-limited ion current density across an accelerating gap.

    ``j_max = (4/9) eps0 sqrt(2q/m_i) V_total^1.5 / d^2``  (A/m^2)

    ``V_total`` is the full potential across the gap, i.e. beam voltage plus the
    magnitude of the accelerator-grid bias. ``d`` is the *effective*
    acceleration length ``l_e = sqrt(l_g^2 + d_s^2/4)`` for a real two-grid set
    (Goebel & Katz eq. 5.3-6), not the bare inter-grid spacing.
    """
    if total_voltage_v <= 0.0 or effective_gap_m <= EPS or ion_mass_kg <= 0.0:
        return 0.0
    charge_factor = math.sqrt(2.0 * ELEMENTARY_CHARGE / ion_mass_kg)
    return (
        (4.0 / 9.0)
        * VACUUM_PERMITTIVITY
        * charge_factor
        * total_voltage_v**1.5
        / (effective_gap_m * effective_gap_m)
    )


def perveance_limited_current(
    total_voltage_v: float,
    effective_gap_m: float,
    ion_mass_kg: float,
    active_area_m2: float,
    perveance_fraction: float = DEFAULT_PERVEANCE_FRACTION,
) -> float:
    """Maximum extractable beam current (A) before direct impingement.

    The ideal one-dimensional Child-Langmuir density scaled by the open grid
    area and by :data:`DEFAULT_PERVEANCE_FRACTION`. Beyond this the beamlets
    over-focus, ions strike the accelerator grid directly, and the engine is
    "perveance limited" -- the classic low-voltage / high-flow failure mode of
    gridded optics.
    """
    j_max = child_langmuir_current_density(total_voltage_v, effective_gap_m, ion_mass_kg)
    return j_max * max(0.0, active_area_m2) * max(0.0, perveance_fraction)


# --- Sputtering --------------------------------------------------------------
def sputter_yield_factor(energy_ev: float, threshold_ev: float) -> float:
    """Threshold-dominated sputter-yield shape, ``(sqrt(E) - sqrt(E_th))^2``.

    Near threshold the yield of a heavy ion on a ceramic or refractory metal
    rises roughly as the square of the excess of ``sqrt(E)`` over
    ``sqrt(E_th)`` (Bohdansky / Yamamura-Tawara threshold form). Returned
    unnormalised: callers scale it against a calibrated reference point, so
    only the *shape* -- the steep rise of erosion with ion energy -- carries
    physical weight here.
    """
    if energy_ev <= threshold_ev or threshold_ev < 0.0:
        return 0.0
    d = math.sqrt(energy_ev) - math.sqrt(threshold_ev)
    return d * d


# --- Thermal -----------------------------------------------------------------
@dataclass(slots=True)
class ThermalNode:
    """Lumped-capacitance radiating node.

    ``m cp dT/dt = Q_in - eps sigma A (T^4 - T_sink^4) - Q_cool``

    The environment hands out steps of hours while the radiative time constant
    of a thruster body is tens of minutes, so :meth:`step` sub-steps and uses a
    linearly-implicit (Rosenbrock-style) Euler update::

        T <- T + h Q(T) / (C + 4 h eps sigma A T^3)

    The denominator is the exact linearisation of the quartic sink, which makes
    the update unconditionally stable and monotone towards equilibrium for any
    ``h`` -- an explicit Euler step at dt = 6 h would oscillate to infinity.
    """

    mass_kg: float
    cp_j_kg_k: float
    area_m2: float
    emissivity: float = 0.85
    temperature_k: float = 293.0
    max_substeps: int = 8

    @property
    def capacitance_j_k(self) -> float:
        return max(self.mass_kg * self.cp_j_kg_k, EPS)

    @property
    def _radiative_conductance(self) -> float:
        return self.emissivity * STEFAN_BOLTZMANN * self.area_m2

    def net_heat_w(self, q_in_w: float, q_cool_w: float, t_sink_k: float) -> float:
        """Instantaneous net heat into the node at the current temperature."""
        k = self._radiative_conductance
        t = self.temperature_k
        return q_in_w - q_cool_w - k * (t**4 - t_sink_k**4)

    def equilibrium_temperature_k(
        self, q_in_w: float, q_cool_w: float, t_sink_k: float = 3.0
    ) -> float:
        """Steady-state temperature for a constant heat load."""
        k = self._radiative_conductance
        if k <= EPS:
            return self.temperature_k
        q = max(0.0, q_in_w - q_cool_w)
        return (q / k + t_sink_k**4) ** 0.25

    def step(
        self, dt_s: float, q_in_w: float, q_cool_w: float, t_sink_k: float = 3.0
    ) -> float:
        """Advance the temperature by ``dt_s`` and return the new value."""
        if dt_s <= 0.0:
            return self.temperature_k
        cap = self.capacitance_j_k
        k = self._radiative_conductance

        # Radiative time constant at the current temperature; sub-step to a
        # fraction of it for accuracy, capped so a 10-year coast stays cheap.
        slope = 4.0 * k * self.temperature_k**3
        tau = cap / slope if slope > EPS else dt_s
        n = int(min(float(self.max_substeps), max(1.0, math.ceil(dt_s / max(0.5 * tau, EPS)))))
        h = dt_s / n

        t = self.temperature_k
        t_sink4 = t_sink_k**4
        # Linearising T^4 around a cold body makes a day-long first step overshoot
        # the equilibrium by tens of thousands of kelvin. Bound the update.
        t_eq = self.equilibrium_temperature_k(q_in_w, q_cool_w, t_sink_k)
        heating = (q_in_w - q_cool_w) >= 0.0
        for _ in range(n):
            q_net = q_in_w - q_cool_w - k * (t**4 - t_sink4)
            t = t + h * q_net / (cap + 4.0 * h * k * max(t, 1.0) ** 3)
            if heating:
                t = min(max(t, t_sink_k), t_eq)
            else:
                t = max(t, t_eq, t_sink_k)
        self.temperature_k = t
        return t


# --- Wear --------------------------------------------------------------------
@dataclass(slots=True)
class WearAccumulator:
    """Multi-mechanism life bookkeeping for an electric thruster.

    Electric propulsion is retired by whichever of three clocks runs out first:

    * **burn time** against the qualification test duration,
    * **propellant throughput** (kg processed) -- the metric life tests are
      actually scored on, because erosion tracks total charge through the
      channel rather than wall-clock hours,
    * **erosion depth** into the channel wall (Hall) or the accelerator grid web
      (gridded ion), which is the mechanism the two clocks above stand in for
      and the only one the agent can move with its voltage command.

    ``fraction`` is the max of the three, so a policy that keeps voltage low
    still cannot outrun the throughput budget.
    """

    qualified_life_s: float
    max_throughput_kg: float
    erosion_depth_limit_m: float = 1.0
    burn_time_s: float = 0.0
    throughput_kg: float = 0.0
    erosion_depth_m: float = 0.0
    restarts: int = 0
    _last_erosion_rate_m_s: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        self.burn_time_s = 0.0
        self.throughput_kg = 0.0
        self.erosion_depth_m = 0.0
        self.restarts = 0
        self._last_erosion_rate_m_s = 0.0

    def accumulate(
        self, dt_s: float, mdot_kg_s: float, erosion_rate_m_s: float
    ) -> None:
        """Add one firing interval. Call only while the thruster is on."""
        if dt_s <= 0.0:
            return
        self.burn_time_s += dt_s
        self.throughput_kg += max(0.0, mdot_kg_s) * dt_s
        self.erosion_depth_m += max(0.0, erosion_rate_m_s) * dt_s
        self._last_erosion_rate_m_s = max(0.0, erosion_rate_m_s)

    @property
    def time_fraction(self) -> float:
        return self.burn_time_s / max(self.qualified_life_s, EPS)

    @property
    def throughput_fraction(self) -> float:
        return self.throughput_kg / max(self.max_throughput_kg, EPS)

    @property
    def erosion_fraction(self) -> float:
        return self.erosion_depth_m / max(self.erosion_depth_limit_m, EPS)

    @property
    def fraction(self) -> float:
        """Worst of the three life clocks, clipped to [0, 1]."""
        f = max(self.time_fraction, self.throughput_fraction, self.erosion_fraction)
        return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)

    @property
    def limiting_mechanism(self) -> str:
        pairs = (
            ("time", self.time_fraction),
            ("throughput", self.throughput_fraction),
            ("erosion", self.erosion_fraction),
        )
        return max(pairs, key=lambda p: p[1])[0]

    @property
    def throughput_remaining_kg(self) -> float:
        return max(0.0, self.max_throughput_kg - self.throughput_kg)

    def remaining_life_s(self, mdot_kg_s: float) -> float:
        """Seconds left at the current operating point, over all three clocks."""
        t_time = max(0.0, self.qualified_life_s - self.burn_time_s)
        t_thru = (
            self.throughput_remaining_kg / mdot_kg_s
            if mdot_kg_s > EPS
            else math.inf
        )
        rate = self._last_erosion_rate_m_s
        t_ero = (
            max(0.0, self.erosion_depth_limit_m - self.erosion_depth_m) / rate
            if rate > EPS
            else math.inf
        )
        return min(t_time, t_thru, t_ero)


__all__ = [
    "VACUUM_PERMITTIVITY",
    "DEFAULT_PERVEANCE_FRACTION",
    "EfficiencyBreakdown",
    "ThermalNode",
    "WearAccumulator",
    "beam_velocity",
    "child_langmuir_current_density",
    "divergence_efficiency",
    "isp_from_beam",
    "mass_correction_beta",
    "perveance_limited_current",
    "sputter_yield_factor",
    "thrust_correction_alpha",
    "thrust_from_power",
    "total_efficiency",
]
