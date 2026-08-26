"""Physical constants and unit conversions.

Single source of truth for every number that is not a design choice.
All values SI unless the name says otherwise. Do not redefine these locally.
"""

from __future__ import annotations

import math

# --- Universal ---------------------------------------------------------------
G0 = 9.80665                  # m/s^2, standard gravity (Isp definition)
BOLTZMANN = 1.380649e-23      # J/K
ELEMENTARY_CHARGE = 1.602176634e-19   # C
AVOGADRO = 6.02214076e23      # 1/mol
R_UNIVERSAL = 8.314462618     # J/(mol K)
STEFAN_BOLTZMANN = 5.670374419e-8     # W/(m^2 K^4)
AMU = 1.66053906660e-27       # kg

# --- Astrodynamics -----------------------------------------------------------
MU_SUN = 1.32712440018e20     # m^3/s^2
MU_EARTH = 3.986004418e14     # m^3/s^2
MU_MARS = 4.282837e13         # m^3/s^2
AU = 1.495978707e11           # m
R_EARTH = 6.371e6             # m
R_MARS = 3.3895e6             # m
EARTH_SMA = 1.000_001_018 * AU        # m, semi-major axis about the Sun
MARS_SMA = 1.523_679 * AU             # m
EARTH_ORBIT_PERIOD = 365.256363004 * 86400.0   # s
MARS_ORBIT_PERIOD = 686.980 * 86400.0          # s
GEO_RADIUS = 4.2164e7         # m
LEO_RADIUS = R_EARTH + 400e3  # m, 400 km reference circular orbit

SOLAR_CONSTANT_1AU = 1361.0   # W/m^2

# --- Time --------------------------------------------------------------------
DAY = 86400.0                 # s
HOUR = 3600.0                 # s
YEAR = 365.25 * DAY           # s

# --- Propellants -------------------------------------------------------------
# Ion masses for electrostatic thrusters.
M_XENON = 131.293 * AMU       # kg
M_KRYPTON = 83.798 * AMU      # kg
M_ARGON = 39.948 * AMU        # kg
M_IODINE = 126.904 * AMU      # kg

# First ionization energies (eV) -- ionization cost floor for EP efficiency.
IONIZATION_EV_XENON = 12.13
IONIZATION_EV_KRYPTON = 14.00
IONIZATION_EV_ARGON = 15.76
IONIZATION_EV_IODINE = 10.45

# Molar masses for thermal (nuclear) propellants.
M_MOLAR_H2 = 2.01588e-3       # kg/mol
GAMMA_H2 = 1.41               # ratio of specific heats, cold H2
H2_DENSITY_LIQUID = 70.85     # kg/m^3 at 20 K
H2_LATENT_HEAT = 448e3        # J/kg, heat of vaporization

# Hydrogen dissociates above ~2500 K, which raises effective Isp. Captured in
# the NTP model via an effective molar mass correction, not here.

# --- Nuclear -----------------------------------------------------------------
U235_FISSION_ENERGY_J = 3.204e-11     # J per fission (~200 MeV)
XE135_YIELD = 0.061                   # cumulative fission yield of Xe-135
I135_YIELD = 0.0639                   # cumulative fission yield of I-135
XE135_DECAY_CONST = math.log(2) / (9.14 * HOUR)   # 1/s
I135_DECAY_CONST = math.log(2) / (6.57 * HOUR)    # 1/s
XE135_SIGMA_A = 2.65e-18              # m^2, thermal absorption cross-section
NEUTRON_GEN_TIME = 1e-4               # s, prompt neutron generation time (thermal)
BETA_EFF = 0.0065                     # delayed neutron fraction, U-235
# Six-group delayed neutron data, U-235 thermal fission.
DELAYED_BETA = (0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273)
DELAYED_LAMBDA = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)   # 1/s
DECAY_HEAT_FRACTION = 0.066           # fraction of full power immediately post-scram

# --- Numerical tolerances ----------------------------------------------------
EPS = 1e-12
TINY_MASS_KG = 1e-6           # below this a tank is considered dry


def isp_to_ve(isp_s: float) -> float:
    """Specific impulse (s) -> effective exhaust velocity (m/s)."""
    return isp_s * G0


def ve_to_isp(ve_m_s: float) -> float:
    """Effective exhaust velocity (m/s) -> specific impulse (s)."""
    return ve_m_s / G0


def solar_flux(distance_m: float) -> float:
    """Solar irradiance (W/m^2) at a heliocentric distance in metres."""
    return SOLAR_CONSTANT_1AU * (AU / max(distance_m, EPS)) ** 2
