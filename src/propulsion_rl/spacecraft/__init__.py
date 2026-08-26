"""Spacecraft physics: trajectory propagation, orbital elements, power bus.

This subpackage is propulsion-agnostic and mission-agnostic on purpose. It
answers three questions and nothing else:

* :mod:`~propulsion_rl.spacecraft.dynamics` -- where does the vehicle go, given
  a thrust command and whatever gravity the mission supplies?
* :mod:`~propulsion_rl.spacecraft.orbit` -- what orbit is that, and what would
  the classical closed-form transfers cost?
* :mod:`~propulsion_rl.spacecraft.power` -- how much electrical power does the
  bus actually have to offer, here, now, this far from the Sun?

plus :mod:`~propulsion_rl.spacecraft.vehicle` for the vehicle's own
configuration and its slice of the observation vector.

Nothing here imports :mod:`propulsion_rl.missions` at module scope; the
integrator reaches a mission only through the ``mu`` / ``gravity`` duck-type it
is handed, so mission modules are free to import this one.
"""

from __future__ import annotations

from .dynamics import (
    DivergedState,
    PropagationStatus,
    inertial_to_rtn,
    is_diverged,
    propagate,
    propagate_with_status,
    rtn_basis,
    rtn_to_inertial,
    specific_energy,
)
from .orbit import (
    OrbitalElements,
    cartesian_to_elements,
    circular_speed,
    edelbaum_delta_v,
    edelbaum_time_of_flight,
    elements_to_cartesian,
    hohmann_delta_v,
    hohmann_transfer_time,
    lambert_delta_v,
    lambert_solve,
    orbital_period,
    phase_angle,
    planet_state,
    synodic_period,
)
from .power import RTG, FixedPower, PowerBus, PowerSource, SolarArray
from .vehicle import (
    VehicleConfig,
    build_initial_state,
    mass_budget,
    vehicle_observation,
    vehicle_observation_labels,
)

__all__ = [
    # dynamics
    "propagate",
    "propagate_with_status",
    "rtn_basis",
    "rtn_to_inertial",
    "inertial_to_rtn",
    "is_diverged",
    "specific_energy",
    "DivergedState",
    "PropagationStatus",
    # orbit
    "OrbitalElements",
    "cartesian_to_elements",
    "elements_to_cartesian",
    "edelbaum_delta_v",
    "edelbaum_time_of_flight",
    "hohmann_delta_v",
    "hohmann_transfer_time",
    "lambert_solve",
    "lambert_delta_v",
    "synodic_period",
    "phase_angle",
    "planet_state",
    "circular_speed",
    "orbital_period",
    # power
    "PowerSource",
    "SolarArray",
    "FixedPower",
    "RTG",
    "PowerBus",
    # vehicle
    "VehicleConfig",
    "build_initial_state",
    "vehicle_observation",
    "vehicle_observation_labels",
    "mass_budget",
]
