"""Readable altitude-stepped mission optimization for preliminary sizing.

The model intentionally stops before propeller selection. Forward-flight
electrical power is estimated from aerodynamic propulsive power and explicit
efficiency assumptions:

    P_propulsive = T_required V
    P_electrical = P_propulsive / (eta_prop eta_motor eta_ESC)

Hover and transition use preliminary total-aircraft electrical powers. A
single affordable electrical-power limit is enforced throughout the mission.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .common import isa
from .phases.phase13_transition import phase13_transition_blending


@dataclass(frozen=True)
class AircraftPerformance:
    """Aircraft inputs used by the wing-borne performance model."""

    mass_kg: float
    wing_area_m2: float
    CD0: float
    aspect_ratio: float
    oswald_efficiency: float
    CL_max: float = 1.30
    CL_limit_fraction: float = 0.90
    CL_allowed_override: Optional[float] = None
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        positive = {
            "mass_kg": self.mass_kg,
            "wing_area_m2": self.wing_area_m2,
            "CD0": self.CD0,
            "aspect_ratio": self.aspect_ratio,
            "oswald_efficiency": self.oswald_efficiency,
            "CL_max": self.CL_max,
            "gravity_m_s2": self.gravity_m_s2,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError("Aircraft performance inputs must be positive.")
        if not 0.0 < self.CL_limit_fraction <= 1.0:
            raise ValueError("CL_limit_fraction must lie in 0 < fraction <= 1.")
        if self.CL_allowed_override is not None:
            if not 0.0 < self.CL_allowed_override <= self.CL_max:
                raise ValueError("CL_allowed_override must lie in 0 < CL <= CL_max.")

    @property
    def weight_N(self) -> float:
        return self.mass_kg * self.gravity_m_s2

    @property
    def induced_drag_factor(self) -> float:
        return 1.0 / (
            np.pi * self.aspect_ratio * self.oswald_efficiency
        )

    @property
    def permitted_CL(self) -> float:
        if self.CL_allowed_override is not None:
            return float(self.CL_allowed_override)
        return self.CL_limit_fraction * self.CL_max


@dataclass(frozen=True)
class MissionPowerAssumptions:
    """Preliminary propulsion assumptions used before component selection."""

    propulsive_efficiency: float = 0.75
    motor_efficiency: float = 0.90
    esc_efficiency: float = 0.95
    max_affordable_electrical_power_W: float = 20000.0
    preliminary_hover_power_W: float = 14000.0
    preliminary_transition_power_W: float = 8000.0

    def __post_init__(self) -> None:
        efficiencies = (
            self.propulsive_efficiency,
            self.motor_efficiency,
            self.esc_efficiency,
        )
        if any(not 0.0 < value <= 1.0 for value in efficiencies):
            raise ValueError("All propulsion efficiencies must lie in 0 < eta <= 1.")
        powers = (
            self.max_affordable_electrical_power_W,
            self.preliminary_hover_power_W,
            self.preliminary_transition_power_W,
        )
        if any(value <= 0.0 for value in powers):
            raise ValueError("Mission power assumptions must be positive.")

    @property
    def total_forward_efficiency(self) -> float:
        return (
            self.propulsive_efficiency
            * self.motor_efficiency
            * self.esc_efficiency
        )


@dataclass(frozen=True)
class MissionProfileConfig:
    """Mission geometry, timing, and integration settings."""

    target_altitude_m: float = 6000.0
    horizontal_range_m: float = 6000.0
    outbound_time_budget_s: float = 600.0
    hover_duration_s: float = 300.0
    altitude_step_m: float = 100.0
    vertical_takeoff_height_m: float = 20.0
    vertical_takeoff_rate_m_s: float = 2.0
    transition_accel_m_s2: float = 1.0
    transition_blend_start_fraction: float = 0.50
    transition_blend_end_fraction: float = 1.20
    transition_cruise_margin_fraction: float = 0.05
    transition_sample_count: int = 9
    minimum_cruise_true_airspeed_m_s: Optional[float] = None
    max_transition_complete_speed_m_s: Optional[float] = 20.0
    max_stall_EAS_m_s: Optional[float] = None
    minimum_power_margin_fraction: float = 0.05
    max_fixed_wing_climb_angle_deg: float = 30.0
    allow_spiral_climb: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.target_altitude_m,
            self.outbound_time_budget_s,
            self.altitude_step_m,
            self.vertical_takeoff_height_m,
            self.vertical_takeoff_rate_m_s,
            self.transition_accel_m_s2,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Mission geometry, time, and integration inputs must be positive.")
        if self.horizontal_range_m < 0.0:
            raise ValueError("horizontal_range_m must be non-negative.")
        if self.hover_duration_s < 0.0:
            raise ValueError("hover_duration_s must be non-negative.")
        if self.vertical_takeoff_height_m >= self.target_altitude_m:
            raise ValueError("Vertical takeoff height must remain below mission altitude.")
        if self.transition_sample_count < 3:
            raise ValueError("transition_sample_count must be at least 3.")
        if (
            self.max_transition_complete_speed_m_s is not None
            and self.max_transition_complete_speed_m_s <= 0.0
        ):
            raise ValueError("max_transition_complete_speed_m_s must be positive when provided.")
        if self.max_stall_EAS_m_s is not None and self.max_stall_EAS_m_s <= 0.0:
            raise ValueError("max_stall_EAS_m_s must be positive when provided.")
        if not 0.0 <= self.minimum_power_margin_fraction < 1.0:
            raise ValueError("minimum_power_margin_fraction must lie in 0 <= fraction < 1.")
        if not 0.0 < self.max_fixed_wing_climb_angle_deg < 90.0:
            raise ValueError("max_fixed_wing_climb_angle_deg must lie between 0 and 90 degrees.")


def equivalent_to_true_airspeed(
        equivalent_airspeed_m_s: float,
        density_kg_m3: float,
        sea_level_density_kg_m3: float = 1.225) -> float:
    """Convert EAS to TAS using ``V_TAS = V_EAS sqrt(rho_0 / rho)``."""
    if equivalent_airspeed_m_s <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("Airspeed and density must be positive.")
    return float(
        equivalent_airspeed_m_s
        * np.sqrt(sea_level_density_kg_m3 / density_kg_m3)
    )


def _required_power_margin_W(
        power: MissionPowerAssumptions,
        config: MissionProfileConfig) -> float:
    return float(
        config.minimum_power_margin_fraction
        * power.max_affordable_electrical_power_W
    )


def _power_margin_failure(
        context: str,
        power_margin_W: float,
        required_power_margin_W: float) -> Optional[str]:
    if power_margin_W >= required_power_margin_W - 1e-6:
        return None
    return (
        f"Required electrical power margin not met {context}: "
        f"{power_margin_W:.1f} W available, "
        f"{required_power_margin_W:.1f} W required."
    )


def aerodynamic_speed_limits(
        aircraft: AircraftPerformance,
        config: MissionProfileConfig,
        isa_fn: Callable = isa) -> Dict:
    """Calculate aerodynamic and transition-driven minimum operating speeds."""
    rho_reference, _, _, _ = isa_fn(0.0)
    rho_transition, _, _, _ = isa_fn(config.vertical_takeoff_height_m)
    rho_target, _, _, _ = isa_fn(config.target_altitude_m)

    def _stall_speed(density: float, CL: float) -> float:
        return float(np.sqrt(
            2.0 * aircraft.weight_N
            / (density * aircraft.wing_area_m2 * CL)
        ))

    stall_EAS = _stall_speed(rho_reference, aircraft.CL_max)
    minimum_climb_EAS = _stall_speed(rho_reference, aircraft.permitted_CL)
    stall_TAS_transition = _stall_speed(rho_transition, aircraft.CL_max)
    minimum_TAS_transition = (
        (1.0 + config.transition_cruise_margin_fraction)
        * config.transition_blend_end_fraction
        * stall_TAS_transition
    )
    minimum_TAS_aerodynamic = _stall_speed(
        rho_target, aircraft.permitted_CL
    )
    minimum_cruise_TAS = max(
        minimum_TAS_transition,
        minimum_TAS_aerodynamic,
        config.minimum_cruise_true_airspeed_m_s or 0.0,
    )
    return {
        "stall_EAS_m_s": float(stall_EAS),
        "minimum_climb_EAS_m_s": float(minimum_climb_EAS),
        "stall_TAS_transition_m_s": float(stall_TAS_transition),
        "minimum_transition_complete_TAS_m_s": float(minimum_TAS_transition),
        "transition_complete_speed_margin_m_s": (
            None
            if config.max_transition_complete_speed_m_s is None
            else float(
                config.max_transition_complete_speed_m_s
                - minimum_TAS_transition
            )
        ),
        "stall_EAS_margin_m_s": (
            None
            if config.max_stall_EAS_m_s is None
            else float(config.max_stall_EAS_m_s - stall_EAS)
        ),
        "max_transition_complete_speed_m_s": (
            None
            if config.max_transition_complete_speed_m_s is None
            else float(config.max_transition_complete_speed_m_s)
        ),
        "max_stall_EAS_m_s": (
            None
            if config.max_stall_EAS_m_s is None
            else float(config.max_stall_EAS_m_s)
        ),
        "minimum_aerodynamic_cruise_TAS_m_s": float(
            minimum_TAS_aerodynamic
        ),
        "minimum_cruise_TAS_m_s": float(minimum_cruise_TAS),
        "rho_reference_kg_m3": float(rho_reference),
        "rho_transition_kg_m3": float(rho_transition),
        "rho_target_kg_m3": float(rho_target),
        "CL_max": float(aircraft.CL_max),
        "CL_allowed": float(aircraft.permitted_CL),
    }


def _default_speed_grid(minimum_speed_m_s: float) -> np.ndarray:
    """Create a coarse search grid from a physical lower speed bound."""
    lower = 1.001 * minimum_speed_m_s
    upper = max(1.8 * lower, lower + 12.0)
    return np.linspace(lower, upper, 7)


def wing_segment_forces(
        aircraft: AircraftPerformance,
        density_kg_m3: float,
        speed_current_m_s: float,
        speed_next_m_s: float,
        climb_angle_rad: float,
        delta_h_m: float,
        delta_s_m: Optional[float] = None) -> Dict:
    """Calculate drag and required thrust for one discrete path segment.

    The mechanical-energy equation is

    ``T = D + [W Delta_h + 0.5 m (V_next^2 - V_current^2)] / Delta_s``.

    For constant speed it reduces to ``T = D + W sin(gamma)``. For level,
    constant-speed flight it reduces to ``T = D``.
    """
    if density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be positive.")
    if speed_current_m_s <= 0.0 or speed_next_m_s <= 0.0:
        raise ValueError("Segment airspeeds must be positive.")
    if delta_h_m < 0.0:
        raise ValueError("This preliminary mission model does not model descent.")
    if delta_s_m is None:
        sine = np.sin(climb_angle_rad)
        if delta_h_m <= 0.0 or sine <= 0.0:
            raise ValueError("delta_s_m is required for a level segment.")
        delta_s_m = delta_h_m / sine
    if delta_s_m <= 0.0:
        raise ValueError("delta_s_m must be positive.")

    speed_m_s = 0.5 * (speed_current_m_s + speed_next_m_s)
    lift_N = aircraft.weight_N * np.cos(climb_angle_rad)
    dynamic_pressure_Pa = 0.5 * density_kg_m3 * speed_m_s**2
    CL = lift_N / (dynamic_pressure_Pa * aircraft.wing_area_m2)
    CD = aircraft.CD0 + aircraft.induced_drag_factor * CL**2
    drag_N = dynamic_pressure_Pa * aircraft.wing_area_m2 * CD
    potential_energy_change_J = aircraft.weight_N * delta_h_m
    kinetic_energy_change_J = 0.5 * aircraft.mass_kg * (
        speed_next_m_s**2 - speed_current_m_s**2
    )
    required_thrust_N = drag_N + (
        potential_energy_change_J + kinetic_energy_change_J
    ) / delta_s_m

    return {
        "speed_m_s": float(speed_m_s),
        "lift_N": float(lift_N),
        "dynamic_pressure_Pa": float(dynamic_pressure_Pa),
        "CL": float(CL),
        "CD": float(CD),
        "drag_N": float(drag_N),
        "required_thrust_N": float(required_thrust_N),
        "potential_energy_change_J": float(potential_energy_change_J),
        "kinetic_energy_change_J": float(kinetic_energy_change_J),
        "mechanical_energy_change_J": float(
            potential_energy_change_J + kinetic_energy_change_J
        ),
    }


def _add_power_model(
        forces: Dict,
        power: MissionPowerAssumptions) -> Dict:
    """Add preliminary shaft/electrical power and affordability margins."""
    propulsive_power_W = max(
        0.0,
        forces["required_thrust_N"] * forces["speed_m_s"],
    )
    shaft_power_W = propulsive_power_W / power.propulsive_efficiency
    electrical_power_W = shaft_power_W / (
        power.motor_efficiency * power.esc_efficiency
    )
    result = dict(forces)
    result.update({
        "propulsive_power_W": float(propulsive_power_W),
        "shaft_power_W": float(shaft_power_W),
        "electrical_power_W": float(electrical_power_W),
        "total_forward_efficiency": float(power.total_forward_efficiency),
        "power_margin_W": float(
            power.max_affordable_electrical_power_W - electrical_power_W
        ),
    })
    return result


def _empty_segment(failure_reason: str, states: Sequence[Dict]) -> Dict:
    return {
        "feasible": False,
        "failure_reason": failure_reason,
        "states": list(states),
        "time_s": float(sum(state["delta_t_s"] for state in states)),
        "distance_m": float(sum(state["delta_x_m"] for state in states)),
        "energy_Wh": float(
            sum(state["delta_electrical_energy_Wh"] for state in states)
        ),
        "mechanical_energy_change_J": float(
            sum(state["mechanical_energy_change_J"] for state in states)
        ),
    }


def _finish_segment(states: Sequence[Dict]) -> Dict:
    time_s = sum(state["delta_t_s"] for state in states)
    distance_m = sum(state["delta_x_m"] for state in states)
    energy_Wh = sum(state["delta_electrical_energy_Wh"] for state in states)
    return {
        "feasible": True,
        "failure_reason": None,
        "states": list(states),
        "time_s": float(time_s),
        "distance_m": float(distance_m),
        "energy_Wh": float(energy_Wh),
        "average_electrical_power_W": float(
            energy_Wh * 3600.0 / time_s if time_s > 0.0 else 0.0
        ),
        "mechanical_energy_change_J": float(
            sum(state["mechanical_energy_change_J"] for state in states)
        ),
        "average_rate_of_climb_m_s": float(
            sum(state["delta_h_m"] for state in states) / time_s
            if time_s > 0.0
            else 0.0
        ),
    }


def simulate_wing_borne_climb(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        start_altitude_m: float,
        end_altitude_m: float,
        equivalent_airspeed_m_s: float,
        climb_angle_rad: float,
        altitude_step_m: float = 100.0,
        initial_true_airspeed_m_s: Optional[float] = None,
        minimum_power_margin_fraction: float = 0.0,
        isa_fn: Callable = isa,
        segment_name: str = "wing_borne_climb") -> Dict:
    """Integrate a constant-EAS, constant-angle wing-borne climb."""
    if not 0.0 <= start_altitude_m < end_altitude_m:
        raise ValueError("Climb altitudes must satisfy 0 <= start < end.")
    if equivalent_airspeed_m_s <= 0.0 or altitude_step_m <= 0.0:
        raise ValueError("Climb EAS and altitude step must be positive.")
    if not 0.0 < climb_angle_rad < 0.5 * np.pi:
        raise ValueError("Climb angle must lie between 0 and 90 degrees.")
    if not 0.0 <= minimum_power_margin_fraction < 1.0:
        raise ValueError("minimum_power_margin_fraction must lie in 0 <= fraction < 1.")

    states = []
    required_power_margin_W = (
        minimum_power_margin_fraction
        * power.max_affordable_electrical_power_W
    )
    altitude = float(start_altitude_m)
    rho_start, _, _, _ = isa_fn(altitude)
    scheduled_start_speed = equivalent_to_true_airspeed(
        equivalent_airspeed_m_s, rho_start
    )
    speed_current = (
        scheduled_start_speed
        if initial_true_airspeed_m_s is None
        else float(initial_true_airspeed_m_s)
    )

    while altitude < end_altitude_m - 1e-9:
        altitude_next = min(altitude + altitude_step_m, end_altitude_m)
        altitude_mid = 0.5 * (altitude + altitude_next)
        rho_mid, mu_mid, speed_of_sound_mid, _ = isa_fn(altitude_mid)
        rho_next, _, _, _ = isa_fn(altitude_next)
        speed_next = equivalent_to_true_airspeed(
            equivalent_airspeed_m_s, rho_next
        )
        delta_h_m = altitude_next - altitude
        delta_s_m = delta_h_m / np.sin(climb_angle_rad)
        delta_x_m = delta_s_m * np.cos(climb_angle_rad)
        forces = wing_segment_forces(
            aircraft,
            rho_mid,
            speed_current,
            speed_next,
            climb_angle_rad,
            delta_h_m,
            delta_s_m=delta_s_m,
        )
        state = _add_power_model(forces, power)
        delta_t_s = delta_s_m / forces["speed_m_s"]
        state.update({
            "segment": segment_name,
            "altitude_start_m": float(altitude),
            "altitude_m": float(altitude_next),
            "altitude_mid_m": float(altitude_mid),
            "density_kg_m3": float(rho_mid),
            "dynamic_viscosity_Pa_s": float(mu_mid),
            "speed_of_sound_m_s": float(speed_of_sound_mid),
            "speed_current_m_s": float(speed_current),
            "speed_next_m_s": float(speed_next),
            "EAS_m_s": float(equivalent_airspeed_m_s),
            "climb_angle_rad": float(climb_angle_rad),
            "climb_angle_deg": float(np.rad2deg(climb_angle_rad)),
            "rate_of_climb_m_s": float(delta_h_m / delta_t_s),
            "delta_h_m": float(delta_h_m),
            "delta_s_m": float(delta_s_m),
            "delta_x_m": float(delta_x_m),
            "delta_t_s": float(delta_t_s),
            "delta_electrical_energy_Wh": float(
                state["electrical_power_W"] * delta_t_s / 3600.0
            ),
            "CL_margin": float(aircraft.permitted_CL - state["CL"]),
        })
        states.append(state)
        if state["CL_margin"] < -1e-9:
            return _empty_segment(
                f"Permitted CL exceeded at {altitude_mid:.0f} m.",
                states,
            )
        if state["power_margin_W"] < -1e-6:
            return _empty_segment(
                f"Affordable electrical power exceeded at {altitude_mid:.0f} m.",
                states,
            )
        margin_failure = _power_margin_failure(
            f"at {altitude_mid:.0f} m",
            state["power_margin_W"],
            required_power_margin_W,
        )
        if margin_failure is not None:
            return _empty_segment(margin_failure, states)
        altitude = altitude_next
        speed_current = speed_next

    return _finish_segment(states)


def simulate_level_cruise(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        altitude_m: float,
        true_airspeed_m_s: float,
        distance_m: float,
        minimum_power_margin_fraction: float = 0.0,
        isa_fn: Callable = isa) -> Dict:
    """Simulate constant-speed level cruise at one altitude."""
    if altitude_m < 0.0 or true_airspeed_m_s <= 0.0 or distance_m < 0.0:
        raise ValueError("Cruise altitude/distance must be non-negative and speed positive.")
    if not 0.0 <= minimum_power_margin_fraction < 1.0:
        raise ValueError("minimum_power_margin_fraction must lie in 0 <= fraction < 1.")
    if distance_m == 0.0:
        return _finish_segment([])

    rho, mu, speed_of_sound, _ = isa_fn(altitude_m)
    forces = wing_segment_forces(
        aircraft,
        rho,
        true_airspeed_m_s,
        true_airspeed_m_s,
        0.0,
        0.0,
        delta_s_m=distance_m,
    )
    state = _add_power_model(forces, power)
    delta_t_s = distance_m / true_airspeed_m_s
    eas_m_s = true_airspeed_m_s * np.sqrt(rho / 1.225)
    state.update({
        "segment": "level_cruise",
        "altitude_start_m": float(altitude_m),
        "altitude_m": float(altitude_m),
        "altitude_mid_m": float(altitude_m),
        "density_kg_m3": float(rho),
        "dynamic_viscosity_Pa_s": float(mu),
        "speed_of_sound_m_s": float(speed_of_sound),
        "speed_current_m_s": float(true_airspeed_m_s),
        "speed_next_m_s": float(true_airspeed_m_s),
        "EAS_m_s": float(eas_m_s),
        "climb_angle_rad": 0.0,
        "climb_angle_deg": 0.0,
        "rate_of_climb_m_s": 0.0,
        "delta_h_m": 0.0,
        "delta_s_m": float(distance_m),
        "delta_x_m": float(distance_m),
        "delta_t_s": float(delta_t_s),
        "delta_electrical_energy_Wh": float(
            state["electrical_power_W"] * delta_t_s / 3600.0
        ),
        "CL_margin": float(aircraft.permitted_CL - state["CL"]),
    })
    if state["CL_margin"] < -1e-9:
        return _empty_segment("Permitted CL exceeded in level cruise.", [state])
    if state["power_margin_W"] < -1e-6:
        return _empty_segment(
            "Affordable electrical power exceeded in level cruise.",
            [state],
        )
    margin_failure = _power_margin_failure(
        "in level cruise",
        state["power_margin_W"],
        minimum_power_margin_fraction
        * power.max_affordable_electrical_power_W,
    )
    if margin_failure is not None:
        return _empty_segment(margin_failure, [state])
    return _finish_segment([state])


def estimate_takeoff_and_transition(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        cruise_true_airspeed_m_s: Optional[float] = None,
        isa_fn: Callable = isa) -> Dict:
    """Estimate low-altitude takeoff and transition using assumed powers."""
    rho0, _, _, _ = isa_fn(0.0)
    rho_transition, _, _, _ = isa_fn(config.vertical_takeoff_height_m)
    stall_EAS_m_s = np.sqrt(
        2.0 * aircraft.weight_N
        / (rho0 * aircraft.wing_area_m2 * aircraft.CL_max)
    )
    stall_speed_m_s = np.sqrt(
        2.0 * aircraft.weight_N
        / (rho_transition * aircraft.wing_area_m2 * aircraft.CL_max)
    )
    transition = phase13_transition_blending(
        stall_speed_m_s,
        V_blend_start_frac=config.transition_blend_start_fraction,
        V_blend_end_frac=config.transition_blend_end_fraction,
        transition_accel_m_s2=config.transition_accel_m_s2,
        V_cruise=cruise_true_airspeed_m_s,
        cruise_speed_margin_frac=config.transition_cruise_margin_fraction,
        sample_count=config.transition_sample_count,
    )
    takeoff_time_s = (
        config.vertical_takeoff_height_m
        / config.vertical_takeoff_rate_m_s
    )
    takeoff_energy_Wh = (
        power.preliminary_hover_power_W * takeoff_time_s / 3600.0
    )
    transition_energy_Wh = (
        power.preliminary_transition_power_W
        * transition["t_transition"]
        / 3600.0
    )
    transition_complete_speed_m_s = (
        (1.0 + config.transition_cruise_margin_fraction)
        * transition["V_blend_end"]
    )
    return {
        "transition_altitude_m": float(config.vertical_takeoff_height_m),
        "takeoff_time_s": float(takeoff_time_s),
        "takeoff_energy_Wh": float(takeoff_energy_Wh),
        "transition": transition,
        "transition_energy_Wh": float(transition_energy_Wh),
        "transition_distance_m": float(transition["distance_transition_m"]),
        "stall_speed_m_s": float(stall_speed_m_s),
        "stall_EAS_m_s": float(stall_EAS_m_s),
        "stall_TAS_transition_m_s": float(stall_speed_m_s),
        "transition_complete_speed_m_s": float(
            transition_complete_speed_m_s
        ),
        "transition_complete_speed_margin_m_s": (
            None
            if config.max_transition_complete_speed_m_s is None
            else float(
                config.max_transition_complete_speed_m_s
                - transition_complete_speed_m_s
            )
        ),
        "stall_EAS_margin_m_s": (
            None
            if config.max_stall_EAS_m_s is None
            else float(config.max_stall_EAS_m_s - stall_EAS_m_s)
        ),
        "transition_density_kg_m3": float(rho_transition),
    }


def _segment_summary(
        time_s: float,
        energy_Wh: float,
        distance_m: float = 0.0) -> Dict:
    return {
        "time_s": float(time_s),
        "distance_m": float(distance_m),
        "energy_Wh": float(energy_Wh),
        "average_electrical_power_W": float(
            energy_Wh * 3600.0 / time_s if time_s > 0.0 else 0.0
        ),
    }


def _failed_candidate(reason: str) -> Dict:
    return {"feasible": False, "failure_reason": reason}


def _complete_candidate_from_climb(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        takeoff_transition: Dict,
        climb: Dict,
        cruise_true_airspeed_m_s: float,
        optimized_eas,
        optimized_gamma,
        isa_fn: Callable = isa) -> Dict:
    if not climb["feasible"]:
        return _failed_candidate(climb["failure_reason"])
    if (
        power.preliminary_hover_power_W
        > power.max_affordable_electrical_power_W + 1e-6
    ):
        return _failed_candidate("Preliminary hover power exceeds affordable electrical power.")
    if (
        power.preliminary_transition_power_W
        > power.max_affordable_electrical_power_W + 1e-6
    ):
        return _failed_candidate("Preliminary transition power exceeds affordable electrical power.")
    required_power_margin_W = _required_power_margin_W(power, config)
    hover_margin_W = (
        power.max_affordable_electrical_power_W
        - power.preliminary_hover_power_W
    )
    transition_power_margin_W = (
        power.max_affordable_electrical_power_W
        - power.preliminary_transition_power_W
    )
    margin_failure = _power_margin_failure(
        "during preliminary hover",
        hover_margin_W,
        required_power_margin_W,
    )
    if margin_failure is not None:
        return _failed_candidate(margin_failure)
    margin_failure = _power_margin_failure(
        "during preliminary transition",
        transition_power_margin_W,
        required_power_margin_W,
    )
    if margin_failure is not None:
        return _failed_candidate(margin_failure)

    transition = takeoff_transition["transition"]
    transition_minimum = (
        (1.0 + config.transition_cruise_margin_fraction)
        * transition["V_blend_end"]
    )
    minimum_cruise_speed = transition_minimum
    if config.minimum_cruise_true_airspeed_m_s is not None:
        minimum_cruise_speed = max(
            minimum_cruise_speed,
            config.minimum_cruise_true_airspeed_m_s,
        )
    if cruise_true_airspeed_m_s < minimum_cruise_speed - 1e-9:
        return _failed_candidate("Cruise speed is below the transition speed requirement.")

    pre_cruise_ground_track_m = (
        takeoff_transition["transition_distance_m"] + climb["distance_m"]
    )
    remaining_distance_m = (
        config.horizontal_range_m
        - pre_cruise_ground_track_m
    )
    if remaining_distance_m < -1e-6 and not config.allow_spiral_climb:
        return _failed_candidate("Wing-borne climb exceeds the horizontal mission range.")
    spiral_excess_ground_track_m = max(0.0, -remaining_distance_m)
    remaining_distance_m = max(0.0, remaining_distance_m)
    cruise = simulate_level_cruise(
        aircraft,
        power,
        config.target_altitude_m,
        cruise_true_airspeed_m_s,
        remaining_distance_m,
        minimum_power_margin_fraction=config.minimum_power_margin_fraction,
        isa_fn=isa_fn,
    )
    if not cruise["feasible"]:
        return _failed_candidate(cruise["failure_reason"])

    outbound_time_s = (
        takeoff_transition["takeoff_time_s"]
        + transition["t_transition"]
        + climb["time_s"]
        + cruise["time_s"]
    )
    if outbound_time_s > config.outbound_time_budget_s + 1e-6:
        return _failed_candidate("Outbound mission time budget exceeded.")

    segment_summaries = {
        "vertical_takeoff": _segment_summary(
            takeoff_transition["takeoff_time_s"],
            takeoff_transition["takeoff_energy_Wh"],
        ),
        "transition": _segment_summary(
            transition["t_transition"],
            takeoff_transition["transition_energy_Wh"],
            takeoff_transition["transition_distance_m"],
        ),
        "wing_borne_climb": _segment_summary(
            climb["time_s"], climb["energy_Wh"], climb["distance_m"]
        ),
        "level_cruise": _segment_summary(
            cruise["time_s"], cruise["energy_Wh"], cruise["distance_m"]
        ),
        "mission_hover": _segment_summary(
            config.hover_duration_s,
            power.preliminary_hover_power_W * config.hover_duration_s / 3600.0,
        ),
    }
    states = list(climb["states"]) + list(cruise["states"])
    cumulative_energy_Wh = (
        takeoff_transition["takeoff_energy_Wh"]
        + takeoff_transition["transition_energy_Wh"]
    )
    for state in states:
        cumulative_energy_Wh += state["delta_electrical_energy_Wh"]
        state["cumulative_mission_energy_Wh"] = float(cumulative_energy_Wh)

    all_power_margins = [state["power_margin_W"] for state in states]
    all_power_margins.extend([
        power.max_affordable_electrical_power_W
        - power.preliminary_hover_power_W,
        power.max_affordable_electrical_power_W
        - power.preliminary_transition_power_W,
    ])
    cl_margins = [state["CL_margin"] for state in states]
    constraint_margins = {
        "CL_margin": float(min(cl_margins) if cl_margins else np.inf),
        "power_margin_W": float(min(all_power_margins)),
        "required_power_margin_W": float(required_power_margin_W),
        "power_margin_over_required_W": float(
            min(all_power_margins) - required_power_margin_W
        ),
        "outbound_time_margin_s": float(
            config.outbound_time_budget_s - outbound_time_s
        ),
        "remaining_cruise_distance_m": float(remaining_distance_m),
        "cruise_speed_margin_m_s": float(
            cruise_true_airspeed_m_s - minimum_cruise_speed
        ),
    }
    total_energy_Wh = sum(
        segment["energy_Wh"] for segment in segment_summaries.values()
    )
    return {
        "feasible": True,
        "failure_reason": None,
        "optimized_climb_EAS_m_s": optimized_eas,
        "optimized_climb_angle_rad": optimized_gamma,
        "cruise_true_airspeed_m_s": float(cruise_true_airspeed_m_s),
        "climb": climb,
        "cruise": cruise,
        "takeoff_transition": takeoff_transition,
        "states": states,
        "segment_summaries": segment_summaries,
        "constraint_margins": constraint_margins,
        "outbound_time_s": float(outbound_time_s),
        "total_mission_time_s": float(
            outbound_time_s + config.hover_duration_s
        ),
        "total_electrical_energy_Wh": float(total_energy_Wh),
        "total_horizontal_distance_m": float(config.horizontal_range_m),
        "required_horizontal_displacement_m": float(config.horizontal_range_m),
        "total_ground_track_distance_m": float(
            pre_cruise_ground_track_m + remaining_distance_m
        ),
        "spiral_excess_ground_track_distance_m": float(
            spiral_excess_ground_track_m
        ),
        "final_altitude_m": float(config.target_altitude_m),
        "peak_electrical_power_W": float(
            max(
                [state["electrical_power_W"] for state in states]
                + [
                    power.preliminary_hover_power_W,
                    power.preliminary_transition_power_W,
                ]
            )
        ),
        "peak_shaft_power_W": float(
            max([state["shaft_power_W"] for state in states] or [0.0])
        ),
        "power_assumptions": asdict(power),
        "assumptions": [
            "Forward-flight electrical power is aerodynamic propulsive power divided by assumed propulsive, motor, and ESC efficiencies.",
            "Hover and transition powers are preliminary total-aircraft electrical-power assumptions.",
            "No propeller map, RPM, advance ratio, motor torque, or current limit is modeled at this design stage.",
            "The affordable electrical-power limit is a design constraint, not a prediction of installed propulsion capability.",
            "When climb ground-track distance exceeds the required point-to-point displacement, an idealized spiral absorbs the excess without bank-angle or turning-drag penalties.",
        ],
    }


def _complete_candidate(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        equivalent_airspeed_m_s: float,
        climb_angle_rad: float,
        cruise_true_airspeed_m_s: float,
        isa_fn: Callable = isa) -> Dict:
    if np.rad2deg(climb_angle_rad) > config.max_fixed_wing_climb_angle_deg + 1e-9:
        return _failed_candidate("Fixed-wing climb angle exceeds the model-validity limit.")
    takeoff_transition = estimate_takeoff_and_transition(
        aircraft,
        power,
        config,
        cruise_true_airspeed_m_s=cruise_true_airspeed_m_s,
        isa_fn=isa_fn,
    )
    transition_complete_speed = (
        (1.0 + config.transition_cruise_margin_fraction)
        * takeoff_transition["transition"]["V_blend_end"]
    )
    if (
        config.max_transition_complete_speed_m_s is not None
        and transition_complete_speed
        > config.max_transition_complete_speed_m_s + 1e-9
    ):
        return _failed_candidate("Transition-complete speed exceeds the selected limit.")
    if (
        config.max_stall_EAS_m_s is not None
        and takeoff_transition["stall_EAS_m_s"]
        > config.max_stall_EAS_m_s + 1e-9
    ):
        return _failed_candidate("Stall EAS exceeds the selected limit.")
    climb = simulate_wing_borne_climb(
        aircraft,
        power,
        takeoff_transition["transition_altitude_m"],
        config.target_altitude_m,
        equivalent_airspeed_m_s,
        climb_angle_rad,
        altitude_step_m=config.altitude_step_m,
        initial_true_airspeed_m_s=takeoff_transition["transition"]["V_exit"],
        minimum_power_margin_fraction=config.minimum_power_margin_fraction,
        isa_fn=isa_fn,
    )
    return _complete_candidate_from_climb(
        aircraft,
        power,
        config,
        takeoff_transition,
        climb,
        cruise_true_airspeed_m_s,
        float(equivalent_airspeed_m_s),
        float(climb_angle_rad),
        isa_fn=isa_fn,
    )


def optimize_constant_eas_mission(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        eas_grid_m_s: Optional[Sequence[float]] = None,
        climb_angle_grid_rad: Optional[Sequence[float]] = None,
        cruise_tas_grid_m_s: Optional[Sequence[float]] = None,
        isa_fn: Callable = isa) -> Dict:
    """Minimize mission electrical energy with an inspectable coarse grid."""
    if (
        power.preliminary_hover_power_W
        > power.max_affordable_electrical_power_W + 1e-6
    ):
        return {
            "feasible": False,
            "failure_reason": "Preliminary hover power exceeds affordable electrical power.",
            "failure_counts": {
                "Preliminary hover power exceeds affordable electrical power.": 1
            },
            "grid": {},
            "power_assumptions": asdict(power),
        }
    if (
        power.preliminary_transition_power_W
        > power.max_affordable_electrical_power_W + 1e-6
    ):
        return {
            "feasible": False,
            "failure_reason": "Preliminary transition power exceeds affordable electrical power.",
            "failure_counts": {
                "Preliminary transition power exceeds affordable electrical power.": 1
            },
            "grid": {},
            "power_assumptions": asdict(power),
        }
    required_power_margin_W = _required_power_margin_W(power, config)
    hover_margin_W = (
        power.max_affordable_electrical_power_W
        - power.preliminary_hover_power_W
    )
    transition_power_margin_W = (
        power.max_affordable_electrical_power_W
        - power.preliminary_transition_power_W
    )
    preliminary_margin_failure = (
        _power_margin_failure(
            "during preliminary hover",
            hover_margin_W,
            required_power_margin_W,
        )
        or _power_margin_failure(
            "during preliminary transition",
            transition_power_margin_W,
            required_power_margin_W,
        )
    )
    if preliminary_margin_failure is not None:
        return {
            "feasible": False,
            "failure_reason": preliminary_margin_failure,
            "failure_counts": {preliminary_margin_failure: 1},
            "grid": {},
            "power_assumptions": asdict(power),
        }
    speed_limits = aerodynamic_speed_limits(aircraft, config, isa_fn=isa_fn)
    if (
        config.max_transition_complete_speed_m_s is not None
        and speed_limits["minimum_transition_complete_TAS_m_s"]
        > config.max_transition_complete_speed_m_s + 1e-9
    ):
        reason = "Transition-complete speed exceeds the selected limit."
        return {
            "feasible": False,
            "failure_reason": reason,
            "failure_counts": {reason: 1},
            "grid": {
                "aerodynamic_speed_limits": speed_limits,
                "max_transition_complete_speed_m_s": float(
                    config.max_transition_complete_speed_m_s
                ),
            },
            "power_assumptions": asdict(power),
        }
    if (
        config.max_stall_EAS_m_s is not None
        and speed_limits["stall_EAS_m_s"] > config.max_stall_EAS_m_s + 1e-9
    ):
        reason = "Stall EAS exceeds the selected limit."
        return {
            "feasible": False,
            "failure_reason": reason,
            "failure_counts": {reason: 1},
            "grid": {
                "aerodynamic_speed_limits": speed_limits,
                "max_stall_EAS_m_s": float(config.max_stall_EAS_m_s),
            },
            "power_assumptions": asdict(power),
        }
    adaptive_eas = eas_grid_m_s is None
    adaptive_cruise = cruise_tas_grid_m_s is None
    eas_grid = np.asarray(
        _default_speed_grid(speed_limits["minimum_climb_EAS_m_s"])
        if adaptive_eas
        else eas_grid_m_s,
        dtype=float,
    )
    angle_grid = np.asarray(
        np.deg2rad(np.arange(
            min(20.0, config.max_fixed_wing_climb_angle_deg),
            config.max_fixed_wing_climb_angle_deg + 1e-9,
            2.5,
        ))
        if climb_angle_grid_rad is None
        else climb_angle_grid_rad,
        dtype=float,
    )
    angle_grid = angle_grid[
        np.rad2deg(angle_grid)
        <= config.max_fixed_wing_climb_angle_deg + 1e-9
    ]
    cruise_grid = np.asarray(
        _default_speed_grid(speed_limits["minimum_cruise_TAS_m_s"])
        if adaptive_cruise
        else cruise_tas_grid_m_s,
        dtype=float,
    )
    if any(grid.size == 0 for grid in (eas_grid, angle_grid, cruise_grid)):
        raise ValueError("Mission optimization grids must not be empty.")

    best = None
    failures = Counter()
    expansion_history = []
    for expansion in range(5):
        best = None
        failures = Counter()
        for eas_m_s, gamma_rad, cruise_m_s in product(
                eas_grid, angle_grid, cruise_grid):
            candidate = _complete_candidate(
                aircraft,
                power,
                config,
                float(eas_m_s),
                float(gamma_rad),
                float(cruise_m_s),
                isa_fn=isa_fn,
            )
            if not candidate["feasible"]:
                failures[candidate["failure_reason"]] += 1
                continue
            if (
                best is None
                or candidate["total_electrical_energy_Wh"]
                < best["total_electrical_energy_Wh"]
            ):
                best = candidate

        expand_eas = (
            adaptive_eas
            and (
                best is None
                or np.isclose(
                    best["optimized_climb_EAS_m_s"],
                    eas_grid[-1],
                    rtol=0.0,
                    atol=1e-9,
                )
            )
        )
        expand_cruise = (
            adaptive_cruise
            and (
                best is None
                or np.isclose(
                    best["cruise_true_airspeed_m_s"],
                    cruise_grid[-1],
                    rtol=0.0,
                    atol=1e-9,
                )
            )
        )
        if expansion >= 4 or not (expand_eas or expand_cruise):
            break
        record = {
            "expansion": expansion + 1,
            "expanded_climb_EAS": bool(expand_eas),
            "expanded_cruise_TAS": bool(expand_cruise),
        }
        if expand_eas:
            eas_grid = np.linspace(eas_grid[0], 1.5 * eas_grid[-1], 7)
            record["new_climb_EAS_max_m_s"] = float(eas_grid[-1])
        if expand_cruise:
            cruise_grid = np.linspace(
                cruise_grid[0], 1.5 * cruise_grid[-1], 7
            )
            record["new_cruise_TAS_max_m_s"] = float(cruise_grid[-1])
        expansion_history.append(record)

    grid = {
        "climb_EAS_m_s": eas_grid.tolist(),
        "climb_angle_deg": np.rad2deg(angle_grid).tolist(),
        "cruise_TAS_m_s": cruise_grid.tolist(),
        "aerodynamic_speed_limits": speed_limits,
        "adaptive_climb_EAS": bool(adaptive_eas),
        "adaptive_cruise_TAS": bool(adaptive_cruise),
        "expansion_history": expansion_history,
        "max_fixed_wing_climb_angle_deg": float(
            config.max_fixed_wing_climb_angle_deg
        ),
        "minimum_power_margin_fraction": float(
            config.minimum_power_margin_fraction
        ),
        "required_power_margin_W": float(required_power_margin_W),
    }
    if best is None:
        common_failure = failures.most_common(1)
        reason = (
            common_failure[0][0]
            if common_failure
            else "No mission candidates were evaluated."
        )
        return {
            "feasible": False,
            "failure_reason": reason,
            "failure_counts": dict(failures),
            "grid": grid,
            "power_assumptions": asdict(power),
        }
    best["failure_counts"] = dict(failures)
    best["grid"] = grid
    return best


def _simulate_scheduled_climb(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        eas_schedule_m_s: Sequence[float],
        angle_schedule_rad: Sequence[float],
        initial_true_airspeed_m_s: float,
        isa_fn: Callable = isa) -> Dict:
    """Simulate three continuous altitude bands for the optional extension."""
    if len(eas_schedule_m_s) != 3 or len(angle_schedule_rad) != 3:
        raise ValueError("The multi-stage climb requires three EAS and angle values.")
    if any(
        np.rad2deg(angle_rad)
        > config.max_fixed_wing_climb_angle_deg + 1e-9
        for angle_rad in angle_schedule_rad
    ):
        return _empty_segment(
            "Fixed-wing climb angle exceeds the model-validity limit.",
            [],
        )
    start = config.vertical_takeoff_height_m
    boundaries = np.linspace(start, config.target_altitude_m, 4)
    states = []
    speed = initial_true_airspeed_m_s
    for index in range(3):
        band = simulate_wing_borne_climb(
            aircraft,
            power,
            float(boundaries[index]),
            float(boundaries[index + 1]),
            float(eas_schedule_m_s[index]),
            float(angle_schedule_rad[index]),
            altitude_step_m=config.altitude_step_m,
            initial_true_airspeed_m_s=speed,
            minimum_power_margin_fraction=config.minimum_power_margin_fraction,
            isa_fn=isa_fn,
            segment_name=f"wing_borne_climb_band_{index + 1}",
        )
        states.extend(band["states"])
        if not band["feasible"]:
            return _empty_segment(band["failure_reason"], states)
        speed = states[-1]["speed_next_m_s"]
    return _finish_segment(states)


def optimize_multistage_mission(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        initial_result: Optional[Dict] = None,
        eas_candidates_m_s: Optional[Sequence[float]] = None,
        climb_angle_candidates_rad: Optional[Sequence[float]] = None,
        isa_fn: Callable = isa) -> Dict:
    """Coordinate-search a readable three-band climb profile."""
    if initial_result is None:
        initial_result = optimize_constant_eas_mission(
            aircraft, power, config, isa_fn=isa_fn
        )
    if not initial_result["feasible"]:
        return initial_result
    base_eas = float(initial_result["optimized_climb_EAS_m_s"])
    base_angle = float(initial_result["optimized_climb_angle_rad"])
    eas_values = (
        np.asarray([0.9 * base_eas, base_eas, 1.1 * base_eas])
        if eas_candidates_m_s is None
        else np.asarray(eas_candidates_m_s, dtype=float)
    )
    angle_values = (
        np.asarray([
            0.9 * base_angle,
            base_angle,
            min(
                np.deg2rad(config.max_fixed_wing_climb_angle_deg),
                1.1 * base_angle,
            ),
        ])
        if climb_angle_candidates_rad is None
        else np.asarray(climb_angle_candidates_rad, dtype=float)
    )
    angle_values = angle_values[
        np.rad2deg(angle_values)
        <= config.max_fixed_wing_climb_angle_deg + 1e-9
    ]
    eas_schedule = [base_eas] * 3
    angle_schedule = [base_angle] * 3
    best = initial_result
    cruise_speed = initial_result["cruise_true_airspeed_m_s"]
    takeoff_transition = estimate_takeoff_and_transition(
        aircraft, power, config, cruise_speed, isa_fn
    )
    for band_index in range(3):
        for eas_m_s, angle_rad in product(eas_values, angle_values):
            trial_eas = list(eas_schedule)
            trial_angles = list(angle_schedule)
            trial_eas[band_index] = float(eas_m_s)
            trial_angles[band_index] = float(angle_rad)
            climb = _simulate_scheduled_climb(
                aircraft,
                power,
                config,
                trial_eas,
                trial_angles,
                takeoff_transition["transition"]["V_exit"],
                isa_fn=isa_fn,
            )
            candidate = _complete_candidate_from_climb(
                aircraft,
                power,
                config,
                takeoff_transition,
                climb,
                cruise_speed,
                trial_eas,
                trial_angles,
                isa_fn=isa_fn,
            )
            if (
                candidate["feasible"]
                and candidate["total_electrical_energy_Wh"]
                < best["total_electrical_energy_Wh"]
            ):
                best = candidate
                eas_schedule = trial_eas
                angle_schedule = trial_angles
    best = dict(best)
    best["optimized_climb_EAS_schedule_m_s"] = list(eas_schedule)
    best["optimized_climb_angle_schedule_rad"] = list(angle_schedule)
    return best


def minimum_time_reference_climb(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        eas_grid_m_s: Optional[Sequence[float]] = None,
        climb_angle_grid_rad: Optional[Sequence[float]] = None,
        isa_fn: Callable = isa) -> Dict:
    """Find the fastest feasible constant-EAS climb under the power cap."""
    eas_grid = np.asarray(
        np.linspace(14.0, 28.0, 8) if eas_grid_m_s is None else eas_grid_m_s,
        dtype=float,
    )
    angle_grid = np.asarray(
        np.deg2rad(np.linspace(
            min(20.0, config.max_fixed_wing_climb_angle_deg),
            config.max_fixed_wing_climb_angle_deg,
            8,
        ))
        if climb_angle_grid_rad is None
        else climb_angle_grid_rad,
        dtype=float,
    )
    angle_grid = angle_grid[
        np.rad2deg(angle_grid)
        <= config.max_fixed_wing_climb_angle_deg + 1e-9
    ]
    transition = estimate_takeoff_and_transition(aircraft, power, config)
    best = None
    for eas_m_s, gamma_rad in product(eas_grid, angle_grid):
        climb = simulate_wing_borne_climb(
            aircraft,
            power,
            transition["transition_altitude_m"],
            config.target_altitude_m,
            float(eas_m_s),
            float(gamma_rad),
            altitude_step_m=config.altitude_step_m,
            initial_true_airspeed_m_s=transition["transition"]["V_exit"],
            minimum_power_margin_fraction=config.minimum_power_margin_fraction,
            isa_fn=isa_fn,
        )
        if climb["feasible"] and (
            best is None or climb["time_s"] < best["time_s"]
        ):
            best = climb
            best["optimized_climb_EAS_m_s"] = float(eas_m_s)
            best["optimized_climb_angle_rad"] = float(gamma_rad)
    if best is None:
        return _failed_candidate("No feasible minimum-time reference climb.")
    return best


def simplified_mission_reference(
        aircraft: AircraftPerformance,
        config: MissionProfileConfig,
        total_forward_efficiency: float,
        cruise_true_airspeed_m_s: float,
        climb_angle_rad: float,
        isa_fn: Callable = isa) -> Dict:
    """Retain the former single-average-atmosphere method as a comparison."""
    if not 0.0 < total_forward_efficiency <= 1.0:
        raise ValueError("total_forward_efficiency must lie in 0 < eta <= 1.")
    h_start = config.vertical_takeoff_height_m
    h_average = 0.5 * (h_start + config.target_altitude_m)
    rho, _, _, _ = isa_fn(h_average)
    delta_h = config.target_altitude_m - h_start
    path_m = delta_h / np.sin(climb_angle_rad)
    forces = wing_segment_forces(
        aircraft,
        rho,
        cruise_true_airspeed_m_s,
        cruise_true_airspeed_m_s,
        climb_angle_rad,
        delta_h,
        delta_s_m=path_m,
    )
    climb_time_s = path_m / cruise_true_airspeed_m_s
    electrical_power_W = (
        forces["required_thrust_N"]
        * cruise_true_airspeed_m_s
        / total_forward_efficiency
    )
    return {
        "climb_time_s": float(climb_time_s),
        "climb_energy_Wh": float(electrical_power_W * climb_time_s / 3600.0),
        "required_thrust_N": float(forces["required_thrust_N"]),
        "electrical_power_W": float(electrical_power_W),
        "average_altitude_m": float(h_average),
        "notes": [
            "This comparison uses one average atmosphere and one constant true airspeed."
        ],
    }


def compare_mission_profiles(**profiles: Dict) -> Dict:
    """Return compact energy/time comparisons for feasible mission results."""
    return {
        name: {
            "feasible": bool(profile.get("feasible", True)),
            "energy_Wh": profile.get("total_electrical_energy_Wh"),
            "outbound_time_s": profile.get("outbound_time_s"),
        }
        for name, profile in profiles.items()
    }


def sweep_target_distances(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config: MissionProfileConfig,
        target_distances_m: Sequence[float],
        eas_grid_m_s: Optional[Sequence[float]] = None,
        climb_angle_grid_rad: Optional[Sequence[float]] = None,
        cruise_tas_grid_m_s: Optional[Sequence[float]] = None,
        isa_fn: Callable = isa) -> Sequence[Dict]:
    """Optimize one fixed aircraft for several target ground distances."""
    results = []
    for distance_m in target_distances_m:
        if distance_m < 0.0:
            raise ValueError("Target distances must be non-negative.")
        mission = optimize_constant_eas_mission(
            aircraft,
            power,
            replace(config, horizontal_range_m=float(distance_m)),
            eas_grid_m_s=eas_grid_m_s,
            climb_angle_grid_rad=climb_angle_grid_rad,
            cruise_tas_grid_m_s=cruise_tas_grid_m_s,
            isa_fn=isa_fn,
        )
        record = {
            "target_distance_m": float(distance_m),
            "feasible": bool(mission["feasible"]),
            "failure_reason": mission.get("failure_reason"),
        }
        if mission["feasible"]:
            record.update({
                "optimized_climb_EAS_m_s": float(
                    mission["optimized_climb_EAS_m_s"]
                ),
                "optimized_climb_angle_deg": float(
                    np.rad2deg(mission["optimized_climb_angle_rad"])
                ),
                "cruise_true_airspeed_m_s": float(
                    mission["cruise_true_airspeed_m_s"]
                ),
                "climb_ground_track_distance_m": float(
                    mission["climb"]["distance_m"]
                ),
                "spiral_excess_ground_track_distance_m": float(
                    mission["spiral_excess_ground_track_distance_m"]
                ),
                "level_cruise_distance_m": float(
                    mission["cruise"]["distance_m"]
                ),
                "total_ground_track_distance_m": float(
                    mission["total_ground_track_distance_m"]
                ),
                "outbound_time_s": float(mission["outbound_time_s"]),
                "total_electrical_energy_Wh": float(
                    mission["total_electrical_energy_Wh"]
                ),
                "peak_electrical_power_W": float(
                    mission["peak_electrical_power_W"]
                ),
            })
        results.append(record)
    return results


def plot_mission_result(result: Dict, output_path=None):
    """Plot the simple mission states without implying component-level detail."""
    import matplotlib.pyplot as plt

    if not result.get("feasible", False):
        raise ValueError("Only feasible mission results can be plotted.")
    states = result["states"]
    altitude = np.asarray([state["altitude_m"] for state in states])
    takeoff_transition = result.get("takeoff_transition", {})
    transition_distance = takeoff_transition.get(
        "transition_distance_m",
        result["segment_summaries"]["transition"].get("distance_m", 0.0),
    )
    x_distance = transition_distance + np.cumsum(
        [state["delta_x_m"] for state in states]
    )

    def values(key):
        return np.asarray([state[key] for state in states])

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    ax = axes.ravel()
    transition_altitude = takeoff_transition.get(
        "transition_altitude_m",
        result.get("h_tr", 0.0),
    )
    ax[0].plot(
        np.concatenate(([0.0, 0.0, transition_distance], x_distance)),
        np.concatenate(([0.0, transition_altitude, transition_altitude], altitude)),
    )
    ax[0].set(xlabel="Horizontal distance [m]", ylabel="Altitude [m]")
    ax[1].plot(altitude, values("speed_m_s"), label="TAS")
    ax[1].plot(altitude, values("EAS_m_s"), label="EAS")
    ax[1].legend()
    ax[1].set(xlabel="Altitude [m]", ylabel="Airspeed [m/s]")
    ax[2].plot(altitude, values("rate_of_climb_m_s"), label="Rate of climb")
    ax[2].plot(altitude, values("climb_angle_deg"), label="Climb angle")
    ax[2].legend()
    ax[2].set(xlabel="Altitude [m]", ylabel="m/s or deg")
    ax[3].plot(altitude, values("required_thrust_N"))
    ax[3].set(xlabel="Altitude [m]", ylabel="Required thrust [N]")
    ax[4].plot(altitude, values("propulsive_power_W") / 1000.0, label="Propulsive")
    ax[4].plot(altitude, values("shaft_power_W") / 1000.0, label="Shaft")
    ax[4].plot(altitude, values("electrical_power_W") / 1000.0, label="Electrical")
    ax[4].axhline(
        result["power_assumptions"]["max_affordable_electrical_power_W"] / 1000.0,
        color="k",
        linestyle="--",
        label="Affordable limit",
    )
    ax[4].legend()
    ax[4].set(xlabel="Altitude [m]", ylabel="Power [kW]")
    ax[5].plot(altitude, values("CL"), label="CL")
    ax[5].plot(altitude, values("CL") + values("CL_margin"), label="Permitted CL")
    ax[5].legend()
    ax[5].set(xlabel="Altitude [m]", ylabel="Lift coefficient [-]")
    ax[6].plot(altitude, values("power_margin_W") / 1000.0)
    ax[6].axhline(0.0, color="k", linestyle="--")
    ax[6].set(xlabel="Altitude [m]", ylabel="Power margin [kW]")
    ax[7].plot(
        np.append(altitude, result.get("final_altitude_m", altitude[-1])),
        np.append(
            values("cumulative_mission_energy_Wh"),
            result.get("total_electrical_energy_Wh", result["E_total_Wh"]),
        ) / 1000.0,
    )
    ax[7].set(xlabel="Altitude [m]", ylabel="Cumulative energy [kWh]")
    for axis in ax:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig
