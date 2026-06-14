"""Course-method climb energy and battery sizing.

Thin wrapper over mission_energy_course.py (the lecture RC_s method): it runs the
altitude-stepped constant-EAS climb, builds the mission profile, and reduces the
result to the fields the sizing loop and report need.
"""

from __future__ import annotations

import math

from mission_energy_course import (
    build_course_mission,
    optimize_course_climb,
    permitted_lift_coefficient,
    aero_drag_coefficient,
)

from sizing.inputs import AIRCRAFT, MASS, MISSION
from sizing.atmosphere import isa_density


def course_method_mission_energy(weight_N, wing, trim_drag=None):
    """Mission energy using the lecture RC_s course-method climb.

    trim_drag, when given, carries the canard area/span and the wing/canard moment
    arms from the previously-converged layout so the two-surface drag model can
    split lift and drag between the surfaces. It is None on the first mass
    iteration (before any canard exists), which falls back to wing-only drag.
    """
    aircraft = dict(AIRCRAFT)
    aircraft["trim_drag"] = trim_drag
    # The wing-borne climb keeps a stall-speed margin of climb_stall_margin_n (flies
    # at V = n * Vstall), so its CL is capped at CL_max / n^2 -- tighter than the
    # global cruise/scissor cap (mission_CL_limit_fraction). This override is applied
    # only to the local mission copy; the global value still feeds the scissor.
    aircraft["mission_CL_limit_fraction"] = 1.0 / AIRCRAFT["climb_stall_margin_n"]**2
    aircraft["course_climb_available_power_W"] = AIRCRAFT["max_affordable_electrical_power_W"]
    aircraft["course_climb_max_thrust_to_weight"] = aircraft.get("course_climb_max_thrust_to_weight", 0.50)

    selected, _ = optimize_course_climb(weight_N, wing, MISSION, aircraft, isa_density)
    if not selected["feasible"]:
        raise RuntimeError(f"No feasible course-method climb: {selected['failure_reason']}")
    if not selected.get("complies_with_time_limit", True):
        raise RuntimeError(selected["failure_reason"])

    mission = build_course_mission(weight_N, wing, MISSION, aircraft, isa_density, selected)
    climb_states = selected["states"]
    final_speed = climb_states[-1]["TAS_m_s"]
    rho_cruise = isa_density(MISSION["altitude_m"])
    q_cruise = 0.5 * rho_cruise * final_speed**2
    CL_cruise = weight_N / (q_cruise * wing["area_m2"])
    CD_cruise = aero_drag_coefficient(
        aircraft, q_cruise, weight_N, wing, MISSION["altitude_m"], final_speed
    )

    states = []
    for state in climb_states:
        states.append({
            "segment": "wing_borne_climb",
            "altitude_mid_m": state["altitude_m"],
            "altitude_m": state["altitude_end_m"],
            "speed_m_s": state["TAS_m_s"],
            "EAS_m_s": state["EAS_m_s"],
            "CL": state["CL"],
            "rate_of_climb_m_s": state["rate_of_climb_m_s"],
            "electrical_power_W": state["electrical_power_used_W"],
            "delta_x_m": state["delta_x_m"],
            "spiral_step": state.get("spiral_step", False),
            "load_factor": state.get("load_factor", 1.0),
        })

    segments = {}
    for name, segment in mission["segments"].items():
        segments[name] = {
            "time_s": segment["time_s"],
            "energy_Wh": segment["energy_Wh"],
            "distance_m": segment["distance_m"],
            "average_electrical_power_W": segment["average_power_W"],
        }

    climb_power = max(state["electrical_power_used_W"] for state in climb_states)
    transition_power = mission["takeoff_transition"]["transition_power_W"]
    CL_allowed = permitted_lift_coefficient(aircraft)
    minimum_climb_EAS = math.sqrt(2.0 * weight_N / (isa_density(0.0) * wing["area_m2"] * CL_allowed))
    return {
        "CD_trim": CD_cruise,
        "CL_cruise": CL_cruise,
        "climb_power_W": climb_power,
        "peak_electrical_power_W": max(AIRCRAFT["hover_power_W"], climb_power, transition_power),
        "total_energy_Wh": mission["total_load_energy_Wh"],
        "installed_battery_energy_Wh": mission["installed_battery_energy_Wh"],
        "battery_mass_kg": mission["battery_mass_kg"],
        "profile": [(time, altitude) for time, _, altitude in mission["profile"]],
        "segment_summaries": segments,
        "states": states,
        "cruise_true_speed_m_s": final_speed,
        "optimized_climb_EAS_m_s": selected["EAS_m_s"],
        "optimized_climb_angle_deg": max(state["climb_angle_deg"] for state in climb_states),
        "outbound_time_s": mission["total_mission_time_s"] - MISSION["hover_time_s"],
        "total_mission_time_s": mission["total_mission_time_s"],
        "climb_horizontal_distance_m": selected["distance_m"],
        "level_cruise_distance_m": mission["cruise"]["distance_m"],
        "spiral_excess_ground_track_distance_m": mission["spiral_excess_distance_m"],
        "spiral_used": selected.get("spiral_used", False),
        "spiral_turn_radius_m": selected.get("spiral_radius_m"),
        "spiral_crossover_altitude_m": selected.get("spiral_crossover_altitude_m"),
        "spiral_arc_m": selected.get("spiral_arc_m", 0.0),
        "spiral_max_load_factor": selected.get("max_load_factor", 1.0),
        "spiral_max_bank_angle_deg": selected.get("max_bank_angle_deg", 0.0),
        "mission_grid": {
            "climb_EAS_m_s": [selected["EAS_m_s"]],
            "aerodynamic_speed_limits": {
                "CL_allowed": CL_allowed,
                "minimum_climb_EAS_m_s": minimum_climb_EAS,
            }
        },
        "course_climb_available_power_W": selected["available_electrical_power_W"],
        "course_climb_average_power_W": selected["average_electrical_power_used_W"],
        "course_climb_time_s": selected["time_s"],
        "course_climb_max_thrust_to_weight": selected["max_thrust_to_weight"],
        "course_climb_thrust_limit": aircraft["course_climb_max_thrust_to_weight"],
        "course_climb_complies_time": selected.get("complies_with_time_limit", True),
    }


def trim_drag_descriptor(result):
    """Canard/arm descriptor for the two-surface drag model, from a solved pass.

    Pulls the selected canard planform and the wing/canard aerodynamic-centre arms
    to the CG (same geometry the scissor uses), plus the aircraft zero-lift
    pitching moment, so the next mission can split lift/drag.
    """
    wing = result["wing"]
    selected = result["selected"]
    canard = selected["canard"]
    mass = selected["mass"]
    coeffs = selected.get("coeffs") or {}
    x_cg = mass["x_cg_fuselage_m"]
    canard_ac_x = MASS["nose_to_canard_m"] + 0.25 * canard["chord_m"]
    wing_ac_x = mass["wing_mac_le_x_m"] + wing["x_ac_m"]
    canard_root_chord = (
        2.0 * canard["area_m2"]
        / (canard["span_m"] * (1.0 + AIRCRAFT["canard_taper"]))
    )
    return {
        "S_c": canard["area_m2"],
        "b_c": canard["span_m"],
        "l_w": abs(wing_ac_x - x_cg),
        "l_c": abs(x_cg - canard_ac_x),
        "Cm_ac": coeffs.get("cmac", AIRCRAFT.get("wing_airfoil_cm0", 0.0)),
        # Geometry for the component drag build-up (canard planform + fuselage).
        "c_c": canard["chord_m"],
        "c_c_root": canard_root_chord,
        "L_fus": mass["fuselage_length_m"],
        "fus_width": MASS["fuselage_width_m"],
    }
