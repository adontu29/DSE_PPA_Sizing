
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mission_energy_course import build_course_mission, optimize_course_climb, permitted_lift_coefficient
from scissor_plot import (
    mach_number,
    datcom_lift_slope,
    aircraft_less_canard_lift_slope,
    aerodynamic_centre_over_mac,
    zero_lift_pitching_moment,
    scissor_cg_limits,
)
from xfoil_wrapper import (
    analyze_airfoil_pair,
    datcom_efficiency_from_section_slope,
    mach_number as xfoil_mach_number,
    reynolds_number,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs")

MISSION = {
    "altitude_m": 6000.0,
    "range_m": 6000.0,
    "time_budget_s": 600.0,
    "hover_time_s": 300.0,
    "vertical_takeoff_height_m": 20.0,
    "vertical_takeoff_rate_m_s": 2.0,
    "transition_time_s": 12.0,
    "average_climb_rate_m_s": 10.0,
    "altitude_step_m": 100.0,
    "allow_spiral_climb": True,
    "mission_equipment_mass_kg": 7.3,
}

AIRCRAFT = {
    "MTOW_kg": 52.78,
    "g_m_s2": 9.80665,
    "wing_area_m2": 6.8,
    "wing_sweep_deg:": 0.0,
    "canard_sweep_deg": 0.0,
    "wing_area_sweep_min_m2": 0.15,
    "wing_area_sweep_max_m2": 8.0,
    "wing_area_sweep_step_m2": 0.05,
    "climb_stall_margin_n": 1.25,
    "sizing_iteration_count": 8,
    "sizing_mass_tolerance_kg": 0.02,
    "wing_aspect_ratio": 7.0,
    "wing_taper": 0.40,
    "wing_CL_max": 1.28,
    "wing_CL_alpha_per_rad": 4.74,
    "canard_CL_max": 1.02,
    "canard_CL_alpha_per_rad": 4.25,
    "canard_CL_limit_fraction": 0.90,
    "canard_aspect_ratio": 5.0,
    "canard_taper": 0.50,
    # Wing longitudinal position is now solved as the canard->wing arm (the
    # canard is pinned to the nose, see MASS["nose_to_canard_m"]). These bound
    # the solve; the wing MAC LE station = nose_to_canard + arm.
    "canard_arm_min_m": 0.30,
    "canard_arm_max_m": 4.10,
    "canard_area_ratio_min": 0.05,
    "canard_area_ratio_max": 0.80,
    "canard_area_ratio_step": 0.005,
    "static_margin": 0.05,
    "cg_envelope_half_width_over_mac": 0.05,
    "cg_margin_over_mac": 0.02,
    "CD0": 0.040,
    "oswald_efficiency": 0.78,
    # --- Scissor-plot aerodynamics (TU Delft AE3211-I, Lectures 7 & 8) ---
    "datcom_eta": 0.95,                   # DATCOM airfoil efficiency (0.90-1.0)
    "wing_sweep_quarter_chord_deg": 0.0,  # straight wing for this UAV
    "wing_sweep_half_chord_deg": 0.0,
    "canard_sweep_half_chord_deg": 0.0,
    "wing_airfoil_cm0": -0.05,            # TODO: airfoil Cm0 from XFOIL (negative if cambered)
    "wing_CL0": 0.20,                     # TODO: aircraft-less-canard CL at alpha=0, from XFOIL
    "wing_datcom_eta": 0.95,
    "canard_datcom_eta": 0.95,
    "use_xfoil_airfoil_updates": True,
    "xfoil_path": "xfoil/xfoilp4.exe",
    "xfoil_sd7037_file": "xfoil/sd7037.dat",
    "xfoil_transition_x_c": 0.50,
    "xfoil_reynolds_rounding": 0.0,
    "xfoil_reynolds_update_threshold": 5000.0,
    "xfoil_mach_rounding": 0.02,
    "xfoil_mach_command_min": 0.20,
    "xfoil_alpha_start_deg": -6.0,
    "xfoil_alpha_end_deg": 18.0,
    "xfoil_alpha_step_deg": 1.0,
    "xfoil_timeout_s": 30.0,
    "xfoil_clmax_to_aircraft_factor": 0.90,
    # Full update step: XFOIL now runs in the OUTER loop (no oscillation risk),
    # so relaxing is unnecessary and only forces extra full sweeps before the
    # coefficients settle. 1.0 = converge in ~2 outer passes.
    "xfoil_update_relaxation": 1.0,
    "xfoil_update_tolerance_fraction": 0.005,
    # XFOIL runs in an OUTER Reynolds-feedback loop around the whole wing-area
    # sweep (not inside the sizing iterations): each outer pass = one full sweep
    # + one XFOIL airfoil-pair run, repeated until the section coefficients stop
    # changing. A handful of XFOIL calls total instead of thousands.
    "xfoil_outer_iteration_count": 4,
    "cruise_true_speed_m_s": 15.0,
    "minimum_cruise_true_speed_m_s": 0.0,
    "mission_CL_limit_fraction": 0.90,
    "cruise_stall_margin_deg": 3.0,
    "max_affordable_electrical_power_W": 18000.0,
    "minimum_power_margin_fraction": 0.05,
    "max_fixed_wing_climb_angle_deg": 30.0,
    "transition_accel_m_s2": 1.0,
    "transition_stall_margin_n": 1.25,
    "transition_wing_lift_fraction_complete": 0.90,
    "transition_thrust_margin": 1.15,
    "transition_blend_start_fraction": 0.50,
    "transition_sample_count": 9,
    "forward_flight_efficiency": 0.75 * 0.90 * 0.95,
    "hover_power_W": 14000.0,
    "battery_specific_energy_Wh_kg": 310.0,
    "battery_usable_fraction": 0.85,
    "battery_efficiency": 0.95,
    "n_rotors": 4,
    "thrust_to_weight": 1.30,
    "disc_loading_N_m2": 170.0,
    # --- Maximum stall speed (the wing-area lower bound) ---
    # If this is a number (m/s, EAS) it is used directly as the stall cap and the
    # back-transition logic below is ignored. Set it to None to instead DERIVE the
    # cap from the back-transition deceleration budget further down.
    "max_stall_EAS_m_s": 16.0,   # hand-set from a separate back-transition calc
    # --- Back-transition requirement that sizes the wing (used only when the
    #     override above is None) ---
    # The aircraft must decelerate from wing-borne flight to a 0 m/s hover
    # within this much horizontal ground track at the transition altitude.
    # SMALLER distance = more demanding = lower stall-speed cap = bigger wing.
    "back_transition_distance_budget_m": 30.0,
    # Optionally specify a time budget (s) instead; if not None it overrides the distance.
    "back_transition_time_budget_s": None,
    # The manoeuvre is entered at this multiple of the stall speed (1.3 = airworthiness-style
    # approach margin; set to 1.0 to decelerate from the stall speed itself).
    "back_transition_approach_speed_factor": 1.30,
     
    "battery_cg_offset_over_mac": -0.25,   # battery CG as fraction of MAC, measured from wing MAC LE
                                       # 0.0 = at MAC LE, 0.25 = at quarter-chord, negative = forward
}

MASS = {
    # Areal/linear densities calibrated to colleague estimates:
    "wing_areal_density_kg_m2": 3.36,    # 12 kg @ 5 m span, AR 7 (3.57 m^2)
    "canard_areal_density_kg_m2": 2.50,  # 2 kg @ 2 m span, AR 5 (0.80 m^2)
    "fuselage_linear_density_kg_m": 2.61,  # 6 kg @ 2.30 m
    # Fuselage length is DERIVED from the layout, not fixed: the canard sits
    # nose_to_canard behind the nose, the wing sits wing_to_tail ahead of the
    # tail, so L_fus = nose_to_canard + arm + wing_to_tail and grows with the
    # canard->wing arm (which is the solved wing position).
    "nose_to_canard_m": 0.35,
    "wing_to_tail_m": 0.0,            # wing essentially at the fuselage tail
    "fuselage_width_m": 0.30,
    "nose_bay_x_m": 0.10,             # station of nose electronics/payload
    "boom_landing_gear_mass_kg": 2.50,   # struts/landing gear, at the wing
    "motor_mass_each_kg": 2.8,        # per motor; on struts from the wing
    "esc_specific_mass_kg_W": 0.00008,
    "prop_mass_coeff_kg_m2": 0.10,
    "avionics_mass_kg": 0.60,         # in the nose bay
    "sensor_mass_kg": 1.00,           # sensors in the nose
    # Balloon-capture subsystem
    "net_gun_mass_kg": 1.50,          # net + netgun, in the nose
    "reel_mass_kg": 0.50,             # reel, ~at the CG (mid-body)
    "parachute_mass_kg": 0.60,        # parachute, just ahead of the wing LE
    "parachute_ahead_of_wing_le_m": 0.10,
    "wiring_fraction": 0.06,
    "contingency_fraction": 0.08,

}

AIRFOIL_AERO_KEYS = (
    "wing_CL_max",
    "wing_CL_alpha_per_rad",
    "wing_airfoil_cm0",
    "wing_CL0",
    "wing_datcom_eta",
    "canard_CL_max",
    "canard_CL_alpha_per_rad",
    "canard_datcom_eta",
)
BASE_AIRFOIL_AERO = {key: AIRCRAFT[key] for key in AIRFOIL_AERO_KEYS}


def reset_airfoil_aero_defaults():
    """Restore the fallback airfoil assumptions before a new sizing run."""
    AIRCRAFT.update(BASE_AIRFOIL_AERO)


def snapshot_aircraft():
    """Copy the current aircraft input state for later reporting."""
    return dict(AIRCRAFT)


def restore_aircraft_snapshot(snapshot):
    """Restore a stored aircraft state without replacing the global object."""
    AIRCRAFT.update(snapshot)


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, rem_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{minutes:.0f}m {rem_seconds:.0f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{hours:.0f}h {rem_minutes:.0f}m"


def progress(message, enabled=True, indent=0):
    if enabled:
        print(f"{'  ' * indent}{message}", flush=True)


# ---------------------------------------------------------------------------
# Atmosphere and wing equations
# ---------------------------------------------------------------------------


def isa_density(altitude_m):
    """Troposphere ISA density."""
    rho0 = 1.225
    temperature0 = 288.15
    lapse = -0.0065
    gas_constant = 287.05
    gravity = AIRCRAFT["g_m_s2"]
    temperature = temperature0 + lapse * altitude_m
    pressure_ratio = (temperature / temperature0) ** (-gravity / (lapse * gas_constant))
    return rho0 * pressure_ratio * temperature0 / temperature


def representative_xfoil_condition(mission):
    """Mission state used for Reynolds-dependent 2D airfoil analysis."""
    candidates = []
    for state in mission.get("states", []):
        if state.get("segment") != "wing_borne_climb":
            continue
        speed = state.get("speed_m_s")
        if speed is None or speed <= 0.0:
            continue
        altitude = state.get("altitude_mid_m", state.get("altitude_m", MISSION["altitude_m"]))
        unit_re = reynolds_number(altitude, speed, 1.0)
        candidates.append({
            "source": "minimum_Re_wing_borne_climb",
            "altitude_m": altitude,
            "true_speed_m_s": speed,
            "mach": xfoil_mach_number(altitude, speed),
            "unit_re_per_m": unit_re,
            "CL": state.get("CL"),
        })

    if candidates:
        return min(candidates, key=lambda item: item["unit_re_per_m"])

    altitude = MISSION["altitude_m"]
    speed = mission.get("cruise_true_speed_m_s", AIRCRAFT["cruise_true_speed_m_s"])
    return {
        "source": "cruise_fallback",
        "altitude_m": altitude,
        "true_speed_m_s": speed,
        "mach": xfoil_mach_number(altitude, speed),
        "unit_re_per_m": reynolds_number(altitude, speed, 1.0),
        "CL": mission.get("CL_cruise"),
    }


def update_airfoil_aerodynamics_from_xfoil(
    wing, mission, selected, show_progress=False, progress_indent=0
):
    """Update section-derived aero inputs using the current mission geometry."""
    enabled = bool(AIRCRAFT.get("use_xfoil_airfoil_updates", False))
    update = {
        "enabled": enabled,
        "changed": False,
        "condition": None,
        "updates": {},
        "wing": None,
        "canard": None,
        "warnings": [],
    }
    if not enabled:
        return update

    canard = selected["canard"]
    condition = representative_xfoil_condition(mission)
    wing_re = condition["unit_re_per_m"] * wing["chord_m"]
    canard_re = condition["unit_re_per_m"] * canard["chord_m"]
    condition = dict(condition)
    condition["wing_reynolds"] = wing_re
    condition["canard_reynolds"] = canard_re
    condition["wing_chord_m"] = wing["chord_m"]
    condition["canard_chord_m"] = canard["chord_m"]
    update["condition"] = condition

    progress(
        (
            "XFOIL airfoils: "
            f"Re_w={wing_re / 1e6:.3f}e6, "
            f"Re_c={canard_re / 1e6:.3f}e6, "
            f"M={condition['mach']:.3f}"
        ),
        show_progress,
        progress_indent,
    )

    analysis = analyze_airfoil_pair(
        xfoil_path=AIRCRAFT["xfoil_path"],
        sd7037_file=AIRCRAFT["xfoil_sd7037_file"],
        wing_reynolds=wing_re,
        canard_reynolds=canard_re,
        mach=condition["mach"],
        x_transition=AIRCRAFT["xfoil_transition_x_c"],
        reynolds_rounding=AIRCRAFT["xfoil_reynolds_rounding"],
        reynolds_update_threshold=AIRCRAFT["xfoil_reynolds_update_threshold"],
        mach_rounding=AIRCRAFT["xfoil_mach_rounding"],
        alpha_start_deg=AIRCRAFT["xfoil_alpha_start_deg"],
        alpha_end_deg=AIRCRAFT["xfoil_alpha_end_deg"],
        alpha_step_deg=AIRCRAFT["xfoil_alpha_step_deg"],
        mach_command_min=AIRCRAFT["xfoil_mach_command_min"],
        timeout_s=AIRCRAFT["xfoil_timeout_s"],
    )
    update["wing"] = analysis["wing"]
    update["canard"] = analysis["canard"]
    update["warnings"] = analysis["warnings"]

    values = {}
    clmax_factor = AIRCRAFT["xfoil_clmax_to_aircraft_factor"]
    if analysis["wing"] is not None:
        wing_eta = datcom_efficiency_from_section_slope(
            analysis["wing"]["cl_alpha_per_rad"]
        )
        values.update({
            "wing_CL_max": clmax_factor * analysis["wing"]["cl_max"],
            "wing_CL0": analysis["wing"]["cl_at_alpha0"],
            "wing_airfoil_cm0": analysis["wing"]["cm_at_zero_lift"],
            "wing_datcom_eta": wing_eta,
            "wing_CL_alpha_per_rad": datcom_lift_slope(
                AIRCRAFT["wing_aspect_ratio"],
                condition["mach"],
                math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]),
                wing_eta,
            ),
        })

    if analysis["canard"] is not None:
        canard_eta = datcom_efficiency_from_section_slope(
            analysis["canard"]["cl_alpha_per_rad"]
        )
        values.update({
            "canard_CL_max": clmax_factor * analysis["canard"]["cl_max"],
            "canard_datcom_eta": canard_eta,
            "canard_CL_alpha_per_rad": datcom_lift_slope(
                AIRCRAFT["canard_aspect_ratio"],
                condition["mach"],
                math.radians(AIRCRAFT["canard_sweep_half_chord_deg"]),
                canard_eta,
            ),
        })

    tolerance = AIRCRAFT["xfoil_update_tolerance_fraction"]
    relaxation = AIRCRAFT["xfoil_update_relaxation"]
    for key, target_value in values.items():
        old_value = AIRCRAFT[key]
        new_value = old_value + relaxation * (target_value - old_value)
        AIRCRAFT[key] = new_value
        limit = tolerance * max(1.0, abs(old_value))
        changed = abs(new_value - old_value) > limit
        update["changed"] = update["changed"] or changed
        update["updates"][key] = {
            "old": old_value,
            "target": target_value,
            "new": new_value,
            "changed": changed,
        }

    if update["updates"]:
        progress(
            (
                "XFOIL update: "
                f"CLmax_w={AIRCRAFT['wing_CL_max']:.3f}, "
                f"CLmax_c={AIRCRAFT['canard_CL_max']:.3f}, "
                f"eta_w={AIRCRAFT['wing_datcom_eta']:.3f}, "
                f"eta_c={AIRCRAFT['canard_datcom_eta']:.3f}, "
                f"changed={update['changed']}"
            ),
            show_progress,
            progress_indent,
        )
    if update["warnings"]:
        progress(
            "XFOIL warning: " + " | ".join(update["warnings"]),
            show_progress,
            progress_indent,
        )

    return update


def back_transition_stall_limit():
    """Largest stall speed the wing may have, derived from the back-transition.

    To decelerate from wing-borne flight to a 0 m/s hover while holding altitude,
    the tailsitter tilts nose-up so the rotor thrust both carries the weight and
    pushes backwards. With thrust-to-weight T/W the largest deceleration that
    still holds altitude is:
        a_max = g * sqrt((T/W)^2 - 1)
    The manoeuvre is entered at (approach_factor * stall speed). Requiring it to
    finish within the distance (or time) budget caps the stall speed. The cap is
    returned as an EAS so it lines up with wing["stall_EAS_m_s"]; the dynamics
    themselves are worked in true airspeed at the transition altitude.
    """
    # If the user pinned a maximum stall speed, that value leads and we skip the
    # back-transition derivation entirely (the dynamics fields are left as None).
    if AIRCRAFT["max_stall_EAS_m_s"] is not None:
        return {
            "source": "user-specified",
            "a_max_m_s2": None,
            "entry_TAS_m_s": None,
            "stall_TAS_m_s": None,
            "stall_EAS_max_m_s": AIRCRAFT["max_stall_EAS_m_s"],
            "transition_time_s": None,
            "transition_distance_m": None,
        }

    g = AIRCRAFT["g_m_s2"]
    thrust_to_weight = AIRCRAFT["thrust_to_weight"]
    approach_factor = AIRCRAFT["back_transition_approach_speed_factor"]

    a_max = g * math.sqrt(thrust_to_weight**2 - 1.0)            # m/s^2, decel holding altitude

    # Highest entry (approach) true airspeed that still stops within the budget.
    if AIRCRAFT["back_transition_time_budget_s"] is not None:
        entry_TAS = a_max * AIRCRAFT["back_transition_time_budget_s"]
    else:
        entry_TAS = math.sqrt(2.0 * a_max * AIRCRAFT["back_transition_distance_budget_m"])

    stall_TAS = entry_TAS / approach_factor                     # true airspeed at altitude

    # Convert the true-airspeed cap to EAS (wing stall is stored as an EAS).
    density_ratio = isa_density(MISSION["altitude_m"]) / isa_density(0.0)
    stall_EAS_max = stall_TAS * math.sqrt(density_ratio)

    return {
        "source": "back-transition",
        "a_max_m_s2": a_max,
        "entry_TAS_m_s": entry_TAS,
        "stall_TAS_m_s": stall_TAS,
        "stall_EAS_max_m_s": stall_EAS_max,
        "transition_time_s": entry_TAS / a_max,
        "transition_distance_m": entry_TAS**2 / (2.0 * a_max),
    }


def propeller_disk_estimate(weight_N):
    thrust_total = AIRCRAFT["thrust_to_weight"] * weight_N
    thrust_per_rotor = thrust_total / AIRCRAFT["n_rotors"]
    disk_area = thrust_per_rotor / AIRCRAFT["disc_loading_N_m2"]
    diameter = 2.0 * math.sqrt(disk_area / math.pi)
    return {
        "thrust_total_N": thrust_total,
        "thrust_per_rotor_N": thrust_per_rotor,
        "disk_area_m2": disk_area,
        "propeller_diameter_m": diameter,
    }


def wing_geometry(weight_N, rho_cruise, wing_area=None):
    """Wing geometry and reference lift condition."""
    if wing_area is None:
        wing_area = AIRCRAFT["wing_area_m2"]
    aspect_ratio = AIRCRAFT["wing_aspect_ratio"]
    span = math.sqrt(wing_area * aspect_ratio)
    chord = wing_area / span
    root_chord = 2.0 * wing_area / (span * (1.0 + AIRCRAFT["wing_taper"]))
    tip_chord = AIRCRAFT["wing_taper"] * root_chord
    x_ac = 0.25 * chord

    rho_sea_level = isa_density(0.0)
    stall_EAS = math.sqrt(2.0 * weight_N / (rho_sea_level * wing_area * AIRCRAFT["wing_CL_max"]))
    cruise_speed = AIRCRAFT["cruise_true_speed_m_s"]
    q_cruise = 0.5 * rho_cruise * cruise_speed**2
    CL_trim = weight_N / (q_cruise * wing_area)

    return {
        "area_m2": wing_area,
        "span_m": span,
        "chord_m": chord,
        "root_chord_m": root_chord,
        "tip_chord_m": tip_chord,
        "x_ac_m": x_ac,
        "stall_EAS_m_s": stall_EAS,
        "cruise_true_speed_m_s": cruise_speed,
        "CL_trim": CL_trim,
        "CL_alpha_per_rad": AIRCRAFT["wing_CL_alpha_per_rad"],
    }


# ---------------------------------------------------------------------------
# Canard, mass, and CG equations
# ---------------------------------------------------------------------------


def canard_geometry(area_ratio, wing):
    """Canard planform from selected area ratio.

    Longitudinal position is no longer set here: the canard is pinned to the
    nose and the wing position (canard->wing arm) is solved in
    canard_and_wing_iteration, so the arm lives in the layout, not the planform.
    """
    area = area_ratio * wing["area_m2"]
    span = math.sqrt(area * AIRCRAFT["canard_aspect_ratio"])
    chord = area / span
    return {
        "area_ratio": area_ratio,
        "area_m2": area,
        "span_m": span,
        "chord_m": chord,
        "CL_alpha_per_rad": AIRCRAFT["canard_CL_alpha_per_rad"],
        "usable_CL": AIRCRAFT["canard_CL_limit_fraction"] * AIRCRAFT["canard_CL_max"],
    }


def longitudinal_layout(wing, canard, wing_le_m):
    """Nose-referenced longitudinal layout for a given wing MAC LE station.

    x is measured aft from the nose. The canard MAC LE is pinned at
    nose_to_canard; the wing MAC LE is at wing_le_m; the fuselage tail is
    wing_to_tail behind the wing, so L_fus = wing_le_m + wing_to_tail and the
    arm (and fuselage length) grow as the wing moves aft.
    """
    c_w = wing["chord_m"]
    c_c = canard["chord_m"]
    nose_to_canard = MASS["nose_to_canard_m"]
    L_fus = wing_le_m + MASS["wing_to_tail_m"]
    return {
        "wing_le_m": wing_le_m,
        "canard_le_m": nose_to_canard,
        "arm_m": wing_le_m - nose_to_canard,                  # canard->wing, > 0
        "L_fus_m": L_fus,
        "wing_quarter_m": wing_le_m + 0.25 * c_w,             # mass/AC station
        "canard_quarter_m": nose_to_canard + 0.25 * c_c,
        "lh_over_mac": (nose_to_canard - wing_le_m) / c_w,    # < 0 (canard ahead)
    }


def mass_and_cg(wing, canard, mission, propeller, wing_le_m):
    """Mass build-up and CG for a wing MAC LE station (measured aft of nose).

    Longitudinal layout (x aft of nose), from colleague mass estimates:
      * nose bay (sensors, avionics, net+netgun) ........ nose_bay_x_m
      * canard ........................................... pinned at nose_to_canard
      * fuselage (uniform body) .......................... centroid L_fus/2
      * reel (balloon subsystem, ~at CG) ................. mid-body
      * parachute ........................................ just ahead of wing LE
      * wing, motors+ESCs+props (on wing struts), booms,
        battery .......................................... wing station
    L_fus = wing_le_m + wing_to_tail grows with the canard->wing arm.
    """
    layout = longitudinal_layout(wing, canard, wing_le_m)
    L_fus = layout["L_fus_m"]
    c_w = wing["chord_m"]
    n_rotors = AIRCRAFT["n_rotors"]
    power_for_motor_sizing = max(
        AIRCRAFT["hover_power_W"],
        mission.get("peak_electrical_power_W", mission["climb_power_W"]),
    )

    nose_x = MASS["nose_bay_x_m"]
    mid_x = 0.5 * L_fus
    wing_x = layout["wing_quarter_m"]
    parachute_x = wing_le_m - MASS["parachute_ahead_of_wing_le_m"]
    # Battery is a large CG-trim mass; its station is selectable. "wing" (aft)
    # is tail-heavy for a canard, "mid"/"nose" move the CG forward to help the
    # scissor close.

    battery_x = wing_le_m + AIRCRAFT.get("battery_cg_offset_over_mac", 0.0) * c_w

    # name -> (mass_kg, station_m aft of nose)
    components = {
        "wing":       (MASS["wing_areal_density_kg_m2"] * wing["area_m2"],      wing_x),
        "canard":     (MASS["canard_areal_density_kg_m2"] * canard["area_m2"],  layout["canard_quarter_m"]),
        "fuselage":   (MASS["fuselage_linear_density_kg_m"] * L_fus,            mid_x),
        "motors":     (n_rotors * MASS["motor_mass_each_kg"],                   wing_x),
        "ESCs":       (MASS["esc_specific_mass_kg_W"] * power_for_motor_sizing, wing_x),
        "propellers": (n_rotors * MASS["prop_mass_coeff_kg_m2"]
                       * propeller["propeller_diameter_m"] ** 2,                wing_x),
        "boom_landing_gear": (MASS["boom_landing_gear_mass_kg"],               wing_x),
        "battery":    (mission["battery_mass_kg"],                             battery_x),
        "avionics":   (MASS["avionics_mass_kg"],                               nose_x),
        "sensors":    (MASS["sensor_mass_kg"],                                 nose_x),
        "net_gun":    (MASS["net_gun_mass_kg"],                                nose_x),
        "reel":       (MASS["reel_mass_kg"],                                   mid_x),
        "parachute":  (MASS["parachute_mass_kg"],                              parachute_x),
    }
    masses = {name: m for name, (m, _) in components.items()}
    locations = {name: x for name, (_, x) in components.items()}

    # Wiring scales with the powered systems; contingency on the full subtotal.
    masses["wiring"] = MASS["wiring_fraction"] * (
        masses["motors"] + masses["ESCs"] + masses["avionics"]
    )
    locations["wiring"] = mid_x
    subtotal = sum(masses.values())
    masses["contingency"] = MASS["contingency_fraction"] * subtotal
    locations["contingency"] = mid_x

    total_mass = sum(masses.values())
    x_cg_nose_m = sum(masses[name] * locations[name] for name in masses) / total_mass
    x_cg_m = x_cg_nose_m - wing_le_m                 # relative to wing MAC LE

    return {
        "total_mass_kg":        total_mass,
        "masses_kg":            masses,
        "locations_fuselage_m": locations,
        "wing_mac_le_x_m":      wing_le_m,
        "arm_m":                layout["arm_m"],
        "fuselage_length_m":    L_fus,
        "x_cg_fuselage_m":      x_cg_nose_m,
        "x_cg_m":               x_cg_m,
        "x_cg_over_mac":        x_cg_m / wing["chord_m"],
    }

def evaluate_wing_station(wing, canard, area_ratio, mission, propeller, wing_le_m):
    """Scissor band, mass CG, and CG-envelope fit clearance at one wing station.

    Both the band (via the arm-dependent lh and fuselage-length-dependent x_ac)
    and the CG depend on wing_le_m, so they are evaluated together. `clearance`
    is the worst-side gap between the operational CG envelope and the scissor
    band (>= 0 means the envelope fits): positive is feasible, and maximising it
    is the right objective for placing the wing, because the band *width* also
    changes with the arm (merely centring the CG can land on a narrow band).
    """
    half_width = AIRCRAFT["cg_envelope_half_width_over_mac"]
    margin = AIRCRAFT["cg_margin_over_mac"]

    layout = longitudinal_layout(wing, canard, wing_le_m)
    coeffs = scissor_coefficients(wing, layout["L_fus_m"], layout["lh_over_mac"])
    scissor = scissor_limits(area_ratio, coeffs)
    mass = mass_and_cg(wing, canard, mission, propeller, wing_le_m)

    x_cg = mass["x_cg_over_mac"]
    lower_clear = (x_cg - half_width) - (scissor["x_forward_over_mac"] + margin)
    upper_clear = (scissor["x_aft_over_mac"] - margin) - (x_cg + half_width)
    return {
        "wing_le_m": wing_le_m,
        "layout": layout,
        "coeffs": coeffs,
        "scissor": scissor,
        "mass": mass,
        "band_center": 0.5 * (scissor["x_forward_over_mac"] + scissor["x_aft_over_mac"]),
        "lower_clearance": lower_clear,
        "upper_clearance": upper_clear,
        "clearance": min(lower_clear, upper_clear),
    }


def best_wing_station_by_clearance(wing, canard, area_ratio, mission, propeller, lo, hi):
    """Return the bounded wing station with the largest CG-envelope clearance."""
    sample_count = 41
    best = None
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        wing_le = lo + fraction * (hi - lo)
        evaluation = evaluate_wing_station(
            wing, canard, area_ratio, mission, propeller, wing_le
        )
        if best is None or evaluation["clearance"] > best["clearance"]:
            best = evaluation

    sample_step = (hi - lo) / (sample_count - 1)
    left = max(lo, best["wing_le_m"] - sample_step)
    right = min(hi, best["wing_le_m"] + sample_step)
    for _ in range(24):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        e1 = evaluate_wing_station(wing, canard, area_ratio, mission, propeller, m1)
        e2 = evaluate_wing_station(wing, canard, area_ratio, mission, propeller, m2)
        for evaluation in (e1, e2):
            if evaluation["clearance"] > best["clearance"]:
                best = evaluation
        if e1["clearance"] < e2["clearance"]:
            left = m1
        else:
            right = m2
        if (right - left) < 1e-3:
            break
    return best


def solve_wing_station(wing, canard, area_ratio, mission, propeller):
    """Solve the smallest arm (shortest, lightest fuselage) that fits the envelope.

    Moving the wing aft lengthens the arm, which monotonically widens the scissor
    band and raises the CG-envelope fit clearance. So the lightest viable design
    is the smallest wing station whose clearance >= 0. We bisect for that lower
    crossing. If the longest arm is still infeasible, a bounded clearance search
    returns the closest candidate so the sweep reports a useful near miss.
    """
    lo = MASS["nose_to_canard_m"] + AIRCRAFT["canard_arm_min_m"]
    hi = MASS["nose_to_canard_m"] + AIRCRAFT["canard_arm_max_m"]

    def evaluate(wing_le):
        return evaluate_wing_station(wing, canard, area_ratio, mission, propeller, wing_le)

    eval_lo, eval_hi = evaluate(lo), evaluate(hi)
    if eval_hi["clearance"] < 0.0:
        return best_wing_station_by_clearance(
            wing, canard, area_ratio, mission, propeller, lo, hi
        )
    if eval_lo["clearance"] >= 0.0:
        return eval_lo                      # shortest arm already fits

    # clearance(lo) < 0 <= clearance(hi): bisect for the smallest feasible arm.
    a, best = lo, eval_hi
    for _ in range(40):
        mid = 0.5 * (a + best["wing_le_m"])
        eval_mid = evaluate(mid)
        if eval_mid["clearance"] >= 0.0:
            best = eval_mid
        else:
            a = mid
        if (best["wing_le_m"] - a) < 1e-3:
            break
    return best


# ---------------------------------------------------------------------------
# Scissor plot equations
# ---------------------------------------------------------------------------


def scissor_coefficients(wing, fuselage_length, lh_over_mac):
    """Course-method aerodynamic inputs for the canard scissor plot.

    `fuselage_length` (= L_fus) and `lh_over_mac` (canard->wing arm / c_bar, < 0)
    come from the longitudinal layout and change as the wing moves, so this is
    re-evaluated per candidate wing station. The equations and the canard sign
    conventions live in scissor_plot.py.
    """
    cruise_speed = wing.get("cruise_true_speed_m_s", AIRCRAFT["cruise_true_speed_m_s"])
    mach = mach_number(cruise_speed, MISSION["altitude_m"])
    wing_eta = AIRCRAFT.get("wing_datcom_eta", AIRCRAFT["datcom_eta"])
    canard_eta = AIRCRAFT.get("canard_datcom_eta", AIRCRAFT["datcom_eta"])
    fuselage_width = MASS["fuselage_width_m"]
    mac = wing["chord_m"]                          # geometric mean chord used as c_bar
    mean_geo_chord = wing["area_m2"] / wing["span_m"]

    cl_alpha_wing = datcom_lift_slope(
        AIRCRAFT["wing_aspect_ratio"], mach,
        math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]), wing_eta,
    )
    cl_alpha_canard = datcom_lift_slope(           # (Vh/V) = 1 for a canard
        AIRCRAFT["canard_aspect_ratio"], mach,
        math.radians(AIRCRAFT["canard_sweep_half_chord_deg"]), canard_eta,
    )
    cl_alpha_A_h = aircraft_less_canard_lift_slope(
        cl_alpha_wing, fuselage_width, wing["span_m"],
        wing["area_m2"], wing["root_chord_m"],
    )

    nose_length = fuselage_length - wing["root_chord_m"]   # estimate of l_fn
    x_ac_over_mac = aerodynamic_centre_over_mac(
        cl_alpha_A_h, fuselage_width, nose_length, wing["area_m2"], mac,
        wing["span_m"], AIRCRAFT["wing_taper"], mean_geo_chord,
        math.radians(AIRCRAFT["wing_sweep_quarter_chord_deg"]),
    )
    cmac = zero_lift_pitching_moment(
        AIRCRAFT["wing_airfoil_cm0"], AIRCRAFT["wing_aspect_ratio"],
        math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]), cl_alpha_A_h,
        fuselage_width, fuselage_length, wing["area_m2"], mac, AIRCRAFT["wing_CL0"],
    )

    # Controllability is sized at the most demanding wing-borne lift, i.e. the
    # slowest wing-borne flight: CL_max limited by the stall margin. For a
    # tailsitter there is no flaps-down approach case (VTOL handles low speed).
    cl_A_h_control = AIRCRAFT["wing_CL_max"] / AIRCRAFT["climb_stall_margin_n"] ** 2

    return {
        "mach": mach,
        "cl_alpha_wing": cl_alpha_wing,
        "cl_alpha_canard": cl_alpha_canard,
        "cl_alpha_A_h": cl_alpha_A_h,
        "x_ac_over_mac": x_ac_over_mac,
        "cmac": cmac,
        # Canard control surface lifts up: CL_h > 0 (its usable max), not -1.
        "cl_h_control": AIRCRAFT["canard_CL_limit_fraction"] * AIRCRAFT["canard_CL_max"],
        "cl_A_h_control": cl_A_h_control,
        "lh_over_mac": lh_over_mac,                              # < 0 for a canard
        "static_margin": AIRCRAFT["static_margin"],
    }


def scissor_limits(area_ratio, coeffs):
    """Forward (controllability) and aft (stability) CG limits at this Sc/Sw."""
    if coeffs["cl_A_h_control"] <= 0.0:
        return None
    x_forward, x_aft = scissor_cg_limits(
        area_ratio,
        x_ac_over_mac=coeffs["x_ac_over_mac"],
        static_margin=coeffs["static_margin"],
        cl_alpha_canard=coeffs["cl_alpha_canard"],
        cl_alpha_A_h=coeffs["cl_alpha_A_h"],
        cl_h_control=coeffs["cl_h_control"],
        cl_A_h_control=coeffs["cl_A_h_control"],
        cmac=coeffs["cmac"],
        lh_over_mac=coeffs["lh_over_mac"],
    )
    return {
        "x_forward_over_mac": x_forward,
        "x_aft_over_mac": x_aft,
        "cg_range_over_mac": x_aft - x_forward,
        "CL_Ah": coeffs["cl_A_h_control"],
    }


def canard_and_wing_iteration(wing, mission, propeller):
    """Find the smallest canard area ratio that fits the operational CG envelope."""
    half_width = AIRCRAFT["cg_envelope_half_width_over_mac"]
    margin = AIRCRAFT["cg_margin_over_mac"]
    required_width = 2.0 * (half_width + margin)

    candidates = []
    best_candidate = None
    steps = int((AIRCRAFT["canard_area_ratio_max"] - AIRCRAFT["canard_area_ratio_min"]) / AIRCRAFT["canard_area_ratio_step"]) + 1
    for i in range(steps):
        area_ratio = AIRCRAFT["canard_area_ratio_min"] + i * AIRCRAFT["canard_area_ratio_step"]
        canard = canard_geometry(area_ratio, wing)

        # Solve the wing station (canard->wing arm) that fits the CG envelope in
        # the band; the band itself moves with the arm, so the two are solved jointly.
        solution = solve_wing_station(wing, canard, area_ratio, mission, propeller)
        scissor = solution["scissor"]
        mass = solution["mass"]
        if scissor is None:
            continue

        band_is_wide_enough = scissor["cg_range_over_mac"] >= required_width
        x_cg = mass["x_cg_over_mac"]
        operational_fwd = x_cg - half_width
        operational_aft = x_cg + half_width
        fits = (
            band_is_wide_enough
            and operational_fwd >= scissor["x_forward_over_mac"] + margin
            and operational_aft <= scissor["x_aft_over_mac"] - margin
        )

        candidate = {
            "canard": canard,
            "scissor": scissor,
            "mass": mass,
            "wing_le_m": solution["wing_le_m"],
            "arm_m": solution["layout"]["arm_m"],
            "coeffs": solution["coeffs"],
            "target_x_cg_over_mac": solution["band_center"],
            "operational_fwd_over_mac": operational_fwd,
            "operational_aft_over_mac": operational_aft,
            "lower_clearance_over_mac": solution["lower_clearance"],
            "upper_clearance_over_mac": solution["upper_clearance"],
            "clearance_over_mac": solution["clearance"],
            "band_is_wide_enough": band_is_wide_enough,
            "feasible": fits,
        }
        candidates.append(candidate)
        if (
            best_candidate is None
            or candidate["clearance_over_mac"] > best_candidate["clearance_over_mac"]
        ):
            best_candidate = candidate
        if fits:
            return candidate, candidates

    if best_candidate is None:
        raise RuntimeError("No canard/scissor candidates could be evaluated.")
    return best_candidate, candidates


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_key_value_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["quantity", "value"])
        for key, value in values.items():
            writer.writerow([key, value])


def write_mass_breakdown(path, mass):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["component", "mass_kg", "x_location_fuselage_m"])
        for name, component_mass in mass["masses_kg"].items():
            writer.writerow([name, component_mass, mass["locations_fuselage_m"][name]])


def write_iteration_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_full_summary(result):
    """Structured dict of geometry, aerodynamics, and mission parameters."""
    aircraft = result.get("aircraft", AIRCRAFT)
    wing = result["wing"]
    mission = result["mission"]
    propeller = result["propeller"]
    selected = result["selected"]
    canard = selected["canard"]
    mass = selected["mass"]
    scissor = selected["scissor"]
    coeffs = selected["coeffs"]   # evaluated at the solved wing station

    return {
        "mass": {
            "MTOW_estimate_kg": mass["total_mass_kg"],
            "MTOW_used_for_final_pass_kg": result["final_mass_used_kg"],
            "mass_closure_error_kg": mass["total_mass_kg"] - result["final_mass_used_kg"],
            "breakdown_kg": mass["masses_kg"],
        },
        "wing": {
            "area_m2": wing["area_m2"],
            "span_m": wing["span_m"],
            "mean_chord_m": wing["chord_m"],
            "root_chord_m": wing["root_chord_m"],
            "tip_chord_m": wing["tip_chord_m"],
            "aspect_ratio": aircraft["wing_aspect_ratio"],
            "taper": aircraft["wing_taper"],
            "x_ac_m_from_mac_le": wing["x_ac_m"],
            "stall_EAS_m_s": wing["stall_EAS_m_s"],
            "CL_max": aircraft["wing_CL_max"],
            "CL_trim_cruise": wing["CL_trim"],
            "mac_le_x_m_from_nose": mass["wing_mac_le_x_m"],
        },
        "canard": {
            "area_ratio_Sc_Sw": canard["area_ratio"],
            "area_m2": canard["area_m2"],
            "span_m": canard["span_m"],
            "chord_m": canard["chord_m"],
            "arm_m": selected["arm_m"],
            "le_m_from_nose": MASS["nose_to_canard_m"],
            "aspect_ratio": aircraft["canard_aspect_ratio"],
            "CL_max": aircraft["canard_CL_max"],
        },
        "fuselage": {
            "length_m": mass["fuselage_length_m"],
            "width_m": MASS["fuselage_width_m"],
        },
        "aerodynamics_scissor": {
            "mach": coeffs["mach"],
            "wing_datcom_eta": aircraft["wing_datcom_eta"],
            "canard_datcom_eta": aircraft["canard_datcom_eta"],
            "CL_alpha_wing_per_rad": coeffs["cl_alpha_wing"],
            "CL_alpha_canard_per_rad": coeffs["cl_alpha_canard"],
            "CL_alpha_aircraft_less_canard_per_rad": coeffs["cl_alpha_A_h"],
            "x_ac_over_mac": coeffs["x_ac_over_mac"],
            "Cmac": coeffs["cmac"],
            "CL_h_control": coeffs["cl_h_control"],
            "CL_Ah_control": coeffs["cl_A_h_control"],
            "lh_over_mac": coeffs["lh_over_mac"],
            "static_margin": coeffs["static_margin"],
            "scissor_forward_limit_x_over_mac": scissor["x_forward_over_mac"],
            "scissor_aft_limit_x_over_mac": scissor["x_aft_over_mac"],
            "scissor_cg_range_over_mac": scissor["cg_range_over_mac"],
            "CD_trim": mission["CD_trim"],
        },
        "cg": {
            "x_cg_over_mac": mass["x_cg_over_mac"],
            "x_cg_m_from_mac_le": mass["x_cg_m"],
            "operational_fwd_over_mac": selected["operational_fwd_over_mac"],
            "operational_aft_over_mac": selected["operational_aft_over_mac"],
        },
        "mission": {
            "cruise_true_speed_m_s": mission["cruise_true_speed_m_s"],
            "CL_cruise": mission["CL_cruise"],
            "optimized_climb_EAS_m_s": mission["optimized_climb_EAS_m_s"],
            "optimized_climb_angle_deg": mission["optimized_climb_angle_deg"],
            "total_energy_Wh": mission["total_energy_Wh"],
            "installed_battery_energy_Wh": mission["installed_battery_energy_Wh"],
            "battery_mass_kg": mission["battery_mass_kg"],
            "peak_electrical_power_W": mission["peak_electrical_power_W"],
            "total_mission_time_s": mission["total_mission_time_s"],
            "climb_horizontal_distance_m": mission["climb_horizontal_distance_m"],
            "level_cruise_distance_m": mission["level_cruise_distance_m"],
        },
        "propeller": {
            "diameter_m": propeller["propeller_diameter_m"],
            "propeller_diameter_m": propeller["propeller_diameter_m"],
            "n_rotors": aircraft["n_rotors"],
        },
        "airfoil_xfoil": result.get("xfoil_airfoil_update", {}),
        "selected_wing_area_m2": result["selected_wing_area_m2"],
    }


def write_json_summary(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(build_full_summary(result), json_file, indent=2)


def write_table_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scissor(path, wing, candidates, selected):
    candidate_area_ratios = [item["canard"]["area_ratio"] for item in candidates]
    ratio_min = min(candidate_area_ratios)
    ratio_max = max(candidate_area_ratios)
    point_count = 100
    if abs(ratio_max - ratio_min) < 1e-12:
        area_ratios = [ratio_min]
    else:
        area_ratios = [
            ratio_min + (ratio_max - ratio_min) * index / (point_count - 1)
            for index in range(point_count)
        ]

    # A scissor plot is defined for one fixed geometry. The sizing loop solves a
    # different wing station for each candidate ratio, so plotting those raw
    # candidates connects many geometries and produces a curved sizing trace.
    coeffs = selected["coeffs"]
    limits = [scissor_limits(area_ratio, coeffs) for area_ratio in area_ratios]
    x_forward = [item["x_forward_over_mac"] for item in limits]
    x_aft = [item["x_aft_over_mac"] for item in limits]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(x_aft, area_ratios, color="#c0392b", label="Stability")
    ax.plot(x_forward, area_ratios, color="#2c3e50", label="Controllability")
    ax.fill_betweenx(area_ratios, x_forward, x_aft, color="#2e8b57", alpha=0.14, label="Feasible CG band")

    x_cg = selected["mass"]["x_cg_over_mac"]
    area_ratio = selected["canard"]["area_ratio"]
    ax.axvline(x_cg, color="#1f77b4", linestyle="-.", label=f"CG x/c={x_cg:.3f}")
    ax.axhline(area_ratio, color="#d35400", linestyle="--", label=f"Selected S_c/S_w={area_ratio:.3f}")
    ax.plot(
        [selected["scissor"]["x_forward_over_mac"], selected["scissor"]["x_aft_over_mac"]],
        [area_ratio, area_ratio],
        color="#d35400",
        linewidth=3,
        alpha=0.7,
    )

    ax.set_xlabel("x_cg / wing chord")
    ax.set_ylabel("canard area ratio S_c / S_w")
    ax.set_title("Canard scissor plot")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mission_profile(path, mission):
    times = [point[0] for point in mission["profile"]]
    altitudes = [point[1] for point in mission["profile"]]
    time_min = [time / 60.0 for time in times]
    altitude_km = [altitude / 1000.0 for altitude in altitudes]
    states = mission["states"]
    segments = mission["segment_summaries"]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    fig.suptitle("Bellona mission profile and energy sizing", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.935,
        (
            f"Load energy {mission['total_energy_Wh'] / 1000.0:.2f} kWh | "
            f"Installed battery {mission['installed_battery_energy_Wh'] / 1000.0:.2f} kWh | "
            f"Climb EAS {mission['optimized_climb_EAS_m_s']:.1f} m/s | "
            f"gamma {mission['optimized_climb_angle_deg']:.1f} deg"
        ),
        ha="center",
        fontsize=10,
        color="#3d4752",
    )

    ax = axes[0, 0]
    ax.plot(time_min, altitude_km, color="#005f73", linewidth=2.4)
    ax.fill_between(time_min, altitude_km, color="#005f73", alpha=0.10)
    ax.set_title("Altitude timeline")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    track_x_km = [0.0, 0.0]
    track_h_km = [0.0, altitude_km[1] if len(altitude_km) > 1 else 0.0]
    x_m = segments["transition"]["distance_m"]
    track_x_km.append(x_m / 1000.0)
    track_h_km.append(track_h_km[-1])
    for state in states:
        x_m += state["delta_x_m"]
        track_x_km.append(x_m / 1000.0)
        track_h_km.append(state["altitude_m"] / 1000.0)
    ax.plot(track_x_km, track_h_km, color="#0a9396", linewidth=2.4)
    ax.fill_between(track_x_km, track_h_km, color="#0a9396", alpha=0.10)
    ax.axvline(mission["climb_horizontal_distance_m"] / 1000.0, color="#94d2bd", linestyle="--", linewidth=1.2)
    ax.set_title("Altitude over ground track")
    ax.set_xlabel("ground track [km]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)

    climb_states = [state for state in states if state["segment"] == "wing_borne_climb"]
    state_altitude_km = [state["altitude_mid_m"] / 1000.0 for state in climb_states]

    ax = axes[0, 2]
    if climb_states:
        ax.plot([state["speed_m_s"] for state in climb_states], state_altitude_km, color="#001219", linewidth=2.2, label="TAS")
        ax.plot([state["EAS_m_s"] for state in climb_states], state_altitude_km, color="#ee9b00", linewidth=2.0, linestyle="--", label="EAS")
    ax.set_title("Climb speed schedule")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if climb_states:
        ax.plot([state["electrical_power_W"] / 1000.0 for state in climb_states], state_altitude_km, color="#bb3e03", linewidth=2.2, label="wing-borne climb")
    ax.axvline(14.0, color="#9b2226", linestyle=":", linewidth=1.8, label="hover assumption")
    ax.set_title("Electrical power")
    ax.set_xlabel("power [kW]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    segment_names = ["vertical_takeoff", "transition", "wing_borne_climb", "level_cruise", "mission_hover"]
    labels = ["takeoff", "transition", "climb", "cruise", "hover"]
    energy_kWh = [segments[name]["energy_Wh"] / 1000.0 for name in segment_names]
    colors = ["#005f73", "#0a9396", "#ee9b00", "#ca6702", "#9b2226"]
    ax.bar(labels, energy_kWh, color=colors, width=0.65)
    ax.set_title("Segment energy")
    ax.set_ylabel("load energy [kWh]")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(energy_kWh):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 2]
    if climb_states:
        CL = [state["CL"] for state in climb_states]
        rate_of_climb = [state["rate_of_climb_m_s"] for state in climb_states]
        ax.plot(CL, state_altitude_km, color="#3d405b", linewidth=2.2, label="CL")
        CL_allowed = mission["mission_grid"]["aerodynamic_speed_limits"]["CL_allowed"]
        ax.axvline(CL_allowed, color="#9b2226", linestyle="--", linewidth=1.5, label="CL limit")
        ax2 = ax.twiny()
        ax2.plot(rate_of_climb, state_altitude_km, color="#2a9d8f", linewidth=1.8, linestyle=":", label="rate of climb")
        ax2.set_xlabel("rate of climb [m/s]", color="#2a9d8f")
        ax2.tick_params(axis="x", colors="#2a9d8f")
    ax.set_title("Lift coefficient and climb rate")
    ax.set_xlabel("CL")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_wing_area_sweep(path, rows, selected_wing_area_m2):
    feasible_rows = [row for row in rows if row["feasible"]]
    if not feasible_rows:
        return
    failed_areas = [row["wing_area_m2"] for row in rows if not row["feasible"]]

    wing_areas = [row["wing_area_m2"] for row in feasible_rows]
    masses = [row["MTOW_mass_estimate_kg"] for row in feasible_rows]
    stall_speeds = [row["wing_stall_EAS_m_s"] for row in feasible_rows]
    energies = [row["mission_energy_Wh"] / 1000.0 for row in feasible_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle("Wing area energy trade", fontsize=15, fontweight="bold")

    if failed_areas:
        failed_start = min(failed_areas) - 0.5 * AIRCRAFT["wing_area_sweep_step_m2"]
        failed_end = max(failed_areas) + 0.5 * AIRCRAFT["wing_area_sweep_step_m2"]
        for ax in axes:
            ax.axvspan(failed_start, failed_end, color="#9b2226", alpha=0.08, label="infeasible")

    axes[0].plot(wing_areas, masses, marker="o", color="#005f73")
    axes[0].axvline(selected_wing_area_m2, color="#9b2226", linestyle="--", label="selected")
    axes[0].set_xlabel("wing area [m2]")
    axes[0].set_ylabel("final mass [kg]")
    axes[0].set_title("Mass closure")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(wing_areas, stall_speeds, marker="o", color="#0a9396")
    axes[1].axvline(selected_wing_area_m2, color="#9b2226", linestyle="--")
    axes[1].set_xlabel("wing area [m2]")
    axes[1].set_ylabel("stall EAS [m/s]")
    axes[1].set_title("Stall speed check")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(wing_areas, energies, marker="o", color="#ee9b00")
    axes[2].axvline(selected_wing_area_m2, color="#9b2226", linestyle="--")
    axes[2].set_xlabel("wing area [m2]")
    axes[2].set_ylabel("mission load energy [kWh]")
    axes[2].set_title("Mission energy")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def course_method_mission_energy(weight_N, wing):
    """Mission energy using the lecture RC_s course-method climb."""
    aircraft = dict(AIRCRAFT)
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
    CD_cruise = AIRCRAFT["CD0"] + CL_cruise**2 / (
        math.pi * AIRCRAFT["wing_aspect_ratio"] * AIRCRAFT["oswald_efficiency"]
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
    minimum_climb_EAS = math.sqrt(2.0 * weight_N / (1.225 * wing["area_m2"] * CL_allowed))
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


def _run_sizing_pass_once(weight_N, wing_area_m2):
    """One pass through mission, canard, wing position, and mass."""
    rho_cruise = isa_density(MISSION["altitude_m"])
    propeller = propeller_disk_estimate(weight_N)
    wing = wing_geometry(weight_N, rho_cruise, wing_area_m2)
    mission = course_method_mission_energy(weight_N, wing)
    wing["cruise_true_speed_m_s"] = mission["cruise_true_speed_m_s"]
    wing["CL_trim"] = mission["CL_cruise"]
    selected, candidates = canard_and_wing_iteration(wing, mission, propeller)

    return {
        "wing": wing,
        "mission": mission,
        "propeller": propeller,
        "selected": selected,
        "candidates": candidates,
    }


def run_sizing_pass(weight_N, wing_area_m2, show_progress=False, progress_indent=0):
    """One sizing pass at the section coefficients currently held in AIRCRAFT.

    XFOIL is no longer called here: the airfoil section data is refreshed once
    per outer Reynolds-feedback iteration in run_sizing(), so every pass inside
    the wing-area sweep and mass loop is pure Python and fast.
    """
    result = _run_sizing_pass_once(weight_N, wing_area_m2)
    result["aircraft"] = snapshot_aircraft()
    return result


def coupled_sizing_iteration(wing_area_m2, show_progress=False, progress_indent=0):
    """Iterate mass, mission energy, and canard sizing for one wing area."""
    mass_kg = AIRCRAFT["MTOW_kg"]
    history = []
    result = None

    for iteration in range(1, AIRCRAFT["sizing_iteration_count"] + 1):
        weight_N = mass_kg * AIRCRAFT["g_m_s2"]
        progress(
            (
                f"Mass iteration {iteration}/{AIRCRAFT['sizing_iteration_count']}: "
                f"input mass={mass_kg:.2f} kg"
            ),
            show_progress,
            progress_indent,
        )
        result = run_sizing_pass(
            weight_N,
            wing_area_m2,
            show_progress=show_progress,
            progress_indent=progress_indent + 1,
        )
        estimated_mass_kg = result["selected"]["mass"]["total_mass_kg"]
        mass_change_kg = estimated_mass_kg - mass_kg

        history.append({
            "iteration": iteration,
            "mass_used_kg": mass_kg,
            "wing_area_m2": result["wing"]["area_m2"],
            "stall_EAS_m_s": result["wing"]["stall_EAS_m_s"],
            "climb_EAS_m_s": result["mission"]["optimized_climb_EAS_m_s"],
            "mission_energy_Wh": result["mission"]["total_energy_Wh"],
            "battery_mass_kg": result["mission"]["battery_mass_kg"],
            "canard_area_ratio": result["selected"]["canard"]["area_ratio"],
            "wing_CL_max": AIRCRAFT["wing_CL_max"],
            "canard_CL_max": AIRCRAFT["canard_CL_max"],
            "wing_datcom_eta": AIRCRAFT["wing_datcom_eta"],
            "canard_datcom_eta": AIRCRAFT["canard_datcom_eta"],
            "estimated_mass_kg": estimated_mass_kg,
            "mass_change_kg": mass_change_kg,
        })

        progress(
            (
                f"Mass estimate={estimated_mass_kg:.2f} kg "
                f"(delta {mass_change_kg:+.2f} kg); "
                f"stall={result['wing']['stall_EAS_m_s']:.2f} m/s, "
                f"climb EAS={result['mission']['optimized_climb_EAS_m_s']:.2f} m/s, "
                f"Sc/Sw={result['selected']['canard']['area_ratio']:.3f}, "
                f"CLmax_w/c={AIRCRAFT['wing_CL_max']:.3f}/{AIRCRAFT['canard_CL_max']:.3f}"
            ),
            show_progress,
            progress_indent,
        )

        if abs(estimated_mass_kg - mass_kg) <= AIRCRAFT["sizing_mass_tolerance_kg"]:
            progress(
                f"Mass converged within {AIRCRAFT['sizing_mass_tolerance_kg']:.2f} kg",
                show_progress,
                progress_indent,
            )
            mass_kg = estimated_mass_kg
            break

        mass_kg = estimated_mass_kg

    weight_N = mass_kg * AIRCRAFT["g_m_s2"]
    progress("Final coupled pass at converged mass", show_progress, progress_indent)
    result = run_sizing_pass(
        weight_N,
        wing_area_m2,
        show_progress=show_progress,
        progress_indent=progress_indent + 1,
    )
    result["iteration_history"] = history
    result["final_mass_used_kg"] = mass_kg
    result["selected_wing_area_m2"] = wing_area_m2
    return result


def wing_area_sweep_values():
    values = []
    area = AIRCRAFT["wing_area_sweep_min_m2"]
    maximum = AIRCRAFT["wing_area_sweep_max_m2"]
    step = AIRCRAFT["wing_area_sweep_step_m2"]
    while area <= maximum + 1e-9:
        values.append(round(area, 6))
        area += step
    return values


def make_summary(result):
    wing = result["wing"]
    mission = result["mission"]
    propeller = result["propeller"]
    selected = result["selected"]
    candidates = result["candidates"]
    xfoil_update = result.get("xfoil_airfoil_update", {})
    xfoil_condition = xfoil_update.get("condition") or {}
    wingborne_states = [state for state in mission["states"] if state["segment"] == "wing_borne_climb"]
    max_climb_CL = max([state["CL"] for state in wingborne_states] or [0.0])
    climb_CL_limit = AIRCRAFT["wing_CL_max"] / AIRCRAFT["climb_stall_margin_n"]**2
    stall_limit = back_transition_stall_limit()

    return {
        "MTOW_input_kg": AIRCRAFT["MTOW_kg"],
        "MTOW_used_for_final_pass_kg": result["final_mass_used_kg"],
        "MTOW_mass_estimate_kg": selected["mass"]["total_mass_kg"],
        "mass_closure_error_kg": selected["mass"]["total_mass_kg"] - result["final_mass_used_kg"],
        "sizing_iterations_used": len(result["iteration_history"]),
        "climb_stall_margin_n": AIRCRAFT["climb_stall_margin_n"],
        "climb_CL_limit": climb_CL_limit,
        "max_climb_CL": max_climb_CL,
        "wing_CL_max": AIRCRAFT["wing_CL_max"],
        "canard_CL_max": AIRCRAFT["canard_CL_max"],
        "wing_CL_alpha_per_rad": AIRCRAFT["wing_CL_alpha_per_rad"],
        "canard_CL_alpha_per_rad": AIRCRAFT["canard_CL_alpha_per_rad"],
        "wing_CL0": AIRCRAFT["wing_CL0"],
        "wing_airfoil_cm0": AIRCRAFT["wing_airfoil_cm0"],
        "wing_datcom_eta": AIRCRAFT["wing_datcom_eta"],
        "canard_datcom_eta": AIRCRAFT["canard_datcom_eta"],
        "xfoil_enabled": xfoil_update.get("enabled", False),
        "xfoil_changed_last_pass": xfoil_update.get("changed", False),
        "xfoil_condition_source": xfoil_condition.get("source", ""),
        "xfoil_condition_altitude_m": xfoil_condition.get("altitude_m", ""),
        "xfoil_condition_true_speed_m_s": xfoil_condition.get("true_speed_m_s", ""),
        "xfoil_condition_mach": xfoil_condition.get("mach", ""),
        "xfoil_wing_Re": xfoil_condition.get("wing_reynolds", ""),
        "xfoil_canard_Re": xfoil_condition.get("canard_reynolds", ""),
        "wing_area_m2": wing["area_m2"],
        "wing_span_m": wing["span_m"],
        "wing_chord_m": wing["chord_m"],
        "wing_stall_EAS_m_s": wing["stall_EAS_m_s"],
        "max_stall_EAS_m_s": stall_limit["stall_EAS_max_m_s"],
        "back_transition_a_max_m_s2": stall_limit["a_max_m_s2"],
        "back_transition_time_s": stall_limit["transition_time_s"],
        "back_transition_distance_m": stall_limit["transition_distance_m"],
        "minimum_climb_EAS_m_s": mission["mission_grid"]["aerodynamic_speed_limits"]["minimum_climb_EAS_m_s"],
        "cruise_true_speed_m_s": mission["cruise_true_speed_m_s"],
        "CL_trim": wing["CL_trim"],
        "CD_trim": mission["CD_trim"],
        "optimized_climb_EAS_m_s": mission["optimized_climb_EAS_m_s"],
        "optimized_climb_angle_deg": mission["optimized_climb_angle_deg"],
        "course_climb_available_power_W": mission["course_climb_available_power_W"],
        "course_climb_average_power_W": mission["course_climb_average_power_W"],
        "course_climb_time_s": mission["course_climb_time_s"],
        "course_climb_max_thrust_to_weight": mission["course_climb_max_thrust_to_weight"],
        "course_climb_thrust_limit": mission["course_climb_thrust_limit"],
        "course_climb_complies_time": mission["course_climb_complies_time"],
        "canard_area_ratio": selected["canard"]["area_ratio"],
        "canard_area_m2": selected["canard"]["area_m2"],
        "canard_span_m": selected["canard"]["span_m"],
        "canard_arm_m": selected["arm_m"],
        "wing_mac_le_x_m": selected["mass"]["wing_mac_le_x_m"],
        "x_CG_over_MAC": selected["mass"]["x_cg_over_mac"],
        "scissor_forward_limit_x_over_c": selected["scissor"]["x_forward_over_mac"],
        "scissor_aft_limit_x_over_c": selected["scissor"]["x_aft_over_mac"],
        "scissor_clearance_over_c": selected.get("clearance_over_mac", ""),
        "operational_cg_forward_x_over_c": selected["operational_fwd_over_mac"],
        "operational_cg_aft_x_over_c": selected["operational_aft_over_mac"],
        "battery_mass_kg": mission["battery_mass_kg"],
        "mission_energy_Wh": mission["total_energy_Wh"],
        "installed_battery_energy_Wh": mission["installed_battery_energy_Wh"],
        "outbound_time_s": mission["outbound_time_s"],
        "total_mission_time_s": mission["total_mission_time_s"],
        "climb_horizontal_distance_m": mission["climb_horizontal_distance_m"],
        "level_cruise_distance_m": mission["level_cruise_distance_m"],
        "spiral_excess_ground_track_distance_m": mission["spiral_excess_ground_track_distance_m"],
        "peak_electrical_power_W": mission["peak_electrical_power_W"],
        "propeller_diameter_m": propeller["propeller_diameter_m"],
        "candidate_count": len(candidates),
    }


def sweep_wing_area(show_progress=False):
    rows = []
    feasible_results = []
    stall_limit = back_transition_stall_limit()   # same for every wing area, so compute once
    wing_areas = wing_area_sweep_values()
    sweep_start = time.perf_counter()

    progress(
        (
            "Wing-area sweep: "
            f"{len(wing_areas)} candidates from {wing_areas[0]:.2f} "
            f"to {wing_areas[-1]:.2f} m^2"
        ),
        show_progress,
    )

    for index, wing_area in enumerate(wing_areas, start=1):
        # Section coefficients are held fixed across the sweep; they are
        # refreshed by the outer XFOIL loop in run_sizing(), not per wing area.
        elapsed_before = time.perf_counter() - sweep_start
        if index > 1:
            average_time = elapsed_before / (index - 1)
            eta_text = format_duration(average_time * (len(wing_areas) - index + 1))
        else:
            eta_text = "estimating"
        progress(
            (
                f"[{index}/{len(wing_areas)} | "
                f"{100.0 * (index - 1) / len(wing_areas):5.1f}% done] "
                f"Wing area {wing_area:.2f} m^2 "
                f"({len(wing_areas) - index} candidates left after this, ETA {eta_text})"
            ),
            show_progress,
        )
        area_start = time.perf_counter()
        try:
            result = coupled_sizing_iteration(
                wing_area,
                show_progress=show_progress,
                progress_indent=1,
            )
            summary = make_summary(result)
            scissor_ok = result["selected"]["feasible"]
            stall_ok = result["wing"]["stall_EAS_m_s"] <= stall_limit["stall_EAS_max_m_s"]
            feasible = scissor_ok and stall_ok

            print(
                f"S={wing_area:.2f} m²  "
                f"stall={result['wing']['stall_EAS_m_s']:.1f}/{stall_limit['stall_EAS_max_m_s']:.1f} m/s  "
                f"scissor={'OK' if scissor_ok else 'FAIL'}  "
                f"stall={'OK' if stall_ok else 'FAIL'}  "
                f"canard_ratio={result['selected']['canard']['area_ratio']:.3f}  "
                f"arm={result['selected']['arm_m']:.2f} m  "
                f"mass={result['selected']['mass']['total_mass_kg']:.1f} kg"
            )

            if not stall_ok:
                failure_reason = "Stall speed above back-transition limit."
            elif not scissor_ok:
                failure_reason = "Scissor constraints not feasible."
            else:
                failure_reason = ""

            row = {
                "wing_area_m2": wing_area,
                "feasible": feasible,
                "failure_reason": failure_reason,
                "MTOW_mass_estimate_kg": summary["MTOW_mass_estimate_kg"],
                "mass_closure_error_kg": summary["mass_closure_error_kg"],
                "wing_span_m": summary["wing_span_m"],
                "wing_stall_EAS_m_s": summary["wing_stall_EAS_m_s"],
                "max_stall_EAS_m_s": stall_limit["stall_EAS_max_m_s"],
                "minimum_climb_EAS_m_s": summary["minimum_climb_EAS_m_s"],
                "optimized_climb_EAS_m_s": summary["optimized_climb_EAS_m_s"],
                "max_climb_CL": summary["max_climb_CL"],
                "climb_CL_limit": summary["climb_CL_limit"],
                "course_climb_available_power_W": summary["course_climb_available_power_W"],
                "course_climb_average_power_W": summary["course_climb_average_power_W"],
                "course_climb_time_s": summary["course_climb_time_s"],
                "course_climb_max_thrust_to_weight": summary["course_climb_max_thrust_to_weight"],
                "course_climb_thrust_limit": summary["course_climb_thrust_limit"],
                "cruise_true_speed_m_s": summary["cruise_true_speed_m_s"],
                "mission_energy_Wh": summary["mission_energy_Wh"],
                "battery_mass_kg": summary["battery_mass_kg"],
                "canard_area_ratio": summary["canard_area_ratio"],
                "canard_area_m2": summary["canard_area_m2"],
                "canard_arm_m": summary["canard_arm_m"],
                "scissor_clearance_over_c": summary["scissor_clearance_over_c"],
                "outbound_time_s": summary["outbound_time_s"],
                "peak_electrical_power_W": summary["peak_electrical_power_W"],
                "propeller_diameter_m": summary["propeller_diameter_m"],
                "sizing_iterations_used": summary["sizing_iterations_used"],
            }
            rows.append(row)
            if feasible:
                feasible_results.append((summary["MTOW_mass_estimate_kg"], result, summary))
            area_elapsed = time.perf_counter() - area_start
            elapsed_total = time.perf_counter() - sweep_start
            average_time = elapsed_total / index
            remaining_time = average_time * (len(wing_areas) - index)
            status = "feasible" if feasible else f"rejected: {failure_reason}"
            progress(
                (
                    f"Result {status}; mass={summary['MTOW_mass_estimate_kg']:.2f} kg, "
                    f"stall={summary['wing_stall_EAS_m_s']:.2f} m/s, "
                    f"Sc/Sw={summary['canard_area_ratio']:.3f}, "
                    f"arm={summary['canard_arm_m']:.2f} m, "
                    f"time={format_duration(area_elapsed)}, "
                    f"remaining ETA={format_duration(remaining_time)}"
                ),
                show_progress,
                1,
            )
        except RuntimeError as error:
            print(f"S={wing_area:.2f} m²  RUNTIME ERROR: {error}")
            rows.append({
                "wing_area_m2": wing_area,
                "feasible": False,
                "failure_reason": str(error),
                "MTOW_mass_estimate_kg": "", "mass_closure_error_kg": "",
                "wing_span_m": "", "wing_stall_EAS_m_s": "", "max_stall_EAS_m_s": "",
                "minimum_climb_EAS_m_s": "", "optimized_climb_EAS_m_s": "",
                "max_climb_CL": "", "climb_CL_limit": "",
                "course_climb_available_power_W": "", "course_climb_average_power_W": "",
                "course_climb_time_s": "", "course_climb_max_thrust_to_weight": "",
                "course_climb_thrust_limit": "", "cruise_true_speed_m_s": "",
                "mission_energy_Wh": "", "battery_mass_kg": "",
                "canard_area_ratio": "", "canard_area_m2": "",
                "canard_arm_m": "", "scissor_clearance_over_c": "",
                "outbound_time_s": "", "peak_electrical_power_W": "",
                "sizing_iterations_used": "",
            })

        except Exception as error:
            import traceback
            print(f"S={wing_area:.2f} m²  UNEXPECTED ERROR: {type(error).__name__}: {error}")
            traceback.print_exc()
            rows.append({
                "wing_area_m2": wing_area,
                "feasible": False,
                "failure_reason": f"{type(error).__name__}: {error}",
                "MTOW_mass_estimate_kg": "", "mass_closure_error_kg": "",
                "wing_span_m": "", "wing_stall_EAS_m_s": "", "max_stall_EAS_m_s": "",
                "minimum_climb_EAS_m_s": "", "optimized_climb_EAS_m_s": "",
                "max_climb_CL": "", "climb_CL_limit": "",
                "course_climb_available_power_W": "", "course_climb_average_power_W": "",
                "course_climb_time_s": "", "course_climb_max_thrust_to_weight": "",
                "course_climb_thrust_limit": "", "cruise_true_speed_m_s": "",
                "mission_energy_Wh": "", "battery_mass_kg": "",
                "canard_area_ratio": "", "canard_area_m2": "",
                "canard_arm_m": "", "scissor_clearance_over_c": "",
                "outbound_time_s": "", "peak_electrical_power_W": "",
                "sizing_iterations_used": "",
            })
            area_elapsed = time.perf_counter() - area_start
            elapsed_total = time.perf_counter() - sweep_start
            average_time = elapsed_total / index
            remaining_time = average_time * (len(wing_areas) - index)
            progress(
                (
                    f"Result failed: {error}; "
                    f"time={format_duration(area_elapsed)}, "
                    f"remaining ETA={format_duration(remaining_time)}"
                ),
                show_progress,
                1,
            )

    if not feasible_results:
        raise RuntimeError("No feasible wing-area point was found in the sweep.")

    feasible_results.sort(key=lambda item: item[0])
    _, result, _ = feasible_results[0]
    restore_aircraft_snapshot(result["aircraft"])
    summary = make_summary(result)
    progress(
        (
            "Selected feasible point: "
            f"S={summary['wing_area_m2']:.2f} m^2, "
            f"mass={summary['MTOW_mass_estimate_kg']:.2f} kg, "
            f"Sc/Sw={summary['canard_area_ratio']:.3f}, "
            f"arm={summary['canard_arm_m']:.2f} m"
        ),
        show_progress,
    )
    return result, summary, rows


def sweep_with_xfoil_feedback(show_progress=True):
    """Outer Reynolds-feedback loop around the wing-area sweep (Option A).

    Each iteration runs one full wing-area sweep at the section coefficients
    currently in AIRCRAFT, then refreshes those coefficients with a single
    XFOIL airfoil-pair run at the best design's representative condition. It
    repeats until XFOIL stops changing the coefficients, so XFOIL is launched
    only a handful of times per sizing run rather than inside every pass.
    """
    xfoil_on = bool(AIRCRAFT.get("use_xfoil_airfoil_updates", False))
    outer_count = max(1, int(AIRCRAFT.get("xfoil_outer_iteration_count", 1))) if xfoil_on else 1
    last_update = {"enabled": xfoil_on, "changed": False, "condition": None}
    result = sweep_rows = None

    for outer in range(1, outer_count + 1):
        if xfoil_on:
            progress(f"XFOIL outer iteration {outer}/{outer_count}", show_progress)
        result, _, sweep_rows = sweep_wing_area(show_progress=show_progress)
        if not xfoil_on:
            break
        last_update = update_airfoil_aerodynamics_from_xfoil(
            result["wing"], result["mission"], result["selected"],
            show_progress=show_progress, progress_indent=1,
        )
        if not last_update["changed"]:
            progress(f"Airfoil aero converged after {outer}/{outer_count} XFOIL passes", show_progress)
            break
        progress("Section coefficients changed; re-running the wing-area sweep", show_progress)

    # The reported design must be sized at the final coefficients. If the loop
    # hit its cap while still changing, run one more sweep to stay consistent.
    if xfoil_on and last_update.get("changed"):
        progress("Final wing-area sweep at the latest section coefficients", show_progress)
        result, _, sweep_rows = sweep_wing_area(show_progress=show_progress)
        last_update = dict(last_update, changed=False)

    if result is not None:
        result["xfoil_airfoil_update"] = last_update
    return result, sweep_rows


def run_sizing(output_dir=OUTPUT_DIR, make_plots=True, use_xfoil=None, show_progress=True):
    output_dir = Path(output_dir)
    reset_airfoil_aero_defaults()
    previous_use_xfoil = AIRCRAFT["use_xfoil_airfoil_updates"]
    if use_xfoil is not None:
        AIRCRAFT["use_xfoil_airfoil_updates"] = bool(use_xfoil)

    try:
        progress("Starting Bellona sizing run", show_progress)
        result, sweep_rows = sweep_with_xfoil_feedback(show_progress=show_progress)
        summary = make_summary(result)
        wing = result["wing"]
        mission = result["mission"]
        propeller = result["propeller"]
        selected = result["selected"]
        candidates = result["candidates"]
        history = result["iteration_history"]

        write_key_value_csv(output_dir / "summary.csv", summary)
        write_mass_breakdown(output_dir / "mass_breakdown.csv", selected["mass"])
        write_iteration_history(output_dir / "iteration_history.csv", history)
        write_table_csv(output_dir / "wing_area_sweep.csv", sweep_rows)
        write_json_summary(output_dir / "aircraft_summary.json", result)
        if make_plots:
            progress("Writing plots", show_progress)
            plot_scissor(output_dir / "scissor_plot.png", wing, candidates, selected)
            plot_mission_profile(output_dir / "mission_profile.png", mission)
            plot_wing_area_sweep(
                output_dir / "wing_area_sweep.png",
                sweep_rows,
                summary["wing_area_m2"],
            )
        progress(f"Outputs written to: {output_dir}", show_progress)

        return {
            "summary": summary,
            "wing": wing,
            "mission": mission,
            "propeller": propeller,
            "selected": selected,
            "candidates": candidates,
            "iteration_history": history,
            "wing_area_sweep": sweep_rows,
        }
    finally:
        if use_xfoil is not None:
            AIRCRAFT["use_xfoil_airfoil_updates"] = previous_use_xfoil


def main():
    result = run_sizing()
    summary = result["summary"]
    print("Bellona simplified sizing")
    print(f"  MTOW estimate: {summary['MTOW_mass_estimate_kg']:.2f} kg")
    print(
        "  Wing: "
        f"{summary['wing_area_m2']:.2f} m^2, span {summary['wing_span_m']:.2f} m, "
        f"stall EAS={summary['wing_stall_EAS_m_s']:.1f} m/s"
    )
    if summary["back_transition_a_max_m_s2"] is None:
        print(
            "  Stall cap: "
            f"stall EAS limit={summary['max_stall_EAS_m_s']:.1f} m/s (user-specified)"
        )
    else:
        print(
            "  Back-transition: "
            f"stall EAS limit={summary['max_stall_EAS_m_s']:.1f} m/s "
            f"(a_max={summary['back_transition_a_max_m_s2']:.2f} m/s^2, "
            f"{summary['back_transition_distance_m']:.0f} m / {summary['back_transition_time_s']:.1f} s)"
        )
    print(
        "  Climb CL: "
        f"max {summary['max_climb_CL']:.3f}, "
        f"limit {summary['climb_CL_limit']:.3f} "
        f"(n={summary['climb_stall_margin_n']:.2f})"
    )
    print(
        "  Mission: "
        f"{summary['mission_energy_Wh'] / 1000.0:.2f} kWh load, "
        f"climb EAS={summary['optimized_climb_EAS_m_s']:.2f} m/s, "
        f"gamma={summary['optimized_climb_angle_deg']:.1f} deg"
    )
    print(
        "  Course climb: "
        f"time={summary['course_climb_time_s'] / 60.0:.2f} min, "
        f"P_avail={summary['course_climb_available_power_W'] / 1000.0:.1f} kW, "
        f"T/W={summary['course_climb_max_thrust_to_weight']:.2f}/"
        f"{summary['course_climb_thrust_limit']:.2f}"
    )
    print(f"  Propeller: diameter {summary['propeller_diameter_m']:.2f} m")
    print(f"  Canard: S_c/S_w={summary['canard_area_ratio']:.3f}, area {summary['canard_area_m2']:.2f} m^2")
    print(f"  CG: x/c={summary['x_CG_over_MAC']:.3f}, wing MAC LE x={summary['wing_mac_le_x_m']:.3f} m")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg")
    print(f"  Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
