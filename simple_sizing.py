
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mission_energy_course import build_course_mission, optimize_course_climb, permitted_lift_coefficient


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
    "canard_arm_over_wing_chord": 2.50,
    "canard_area_ratio_min": 0.05,
    "canard_area_ratio_max": 0.80,
    "canard_area_ratio_step": 0.001,
    "static_margin": 0.10,
    "cg_envelope_half_width_over_mac": 0.05,
    "cg_margin_over_mac": 0.02,
    "CD0": 0.040,
    "oswald_efficiency": 0.78,
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
    # --- Back-transition requirement that sizes the wing ---
    # The aircraft must decelerate from wing-borne flight to a 0 m/s hover
    # within this much horizontal ground track at the transition altitude.
    # SMALLER distance = more demanding = lower stall-speed cap = bigger wing.
    "back_transition_distance_budget_m": 30.0,
    # Optionally specify a time budget (s) instead; if not None it overrides the distance.
    "back_transition_time_budget_s": None,
    # The manoeuvre is entered at this multiple of the stall speed (1.3 = airworthiness-style
    # approach margin; set to 1.0 to decelerate from the stall speed itself).
    "back_transition_approach_speed_factor": 1.30,
}

MASS = {
    "wing_areal_density_kg_m2": 2.00,
    "canard_areal_density_kg_m2": 1.70,
    "fuselage_length_m": 2.20,
    "fuselage_linear_density_kg_m": 2.20,
    "boom_landing_gear_mass_kg": 2.50,
    "motor_specific_mass_kg_W": 0.00027,
    "esc_specific_mass_kg_W": 0.00008,
    "prop_mass_coeff_kg_m2": 0.10,
    "avionics_mass_kg": 1.50,
    "wiring_fraction": 0.06,
    "contingency_fraction": 0.08,
}


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
        "a_max_m_s2": a_max,
        "entry_TAS_m_s": entry_TAS,
        "stall_TAS_m_s": stall_TAS,
        "stall_EAS_max_m_s": stall_EAS_max,
        "transition_time_s": entry_TAS / a_max,
        "transition_distance_m": entry_TAS**2 / (2.0 * a_max),
    }


def propeller_disk_estimate(weight_N):
    """Disk area from selected disk loading."""
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
    """Canard planform from selected area ratio."""
    area = area_ratio * wing["area_m2"]
    span = math.sqrt(area * AIRCRAFT["canard_aspect_ratio"])
    chord = area / span
    arm = AIRCRAFT["canard_arm_over_wing_chord"] * wing["chord_m"]
    return {
        "area_ratio": area_ratio,
        "area_m2": area,
        "span_m": span,
        "chord_m": chord,
        "arm_m": arm,
        "x_ac_m": wing["x_ac_m"] - arm,
        "CL_alpha_per_rad": AIRCRAFT["canard_CL_alpha_per_rad"],
        "usable_CL": AIRCRAFT["canard_CL_limit_fraction"] * AIRCRAFT["canard_CL_max"],
    }


def mass_and_cg(wing, canard, mission, propeller, wing_mac_le_x_m):
    """Mass estimate and longitudinal CG relative to the wing MAC leading edge."""
    power_for_motor_sizing = max(
        AIRCRAFT["hover_power_W"],
        mission.get("peak_electrical_power_W", mission["climb_power_W"]),
    )

    masses = {
        "wing": MASS["wing_areal_density_kg_m2"] * wing["area_m2"],
        "canard": MASS["canard_areal_density_kg_m2"] * canard["area_m2"],
        "fuselage": MASS["fuselage_linear_density_kg_m"] * MASS["fuselage_length_m"],
        "boom_landing_gear": MASS["boom_landing_gear_mass_kg"],
        "motors": MASS["motor_specific_mass_kg_W"] * power_for_motor_sizing,
        "ESCs": MASS["esc_specific_mass_kg_W"] * power_for_motor_sizing,
        "propellers": AIRCRAFT["n_rotors"] * MASS["prop_mass_coeff_kg_m2"] * propeller["propeller_diameter_m"] ** 2,
        "battery": mission["battery_mass_kg"],
        "avionics": MASS["avionics_mass_kg"],
        "mission_equipment": MISSION["mission_equipment_mass_kg"],
    }
    masses["wiring"] = MASS["wiring_fraction"] * (masses["motors"] + masses["ESCs"] + masses["avionics"])
    masses["contingency"] = MASS["contingency_fraction"] * sum(masses.values())

    locations = {name: 0.0 for name in masses}
    locations["wing"] = wing_mac_le_x_m + wing["x_ac_m"]
    locations["canard"] = wing_mac_le_x_m + canard["x_ac_m"]

    total_mass = sum(masses.values())
    x_cg_fuselage_m = sum(masses[name] * locations[name] for name in masses) / total_mass
    x_cg_m = x_cg_fuselage_m - wing_mac_le_x_m

    return {
        "total_mass_kg": total_mass,
        "masses_kg": masses,
        "locations_fuselage_m": locations,
        "wing_mac_le_x_m": wing_mac_le_x_m,
        "x_cg_fuselage_m": x_cg_fuselage_m,
        "x_cg_m": x_cg_m,
        "x_cg_over_mac": x_cg_m / wing["chord_m"],
    }


def solve_wing_position(wing, canard, mission, propeller, target_x_cg_over_mac):
    """Shift the wing group until the mass CG reaches the scissor target."""
    mass_at_zero = mass_and_cg(wing, canard, mission, propeller, wing_mac_le_x_m=0.0)
    moving_mass = mass_at_zero["masses_kg"]["wing"] + mass_at_zero["masses_kg"]["canard"]
    slope = moving_mass / mass_at_zero["total_mass_kg"] - 1.0
    target_x_cg_m = target_x_cg_over_mac * wing["chord_m"]
    wing_shift_m = (target_x_cg_m - mass_at_zero["x_cg_m"]) / slope
    return mass_and_cg(wing, canard, mission, propeller, wing_mac_le_x_m=wing_shift_m)


# ---------------------------------------------------------------------------
# Scissor plot equations
# ---------------------------------------------------------------------------


def scissor_limits(area_ratio, wing, canard):
    """Lecture canard scissor equations."""
    CL_h = canard["usable_CL"]
    CL_Ah = wing["CL_trim"] - CL_h * area_ratio
    if CL_Ah <= 0.0:
        return None

    l_h_over_c = -AIRCRAFT["canard_arm_over_wing_chord"]
    x_ac = wing["x_ac_m"] / wing["chord_m"]
    stability_slope = canard["CL_alpha_per_rad"] / wing["CL_alpha_per_rad"] * l_h_over_c
    control_slope = CL_h / CL_Ah * l_h_over_c

    x_aft = x_ac + stability_slope * area_ratio - AIRCRAFT["static_margin"]
    x_forward = x_ac + control_slope * area_ratio
    return {
        "x_forward_over_mac": x_forward,
        "x_aft_over_mac": x_aft,
        "cg_range_over_mac": x_aft - x_forward,
        "CL_Ah": CL_Ah,
    }


def canard_and_wing_iteration(wing, mission, propeller):
    """Find the smallest canard area ratio that fits the operational CG envelope."""
    half_width = AIRCRAFT["cg_envelope_half_width_over_mac"]
    margin = AIRCRAFT["cg_margin_over_mac"]
    required_width = 2.0 * (half_width + margin)

    candidates = []
    steps = int((AIRCRAFT["canard_area_ratio_max"] - AIRCRAFT["canard_area_ratio_min"]) / AIRCRAFT["canard_area_ratio_step"]) + 1
    for i in range(steps):
        area_ratio = AIRCRAFT["canard_area_ratio_min"] + i * AIRCRAFT["canard_area_ratio_step"]
        canard = canard_geometry(area_ratio, wing)
        scissor = scissor_limits(area_ratio, wing, canard)
        if scissor is None:
            continue

        band_is_wide_enough = scissor["cg_range_over_mac"] >= required_width
        target_center = 0.5 * (
            scissor["x_forward_over_mac"]
            + margin
            + half_width
            + scissor["x_aft_over_mac"]
            - margin
            - half_width
        )
        mass = solve_wing_position(wing, canard, mission, propeller, target_center)
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
            "target_x_cg_over_mac": target_center,
            "operational_fwd_over_mac": operational_fwd,
            "operational_aft_over_mac": operational_aft,
            "feasible": fits,
        }
        candidates.append(candidate)
        if fits:
            return candidate, candidates

    return candidates[-1], candidates


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


def write_table_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scissor(path, wing, candidates, selected):
    area_ratios = [item["canard"]["area_ratio"] for item in candidates]
    x_forward = [item["scissor"]["x_forward_over_mac"] for item in candidates]
    x_aft = [item["scissor"]["x_aft_over_mac"] for item in candidates]

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


def run_sizing_pass(weight_N, wing_area_m2):
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


def coupled_sizing_iteration(wing_area_m2):
    """Iterate mass, mission energy, and canard sizing for one wing area."""
    mass_kg = AIRCRAFT["MTOW_kg"]
    history = []
    result = None

    for iteration in range(1, AIRCRAFT["sizing_iteration_count"] + 1):
        weight_N = mass_kg * AIRCRAFT["g_m_s2"]
        result = run_sizing_pass(weight_N, wing_area_m2)
        estimated_mass_kg = result["selected"]["mass"]["total_mass_kg"]

        history.append({
            "iteration": iteration,
            "mass_used_kg": mass_kg,
            "wing_area_m2": result["wing"]["area_m2"],
            "stall_EAS_m_s": result["wing"]["stall_EAS_m_s"],
            "climb_EAS_m_s": result["mission"]["optimized_climb_EAS_m_s"],
            "mission_energy_Wh": result["mission"]["total_energy_Wh"],
            "battery_mass_kg": result["mission"]["battery_mass_kg"],
            "canard_area_ratio": result["selected"]["canard"]["area_ratio"],
            "estimated_mass_kg": estimated_mass_kg,
            "mass_change_kg": estimated_mass_kg - mass_kg,
        })

        if abs(estimated_mass_kg - mass_kg) <= AIRCRAFT["sizing_mass_tolerance_kg"]:
            mass_kg = estimated_mass_kg
            break

        mass_kg = estimated_mass_kg

    weight_N = mass_kg * AIRCRAFT["g_m_s2"]
    result = run_sizing_pass(weight_N, wing_area_m2)
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
        "wing_mac_le_x_m": selected["mass"]["wing_mac_le_x_m"],
        "x_CG_over_MAC": selected["mass"]["x_cg_over_mac"],
        "scissor_forward_limit_x_over_c": selected["scissor"]["x_forward_over_mac"],
        "scissor_aft_limit_x_over_c": selected["scissor"]["x_aft_over_mac"],
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


def sweep_wing_area():
    rows = []
    feasible_results = []
    stall_limit = back_transition_stall_limit()   # same for every wing area, so compute once

    for wing_area in wing_area_sweep_values():
        try:
            result = coupled_sizing_iteration(wing_area)
            summary = make_summary(result)
            scissor_ok = result["selected"]["feasible"]
            stall_ok = result["wing"]["stall_EAS_m_s"] <= stall_limit["stall_EAS_max_m_s"]
            feasible = scissor_ok and stall_ok
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
                "outbound_time_s": summary["outbound_time_s"],
                "peak_electrical_power_W": summary["peak_electrical_power_W"],
                "sizing_iterations_used": summary["sizing_iterations_used"],
            }
            rows.append(row)
            if feasible:
                feasible_results.append((summary["MTOW_mass_estimate_kg"], result, summary))
        except RuntimeError as error:
            rows.append({
                "wing_area_m2": wing_area,
                "feasible": False,
                "failure_reason": str(error),
                "MTOW_mass_estimate_kg": "",
                "mass_closure_error_kg": "",
                "wing_span_m": "",
                "wing_stall_EAS_m_s": "",
                "max_stall_EAS_m_s": "",
                "minimum_climb_EAS_m_s": "",
                "optimized_climb_EAS_m_s": "",
                "max_climb_CL": "",
                "climb_CL_limit": "",
                "course_climb_available_power_W": "",
                "course_climb_average_power_W": "",
                "course_climb_time_s": "",
                "course_climb_max_thrust_to_weight": "",
                "course_climb_thrust_limit": "",
                "cruise_true_speed_m_s": "",
                "mission_energy_Wh": "",
                "battery_mass_kg": "",
                "canard_area_ratio": "",
                "canard_area_m2": "",
                "outbound_time_s": "",
                "peak_electrical_power_W": "",
                "sizing_iterations_used": "",
            })

    if not feasible_results:
        raise RuntimeError("No feasible wing-area point was found in the sweep.")

    feasible_results.sort(key=lambda item: item[0])
    _, result, summary = feasible_results[0]
    return result, summary, rows


def run_sizing(output_dir=OUTPUT_DIR, make_plots=True):
    output_dir = Path(output_dir)

    result, summary, sweep_rows = sweep_wing_area()
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
    if make_plots:
        plot_scissor(output_dir / "scissor_plot.png", wing, candidates, selected)
        plot_mission_profile(output_dir / "mission_profile.png", mission)
        plot_wing_area_sweep(
            output_dir / "wing_area_sweep.png",
            sweep_rows,
            summary["wing_area_m2"],
        )

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
    print(f"  Canard: S_c/S_w={summary['canard_area_ratio']:.3f}, area {summary['canard_area_m2']:.2f} m^2")
    print(f"  CG: x/c={summary['x_CG_over_MAC']:.3f}, wing MAC LE x={summary['wing_mac_le_x_m']:.3f} m")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg")
    print(f"  Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
