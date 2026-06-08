"""Bellona simplified sizing script.

The inputs are grouped at the top. The workflow at the bottom runs the sizing
calculation and writes the report tables and figures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    "mission_equipment_mass_kg": 7.3,
}

AIRCRAFT = {
    "MTOW_kg": 52.78,
    "g_m_s2": 9.80665,
    "wing_area_m2": 6.8,
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
    "forward_flight_efficiency": 0.75 * 0.90 * 0.95,
    "hover_power_W": 14000.0,
    "transition_power_W": 8000.0,
    "battery_specific_energy_Wh_kg": 310.0,
    "battery_usable_fraction": 0.85,
    "battery_efficiency": 0.95,
    "n_rotors": 4,
    "thrust_to_weight": 1.30,
    "disc_loading_N_m2": 170.0,
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
# Atmosphere, mission, and wing equations
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


def wing_geometry(weight_N, rho_cruise):
    """Wing geometry and reference lift condition."""
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


def mission_energy(weight_N, wing, rho_cruise):
    """Energy estimate for hover, transition, and wing-borne climb."""
    speed = wing["cruise_true_speed_m_s"]
    q = 0.5 * rho_cruise * speed**2
    induced_factor = 1.0 / (math.pi * AIRCRAFT["wing_aspect_ratio"] * AIRCRAFT["oswald_efficiency"])
    CD = AIRCRAFT["CD0"] + induced_factor * wing["CL_trim"] ** 2
    drag_N = q * wing["area_m2"] * CD

    climb_rate = MISSION["average_climb_rate_m_s"]
    climb_time_s = MISSION["altitude_m"] / climb_rate
    climb_power_W = (drag_N * speed + weight_N * climb_rate) / AIRCRAFT["forward_flight_efficiency"]

    takeoff_time_s = MISSION["vertical_takeoff_height_m"] / MISSION["vertical_takeoff_rate_m_s"]
    transition_time_s = MISSION["transition_time_s"]
    hover_time_s = MISSION["hover_time_s"]

    energies = {
        "vertical_takeoff_Wh": AIRCRAFT["hover_power_W"] * takeoff_time_s / 3600.0,
        "transition_Wh": AIRCRAFT["transition_power_W"] * transition_time_s / 3600.0,
        "wing_borne_climb_Wh": climb_power_W * climb_time_s / 3600.0,
        "mission_hover_Wh": AIRCRAFT["hover_power_W"] * hover_time_s / 3600.0,
    }
    total_energy_Wh = sum(energies.values())
    battery_mass_kg = (
        total_energy_Wh
        / AIRCRAFT["battery_specific_energy_Wh_kg"]
        / AIRCRAFT["battery_usable_fraction"]
        / AIRCRAFT["battery_efficiency"]
    )

    profile = [
        (0.0, 0.0),
        (takeoff_time_s, MISSION["vertical_takeoff_height_m"]),
        (takeoff_time_s + transition_time_s, MISSION["vertical_takeoff_height_m"]),
        (takeoff_time_s + transition_time_s + climb_time_s, MISSION["altitude_m"]),
        (takeoff_time_s + transition_time_s + climb_time_s + hover_time_s, MISSION["altitude_m"]),
    ]

    return {
        "drag_N": drag_N,
        "CD_trim": CD,
        "climb_power_W": climb_power_W,
        "total_energy_Wh": total_energy_Wh,
        "battery_mass_kg": battery_mass_kg,
        "profile": profile,
        **energies,
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
        AIRCRAFT["transition_power_W"],
        mission["climb_power_W"],
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
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(times, altitudes, marker="o", color="#1f77b4")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("altitude [m]")
    ax.set_title("Simplified mission profile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_sizing(output_dir=OUTPUT_DIR, make_plots=True):
    output_dir = Path(output_dir)

    weight_N = AIRCRAFT["MTOW_kg"] * AIRCRAFT["g_m_s2"]
    rho_cruise = isa_density(MISSION["altitude_m"])
    propeller = propeller_disk_estimate(weight_N)
    wing = wing_geometry(weight_N, rho_cruise)
    mission = mission_energy(weight_N, wing, rho_cruise)
    selected, candidates = canard_and_wing_iteration(wing, mission, propeller)

    summary = {
        "MTOW_input_kg": AIRCRAFT["MTOW_kg"],
        "MTOW_mass_estimate_kg": selected["mass"]["total_mass_kg"],
        "wing_area_m2": wing["area_m2"],
        "wing_span_m": wing["span_m"],
        "wing_chord_m": wing["chord_m"],
        "wing_stall_EAS_m_s": wing["stall_EAS_m_s"],
        "CL_trim": wing["CL_trim"],
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
        "propeller_diameter_m": propeller["propeller_diameter_m"],
        "candidate_count": len(candidates),
    }

    write_key_value_csv(output_dir / "summary.csv", summary)
    write_mass_breakdown(output_dir / "mass_breakdown.csv", selected["mass"])
    if make_plots:
        plot_scissor(output_dir / "scissor_plot.png", wing, candidates, selected)
        plot_mission_profile(output_dir / "mission_profile.png", mission)

    return {
        "summary": summary,
        "wing": wing,
        "mission": mission,
        "propeller": propeller,
        "selected": selected,
        "candidates": candidates,
    }


def main():
    result = run_sizing()
    summary = result["summary"]
    print("Bellona simplified sizing")
    print(f"  MTOW estimate: {summary['MTOW_mass_estimate_kg']:.2f} kg")
    print(f"  Wing: {summary['wing_area_m2']:.2f} m^2, span {summary['wing_span_m']:.2f} m")
    print(f"  Canard: S_c/S_w={summary['canard_area_ratio']:.3f}, area {summary['canard_area_m2']:.2f} m^2")
    print(f"  CG: x/c={summary['x_CG_over_MAC']:.3f}, wing MAC LE x={summary['wing_mac_le_x_m']:.3f} m")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg")
    print(f"  Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
