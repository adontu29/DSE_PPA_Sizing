"""Report outputs: a concise summary table, the mass breakdown, the wing-area
trade table, and the four report figures.

The summary is deliberately short -- the values an engineer quotes in a sizing
report, not every diagnostic. Heavy traceability dumps (per-XFOIL-pass Reynolds,
full coefficient histories, JSON) are intentionally not produced here.
"""

from __future__ import annotations

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from drag_buildup import build_drag_geometry, parasite_drag_buildup
from mission_energy_course import trim_lift_split
from xfoil_wrapper import mach_number as xfoil_mach_number, reynolds_number

from sizing.inputs import AIRCRAFT, MASS, MISSION
from sizing.atmosphere import isa_density
from sizing.mission import trim_drag_descriptor
from sizing.scissor import scissor_limits


# ---------------------------------------------------------------------------
# Progress / formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, rem_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{minutes:.0f}m {rem_seconds:.0f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{hours:.0f}h {rem_minutes:.0f}m"


def _fmt(value, spec=".1f"):
    """Format a number, or pass through missing optional fields as 'n/a'."""
    if value is None or value == "":
        return "n/a"
    return format(value, spec)


def progress(message, enabled=True, indent=0):
    if enabled:
        print(f"{'  ' * indent}{message}", flush=True)


# ---------------------------------------------------------------------------
# Cruise drag breakdown (for the summary)
# ---------------------------------------------------------------------------


def drag_buildup_summary(result, aircraft):
    """Component drag build-up and trimmed lift split at the cruise condition."""
    wing = result["wing"]
    mission = result["mission"]
    trim = trim_drag_descriptor(result)
    weight_N = result["final_mass_used_kg"] * aircraft["g_m_s2"]
    altitude_m = MISSION["altitude_m"]
    true_speed = mission["cruise_true_speed_m_s"]
    q = 0.5 * isa_density(altitude_m) * true_speed**2
    S_w = wing["area_m2"]
    S_c = trim["S_c"]

    L_w, L_c = trim_lift_split(weight_N, q, wing, trim)
    fields = {
        "CD0_fixed": aircraft["CD0"],
        "oswald_wing": aircraft["oswald_efficiency"],
        "oswald_canard": aircraft.get("canard_oswald_efficiency", aircraft["oswald_efficiency"]),
        "wing_CL_trim": L_w / (q * S_w),
        "canard_CL_trim": L_c / (q * S_c) if S_c > 0.0 else "",
    }

    geom = build_drag_geometry(aircraft, wing, trim)
    if geom is not None:
        breakdown = parasite_drag_buildup(
            geom,
            xfoil_mach_number(altitude_m, true_speed),
            reynolds_number(altitude_m, true_speed, 1.0),
        )
        fields.update({
            "CD0_component": breakdown["CD0_component"],
            "CD0_wing": breakdown["CD0_wing"],
            "CD0_canard": breakdown["CD0_canard"],
            "CD0_fuselage": breakdown["CD0_fuselage"],
            "CD0_hardware": breakdown["CD0_hardware"],
            "CD0_misc": breakdown["CD0_misc"],
        })
    else:
        for key in ("CD0_component", "CD0_wing", "CD0_canard",
                    "CD0_fuselage", "CD0_hardware", "CD0_misc"):
            fields[key] = ""
    return fields


def _sim_leg_field(stall_limit, leg_key, field):
    """Read a per-leg field from a transition-sim stall limit, or '' if absent."""
    leg = stall_limit.get(leg_key)
    if not leg:
        return ""
    value = leg.get(field, "")
    if isinstance(value, list):
        return "; ".join(value)
    return value


# ---------------------------------------------------------------------------
# Concise report summary
# ---------------------------------------------------------------------------


def make_summary(result):
    """Report-ready summary of the selected design (the values quoted in a report)."""
    wing = result["wing"]
    mission = result["mission"]
    propeller = result["propeller"]
    selected = result["selected"]
    candidates = result["candidates"]
    stall_limit = result["stall_limit"]
    drag = drag_buildup_summary(result, result.get("aircraft", AIRCRAFT))

    wingborne_states = [s for s in mission["states"] if s["segment"] == "wing_borne_climb"]
    max_climb_CL = max([s["CL"] for s in wingborne_states] or [0.0])
    climb_CL_limit = AIRCRAFT["wing_CL_max"] / AIRCRAFT["climb_stall_margin_n"]**2

    return {
        # --- Mass closure ---
        "MTOW_input_kg": AIRCRAFT["MTOW_kg"],
        "MTOW_used_for_final_pass_kg": result["final_mass_used_kg"],
        "MTOW_mass_estimate_kg": selected["mass"]["total_mass_kg"],
        "mass_closure_error_kg": selected["mass"]["total_mass_kg"] - result["final_mass_used_kg"],
        "sizing_iterations_used": len(result["iteration_history"]),
        "xfoil_enabled": result.get("xfoil_airfoil_update", {}).get("enabled", False),

        # --- Wing and cruise aerodynamics ---
        "wing_area_m2": wing["area_m2"],
        "wing_span_m": wing["span_m"],
        "wing_chord_m": wing["chord_m"],
        "wing_aspect_ratio": AIRCRAFT["wing_aspect_ratio"],
        "wing_stall_EAS_m_s": wing["stall_EAS_m_s"],
        "wing_CL_max": AIRCRAFT["wing_CL_max"],
        "canard_CL_max": AIRCRAFT["canard_CL_max"],
        "CL_trim": wing["CL_trim"],
        "CD_trim": mission["CD_trim"],
        "CD0_component": drag["CD0_component"],
        "wing_CL_trim": drag["wing_CL_trim"],
        "canard_CL_trim": drag["canard_CL_trim"],
        "cruise_true_speed_m_s": mission["cruise_true_speed_m_s"],

        # --- Climb ---
        "climb_stall_margin_n": AIRCRAFT["climb_stall_margin_n"],
        "climb_CL_limit": climb_CL_limit,
        "max_climb_CL": max_climb_CL,
        "minimum_climb_EAS_m_s": mission["mission_grid"]["aerodynamic_speed_limits"]["minimum_climb_EAS_m_s"],
        "optimized_climb_EAS_m_s": mission["optimized_climb_EAS_m_s"],
        "optimized_climb_angle_deg": mission["optimized_climb_angle_deg"],
        "course_climb_time_s": mission["course_climb_time_s"],
        "course_climb_available_power_W": mission["course_climb_available_power_W"],
        "course_climb_max_thrust_to_weight": mission["course_climb_max_thrust_to_weight"],
        "course_climb_thrust_limit": mission["course_climb_thrust_limit"],

        # --- Maximum-stall-speed requirement (transition simulation) ---
        "max_stall_EAS_m_s": stall_limit["stall_EAS_max_m_s"],
        "stall_margin_m_s": stall_limit["stall_EAS_max_m_s"] - wing["stall_EAS_m_s"],
        "stall_limit_source": stall_limit["source"],
        "transition_sim_binding_leg": stall_limit.get("binding_leg", ""),
        "transition_sim_forward_cap_EAS_m_s": stall_limit.get("forward_stall_EAS_max_m_s", ""),
        "transition_sim_back_cap_EAS_m_s": stall_limit.get("back_stall_EAS_max_m_s", ""),
        "transition_sim_cap_time_s": stall_limit.get("sim_cap_time_s", ""),
        "transition_sim_cap_distance_m": stall_limit.get("sim_cap_distance_m", ""),
        "transition_sim_cap_max_alpha_deg": stall_limit.get("sim_cap_max_alpha_deg", ""),
        "transition_sim_forward_actual_success": _sim_leg_field(stall_limit, "forward_sim", "sim_actual_success"),
        "transition_sim_back_actual_success": _sim_leg_field(stall_limit, "back_sim", "sim_actual_success"),

        # --- Canard, layout and CG (scissor) ---
        "canard_area_ratio": selected["canard"]["area_ratio"],
        "canard_area_m2": selected["canard"]["area_m2"],
        "canard_span_m": selected["canard"]["span_m"],
        "canard_arm_m": selected["arm_m"],
        "wing_mac_le_x_m": selected["mass"]["wing_mac_le_x_m"],
        "fuselage_length_m": selected["mass"]["fuselage_length_m"],
        "x_CG_over_MAC": selected["mass"]["x_cg_over_mac"],
        "scissor_forward_limit_x_over_c": selected["scissor"]["x_forward_over_mac"],
        "scissor_aft_limit_x_over_c": selected["scissor"]["x_aft_over_mac"],
        "scissor_clearance_over_c": selected.get("clearance_over_mac", ""),
        "achieved_static_margin_over_mac": selected["achieved_static_margin_over_mac"],
        "statically_stable": selected["statically_stable"],

        # --- Energy, power and mission ---
        "battery_mass_kg": mission["battery_mass_kg"],
        "mission_energy_Wh": mission["total_energy_Wh"],
        "installed_battery_energy_Wh": mission["installed_battery_energy_Wh"],
        "peak_electrical_power_W": mission["peak_electrical_power_W"],
        "total_mission_time_s": mission["total_mission_time_s"],
        "outbound_time_s": mission["outbound_time_s"],
        "climb_horizontal_distance_m": mission["climb_horizontal_distance_m"],
        "level_cruise_distance_m": mission["level_cruise_distance_m"],
        "spiral_used": mission["spiral_used"],
        "spiral_crossover_altitude_m": mission["spiral_crossover_altitude_m"],
        "spiral_turn_radius_m": mission["spiral_turn_radius_m"],
        "spiral_arc_m": mission["spiral_arc_m"],
        "spiral_max_load_factor": mission["spiral_max_load_factor"],
        "spiral_max_bank_angle_deg": mission["spiral_max_bank_angle_deg"],

        # --- Propulsion geometry ---
        "propeller_diameter_m": propeller["propeller_diameter_m"],
        "candidate_count": len(candidates),
    }


# ---------------------------------------------------------------------------
# CSV writers
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


def write_table_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rows are heterogeneous (feasible rows carry more fields than error rows), so
    # the header is the union of every row's keys in first-seen order.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


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
    # different wing station for each candidate ratio, so the curves are drawn at
    # the selected design's coefficients.
    coeffs = selected["coeffs"]
    limits = [scissor_limits(area_ratio, coeffs) for area_ratio in area_ratios]
    x_forward = [item["x_forward_over_mac"] for item in limits]
    x_aft = [item["x_aft_over_mac"] for item in limits]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(x_aft, area_ratios, color="#c0392b", label="Stability (aft CG limit)")
    ax.plot(x_forward, area_ratios, color="#2c3e50", label="Controllability (fwd CG limit)")
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

    ax.set_xlabel("CG position  x_cg / wing chord  [-]")
    ax.set_ylabel("canard area ratio  S_c / S_w  [-]")
    ax.set_title("Canard scissor plot")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mission_trajectory(path, mission):
    """3D physical trajectory: launch -> spiral-up -> straight climb-out -> intercept."""
    R = mission.get("spiral_turn_radius_m") or 0.0
    spiral_used = mission.get("spiral_used", False)
    states = mission["states"]
    h_takeoff = MISSION["vertical_takeoff_height_m"]
    trans_dist = mission["segment_summaries"]["transition"]["distance_m"]
    target_x = MISSION["range_m"]
    target_z = MISSION["altitude_m"]

    # Ordered path of (x, y, z, segment), x toward the target, z = altitude.
    pts = [(0.0, 0.0, 0.0, "takeoff"), (0.0, 0.0, h_takeoff, "takeoff")]
    pts.append((trans_dist, 0.0, h_takeoff, "transition"))

    cx, cy = trans_dist, R                      # spiral centre; entry tangent at (trans_dist, 0)
    arc = 0.0
    exit_x, exit_y, exit_z = trans_dist, 0.0, h_takeoff
    straight_states = []
    for st in states:
        z = st["altitude_m"]
        if spiral_used and st.get("spiral_step") and R > 0.0:
            arc += st["delta_x_m"]
            phi = arc / R
            x = cx + R * math.sin(phi)
            y = cy - R * math.cos(phi)
            pts.append((x, y, z, "spiral"))
            exit_x, exit_y, exit_z = x, y, z
        else:
            straight_states.append(st)

    if spiral_used and straight_states:
        dz = target_z - exit_z
        for st in straight_states:
            f = min(1.0, max(0.0, (st["altitude_m"] - exit_z) / dz)) if dz > 0 else 1.0
            pts.append((exit_x + f * (target_x - exit_x), exit_y * (1.0 - f), st["altitude_m"], "climb"))
    elif not spiral_used:
        x = trans_dist
        for st in states:
            x += st["delta_x_m"]
            pts.append((x, 0.0, st["altitude_m"], "climb"))
        if mission["level_cruise_distance_m"] > 0.0:
            pts.append((x + mission["level_cruise_distance_m"], 0.0, target_z, "cruise"))
    pts.append((target_x, 0.0, target_z, "intercept"))

    colors = {
        "takeoff": "#94d2bd", "transition": "#0a9396", "spiral": "#ee9b00",
        "climb": "#005f73", "cruise": "#9b2226", "intercept": "#9b2226",
    }
    X = [p[0] / 1000.0 for p in pts]
    Y = [p[1] / 1000.0 for p in pts]
    Z = [p[2] / 1000.0 for p in pts]
    seg = [p[3] for p in pts]

    fig = plt.figure(figsize=(14.0, 7.0))
    ax = fig.add_subplot(121, projection="3d")
    z_floor = min(Z)
    for i in range(len(pts) - 1):
        ax.plot(X[i:i + 2], Y[i:i + 2], Z[i:i + 2], color=colors[seg[i + 1]], linewidth=2.2)
        ax.plot(X[i:i + 2], Y[i:i + 2], [z_floor, z_floor], color="#adb5bd", linewidth=0.8, alpha=0.6)

    ax.scatter([0.0], [0.0], [0.0], color="#001219", s=45)
    ax.scatter([X[-1]], [Y[-1]], [Z[-1]], color="#9b2226", marker="*", s=240)
    ax.plot([X[-1], X[-1]], [Y[-1], Y[-1]], [z_floor, Z[-1]], color="#9b2226", linestyle=":", linewidth=1.0)
    ax.set_xlabel("range toward target [km]")
    ax.set_ylabel("cross-track [km]")
    ax.set_zlabel("altitude [km]")
    ax.set_box_aspect((6.0, 1.2, 4.0))
    ax.view_init(elev=18, azim=-60)
    ax.set_title("3D trajectory", fontsize=11)

    # 2D side view (altitude vs downrange) -- the readable "line to interception".
    ax2 = fig.add_subplot(122)
    for i in range(len(pts) - 1):
        ax2.plot(X[i:i + 2], Z[i:i + 2], color=colors[seg[i + 1]], linewidth=2.4)
    ax2.scatter([0.0], [0.0], color="#001219", s=45)
    ax2.scatter([X[-1]], [Z[-1]], color="#9b2226", marker="*", s=260, zorder=5)
    ax2.annotate("balloon", (X[-1], Z[-1]), textcoords="offset points", xytext=(-46, 4), fontsize=9, color="#9b2226")
    ax2.set_xlabel("downrange toward target [km]")
    ax2.set_ylabel("altitude [km]")
    ax2.set_title("Side view (altitude vs downrange)", fontsize=11)
    ax2.grid(True, alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], color=colors["takeoff"], lw=2.4, label="vertical takeoff"),
        plt.Line2D([0], [0], color=colors["transition"], lw=2.4, label="transition"),
        plt.Line2D([0], [0], color=colors["spiral"], lw=2.4, label="spiral climb (in place)"),
        plt.Line2D([0], [0], color=colors["climb"], lw=2.4, label="straight climb-out"),
        plt.Line2D([0], [0], color="#9b2226", marker="*", lw=0, markersize=12, label="intercept"),
    ]
    if not spiral_used and mission["level_cruise_distance_m"] > 0.0:
        legend_handles.insert(4, plt.Line2D([0], [0], color=colors["cruise"], lw=2.4, label="level cruise"))
    ax2.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)

    if spiral_used:
        subtitle = (
            f"spiral up to {mission['spiral_crossover_altitude_m'] / 1000.0:.2f} km over launch "
            f"(R={R:.0f} m, n_max={mission['spiral_max_load_factor']:.2f}, "
            f"{mission['spiral_arc_m'] / 1000.0:.1f} km arc), then climb-out to the balloon"
        )
    else:
        subtitle = "straight climb-out to the balloon (no spiral needed)"
    fig.suptitle("Bellona mission trajectory to interception\n" + subtitle, fontsize=13, fontweight="bold")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
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
        0.5, 0.935,
        (
            f"Load energy {mission['total_energy_Wh'] / 1000.0:.2f} kWh | "
            f"Installed battery {mission['installed_battery_energy_Wh'] / 1000.0:.2f} kWh | "
            f"Climb EAS {mission['optimized_climb_EAS_m_s']:.1f} m/s | "
            f"gamma {mission['optimized_climb_angle_deg']:.1f} deg"
        ),
        ha="center", fontsize=10, color="#3d4752",
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
        ax.plot([s["speed_m_s"] for s in climb_states], state_altitude_km, color="#001219", linewidth=2.2, label="TAS")
        ax.plot([s["EAS_m_s"] for s in climb_states], state_altitude_km, color="#ee9b00", linewidth=2.0, linestyle="--", label="EAS")
    ax.set_title("Climb speed schedule")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if climb_states:
        ax.plot([s["electrical_power_W"] / 1000.0 for s in climb_states], state_altitude_km, color="#bb3e03", linewidth=2.2, label="wing-borne climb")
    ax.axvline(AIRCRAFT["hover_power_W"] / 1000.0, color="#9b2226", linestyle=":", linewidth=1.8, label="hover assumption")
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
        CL = [s["CL"] for s in climb_states]
        rate_of_climb = [s["rate_of_climb_m_s"] for s in climb_states]
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
