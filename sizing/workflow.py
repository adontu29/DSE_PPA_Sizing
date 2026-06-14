"""Top-level sizing workflow: run_sizing() and main().

The method reads top to bottom:

    1. reset the airfoil assumptions to their fallbacks
    2. sweep the wing area (optionally with the XFOIL outer feedback loop) and
       keep the lightest feasible design
    3. summarise it and write the report tables and figures
"""

from __future__ import annotations

from pathlib import Path

from sizing.inputs import AIRCRAFT, MISSION, reset_airfoil_aero_defaults
from sizing.airfoil import sweep_with_xfoil_feedback
from sizing.report import (
    make_summary,
    progress,
    _fmt,
    write_key_value_csv,
    write_mass_breakdown,
    write_table_csv,
    plot_scissor,
    plot_mission_profile,
    plot_mission_trajectory,
    plot_wing_area_sweep,
)


OUTPUT_DIR = Path("outputs")


def run_sizing(output_dir=OUTPUT_DIR, make_plots=True, use_xfoil=None, show_progress=True):
    """Run the full sizing workflow and write the report outputs."""
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
        write_table_csv(output_dir / "wing_area_sweep.csv", sweep_rows)
        if make_plots:
            progress("Writing figures", show_progress)
            plot_scissor(output_dir / "scissor_plot.png", wing, candidates, selected)
            plot_mission_profile(output_dir / "mission_profile.png", mission)
            plot_mission_trajectory(output_dir / "mission_trajectory.png", mission)
            plot_wing_area_sweep(
                output_dir / "wing_area_sweep.png", sweep_rows, summary["wing_area_m2"]
            )
        progress(f"Outputs written to: {output_dir}", show_progress)

        return {
            "summary": summary,
            "wing": wing,
            "mission": mission,
            "propeller": propeller,
            "selected": selected,
            "candidates": candidates,
            "stall_limit": result["stall_limit"],
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
    print(
        f"  Stall cap ({summary['stall_limit_source']}): "
        f"stall EAS limit={summary['max_stall_EAS_m_s']:.1f} m/s "
        f"(forward {_fmt(summary['transition_sim_forward_cap_EAS_m_s'])} / "
        f"back {_fmt(summary['transition_sim_back_cap_EAS_m_s'])} m/s; "
        f"binding={summary['transition_sim_binding_leg']}, "
        f"t={_fmt(summary['transition_sim_cap_time_s'])} s, "
        f"d={_fmt(summary['transition_sim_cap_distance_m'], '.0f')} m, "
        f"alpha_max={_fmt(summary['transition_sim_cap_max_alpha_deg'])} deg)"
    )
    print(
        "    At actual stall speed: "
        f"forward={'OK' if summary['transition_sim_forward_actual_success'] else 'FAIL'}, "
        f"back={'OK' if summary['transition_sim_back_actual_success'] else 'FAIL'}"
    )
    print(
        "  Climb CL: "
        f"max {summary['max_climb_CL']:.3f}, limit {summary['climb_CL_limit']:.3f} "
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
    if summary.get("spiral_used"):
        print(
            "  Spiral climb: "
            f"spiral up in place to {summary['spiral_crossover_altitude_m'] / 1000.0:.2f} km "
            f"(R={summary['spiral_turn_radius_m']:.0f} m, n_max={summary['spiral_max_load_factor']:.2f}, "
            f"bank_max={summary['spiral_max_bank_angle_deg']:.1f} deg, "
            f"arc {summary['spiral_arc_m'] / 1000.0:.1f} km), "
            f"then straight climb-out {summary['climb_horizontal_distance_m'] / 1000.0:.1f} km to target"
        )
    else:
        print(
            "  Spiral climb: not engaged "
            f"(straight climb track {summary['climb_horizontal_distance_m'] / 1000.0:.1f} km <= "
            f"{MISSION['range_m'] / 1000.0:.0f} km range)"
        )
    print(f"  Propeller: diameter {summary['propeller_diameter_m']:.2f} m")
    print(f"  Canard: S_c/S_w={summary['canard_area_ratio']:.3f}, area {summary['canard_area_m2']:.2f} m^2")
    print(f"  CG: x/c={summary['x_CG_over_MAC']:.3f}, wing MAC LE x={summary['wing_mac_le_x_m']:.3f} m")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg")
    print(f"  Outputs written to: {OUTPUT_DIR}")
