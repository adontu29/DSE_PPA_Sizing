"""Command-line interface, JSON output, and compact console summary."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .models import Assumptions, Mission
from .workflows.coupled import iterate_phases_1_to_15
from .workflows.mtow import converge_design


def _json_safe(value):
    """Convert numpy and Path values into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_summary(result: Dict) -> Dict:
    """Pull the most useful Phase 1-16 outputs into one compact dictionary."""
    phase1 = result["phase1"]
    phase2 = result["phase2_hover"]
    phase3 = result["phase3"]
    phase4 = result["phase4"]
    phase5 = result["phase5"]
    phase8 = result["phase8"]
    phase9 = result["phase9"]
    phase10 = result.get("phase10", {})
    phase11 = result.get("phase11", {})
    phase12 = result.get("phase12", {})
    phase13 = result.get("phase13", {})
    phase14 = result.get("phase14", {})
    phase15 = result.get("phase15", {})
    phase16 = result.get("phase16", {})
    canard_search = result.get("canard_cg_search", {})
    sanity_checks = result.get("sanity_checks", {})
    effective_assumptions = result.get("effective_assumptions", {})
    wing_layout = result.get("wing_layout_solver", phase15.get("wing_layout_solver", {}))

    n_rotors = phase1["T_total"] / phase1["T_per_rotor"]
    summary = {
        "converged": result["converged"],
        "inner_loop_converged": result.get("inner_converged", result["converged"]),
        "outer_loop_converged": result.get("outer_converged"),
        "stopped_reason": result["stopped_reason"],
        "iterations": len(result["history"]),
        "outer_iterations": result.get("outer_iterations"),
        "control_closure_iterations": len(result.get("control_mass_history", [])),
        "MTOW_kg": result["MTOW_kg"],
        "MTOW_seed_kg": result.get("MTOW_seed_kg"),
        "MTOW_converged_kg": result.get("MTOW_converged_kg"),
        "effective_thrust_to_weight": effective_assumptions.get("thrust_to_weight"),
        "effective_hover_pitch_arm_fraction_fuselage": effective_assumptions.get("hover_pitch_arm_fraction_fuselage"),
        "effective_hover_roll_arm_fraction_span": effective_assumptions.get("hover_roll_arm_fraction_span"),
        "propeller_diameter_m": phase1["D_prop"],
        "propeller_diameter_max_m": phase1.get("prop_diameter_max_m"),
        "propeller_diameter_limited": phase1.get("prop_diameter_limited", False),
        "disc_loading_target_N_m2": phase1.get("disc_loading_target", phase1["disc_loading"]),
        "disc_loading_actual_N_m2": phase1["disc_loading"],
        "cruise_speed_m_s": phase3["V_cruise"],
        "cruise_speed_min_required_m_s": phase3["V_min_required"],
        "cruise_speed_min_requirement_active": phase3["V_min_requirement_active"],
        "fixed_wing_ROC_m_s": phase3["ROC"],
        "cruise_advance_ratio": phase4["J"],
        "cruise_rpm": result["cruise_rpm"],
        "hover_power_total_W": phase2["P_elec"] * n_rotors,
        "battery_mass_kg": phase5["m_batt_kg"],
        "installed_battery_energy_Wh": phase5["E_total_Wh"],
        "phase5_transition_energy_source": phase5.get("transition_energy_source", "phase3_mission_optimise"),
        "wing_area_m2": phase8["S"],
        "wing_span_m": phase8["b"],
        "wing_mean_chord_m": phase8["c_bar"],
        "wing_stall_speed_m_s": phase8["V_stall"],
        "canard_area_m2": phase9["S_c"],
        "canard_span_m": phase9["b_c"],
        "canard_mean_chord_m": phase9["c_bar_c"],
        "Re_main": result["Re_main"],
        "Re_canard": result["Re_canard"],
    }
    if phase10:
        summary.update({
            "neutral_point_x_over_c": phase10["x_np_over_c"],
            "cg_forward_x_over_c": phase10["x_cg_fwd_over_c"],
            "cg_aft_x_over_c": phase10["x_cg_aft_over_c"],
            "cg_range_pct_mac": phase10["CG_range_pct"],
            "cg_range_feasible": phase10["feasible_preliminary_CG_range"],
            "operational_cg_feasible": phase10.get("operational_CG_feasible"),
            "operational_cg_forward_x_over_c": phase10.get("x_cg_fwd_operational_over_mac"),
            "operational_cg_aft_x_over_c": phase10.get("x_cg_aft_operational_over_mac"),
            "operational_cg_range_over_mac": phase10.get("operational_CG_range_over_mac"),
            "operational_cg_fwd_margin_over_mac": phase10.get("operational_fwd_margin_over_mac"),
            "operational_cg_aft_margin_over_mac": phase10.get("operational_aft_margin_over_mac"),
            "mass_cg_inside_theoretical": phase10.get("mass_CG_inside_theoretical"),
            "required_CG_shift_over_mac": phase10.get("required_CG_shift_over_mac"),
        })
    if phase11:
        summary.update({
            "elevon_chord_fraction": phase11["c_e_over_c"],
            "elevon_span_fraction": phase11["b_e_over_b"],
            "elevon_area_total_m2": phase11["S_e_total"],
            "elevon_area_ratio": phase11["S_e_over_S"],
            "elevon_binding_case": phase11["binding_case"],
            "elevon_feasible": phase11["feasible_preliminary_elevon"],
            "roll_rate_achievable_deg_s": phase11["p_achievable_deg_s"],
            "roll_rate_required_deg_s": phase11["p_required_deg_s"],
            "Cm_de_abs": phase11["Cm_de_abs"],
            "Cm_de_required": phase11["Cm_de_required"],
            "elevon_pitch_margin": phase11["pitch_margin"],
            "elevon_roll_margin": phase11["roll_margin"],
        })
    if phase12:
        summary.update({
            "hover_control_feasible": phase12["feasible_preliminary_hover_control"],
            "hover_control_binding_case": phase12["binding_case"],
            "hover_pitch_margin": phase12["pitch_margin"],
            "hover_roll_margin": phase12["roll_margin"],
            "hover_yaw_margin": phase12["yaw_margin"],
            "hover_delta_T_available_N": phase12["delta_T_available"],
            "hover_delta_T_pitch_required_N": phase12["delta_T_pitch_required"],
            "hover_delta_T_roll_required_N": phase12["delta_T_roll_required"],
            "hover_pitch_arm_m": phase12["d_x_rotor"],
            "hover_pitch_arm_required_m": phase12["pitch_arm_required_m"],
            "hover_roll_arm_m": phase12["d_y_rotor"],
            "hover_roll_arm_required_m": phase12["roll_arm_required_m"],
            "hover_thrust_to_weight_required_pitch": phase12["thrust_to_weight_required_pitch"],
            "hover_thrust_to_weight_required_roll": phase12["thrust_to_weight_required_roll"],
            "hover_pitch_angular_accel_available_rad_s2": phase12["pitch_angular_accel_available_rad_s2"],
            "hover_roll_angular_accel_available_rad_s2": phase12["roll_angular_accel_available_rad_s2"],
            "hover_Iyy_max_for_pitch_kg_m2": phase12["Iyy_max_for_pitch_kg_m2"],
            "hover_Ixx_max_for_roll_kg_m2": phase12["Ixx_max_for_roll_kg_m2"],
            "hover_yaw_elevon_area_required_m2": phase12["S_e_yaw_required"],
            "hover_slipstream_speed_m_s": phase12["V_slip_hover"],
        })
    if phase13:
        summary.update({
            "transition_blend_start_m_s": phase13["V_blend_start"],
            "transition_blend_end_m_s": phase13["V_blend_end"],
            "transition_time_s": phase13["t_transition"],
            "transition_distance_m": phase13["distance_transition_m"],
            "transition_energy_estimate_Wh": phase13["E_transition_estimate_Wh"],
            "transition_cruise_required_for_margin_m_s": phase13["V_cruise_required_for_margin"],
            "transition_cruise_margin_over_blend_end": phase13["cruise_speed_margin_over_blend_end"],
            "transition_alpha_fw_at_cruise": phase13["alpha_fw_at_cruise"],
            "transition_alpha_hover_at_cruise": phase13["alpha_hover_at_cruise"],
        })
    if phase14:
        summary.update({
            "dynamic_stability_preliminary": phase14["level_meets_8785C_preliminary"],
            "short_period_meets": phase14["short_period_meets"],
            "short_period_omega_rad_s": phase14["omega_sp_rad_s"],
            "short_period_zeta": phase14["zeta_sp"],
            "phugoid_meets": phase14["phugoid_meets"],
            "phugoid_omega_rad_s": phase14["omega_ph_rad_s"],
            "phugoid_zeta": phase14["zeta_ph"],
            "dutch_roll_meets": phase14["dutch_roll_meets"],
            "dutch_roll_omega_rad_s": phase14["omega_dr_rad_s"],
            "dutch_roll_zeta": phase14["zeta_dr"],
            "spiral_meets": phase14["spiral_meets"],
            "spiral_stable": phase14["spiral_stable"],
            "spiral_time_to_double_s": phase14["spiral_time_to_double_s"],
            "roll_subsidence_tau_s": phase14["roll_subsidence_tau_s"],
            "Cm_alpha": phase14["Cm_alpha"],
            "Cm_q_effective": phase14["Cm_q_effective"],
        })
    if phase15:
        summary.update({
            "MTOW_mass_estimate_kg": phase15["MTOW_estimate_kg"],
            "MTOW_mass_delta_kg": phase15["MTOW_estimate_kg"] - result["MTOW_kg"],
            "MTOW_mass_delta_fraction": (
                (phase15["MTOW_estimate_kg"] - result["MTOW_kg"]) / result["MTOW_kg"]
            ),
            "empty_mass_no_battery_no_mission_kg": phase15["empty_mass_no_battery_no_mission_kg"],
            "structure_mass_kg": phase15["structure_mass_kg"],
            "propulsion_mass_kg": phase15["propulsion_mass_kg"],
            "mission_equipment_mass_kg": phase15["m_mission_equipment_kg"],
            "wing_mac_le_x_m": phase15.get("wing_mac_le_x_m"),
            "mass_model_x_CG_m": phase15["x_CG_m"],
            "mass_model_x_CG_fuselage_m": phase15.get("x_CG_fuselage_m"),
            "mass_model_x_CG_over_wing_mac": phase15["x_CG_over_wing_mac"],
            "mass_model_Ixx_kg_m2": phase15["I_tensor_kg_m2"]["Ixx"],
            "mass_model_Iyy_kg_m2": phase15["I_tensor_kg_m2"]["Iyy"],
            "mass_model_Izz_kg_m2": phase15["I_tensor_kg_m2"]["Izz"],
            "external_tow_load_included_in_MTOW": phase15["external_tow_load_included_in_MTOW"],
        })
    if wing_layout:
        summary.update({
            "wing_layout_solve_enabled": wing_layout["enabled"],
            "wing_layout_solved": wing_layout["solved"],
            "wing_layout_required_shift_m": wing_layout["required_wing_shift_m"],
            "wing_layout_initial_x_CG_over_mac": wing_layout["initial_x_CG_over_mac"],
            "wing_layout_final_x_CG_over_mac": wing_layout["final_x_CG_over_mac"],
            "wing_layout_target_x_CG_over_mac": wing_layout["target_x_CG_over_mac"],
            "wing_layout_station_over_mac": wing_layout["wing_station_over_mac"],
        })
    if canard_search:
        summary.update({
            "canard_selected_volume_coeff": canard_search["selected_V_bar_c"],
            "canard_default_volume_coeff": canard_search["default_V_bar_c"],
            "canard_cg_search_candidate_count": canard_search["candidate_count"],
            "canard_default_operational_CG_feasible": canard_search["default_operational_CG_feasible"],
        })
    if phase16:
        summary.update({
            "phase16_MTOW_old_kg": phase16["MTOW_old_kg"],
            "phase16_MTOW_new_kg": phase16["MTOW_new_kg"],
            "phase16_MTOW_next_kg": phase16["MTOW_next_kg"],
            "phase16_delta_kg": phase16["delta_kg"],
            "phase16_relative_error": phase16["relative_error"],
            "phase16_damping": phase16["damping"],
            "phase16_tol": phase16["tol"],
            "phase16_converged": phase16["converged"],
        })
    if sanity_checks:
        summary.update({
            "design_sanity_all_passed": sanity_checks["all_passed"],
            "design_sanity_issue_count": sanity_checks["issue_count"],
            "design_sanity_issues": sanity_checks["issues"],
        })
    return summary


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the completed Bellona Phase 1-16 propulsion, aerodynamics, controls, transition, stability, mass, and MTOW convergence loop."
    )
    parser.add_argument("--mtow-kg", type=float, default=50.0,
                        help="MTOW seed for the Phase 1-16 loop, or fixed MTOW when --fixed-mtow is used.")
    parser.add_argument("--fixed-mtow", action="store_true",
                        help="Run one fixed-MTOW Phase 1-15 pass instead of the Phase 16 outer loop.")
    parser.add_argument("--altitude-m", type=float, default=Mission.altitude_m)
    parser.add_argument("--range-m", type=float, default=Mission.range_m)
    parser.add_argument("--time-budget-s", type=float, default=Mission.time_budget_s)
    parser.add_argument("--hover-time-s", type=float, default=Mission.hover_time_s)
    parser.add_argument("--mission-equipment-mass-kg", type=float,
                        default=Mission.mission_equipment_mass_kg)
    parser.add_argument("--external-tow-load-n", type=float,
                        default=Mission.external_tow_load_N)

    parser.add_argument("--disc-loading-target-n-m2", type=float,
                        default=Assumptions.disc_loading_target_N_m2)
    parser.add_argument("--thrust-to-weight", type=float,
                        default=Assumptions.thrust_to_weight)
    parser.add_argument("--prop-diameter-max-m", type=float, default=None,
                        help="Optional CAD/geometry propeller diameter cap.")
    parser.add_argument("--battery-specific-energy-wh-kg", type=float,
                        default=Assumptions.battery_specific_energy_Wh_kg)
    parser.add_argument("--stall-speed-target-max-m-s", type=float,
                        default=Assumptions.stall_speed_target_max_m_s)
    parser.add_argument("--wing-area-seed-m2", type=float,
                        default=Assumptions.preliminary_wing_area_m2)
    parser.add_argument("--cruise-j-target", type=float,
                        default=Assumptions.cruise_J_target)
    parser.add_argument("--canard-volume-coeff", type=float,
                        default=Assumptions.canard_volume_coeff)
    parser.add_argument("--canard-volume-grid-min", type=float,
                        default=Assumptions.canard_volume_grid_min)
    parser.add_argument("--canard-volume-grid-max", type=float,
                        default=Assumptions.canard_volume_grid_max)
    parser.add_argument("--canard-volume-grid-step", type=float,
                        default=Assumptions.canard_volume_grid_step)
    parser.add_argument("--cg-envelope-half-width-over-mac", type=float,
                        default=Assumptions.cg_envelope_half_width_over_mac)
    parser.add_argument("--cg-required-margin-over-mac", type=float,
                        default=Assumptions.cg_required_margin_over_mac)
    parser.add_argument("--wing-mac-le-x-m", type=float, default=None,
                        help="Optional initial/fixed wing MAC leading-edge station from the provisional fuselage/equipment datum.")
    parser.add_argument("--disable-wing-position-solve", action="store_true",
                        help="Keep the provided wing MAC station instead of solving it from the scissor CG target.")
    parser.add_argument("--roll-rate-required-deg-s", type=float,
                        default=Assumptions.roll_rate_required_deg_s)
    parser.add_argument("--elevon-max-deflection-deg", type=float,
                        default=Assumptions.elevon_max_deflection_deg)
    parser.add_argument("--elevon-q-slipstream-ratio", type=float,
                        default=Assumptions.elevon_q_slipstream_ratio)
    parser.add_argument("--hover-angular-accel-rad-s2", type=float,
                        default=Assumptions.hover_angular_accel_required_rad_s2)
    parser.add_argument("--hover-yaw-rate-required-deg-s", type=float,
                        default=Assumptions.hover_yaw_rate_required_deg_s)
    parser.add_argument("--hover-yaw-response-time-s", type=float,
                        default=Assumptions.hover_yaw_response_time_s)
    parser.add_argument("--hover-pitch-arm-m", type=float, default=None,
                        help="Optional pitch rotor moment arm from CG to rotor pair.")
    parser.add_argument("--hover-roll-arm-m", type=float, default=None,
                        help="Optional roll rotor moment arm from CG to rotor pair.")
    parser.add_argument("--hover-yaw-arm-m", type=float, default=None,
                        help="Optional effective elevon yaw moment arm in hover.")
    parser.add_argument("--hover-control-margin-min", type=float,
                        default=Assumptions.hover_control_margin_min)
    parser.add_argument("--hover-pitch-arm-fraction-fuselage-max", type=float,
                        default=Assumptions.hover_pitch_arm_fraction_fuselage_max)
    parser.add_argument("--hover-roll-arm-fraction-span-max", type=float,
                        default=Assumptions.hover_roll_arm_fraction_span_max)
    parser.add_argument("--thrust-to-weight-max", type=float,
                        default=Assumptions.thrust_to_weight_max)
    parser.add_argument("--control-closure-max-iter", type=int,
                        default=Assumptions.control_closure_max_iter)
    parser.add_argument("--transition-blend-start-frac", type=float,
                        default=Assumptions.transition_blend_start_frac)
    parser.add_argument("--transition-blend-end-frac", type=float,
                        default=Assumptions.transition_blend_end_frac)
    parser.add_argument("--transition-cruise-margin-frac", type=float,
                        default=Assumptions.transition_cruise_margin_frac)
    parser.add_argument("--transition-accel-m-s2", type=float,
                        default=Assumptions.transition_accel_m_s2)
    parser.add_argument("--dynamic-cm-q", type=float, default=None,
                        help="Optional pitch damping derivative for Phase 14.")
    parser.add_argument("--dynamic-cm-alpha-dot", type=float,
                        default=Assumptions.dynamic_Cm_alpha_dot)
    parser.add_argument("--dynamic-cl-beta", type=float,
                        default=Assumptions.dynamic_Cl_beta)
    parser.add_argument("--dynamic-cl-p", type=float, default=None,
                        help="Optional roll damping derivative for Phase 14.")
    parser.add_argument("--dynamic-cl-r", type=float,
                        default=Assumptions.dynamic_Cl_r)
    parser.add_argument("--dynamic-cn-beta", type=float,
                        default=Assumptions.dynamic_Cn_beta)
    parser.add_argument("--dynamic-cn-p", type=float,
                        default=Assumptions.dynamic_Cn_p)
    parser.add_argument("--dynamic-cn-r", type=float,
                        default=Assumptions.dynamic_Cn_r)

    parser.add_argument("--use-xfoil", action="store_true",
                        help="Try to run XFOIL; falls back to project table values if unavailable.")
    parser.add_argument("--xfoil-path", default=None,
                        help="Path to an XFOIL executable.")
    parser.add_argument("--airfoil-main-file", default=None,
                        help="Optional SD7037 coordinate file for XFOIL.")
    parser.add_argument("--airfoil-canard-file", default=None,
                        help="Optional NACA 0012 coordinate file for XFOIL.")

    parser.add_argument("--max-inner-iter", type=int, default=15)
    parser.add_argument("--outer-max-iter", type=int, default=30)
    parser.add_argument("--mtow-tol", type=float, default=0.01)
    parser.add_argument("--mtow-damping", type=float, default=0.4)
    parser.add_argument("--j-tol", type=float, default=0.01)
    parser.add_argument("--area-tol", type=float, default=0.01)
    parser.add_argument("--re-tol", type=float, default=0.02)
    parser.add_argument("--plot", default=None,
                        help="Optional path for the Phase 6 constraint diagram.")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Optional path to write full JSON results.")
    return parser


def _inputs_from_args(args) -> Tuple[Mission, Assumptions, Dict[str, str]]:
    mission = Mission(
        altitude_m=args.altitude_m,
        range_m=args.range_m,
        time_budget_s=args.time_budget_s,
        hover_time_s=args.hover_time_s,
        mission_equipment_mass_kg=args.mission_equipment_mass_kg,
        external_tow_load_N=args.external_tow_load_n,
    )
    assumptions = Assumptions(
        thrust_to_weight=args.thrust_to_weight,
        disc_loading_target_N_m2=args.disc_loading_target_n_m2,
        prop_diameter_max_m=args.prop_diameter_max_m,
        battery_specific_energy_Wh_kg=args.battery_specific_energy_wh_kg,
        preliminary_wing_area_m2=args.wing_area_seed_m2,
        stall_speed_target_max_m_s=args.stall_speed_target_max_m_s,
        cruise_J_target=args.cruise_j_target,
        canard_volume_coeff=args.canard_volume_coeff,
        canard_volume_grid_min=args.canard_volume_grid_min,
        canard_volume_grid_max=args.canard_volume_grid_max,
        canard_volume_grid_step=args.canard_volume_grid_step,
        cg_envelope_half_width_over_mac=args.cg_envelope_half_width_over_mac,
        cg_required_margin_over_mac=args.cg_required_margin_over_mac,
        wing_mac_le_x_m=args.wing_mac_le_x_m,
        solve_wing_position_for_cg=not args.disable_wing_position_solve,
        roll_rate_required_deg_s=args.roll_rate_required_deg_s,
        elevon_max_deflection_deg=args.elevon_max_deflection_deg,
        elevon_q_slipstream_ratio=args.elevon_q_slipstream_ratio,
        hover_angular_accel_required_rad_s2=args.hover_angular_accel_rad_s2,
        hover_yaw_rate_required_deg_s=args.hover_yaw_rate_required_deg_s,
        hover_yaw_response_time_s=args.hover_yaw_response_time_s,
        hover_pitch_arm_m=args.hover_pitch_arm_m,
        hover_roll_arm_m=args.hover_roll_arm_m,
        hover_yaw_arm_m=args.hover_yaw_arm_m,
        hover_control_margin_min=args.hover_control_margin_min,
        hover_pitch_arm_fraction_fuselage_max=args.hover_pitch_arm_fraction_fuselage_max,
        hover_roll_arm_fraction_span_max=args.hover_roll_arm_fraction_span_max,
        thrust_to_weight_max=args.thrust_to_weight_max,
        control_closure_max_iter=args.control_closure_max_iter,
        transition_blend_start_frac=args.transition_blend_start_frac,
        transition_blend_end_frac=args.transition_blend_end_frac,
        transition_cruise_margin_frac=args.transition_cruise_margin_frac,
        transition_accel_m_s2=args.transition_accel_m_s2,
        dynamic_Cm_q=args.dynamic_cm_q,
        dynamic_Cm_alpha_dot=args.dynamic_cm_alpha_dot,
        dynamic_Cl_beta=args.dynamic_cl_beta,
        dynamic_Cl_p=args.dynamic_cl_p,
        dynamic_Cl_r=args.dynamic_cl_r,
        dynamic_Cn_beta=args.dynamic_cn_beta,
        dynamic_Cn_p=args.dynamic_cn_p,
        dynamic_Cn_r=args.dynamic_cn_r,
        use_xfoil=args.use_xfoil,
        xfoil_path=args.xfoil_path,
    )

    airfoil_files = {}
    if args.airfoil_main_file:
        airfoil_files["main"] = args.airfoil_main_file
    if args.airfoil_canard_file:
        airfoil_files["canard"] = args.airfoil_canard_file
    return mission, assumptions, airfoil_files


def _print_summary(summary: Dict) -> None:
    if "phase16_relative_error" in summary:
        print("Bellona Phase 1-16 propulsion/aerodynamics/canard/control/stability/mass sizing")
    else:
        print("Bellona Phase 1-15 fixed-MTOW propulsion/aerodynamics/canard/control/stability/mass sizing")
    print(f"  Converged: {summary['converged']} ({summary['stopped_reason']})")
    if summary.get("outer_iterations") is None:
        print(f"  Iterations: {summary['iterations']}")
    else:
        print(
            "  Iterations: "
            f"outer={summary['outer_iterations']}, "
            f"inner-final={summary['iterations']}, "
            f"control-closure={summary['control_closure_iterations']}"
        )
    print(f"  Propeller diameter: {summary['propeller_diameter_m']:.3f} m")
    if summary["propeller_diameter_limited"]:
        print(
            "  Disc loading: "
            f"{summary['disc_loading_actual_N_m2']:.1f} N/m^2 "
            f"(target {summary['disc_loading_target_N_m2']:.1f}, diameter capped)"
        )
    else:
        print(
            "  Disc loading: "
            f"{summary['disc_loading_actual_N_m2']:.1f} N/m^2 "
            "(target met; no diameter cap active)"
        )
    print(
        "  Cruise speed: "
        f"{summary['cruise_speed_m_s']:.2f} m/s "
        f"(min {summary['cruise_speed_min_required_m_s']:.2f})"
    )
    print(f"  Cruise RPM / J: {summary['cruise_rpm']:.0f} rpm / {summary['cruise_advance_ratio']:.3f}")
    if summary.get("effective_thrust_to_weight") is not None:
        print(
            "  Effective T/W and arms: "
            f"T/W={summary['effective_thrust_to_weight']:.3f}, "
            f"pitch arm frac={summary['effective_hover_pitch_arm_fraction_fuselage']:.3f}, "
            f"roll arm frac={summary['effective_hover_roll_arm_fraction_span']:.3f}"
        )
    print(f"  Hover power: {summary['hover_power_total_W'] / 1000.0:.2f} kW")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg, {summary['installed_battery_energy_Wh'] / 1000.0:.2f} kWh")
    print(f"  Wing: {summary['wing_area_m2']:.2f} m^2, span {summary['wing_span_m']:.2f} m")
    print(f"  Canard: {summary['canard_area_m2']:.2f} m^2, span {summary['canard_span_m']:.2f} m")
    if summary.get("wing_mac_le_x_m") is not None:
        print(
            "  Wing layout: "
            f"MAC LE x={summary['wing_mac_le_x_m']:.3f} m, "
            f"CG={summary['mass_model_x_CG_over_wing_mac']:.3f} x/c"
        )
    if summary.get("wing_layout_solve_enabled") is not None:
        print(
            "  Wing-CG solve: "
            f"enabled={summary['wing_layout_solve_enabled']}, "
            f"solved={summary['wing_layout_solved']}, "
            f"target={summary['wing_layout_target_x_CG_over_mac']:.3f} x/c"
        )
    if "canard_selected_volume_coeff" in summary:
        print(
            "  Canard search: "
            f"Vbar={summary['canard_selected_volume_coeff']:.3f} "
            f"(default {summary['canard_default_volume_coeff']:.3f})"
        )
    if "neutral_point_x_over_c" in summary:
        print(
            "  CG range: "
            f"{summary['cg_forward_x_over_c']:.3f} to {summary['cg_aft_x_over_c']:.3f} x/c "
            f"({summary['cg_range_pct_mac']:.1f}% MAC, feasible={summary['cg_range_feasible']})"
        )
        if summary.get("operational_cg_feasible") is not None:
            print(
                "  Operational CG: "
                f"{summary['operational_cg_forward_x_over_c']:.3f} to "
                f"{summary['operational_cg_aft_x_over_c']:.3f} x/c, "
                f"feasible={summary['operational_cg_feasible']}"
            )
        print(f"  Neutral point: {summary['neutral_point_x_over_c']:.3f} x/c")
    if "elevon_chord_fraction" in summary:
        print(
            "  Elevon: "
            f"c_e/c={summary['elevon_chord_fraction']:.3f}, "
            f"b_e/b={summary['elevon_span_fraction']:.3f}, "
            f"S_e={summary['elevon_area_total_m2']:.2f} m^2, "
            f"binding={summary['elevon_binding_case']}, feasible={summary['elevon_feasible']}"
        )
        print(
            "  Roll rate: "
            f"{summary['roll_rate_achievable_deg_s']:.1f} deg/s "
            f"(required {summary['roll_rate_required_deg_s']:.1f})"
        )
        print(
            "  Elevon margins: "
            f"pitch={summary['elevon_pitch_margin']:.2f}, "
            f"roll={summary['elevon_roll_margin']:.2f}"
        )
    if "hover_control_feasible" in summary:
        print(
            "  Hover control: "
            f"binding={summary['hover_control_binding_case']}, "
            f"feasible={summary['hover_control_feasible']}, "
            f"margins P/R/Y="
            f"{summary['hover_pitch_margin']:.2f}/"
            f"{summary['hover_roll_margin']:.2f}/"
            f"{summary['hover_yaw_margin']:.2f}"
        )
    if "transition_blend_start_m_s" in summary:
        print(
            "  Transition blend: "
            f"{summary['transition_blend_start_m_s']:.2f}-"
            f"{summary['transition_blend_end_m_s']:.2f} m/s, "
            f"t={summary['transition_time_s']:.1f} s, "
            f"alpha_fw@cruise={summary['transition_alpha_fw_at_cruise']:.2f}"
        )
    if "dynamic_stability_preliminary" in summary:
        print(
            "  Dynamic stability: "
            f"prelim={summary['dynamic_stability_preliminary']}, "
            f"SP zeta/omega={summary['short_period_zeta']:.2f}/"
            f"{summary['short_period_omega_rad_s']:.2f}, "
            f"DR zeta/omega={summary['dutch_roll_zeta']:.2f}/"
            f"{summary['dutch_roll_omega_rad_s']:.2f}"
        )
    if "MTOW_mass_estimate_kg" in summary:
        print(
            "  Mass estimate: "
            f"{summary['MTOW_mass_estimate_kg']:.2f} kg "
            f"(delta {summary['MTOW_mass_delta_kg']:+.2f} kg vs sizing MTOW)"
        )
    if "phase16_relative_error" in summary:
        print(
            "  MTOW loop: "
            f"old={summary['phase16_MTOW_old_kg']:.2f} kg, "
            f"new={summary['phase16_MTOW_new_kg']:.2f} kg, "
            f"rel err={summary['phase16_relative_error']:.4f}"
        )
    if "design_sanity_issue_count" in summary:
        print(
            "  Sanity checks: "
            f"passed={summary['design_sanity_all_passed']}, "
            f"issues={summary['design_sanity_issue_count']}"
        )
    print(f"  Reynolds main/canard: {summary['Re_main']:.0f} / {summary['Re_canard']:.0f}")


def main(argv=None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    mission, assumptions, airfoil_files = _inputs_from_args(args)

    if args.fixed_mtow:
        result = iterate_phases_1_to_15(
            MTOW_kg=args.mtow_kg,
            mission=mission,
            assumptions=assumptions,
            max_inner_iter=args.max_inner_iter,
            j_tol=args.j_tol,
            area_tol=args.area_tol,
            re_tol=args.re_tol,
            constraint_plot_path=args.plot,
            airfoil_files=airfoil_files,
        )
        description = "Bellona Phase 1-15 fixed-MTOW propulsion/aerodynamics/canard/control/transition/stability/mass sizing result."
    else:
        result = converge_design(
            MTOW_seed_kg=args.mtow_kg,
            mission=mission,
            assumptions=assumptions,
            max_iter=args.outer_max_iter,
            tol=args.mtow_tol,
            damping=args.mtow_damping,
            max_inner_iter=args.max_inner_iter,
            j_tol=args.j_tol,
            area_tol=args.area_tol,
            re_tol=args.re_tol,
            constraint_plot_path=args.plot,
            airfoil_files=airfoil_files,
        )
        description = "Bellona Phase 1-16 MTOW-converged propulsion/aerodynamics/canard/control/transition/stability/mass sizing result."
    summary = _build_summary(result)
    payload = {
        "description": description,
        "inputs": {
            "MTOW_seed_kg": args.mtow_kg,
            "fixed_mtow": args.fixed_mtow,
            "outer_max_iter": args.outer_max_iter,
            "mtow_tol": args.mtow_tol,
            "mtow_damping": args.mtow_damping,
            "mission": asdict(mission),
            "assumptions": asdict(assumptions),
            "airfoil_files": airfoil_files,
            "constraint_plot_path": args.plot,
        },
        "summary": summary,
        "result": result,
    }

    _print_summary(summary)
    if args.json_path:
        json_path = Path(args.json_path)
        if json_path.parent != Path("."):
            json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(_json_safe(payload), indent=2),
            encoding="utf-8",
        )
        print(f"  JSON written to: {json_path}")
    if args.plot:
        print(f"  Constraint diagram written to: {args.plot}")
    return 0
