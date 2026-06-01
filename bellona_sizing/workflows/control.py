"""Control and transition coupling helpers: operational-CG elevons, Phase 15 inertias, transition energy feedback, and hover-control closure."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from ..common import isa
from ..models import Assumptions, Mission
from ..phases.phase05_energy import phase5_energy_battery
from ..phases.phase11_elevon import phase11_elevon_FW
from ..phases.phase12_hover_control import phase12_hover_control
from ..phases.phase13_transition import phase13_transition_blending
from ..phases.phase14_dynamic_stability import phase14_dynamic_stability


def _phase5_with_transition_energy(result: Dict, phase13: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Recompute Phase 5 using Phase 13 transition energy when available."""
    if phase13.get("P_transition_average_W") is not None:
        transition_segment = (
            phase13["P_transition_average_W"],
            phase13["t_transition"],
        )
        source = "phase13_transition_blending"
    else:
        transition_segment = (
            result["phase3"]["P_transition"],
            result["phase3"]["t_transition"],
        )
        source = "phase3_mission_optimise_fallback"

    phase5 = phase5_energy_battery(
        {
            "transition": transition_segment,
            "fixed_wing_climb": (result["phase3"]["P_fw"], result["phase3"]["t_fw"]),
            "mission_hover": (result["phase3"]["P_hover_target"], mission.hover_time_s),
        },
        assumptions.eta_batt,
        assumptions.usable_battery_fraction,
        assumptions.battery_specific_energy_Wh_kg,
    )
    phase5["transition_energy_source"] = source
    phase5["phase3_transition_energy_Wh"] = result["phase3"]["E_transition_Wh"]
    phase5["phase13_transition_energy_Wh"] = phase13.get("E_transition_estimate_Wh")
    return phase5


def _run_phase11_with_operational_cg(result: Dict, phase10: Dict,
                                     assumptions: Assumptions) -> Dict:
    """Run Phase 11 with operational CG travel instead of full theoretical range."""
    phase8 = result["phase8"]
    return phase11_elevon_FW(
        phase8["S"],
        phase8["b"],
        phase8["c_bar"],
        phase8["CL_a"],
        result["phase3"]["V_cruise"],
        phase8["V_stall"],
        phase10.get("operational_CG_range_m", phase10["CG_range_m"]),
        q_slipstream_ratio=assumptions.elevon_q_slipstream_ratio,
        p_required_rad_s=np.deg2rad(assumptions.roll_rate_required_deg_s),
        Cm_de_required=None,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
        tau_e=assumptions.elevon_tau,
        eta_e=assumptions.elevon_eta,
        l_e_over_c=assumptions.elevon_l_e_over_c,
        CL_trim=phase10["CL_trim"],
        pitch_trim_margin=assumptions.elevon_pitch_trim_margin,
        control_margin_min=1.05,
        chord_fraction_bounds=(
            assumptions.elevon_chord_fraction_min,
            assumptions.elevon_chord_fraction_max,
        ),
        span_fraction_bounds=(
            assumptions.elevon_span_fraction_min,
            assumptions.elevon_span_fraction_max,
        ),
        grid_points=assumptions.elevon_grid_points,
    )


def _run_phase12_with_mass_inertia(result: Dict, phase11: Dict, phase15: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Run Phase 12 using Phase 15 inertias."""
    phase1 = result["phase1"]
    phase8 = result["phase8"]
    rho_target, _, _, _ = isa(mission.altitude_m)
    d_x_rotor = (
        assumptions.hover_pitch_arm_m
        if assumptions.hover_pitch_arm_m is not None
        else assumptions.hover_pitch_arm_fraction_fuselage * assumptions.fuselage_length_m
    )
    d_y_rotor = (
        assumptions.hover_roll_arm_m
        if assumptions.hover_roll_arm_m is not None
        else assumptions.hover_roll_arm_fraction_span * phase8["b"]
    )
    l_z_elev = (
        assumptions.hover_yaw_arm_m
        if assumptions.hover_yaw_arm_m is not None
        else assumptions.hover_yaw_arm_fraction_chord * phase8["c_bar"]
    )
    CL_de = (
        assumptions.hover_control_CL_de
        if assumptions.hover_control_CL_de is not None
        else phase8["CL_a"] * assumptions.elevon_tau
    )
    phase12 = phase12_hover_control(
        result["MTOW_N"] / assumptions.n_rotors,
        phase1["A_disc"],
        rho_target,
        d_x_rotor,
        d_y_rotor,
        phase15["I_tensor_kg_m2"]["Ixx"],
        phase15["I_tensor_kg_m2"]["Iyy"],
        phase15["I_tensor_kg_m2"]["Izz"],
        phase11["S_e_total"],
        phase11["b_e_over_b"] * phase8["b"],
        l_z_elev,
        omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_rate_required=np.deg2rad(assumptions.hover_yaw_rate_required_deg_s),
        T_available_per_rotor=phase1["T_per_rotor"],
        roll_omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_response_time_s=assumptions.hover_yaw_response_time_s,
        CL_de=CL_de,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
    )
    phase12["inertia_source"] = {
        "I_xx": "phase15_mass_model",
        "I_yy": "phase15_mass_model",
        "I_zz": "phase15_mass_model",
    }
    phase12["arm_source"] = {
        "d_x_rotor": "assumption" if assumptions.hover_pitch_arm_m is not None else "fuselage_fraction_estimate",
        "d_y_rotor": "assumption" if assumptions.hover_roll_arm_m is not None else "span_fraction_estimate",
        "l_z_elev": "assumption" if assumptions.hover_yaw_arm_m is not None else "chord_fraction_estimate",
    }
    return phase12


def _run_phase13_from_result(result: Dict, assumptions: Assumptions) -> Dict:
    """Run Phase 13 transition blending."""
    return phase13_transition_blending(
        result["phase8"]["V_stall"],
        assumptions.transition_blend_start_frac,
        assumptions.transition_blend_end_frac,
        V_entry=0.0,
        V_exit=None,
        transition_accel_m_s2=assumptions.transition_accel_m_s2,
        V_cruise=result["phase3"]["V_cruise"],
        cruise_speed_margin_frac=assumptions.transition_cruise_margin_frac,
        P_hover_W=result["phase3"]["P_hover_target"],
        P_fw_W=result["phase3"]["P_fw"],
        sample_count=assumptions.transition_sample_count,
    )


def _run_phase14_with_mass_inertia(result: Dict, phase10: Dict, phase15: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Run Phase 14 using Phase 15 inertias."""
    rho_cruise, _, _, _ = isa(mission.altitude_m)
    phase8 = result["phase8"]
    phase9 = result["phase9"]
    CL_alpha_total = phase10["a_w_eff"] + phase10["a_c_eff"] * phase10["area_ratio"]
    Cm_alpha = -phase10["SM_min"] * CL_alpha_total
    if assumptions.dynamic_Cm_q is None:
        Cm_q = -2.0 * phase9["CL_a_c"] * phase10["V_bar_c"] * (
            phase9["l_c"] / phase8["c_bar"]
        )
    else:
        Cm_q = assumptions.dynamic_Cm_q
    phase14 = phase14_dynamic_stability(
        Cm_alpha,
        Cm_q,
        assumptions.dynamic_Cm_alpha_dot,
        CL_alpha_total,
        phase15["I_tensor_kg_m2"]["Iyy"],
        phase15["I_tensor_kg_m2"]["Ixx"],
        phase15["I_tensor_kg_m2"]["Izz"],
        phase8["S"],
        phase8["c_bar"],
        phase8["b"],
        result["phase3"]["V_cruise"],
        rho_cruise,
        CL_trim=phase10["CL_trim"],
        CD_trim=result["phase3"]["CD_cruise"],
        C_l_beta=assumptions.dynamic_Cl_beta,
        C_l_p=assumptions.dynamic_Cl_p or result["phase11"].get("C_l_p"),
        C_l_r=assumptions.dynamic_Cl_r,
        C_n_beta=assumptions.dynamic_Cn_beta,
        C_n_p=assumptions.dynamic_Cn_p,
        C_n_r=assumptions.dynamic_Cn_r,
        g=assumptions.g,
        short_period_zeta_min=assumptions.short_period_zeta_min,
        short_period_zeta_max=assumptions.short_period_zeta_max,
        short_period_omega_min_rad_s=assumptions.short_period_omega_min_rad_s,
        phugoid_zeta_min=assumptions.phugoid_zeta_min,
        dutch_roll_zeta_min=assumptions.dutch_roll_zeta_min,
        dutch_roll_omega_min_rad_s=assumptions.dutch_roll_omega_min_rad_s,
        spiral_time_to_double_min_s=assumptions.spiral_time_to_double_min_s,
    )
    phase14["derivative_sources"] = {
        "Cm_alpha": "static_margin_times_total_lift_curve_slope",
        "Cm_q": "assumption" if assumptions.dynamic_Cm_q is not None else "canard_volume_estimate",
        "inertias": "phase15_mass_model",
    }
    return phase14


def _hover_control_closure_update(result: Dict, assumptions: Assumptions) -> Tuple[Assumptions, Dict]:
    """Compute geometry-first hover-control updates for the next closure pass."""
    phase12 = result["phase12"]
    phase8 = result["phase8"]
    margin_min = assumptions.hover_control_margin_min
    update_margin = margin_min + 0.005
    updates = {}
    messages = []

    if phase12["pitch_margin"] < margin_min:
        required_fraction = (
            phase12["pitch_arm_required_m"] * update_margin / assumptions.fuselage_length_m
        )
        new_fraction = min(
            assumptions.hover_pitch_arm_fraction_fuselage_max,
            max(assumptions.hover_pitch_arm_fraction_fuselage, required_fraction),
        )
        if new_fraction > assumptions.hover_pitch_arm_fraction_fuselage + 1e-9:
            updates["hover_pitch_arm_fraction_fuselage"] = float(new_fraction)
            messages.append(f"pitch arm fraction -> {new_fraction:.3f}")

    if phase12["roll_margin"] < margin_min:
        required_fraction = phase12["roll_arm_required_m"] * update_margin / phase8["b"]
        new_fraction = min(
            assumptions.hover_roll_arm_fraction_span_max,
            max(assumptions.hover_roll_arm_fraction_span, required_fraction),
        )
        if new_fraction > assumptions.hover_roll_arm_fraction_span + 1e-9:
            updates["hover_roll_arm_fraction_span"] = float(new_fraction)
            messages.append(f"roll arm fraction -> {new_fraction:.3f}")

    pitch_at_cap = (
        assumptions.hover_pitch_arm_fraction_fuselage
        >= assumptions.hover_pitch_arm_fraction_fuselage_max - 1e-9
    )
    roll_at_cap = (
        assumptions.hover_roll_arm_fraction_span
        >= assumptions.hover_roll_arm_fraction_span_max - 1e-9
    )
    if (
        (phase12["pitch_margin"] < margin_min and pitch_at_cap)
        or (phase12["roll_margin"] < margin_min and roll_at_cap)
    ):
        T_hover = phase12["T_hover_per_rotor"]
        T_W_pitch = 1.0 + update_margin * phase12["delta_T_pitch_required"] / T_hover
        T_W_roll = 1.0 + update_margin * phase12["delta_T_roll_required"] / T_hover
        T_W_required = max(T_W_pitch, T_W_roll, assumptions.thrust_to_weight)
        new_T_W = min(assumptions.thrust_to_weight_max, T_W_required)
        if new_T_W > assumptions.thrust_to_weight + 1e-9:
            updates["thrust_to_weight"] = float(new_T_W)
            messages.append(f"T/W -> {new_T_W:.3f}")

    updated = replace(assumptions, **updates) if updates else assumptions
    return updated, {
        "updated": bool(updates),
        "updates": updates,
        "messages": messages,
        "pitch_margin": phase12["pitch_margin"],
        "roll_margin": phase12["roll_margin"],
        "yaw_margin": phase12["yaw_margin"],
        "target_margin": margin_min,
        "thrust_to_weight": assumptions.thrust_to_weight,
        "hover_pitch_arm_fraction_fuselage": assumptions.hover_pitch_arm_fraction_fuselage,
        "hover_roll_arm_fraction_span": assumptions.hover_roll_arm_fraction_span,
    }
