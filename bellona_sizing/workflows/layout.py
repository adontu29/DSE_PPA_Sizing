"""Layout and CG coupling helpers: Phase 15 mass locations, Phase 10 scissor checks, wing station solve, and canard grid search."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..common import isa
from ..models import Assumptions, Mission
from ..phases.phase09_canard import phase9_canard
from ..phases.phase10_scissor import phase10_scissor_canard
from ..phases.phase15_mass import phase15_mass


def _estimate_phase15_from_result(result: Dict, phase5: Dict, phase9: Dict,
                                  mission: Mission, assumptions: Assumptions,
                                  wing_mac_le_x_m: Optional[float] = None) -> Dict:
    """Run Phase 15 from an existing sizing result and a selected canard."""
    wing_x = (
        0.0
        if wing_mac_le_x_m is None
        else float(wing_mac_le_x_m)
    )
    return phase15_mass(
        result["phase8"]["S"],
        phase9["S_c"],
        assumptions.fuselage_length_m,
        phase5["P_motor_cont"],
        phase5["m_batt_kg"],
        mission.mission_equipment_mass_kg,
        n_rotors=assumptions.n_rotors,
        prop_diameter_m=result["phase1"]["D_prop"],
        b_w=result["phase8"]["b"],
        c_bar_w=result["phase8"]["c_bar"],
        x_ac_w=result["phase8"]["x_ac_w"],
        x_ac_c=phase9["x_ac_c"],
        wing_mac_le_x_m=wing_x,
        external_tow_load_N=mission.external_tow_load_N,
        g=assumptions.g,
        wing_areal_density_kg_m2=assumptions.wing_areal_density_kg_m2,
        canard_areal_density_kg_m2=assumptions.canard_areal_density_kg_m2,
        fuselage_linear_density_kg_m=assumptions.fuselage_linear_density_kg_m,
        boom_landing_gear_mass_kg=assumptions.boom_landing_gear_mass_kg,
        motor_specific_mass_kg_W=assumptions.motor_specific_mass_kg_W,
        esc_specific_mass_kg_W=assumptions.esc_specific_mass_kg_W,
        prop_mass_coeff_kg_m2=assumptions.prop_mass_coeff_kg_m2,
        avionics_mass_kg=assumptions.avionics_mass_kg,
        wiring_fraction=assumptions.wiring_fraction,
        mass_contingency_fraction=assumptions.mass_contingency_fraction,
        Ixx_radius_fraction_span=assumptions.hover_Ixx_radius_fraction_span,
        Iyy_radius_fraction_fuselage=assumptions.hover_Iyy_radius_fraction_fuselage,
        Izz_radius_fraction_span=assumptions.hover_Izz_radius_fraction_span,
    )


def _operational_cg_envelope(phase15: Dict, phase8: Dict,
                             assumptions: Assumptions) -> Dict:
    """Build a simple operational CG envelope around the Phase 15 mass CG."""
    c_bar = phase8["c_bar"]
    half_width_m = assumptions.cg_envelope_half_width_over_mac * c_bar
    margin_m = assumptions.cg_required_margin_over_mac * c_bar
    x_cg = phase15["x_CG_m"]
    x_fwd = x_cg - half_width_m
    x_aft = x_cg + half_width_m
    return {
        "x_cg_mass_m": float(x_cg),
        "x_cg_mass_over_mac": float(x_cg / c_bar),
        "x_cg_fwd_operational_m": float(x_fwd),
        "x_cg_aft_operational_m": float(x_aft),
        "x_cg_fwd_operational_over_mac": float(x_fwd / c_bar),
        "x_cg_aft_operational_over_mac": float(x_aft / c_bar),
        "cg_envelope_half_width_m": float(half_width_m),
        "cg_envelope_half_width_over_mac": float(assumptions.cg_envelope_half_width_over_mac),
        "cg_required_margin_m": float(margin_m),
        "cg_required_margin_over_mac": float(assumptions.cg_required_margin_over_mac),
        "operational_CG_range_m": float(2.0 * half_width_m),
        "operational_CG_range_over_mac": float(2.0 * assumptions.cg_envelope_half_width_over_mac),
    }


def _attach_operational_cg_to_phase10(phase10: Dict, cg_envelope: Dict,
                                      phase8: Dict) -> Dict:
    """Add operational CG-envelope checks to a Phase 10 scissor result."""
    c_bar = phase8["c_bar"]
    margin_m = cg_envelope["cg_required_margin_m"]
    half_width_m = cg_envelope["cg_envelope_half_width_m"]

    fwd_margin_m = cg_envelope["x_cg_fwd_operational_m"] - phase10["x_cg_fwd"]
    aft_margin_m = phase10["x_cg_aft"] - cg_envelope["x_cg_aft_operational_m"]
    mass_fwd_margin_m = cg_envelope["x_cg_mass_m"] - phase10["x_cg_fwd"]
    mass_aft_margin_m = phase10["x_cg_aft"] - cg_envelope["x_cg_mass_m"]
    feasible = fwd_margin_m >= margin_m and aft_margin_m >= margin_m
    mass_inside = mass_fwd_margin_m >= 0.0 and mass_aft_margin_m >= 0.0

    center_min = phase10["x_cg_fwd"] + margin_m + half_width_m
    center_max = phase10["x_cg_aft"] - margin_m - half_width_m
    if center_min <= center_max:
        target_center = float(np.clip(cg_envelope["x_cg_mass_m"], center_min, center_max))
        required_shift_m = target_center - cg_envelope["x_cg_mass_m"]
        envelope_width_excess_m = 0.0
    else:
        required_shift_m = np.nan
        envelope_width_excess_m = center_min - center_max

    result = dict(phase10)
    result.update(cg_envelope)
    result.update({
        "operational_CG_feasible": bool(feasible),
        "mass_CG_inside_theoretical": bool(mass_inside),
        "operational_fwd_margin_m": float(fwd_margin_m),
        "operational_aft_margin_m": float(aft_margin_m),
        "operational_fwd_margin_over_mac": float(fwd_margin_m / c_bar),
        "operational_aft_margin_over_mac": float(aft_margin_m / c_bar),
        "mass_CG_fwd_margin_m": float(mass_fwd_margin_m),
        "mass_CG_aft_margin_m": float(mass_aft_margin_m),
        "cg_center_min_with_margin_m": float(center_min),
        "cg_center_max_with_margin_m": float(center_max),
        "required_CG_shift_m": None if not np.isfinite(required_shift_m) else float(required_shift_m),
        "required_CG_shift_over_mac": (
            None if not np.isfinite(required_shift_m) else float(required_shift_m / c_bar)
        ),
        "cg_envelope_width_excess_m": float(envelope_width_excess_m),
        "cg_envelope_width_excess_over_mac": float(envelope_width_excess_m / c_bar),
    })
    result["warnings"] = list(result.get("warnings", []))
    if not feasible:
        result["warnings"].append(
            "Operational CG envelope does not fit inside the theoretical Phase 10 CG limits with the required margin."
        )
    if not mass_inside:
        result["warnings"].append(
            "Mass-model CG lies outside the theoretical Phase 10 CG range."
        )
    return result


def _run_phase10_for_phase9(result: Dict, phase9: Dict, phase15: Dict,
                            assumptions: Assumptions) -> Dict:
    """Run Phase 10 and attach operational CG-envelope data."""
    phase8 = result["phase8"]
    phase10 = phase10_scissor_canard(
        phase8["S"],
        phase8["c_bar"],
        phase8["x_ac_w"],
        phase8["CL_a"],
        phase9["S_c"],
        phase9["l_c"],
        phase9["CL_a_c"],
        assumptions.canard_eps_alpha_c,
        assumptions.wing_eps_alpha_w,
        phase9["CL_max_3D_c"],
        result["phase3"]["CL_cruise"],
        assumptions.static_margin_min,
    )
    return _attach_operational_cg_to_phase10(
        phase10,
        _operational_cg_envelope(phase15, phase8, assumptions),
        phase8,
    )


def _target_cg_center_from_phase10(phase10: Dict, phase8: Dict,
                                   assumptions: Assumptions) -> Dict:
    """Find the wing-local CG target that centers the operational envelope in Phase 10."""
    c_bar = phase8["c_bar"]
    half_width_m = assumptions.cg_envelope_half_width_over_mac * c_bar
    margin_m = assumptions.cg_required_margin_over_mac * c_bar
    center_min = phase10["x_cg_fwd"] + margin_m + half_width_m
    center_max = phase10["x_cg_aft"] - margin_m - half_width_m
    feasible_width = center_min <= center_max
    if feasible_width:
        target_center = 0.5 * (center_min + center_max)
    else:
        target_center = 0.5 * (phase10["x_cg_fwd"] + phase10["x_cg_aft"])

    return {
        "target_x_cg_m": float(target_center),
        "target_x_cg_over_mac": float(target_center / c_bar),
        "target_center_min_m": float(center_min),
        "target_center_max_m": float(center_max),
        "target_center_min_over_mac": float(center_min / c_bar),
        "target_center_max_over_mac": float(center_max / c_bar),
        "target_width_feasible": bool(feasible_width),
        "required_theoretical_cg_range_over_mac": float(
            2.0 * assumptions.cg_envelope_half_width_over_mac
            + 2.0 * assumptions.cg_required_margin_over_mac
        ),
        "theoretical_cg_range_over_mac": float(phase10["CG_range_m"] / c_bar),
    }


def _phase15_phase10_with_wing_layout(result: Dict, phase5: Dict, phase9: Dict,
                                      mission: Mission,
                                      assumptions: Assumptions
                                      ) -> Tuple[Dict, Dict, Dict]:
    """Solve the wing MAC station needed to place mass CG inside the scissor range."""
    initial_wing_x = (
        0.0
        if assumptions.wing_mac_le_x_m is None
        else float(assumptions.wing_mac_le_x_m)
    )
    phase15_initial = _estimate_phase15_from_result(
        result,
        phase5,
        phase9,
        mission,
        assumptions,
        wing_mac_le_x_m=initial_wing_x,
    )
    phase10_initial = _run_phase10_for_phase9(
        result,
        phase9,
        phase15_initial,
        assumptions,
    )
    target = _target_cg_center_from_phase10(
        phase10_initial,
        result["phase8"],
        assumptions,
    )

    phase15 = phase15_initial
    phase10 = phase10_initial
    solved_wing_x = initial_wing_x
    solved = False
    warnings = []

    lift_mass = phase15_initial["m_wing_kg"] + phase15_initial["m_canard_kg"]
    total_mass = phase15_initial["MTOW_estimate_kg"]
    # Moving the wing station moves the wing and canard absolute locations, but
    # the nonlifting equipment datum is held fixed until CAD replaces it.
    slope = lift_mass / total_mass - 1.0
    required_wing_shift_m = None

    if assumptions.solve_wing_position_for_cg:
        if not target["target_width_feasible"]:
            warnings.append(
                "The Phase 10 scissor range is too narrow for the requested operational CG envelope and margin; wing station alone cannot make it feasible."
            )
        elif abs(slope) < 1e-9:
            warnings.append(
                "Wing station solve is singular because nearly all mass moves with the lifting group."
            )
        else:
            # Linear one-step solve:
            #   x_CG_wing_new = x_CG_wing_old + slope * dx_wing
            required_wing_shift_m = (
                target["target_x_cg_m"] - phase15_initial["x_CG_m"]
            ) / slope
            solved_wing_x = initial_wing_x + required_wing_shift_m
            if np.isfinite(solved_wing_x):
                phase15 = _estimate_phase15_from_result(
                    result,
                    phase5,
                    phase9,
                    mission,
                    assumptions,
                    wing_mac_le_x_m=solved_wing_x,
                )
                phase10 = _run_phase10_for_phase9(
                    result,
                    phase9,
                    phase15,
                    assumptions,
                )
                solved = True
            else:
                solved_wing_x = initial_wing_x
                warnings.append(
                    "Wing station solve produced a non-finite location; keeping the initial wing station."
                )

    c_bar = result["phase8"]["c_bar"]
    layout_report = {
        "enabled": bool(assumptions.solve_wing_position_for_cg),
        "solved": bool(solved),
        "initial_wing_mac_le_x_m": float(initial_wing_x),
        "solved_wing_mac_le_x_m": float(solved_wing_x),
        "required_wing_shift_m": (
            None if required_wing_shift_m is None else float(required_wing_shift_m)
        ),
        "initial_x_CG_over_mac": float(phase15_initial["x_CG_over_wing_mac"]),
        "final_x_CG_over_mac": float(phase15["x_CG_over_wing_mac"]),
        "initial_x_CG_fuselage_m": float(phase15_initial["x_CG_fuselage_m"]),
        "final_x_CG_fuselage_m": float(phase15["x_CG_fuselage_m"]),
        "target_x_CG_over_mac": target["target_x_cg_over_mac"],
        "target_x_CG_m": target["target_x_cg_m"],
        "target_center_min_over_mac": target["target_center_min_over_mac"],
        "target_center_max_over_mac": target["target_center_max_over_mac"],
        "target_width_feasible": target["target_width_feasible"],
        "cg_change_per_wing_shift": float(slope),
        "lifting_group_mass_fraction": float(lift_mass / total_mass),
        "operational_CG_feasible_initial": phase10_initial["operational_CG_feasible"],
        "operational_CG_feasible_final": phase10["operational_CG_feasible"],
        "final_operational_fwd_margin_over_mac": phase10["operational_fwd_margin_over_mac"],
        "final_operational_aft_margin_over_mac": phase10["operational_aft_margin_over_mac"],
        "wing_station_over_mac": float(solved_wing_x / c_bar),
        "notes": [
            "This solve shifts the wing MAC leading edge relative to a provisional fuselage/equipment mass datum.",
            "The canard arm ratio is held fixed, so the first-cut solve moves the lifting group as a block.",
            "CAD component locations should replace the provisional mass datum before final acceptance.",
        ],
        "warnings": warnings,
    }

    phase15 = dict(phase15)
    phase15["wing_layout_solver"] = layout_report
    phase10 = dict(phase10)
    phase10["wing_layout_solver"] = layout_report
    return phase15, phase10, layout_report


def _canard_cg_grid_search(result: Dict, phase5: Dict, mission: Mission,
                           assumptions: Assumptions) -> Tuple[Dict, Dict, Dict, Dict]:
    """Select the smallest canard volume that fits the operational CG envelope."""
    phase8 = result["phase8"]
    phase7 = result["phase7"]
    rho_target, mu_target, a_sound_target, _ = isa(mission.altitude_m)
    mach_cruise = result["phase3"]["V_cruise"] / a_sound_target
    l_c = assumptions.canard_arm_chord_ratio * phase8["c_bar"]
    default_phase9 = result["phase9"]
    default_phase15, default_phase10, default_layout = _phase15_phase10_with_wing_layout(
        result,
        phase5,
        default_phase9,
        mission,
        assumptions,
    )

    start = assumptions.canard_volume_grid_min
    stop = assumptions.canard_volume_grid_max
    step = assumptions.canard_volume_grid_step
    if step <= 0.0 or stop < start:
        raise ValueError("Canard volume grid bounds are invalid.")

    candidates = []
    selected = None
    for V_bar_c in np.arange(start, stop + 0.5 * step, step):
        phase9_candidate = phase9_canard(
            phase8["S"],
            phase8["c_bar"],
            phase8["x_ac_w"],
            l_c,
            float(V_bar_c),
            assumptions.canard_AR,
            assumptions.canard_taper,
            assumptions.canard_sweep_c4_rad,
            phase7["canard"]["cl_max"],
            mach_cruise,
        )
        phase15_candidate, phase10_candidate, layout_candidate = _phase15_phase10_with_wing_layout(
            result,
            phase5,
            phase9_candidate,
            mission,
            assumptions,
        )
        min_margin = min(
            phase10_candidate["operational_fwd_margin_over_mac"],
            phase10_candidate["operational_aft_margin_over_mac"],
        )
        candidate_record = {
            "V_bar_c": float(V_bar_c),
            "S_c": phase9_candidate["S_c"],
            "MTOW_estimate_kg": phase15_candidate["MTOW_estimate_kg"],
            "wing_mac_le_x_m": phase15_candidate["wing_mac_le_x_m"],
            "x_CG_over_mac": phase15_candidate["x_CG_over_wing_mac"],
            "target_x_CG_over_mac": layout_candidate["target_x_CG_over_mac"],
            "operational_CG_feasible": phase10_candidate["operational_CG_feasible"],
            "min_operational_margin_over_mac": float(min_margin),
            "required_CG_shift_over_mac": phase10_candidate["required_CG_shift_over_mac"],
            "cg_envelope_width_excess_over_mac": phase10_candidate["cg_envelope_width_excess_over_mac"],
            "wing_layout_solved": layout_candidate["solved"],
        }
        candidates.append(candidate_record)
        if selected is None and phase10_candidate["operational_CG_feasible"]:
            selected = (phase9_candidate, phase15_candidate, phase10_candidate, layout_candidate)

    warnings = []
    if selected is None:
        selected = (default_phase9, default_phase15, default_phase10, default_layout)
        warnings.append(
            "No canard-volume candidate fit the operational CG envelope; keeping the current canard and reporting required CG shift."
        )

    selected_phase9, selected_phase15, selected_phase10, selected_layout = selected
    Re_canard = (
        rho_target
        * result["phase3"]["V_cruise"]
        * selected_phase9["c_bar_c"]
        / mu_target
    )
    search = {
        "selected_V_bar_c": selected_phase9["V_bar_c"],
        "selected_S_c": selected_phase9["S_c"],
        "selected_Re_canard": float(Re_canard),
        "selected_wing_layout": selected_layout,
        "default_V_bar_c": default_phase9["V_bar_c"],
        "default_operational_CG_feasible": default_phase10["operational_CG_feasible"],
        "default_wing_layout": default_layout,
        "wing_position_solve_enabled": bool(assumptions.solve_wing_position_for_cg),
        "grid_min": float(start),
        "grid_max": float(stop),
        "grid_step": float(step),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warnings": warnings,
        "notes": [
            "The canard arm ratio is held fixed during this grid search.",
            "The first feasible candidate is selected so canard area is kept as small as possible.",
            "The selected canard Reynolds number is reported; final XFOIL or measured-polar checks should use that selected value.",
        ],
    }
    return selected_phase9, selected_phase15, selected_phase10, search
