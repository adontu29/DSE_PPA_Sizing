"""Design sanity checks that flag infeasible or unphysical sizing outputs."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..models import Assumptions, Mission


def _design_sanity_checks(result: Dict, mission: Mission,
                          assumptions: Assumptions) -> Dict:
    """Collect design sanity checks that flag infeasible or unphysical outputs."""
    checks = {}
    issues = []

    def add(name: str, passed, message: str, value=None, limit=None, severity: str = "warning"):
        checks[name] = {
            "passed": None if passed is None else bool(passed),
            "message": message,
            "value": value,
            "limit": limit,
            "severity": severity,
        }
        if passed is False:
            issues.append(f"{name}: {message}")

    phase1 = result["phase1"]
    phase3 = result["phase3"]
    phase5 = result["phase5"]
    phase7 = result["phase7"]
    phase8 = result["phase8"]
    phase10 = result.get("phase10", {})
    phase11 = result.get("phase11", {})
    phase12 = result.get("phase12", {})
    phase13 = result.get("phase13", {})
    phase15 = result.get("phase15", {})
    phase16 = result.get("phase16", {})
    canard_search = result.get("canard_cg_search", {})
    sanity_checks = result.get("sanity_checks", {})
    effective_assumptions = result.get("effective_assumptions", {})
    wing_layout = result.get("wing_layout_solver", phase15.get("wing_layout_solver", {}))

    add(
        "mtow_convergence",
        None if not phase16 else phase16["converged"],
        "Phase 16 MTOW relative error must be below tolerance.",
        None if not phase16 else phase16["relative_error"],
        None if not phase16 else phase16["tol"],
    )
    if phase10:
        add(
            "operational_cg_inside_theoretical",
            phase10.get("operational_CG_feasible"),
            "Operational CG envelope must fit inside Phase 10 limits with margin.",
            min(
                phase10.get("operational_fwd_margin_over_mac", np.nan),
                phase10.get("operational_aft_margin_over_mac", np.nan),
            ),
            assumptions.cg_required_margin_over_mac,
        )
        add(
            "mass_cg_inside_theoretical",
            phase10.get("mass_CG_inside_theoretical"),
            "Mass-model CG must lie inside the theoretical scissor range.",
            phase10.get("x_cg_mass_over_mac"),
            [phase10.get("x_cg_fwd_over_c"), phase10.get("x_cg_aft_over_c")],
        )
        add(
            "static_margin_min",
            phase10.get("SM_min", 0.0) >= assumptions.static_margin_min,
            "Static margin must meet the selected minimum.",
            phase10.get("SM_min"),
            assumptions.static_margin_min,
        )
    if phase11:
        add(
            "phase11_pitch_margin",
            phase11["pitch_margin"] >= 1.05,
            "Fixed-wing pitch-control margin must be at least 1.05.",
            phase11["pitch_margin"],
            1.05,
        )
        add(
            "phase11_roll_margin",
            phase11["roll_margin"] >= 1.05,
            "Fixed-wing roll-control margin must be at least 1.05.",
            phase11["roll_margin"],
            1.05,
        )
        add(
            "phase11_elevon_chord_not_at_cap",
            phase11["c_e_over_c"] < assumptions.elevon_chord_fraction_max - 1e-9,
            "Selected elevon chord should not sit on the maximum bound.",
            phase11["c_e_over_c"],
            assumptions.elevon_chord_fraction_max,
        )
        add(
            "phase11_elevon_span_not_at_cap",
            phase11["b_e_over_b"] < assumptions.elevon_span_fraction_max - 1e-9,
            "Selected elevon span should not sit on the maximum bound.",
            phase11["b_e_over_b"],
            assumptions.elevon_span_fraction_max,
        )
    if phase12:
        add(
            "phase12_pitch_margin",
            phase12["pitch_margin"] >= assumptions.hover_control_margin_min,
            "Hover pitch margin must meet the selected margin target.",
            phase12["pitch_margin"],
            assumptions.hover_control_margin_min,
        )
        add(
            "phase12_roll_margin",
            phase12["roll_margin"] >= assumptions.hover_control_margin_min,
            "Hover roll margin must meet the selected margin target.",
            phase12["roll_margin"],
            assumptions.hover_control_margin_min,
        )
        add(
            "phase12_yaw_margin",
            phase12["yaw_margin"] >= assumptions.hover_control_margin_min,
            "Hover yaw margin must meet the selected margin target.",
            phase12["yaw_margin"],
            assumptions.hover_control_margin_min,
        )
        add(
            "phase12_required_thrust_to_weight",
            max(
                phase12["thrust_to_weight_required_pitch"],
                phase12["thrust_to_weight_required_roll"],
            ) <= assumptions.thrust_to_weight_max,
            "Required hover-control T/W must remain below the selected cap.",
            max(
                phase12["thrust_to_weight_required_pitch"],
                phase12["thrust_to_weight_required_roll"],
            ),
            assumptions.thrust_to_weight_max,
        )
        add(
            "phase12_required_pitch_arm",
            phase12["pitch_arm_required_m"] <= assumptions.hover_pitch_arm_fraction_fuselage_max * assumptions.fuselage_length_m,
            "Required pitch arm must fit inside the selected geometry cap.",
            phase12["pitch_arm_required_m"],
            assumptions.hover_pitch_arm_fraction_fuselage_max * assumptions.fuselage_length_m,
        )
        add(
            "phase12_required_roll_arm",
            phase12["roll_arm_required_m"] <= assumptions.hover_roll_arm_fraction_span_max * phase8["b"],
            "Required roll arm must fit inside the selected geometry cap.",
            phase12["roll_arm_required_m"],
            assumptions.hover_roll_arm_fraction_span_max * phase8["b"],
        )
        add(
            "prop_disk_lateral_overlap",
            phase1["D_prop"] <= 2.0 * phase12["d_y_rotor"],
            "Propeller diameter must not exceed lateral rotor-center spacing.",
            phase1["D_prop"],
            2.0 * phase12["d_y_rotor"],
        )
    add(
        "disc_loading_range",
        100.0 <= phase1["disc_loading"] <= 250.0,
        "Disc loading should remain inside the preliminary 100-250 N/m^2 range.",
        phase1["disc_loading"],
        [100.0, 250.0],
    )
    add(
        "thrust_to_weight_range",
        assumptions.thrust_to_weight_min <= assumptions.thrust_to_weight <= assumptions.thrust_to_weight_max,
        "Installed T/W should stay inside the selected preliminary range.",
        assumptions.thrust_to_weight,
        [assumptions.thrust_to_weight_min, assumptions.thrust_to_weight_max],
    )
    if phase13:
        cf3 = phase3.get("carry_forward", {})
        ref_TAS = cf3.get("reference_level_flight", {}).get(
            "TAS_m_s", phase3.get("V_cruise", 0.0)
        )
        add(
            "cruise_speed_transition_margin",
            ref_TAS >= phase13["V_cruise_required_for_margin"],
            "Reference level-flight speed must clear transition blend end plus margin.",
            ref_TAS,
            phase13["V_cruise_required_for_margin"],
        )
    if phase15:
        battery_fraction = phase5["m_batt_kg"] / phase15["MTOW_estimate_kg"]
        if wing_layout:
            add(
                "wing_layout_operational_cg_final",
                wing_layout.get("operational_CG_feasible_final"),
                "Wing station solve should place the operational CG envelope inside Phase 10 limits.",
                min(
                    wing_layout.get("final_operational_fwd_margin_over_mac", np.nan),
                    wing_layout.get("final_operational_aft_margin_over_mac", np.nan),
                ),
                assumptions.cg_required_margin_over_mac,
            )
            add(
                "wing_layout_station_reported",
                None,
                "Wing MAC leading-edge station is reported relative to the provisional fuselage/equipment datum.",
                phase15.get("wing_mac_le_x_m"),
                "replace with CAD datum",
            )
        add(
            "battery_mass_fraction",
            battery_fraction <= 0.35,
            "Battery mass fraction should not exceed 35%.",
            battery_fraction,
            0.35,
        )
        add(
            "mission_equipment_included",
            abs(phase15["m_mission_equipment_kg"] - mission.mission_equipment_mass_kg) < 1e-9,
            "mission_equipment_mass_kg must be included in UAV MTOW.",
            phase15["m_mission_equipment_kg"],
            mission.mission_equipment_mass_kg,
        )
        add(
            "external_tow_load_excluded",
            phase15["external_tow_load_included_in_MTOW"] is False,
            "External tow load must remain outside UAV MTOW.",
            phase15["external_tow_load_included_in_MTOW"],
            False,
        )
    if phase13 and phase13.get("E_transition_estimate_Wh") is not None:
        # Compare against the energy-critical case transition energy.
        cf3 = phase3.get("carry_forward", {})
        phase3_E = float(
            cf3.get("energy_sizing_case", {}).get(
                "segment_summaries", {}
            ).get("transition", {}).get("energy_Wh",
                  phase3.get("E_transition_Wh", 0.0))
        )
        phase13_E = phase13["E_transition_estimate_Wh"]
        diff = abs(phase13_E - phase3_E) / max(phase3_E, 1e-9)
        add(
            "transition_energy_consistency",
            diff <= 0.20,
            "Phase 13 transition energy should be within 20% of the Phase 3 estimate.",
            diff,
            0.20,
        )
    canard = phase7["canard"]
    re_ratio = result["Re_canard"] / canard.get("Re_reference", result["Re_canard"])
    add(
        "canard_reynolds_fallback_range",
        0.75 <= re_ratio <= 1.25,
        "Canard Reynolds number should remain within 25% of the fallback table reference.",
        re_ratio,
        [0.75, 1.25],
    )
    airfoil_sources = [phase7["main"]["source"], phase7["canard"]["source"]]
    add(
        "airfoil_verified_not_fallback",
        all(source == "xfoil" for source in airfoil_sources),
        "Final reporting should use XFOIL or measured polar data instead of fallback values.",
        airfoil_sources,
        "xfoil",
    )
    cf3 = phase3.get("carry_forward", {})
    transition_ref = cf3.get("transition_reference", {})
    transition_margin = transition_ref.get("transition_complete_speed_margin_m_s")
    add(
        "transition_complete_speed_limit",
        None if transition_margin is None else transition_margin >= -1e-9,
        "Transition-complete speed must not exceed the selected tailsitter handoff limit.",
        transition_ref.get("minimum_transition_complete_TAS_m_s"),
        transition_ref.get("max_transition_complete_speed_m_s"),
    )
    stall_eas_margin = transition_ref.get("stall_EAS_margin_m_s")
    add(
        "stall_eas_limit",
        None if stall_eas_margin is None else stall_eas_margin >= -1e-9,
        "Calculated stall EAS must not exceed the optional selected limit.",
        transition_ref.get("stall_EAS_m_s"),
        transition_ref.get("max_stall_EAS_m_s"),
    )
    power_margin_over_required = cf3.get("minimum_constraint_margins", {}).get(
        "power_margin_over_required_W"
    )
    add(
        "mission_power_reserve_margin",
        None if power_margin_over_required is None else power_margin_over_required >= -1e-6,
        "Mission power margin must exceed the selected reserve fraction.",
        power_margin_over_required,
        0.0,
    )
    max_climb_angle = max(
        (
            row.get("optimized_climb_angle_deg", -np.inf)
            for row in phase3.get("diagnostics", {}).get("distance_sweep", [])
            if row.get("feasible")
        ),
        default=phase3.get("optimized_climb_angle_deg"),
    )
    add(
        "fixed_wing_climb_angle_limit",
        None if max_climb_angle is None else max_climb_angle <= assumptions.max_fixed_wing_climb_angle_deg + 1e-9,
        "Fixed-wing climb angle must remain inside the selected model-validity bound.",
        max_climb_angle,
        assumptions.max_fixed_wing_climb_angle_deg,
    )
    mission_cl_margin = cf3.get("minimum_constraint_margins", {}).get(
        "CL_margin"
    )
    add(
        "mission_cl_allowed_compliance",
        None if mission_cl_margin is None else mission_cl_margin >= -1e-9,
        "Every required mission state must remain at or below CL_allowed.",
        mission_cl_margin,
        0.0,
    )
    ref_lf = cf3.get("reference_level_flight", {})
    CL_max = phase8.get("CL_max_3D")
    cruise_CL = ref_lf.get("CL")
    cruise_stall_margin_deg = None
    if CL_max is not None and cruise_CL is not None and phase8.get("CL_a", 0.0) > 0.0:
        cruise_stall_margin_deg = float(
            np.rad2deg((CL_max - cruise_CL) / phase8["CL_a"])
        )
    add(
        "cruise_stall_margin_estimate",
        None if cruise_stall_margin_deg is None else cruise_stall_margin_deg >= assumptions.cruise_stall_margin_deg,
        "Reference cruise condition should retain the selected Stone-style stall-angle margin estimate.",
        cruise_stall_margin_deg,
        assumptions.cruise_stall_margin_deg,
    )
    canard_stall_margin_deg = None
    if (
        phase7.get("main", {}).get("alpha_stall_deg") is not None
        and phase7.get("canard", {}).get("alpha_stall_deg") is not None
    ):
        canard_stall_margin_deg = float(
            phase7["main"]["alpha_stall_deg"]
            - phase7["canard"]["alpha_stall_deg"]
        )
    add(
        "canard_stalls_before_wing_estimate",
        None if canard_stall_margin_deg is None else canard_stall_margin_deg >= assumptions.canard_stall_before_wing_margin_deg,
        "Canard stall angle should precede wing stall angle by the selected Stone-style margin.",
        canard_stall_margin_deg,
        assumptions.canard_stall_before_wing_margin_deg,
    )
    wing_area_optimization = result.get("wing_area_optimization", {})
    if wing_area_optimization:
        add(
            "wing_area_optimization_convergence",
            wing_area_optimization.get("converged"),
            "Wing-area optimization must converge or be explicitly fixed.",
            wing_area_optimization.get("selected_area_m2"),
            wing_area_optimization.get("area_tolerance"),
        )
        add(
            "wing_area_local_minimum_verified",
            wing_area_optimization.get("local_minimum_verified"),
            "Neighboring wing areas must be heavier or infeasible.",
            wing_area_optimization.get("verification_neighbors"),
            "selected local minimum",
        )

    passed_values = [item["passed"] for item in checks.values() if item["passed"] is not None]
    return {
        "all_passed": bool(all(passed_values)) if passed_values else True,
        "checks": checks,
        "issues": issues,
        "issue_count": len(issues),
    }
