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
        add(
            "cruise_speed_transition_margin",
            phase3["V_cruise"] >= phase13["V_cruise_required_for_margin"],
            "Cruise speed must clear transition blend end plus margin.",
            phase3["V_cruise"],
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
        phase3_E = phase3["E_transition_Wh"]
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
    add(
        "stall_speed_limit",
        phase8["V_stall"] <= assumptions.stall_speed_target_max_m_s,
        "Stall speed must not exceed the selected target.",
        phase8["V_stall"],
        assumptions.stall_speed_target_max_m_s,
    )

    passed_values = [item["passed"] for item in checks.values() if item["passed"] is not None]
    return {
        "all_passed": bool(all(passed_values)) if passed_values else True,
        "checks": checks,
        "issues": issues,
        "issue_count": len(issues),
    }
