"""Short coupled fixed-MTOW orchestrator that assembles the layout, control, transition, mass, and sanity-check helper workflows."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from ..models import Assumptions, Mission
from ..phases.phase06_constraints import phase6_constraint_diagram
from .control import (
    _hover_control_closure_update,
    _phase5_with_transition_energy,
    _run_phase11_with_operational_cg,
    _run_phase12_with_mass_inertia,
    _run_phase13_from_result,
    _run_phase14_with_mass_inertia,
)
from .layout import _canard_cg_grid_search, _phase15_phase10_with_wing_layout
from .phase1_to9 import iterate_phases_1_to_9
from .sanity import _design_sanity_checks
from .wing_area import optimize_wing_area


def _write_selected_constraint_plot(result: Dict, assumptions: Assumptions,
                                    plot_path) -> Dict:
    """Regenerate Phase 6 only for the selected wing-area candidate."""
    cf3 = result["phase3"]["carry_forward"]
    hover_power_loading = (
        result["phase2_hover"]["P_elec"]
        * assumptions.n_rotors
        / result["MTOW_N"]
    )
    return phase6_constraint_diagram(
        np.asarray(result["phase6"]["W_S_range"], dtype=float),
        np.array([]),
        cf3["reference_level_flight"]["TAS_m_s"],
        cf3["minimum_required_average_ROC_m_s"],
        cf3["gamma_reference_rad"],
        cf3["reference_level_flight"]["density_kg_m3"],
        assumptions.CD0,
        assumptions.AR,
        assumptions.oswald_e,
        result["phase3"]["eta_fw"],
        assumptions.thrust_to_weight,
        selected_W_S=result["phase8"]["W_S_design"],
        hover_power_loading_W_N=hover_power_loading,
        plot_path=plot_path,
    )


def _run_coupled_fixed_mtow_once(MTOW_kg: float,
                                 mission: Mission,
                                 assumptions: Assumptions,
                                 max_inner_iter: int,
                                 area_tol: float,
                                 re_tol: float,
                                 constraint_plot_path=None,
                                 airfoil_files: Optional[Dict[str, str]] = None,
                                 wing_area_m2: Optional[float] = None) -> Dict:
    """Run one coupled fixed-MTOW pass with mass/CG/control feedback."""
    # First close the preliminary mission/aerodynamics loop at the fixed MTOW.
    result = iterate_phases_1_to_9(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
        wing_area_m2=wing_area_m2,
    )

    # Use the preliminary battery mass to place the CG, choose a canard, and
    # solve the wing station against the Phase 10 scissor limits.
    phase5_preliminary = result["phase5"]
    phase9, phase15_preliminary, phase10, canard_search = _canard_cg_grid_search(
        result,
        phase5_preliminary,
        mission,
        assumptions,
    )
    result = dict(result)
    result["phase9"] = phase9
    result["phase10"] = phase10
    result["phase15_preliminary"] = phase15_preliminary
    result["phase5_preliminary"] = phase5_preliminary
    result["canard_cg_search"] = canard_search
    result["wing_layout_solver"] = canard_search.get("selected_wing_layout")
    result["Re_canard"] = canard_search["selected_Re_canard"]

    # Initial control checks use the preliminary mass/inertia. Phase 13 then
    # replaces the coarse Phase 3 transition energy before the final mass pass.
    phase11 = _run_phase11_with_operational_cg(result, phase10, assumptions)
    result["phase11"] = phase11
    phase12_preliminary = _run_phase12_with_mass_inertia(
        result,
        phase11,
        phase15_preliminary,
        assumptions,
        mission,
    )
    result["phase12_preliminary"] = phase12_preliminary

    phase13 = _run_phase13_from_result(result, assumptions)
    result["phase13"] = phase13
    phase5 = _phase5_with_transition_energy(result, phase13, assumptions, mission)
    result["phase5"] = phase5

    # Re-run mass, CG, scissor, controls, and dynamics after transition energy
    # has fed back into the battery mass estimate.
    phase15, phase10, wing_layout = _phase15_phase10_with_wing_layout(
        result,
        phase5,
        phase9,
        mission,
        assumptions,
    )
    result["phase15"] = phase15
    result["phase10"] = phase10
    result["wing_layout_solver"] = wing_layout
    phase11 = _run_phase11_with_operational_cg(result, phase10, assumptions)
    result["phase11"] = phase11
    phase12 = _run_phase12_with_mass_inertia(
        result,
        phase11,
        phase15,
        assumptions,
        mission,
    )
    result["phase12"] = phase12
    phase14 = _run_phase14_with_mass_inertia(result, phase10, phase15, assumptions, mission)
    result["phase14"] = phase14

    warnings = []
    for phase_name in ("phase9", "phase10", "phase11", "phase12", "phase13", "phase14", "phase15"):
        phase = result.get(phase_name, {})
        for warning in phase.get("warnings", []):
            warnings.append(f"{phase_name}: {warning}")
    for warning in canard_search.get("warnings", []):
        warnings.append(f"canard_cg_search: {warning}")
    for warning in result.get("wing_layout_solver", {}).get("warnings", []):
        warnings.append(f"wing_layout_solver: {warning}")

    result["warnings"] = list(result.get("warnings", [])) + warnings
    result["notes"] = list(result.get("notes", [])) + [
        "The coupled fixed-MTOW pass uses Phase 15 CG/inertia before Phases 10-12 and 14.",
        "Phase 5 is recomputed with Phase 13 transition energy before the final Phase 15 mass estimate.",
    ]
    result["sanity_checks"] = _design_sanity_checks(result, mission, assumptions)
    result["effective_assumptions"] = asdict(assumptions)

    if result.get("converged"):
        open_issues = []
        if not phase10.get("operational_CG_feasible", False):
            open_issues.append("operational CG")
        if not phase11.get("feasible_preliminary_elevon", False):
            open_issues.append("Phase 11 fixed-wing elevon")
        if not phase12.get("feasible_preliminary_hover_control", False):
            open_issues.append("Phase 12 hover control")
        if not phase14.get("level_meets_8785C_preliminary", False):
            open_issues.append("Phase 14 dynamic stability")
        if open_issues:
            result["stopped_reason"] = (
                "coupled fixed-MTOW pass completed, but "
                + ", ".join(open_issues)
                + " remains infeasible/preliminary-failed"
            )
        else:
            result["stopped_reason"] = "coupled fixed-MTOW Phase 1-15 pass completed"
    return result


def iterate_phases_1_to_15(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run the coupled fixed-MTOW sizing pass with hover-control closure."""
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()
    if assumptions.control_closure_max_iter <= 0:
        raise ValueError("control_closure_max_iter must be positive.")

    effective = assumptions
    control_mass_history = []
    final_result = None
    for closure_it in range(assumptions.control_closure_max_iter):
        # The hover closure changes geometry/TW assumptions, then repeats the
        # fixed-MTOW pass until pitch/roll/yaw margins stop asking for changes.
        result, wing_area_optimization = optimize_wing_area(
            lambda area_m2: _run_coupled_fixed_mtow_once(
                MTOW_kg,
                mission,
                effective,
                max_inner_iter,
                area_tol,
                re_tol,
                constraint_plot_path=None,
                airfoil_files=airfoil_files,
                wing_area_m2=area_m2,
            ),
            effective.preliminary_wing_area_m2,
            area_tol=area_tol,
            fixed_area_m2=effective.wing_area_m2,
            hover_control_margin_min=effective.hover_control_margin_min,
        )
        result = dict(result)
        result["wing_area_optimization"] = wing_area_optimization
        updated, closure_record = _hover_control_closure_update(result, effective)
        closure_record["closure_iteration"] = closure_it + 1
        closure_record["MTOW_kg"] = float(MTOW_kg)
        closure_record["phase15_MTOW_estimate_kg"] = result["phase15"]["MTOW_estimate_kg"]
        closure_record["selected_wing_area_m2"] = result["phase8"]["S"]
        closure_record["wing_area_optimization_converged"] = (
            wing_area_optimization["converged"]
        )
        control_mass_history.append(closure_record)
        final_result = result
        if not closure_record["updated"]:
            break
        effective = updated

    final_result = dict(final_result)
    if constraint_plot_path:
        final_result["phase6"] = _write_selected_constraint_plot(
            final_result, effective, constraint_plot_path
        )
    final_result["control_mass_history"] = control_mass_history
    final_result["effective_assumptions"] = asdict(effective)
    final_result["wing_area_converged"] = bool(
        final_result.get("wing_area_optimization", {}).get("converged", False)
    )
    final_result["converged"] = bool(
        final_result.get("converged", False)
        and final_result["wing_area_converged"]
    )
    final_result["sanity_checks"] = _design_sanity_checks(final_result, mission, effective)
    if not final_result["sanity_checks"]["all_passed"]:
        final_result["warnings"] = list(final_result.get("warnings", [])) + [
            f"sanity: {issue}" for issue in final_result["sanity_checks"]["issues"]
        ]
    if (
        final_result.get("wing_area_optimization", {}).get("mode")
        == "optimized"
        and not final_result["wing_area_converged"]
    ):
        final_result["stopped_reason"] = (
            "coupled fixed-MTOW pass completed, but wing-area optimization "
            "did not converge"
        )
    return final_result
