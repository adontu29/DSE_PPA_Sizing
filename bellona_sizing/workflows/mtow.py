"""Outer MTOW convergence loop that wraps the coupled fixed-MTOW pass."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional

from ..models import Assumptions, Mission
from ..phases.phase16_mtow import phase16_mtow_converge
from .coupled import iterate_phases_1_to_15
from .sanity import _design_sanity_checks


def converge_design(MTOW_seed_kg: float = 50.0,
                    mission: Optional[Mission] = None,
                    assumptions: Optional[Assumptions] = None,
                    max_iter: int = 30,
                    tol: float = 0.01,
                    damping: float = 0.4,
                    max_inner_iter: int = 15,
                    area_tol: float = 0.01,
                    re_tol: float = 0.02,
                    constraint_plot_path=None,
                    airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Outer MTOW loop orchestrating all 16 phases.
    Sequence (dependency-graph order, NOT constraint-diagram-first):
      0. ISA tables
      1. Phase 1  prop diameter
      2. Phase 2  hover/climb power
      3. Wing-area search evaluates coupled fixed-area candidates.
      4. Phase 3  mission optimise  (V_cruise, gamma, h_tr DERIVED here)
      5. Phase 4  preliminary mission-power validation
      6. Phase 7  airfoil  inner Re loop seeded from Ph3 V_cruise
      7. Phase 8  wing  stall speed calculated from selected wing area
      8. Phase 7' Re-loop refine
      9. Phase 9  canard
     10. Phase 10 scissor (canard form)  inner CGcanard loop with Ph9
     11. Phase 11 FW elevon
     12. Phase 12 hover elevon  MAX-of with Ph11 closes elevon spec
     13. Phase 13 transition blending
     14. Phase 5  energy / battery
     15. Phase 15 mass  produces new MTOW
     16. Phase 16 converge check (damping 0.4) plus wing-area convergence
     17. Phase 6  constraint diagram VERIFY (final)
     18. Phase 14 dynamic stability VERIFY (final)
    """
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()
    if MTOW_seed_kg <= 0.0:
        raise ValueError("MTOW_seed_kg must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in the range 0 < damping <= 1.")

    MTOW_current = float(MTOW_seed_kg)
    outer_history = []
    final_result = None
    outer_converged = False
    wing_area_converged = False
    outer_failure_reason = None
    previous_wing_area_m2 = None

    for outer_it in range(max_iter):
        try:
            result = iterate_phases_1_to_15(
                MTOW_kg=MTOW_current,
                mission=mission,
                assumptions=assumptions,
                max_inner_iter=max_inner_iter,
                area_tol=area_tol,
                re_tol=re_tol,
                constraint_plot_path=constraint_plot_path,
                airfoil_files=airfoil_files,
            )
        except ValueError as exc:
            if final_result is None:
                raise
            outer_failure_reason = str(exc)
            outer_history.append({
                "outer_iteration": outer_it + 1,
                "MTOW_input_kg": float(MTOW_current),
                "MTOW_mass_estimate_kg": None,
                "MTOW_next_kg": None,
                "mass_delta_kg": None,
                "relative_error": None,
                "inner_converged": False,
                "inner_iterations": 0,
                "control_closure_iterations": 0,
                "stopped_reason": outer_failure_reason,
            })
            break
        MTOW_estimate = result["phase15"]["MTOW_estimate_kg"]
        phase16 = phase16_mtow_converge(
            MTOW_current,
            MTOW_estimate,
            damping=damping,
            tol=tol,
        )
        result = dict(result)
        result["phase16"] = phase16
        final_result = result
        selected_wing_area_m2 = float(result["phase8"]["S"])
        wing_area_relative_change = (
            None
            if previous_wing_area_m2 is None
            else abs(selected_wing_area_m2 - previous_wing_area_m2)
            / previous_wing_area_m2
        )
        wing_area_converged = (
            assumptions.wing_area_m2 is not None
            or (
                wing_area_relative_change is not None
                and wing_area_relative_change <= area_tol
                and result.get("wing_area_optimization", {}).get(
                    "converged", False
                )
            )
        )

        outer_history.append({
            "outer_iteration": outer_it + 1,
            "MTOW_input_kg": float(MTOW_current),
            "MTOW_mass_estimate_kg": float(MTOW_estimate),
            "MTOW_next_kg": phase16["MTOW_next_kg"],
            "mass_delta_kg": phase16["delta_kg"],
            "relative_error": phase16["relative_error"],
            "selected_wing_area_m2": selected_wing_area_m2,
            "wing_area_relative_change": wing_area_relative_change,
            "wing_area_converged": bool(wing_area_converged),
            "wing_area_candidate_count": result.get(
                "wing_area_optimization", {}
            ).get("evaluation_count"),
            "inner_converged": bool(result.get("converged", False)),
            "inner_iterations": len(result.get("history", [])),
            "control_closure_iterations": len(result.get("control_mass_history", [])),
            "effective_thrust_to_weight": result.get("effective_assumptions", {}).get("thrust_to_weight"),
            "effective_hover_pitch_arm_fraction_fuselage": result.get("effective_assumptions", {}).get("hover_pitch_arm_fraction_fuselage"),
            "effective_hover_roll_arm_fraction_span": result.get("effective_assumptions", {}).get("hover_roll_arm_fraction_span"),
            "wing_mac_le_x_m": result.get("phase15", {}).get("wing_mac_le_x_m"),
            "operational_cg_feasible": result.get("phase10", {}).get("operational_CG_feasible"),
            "sanity_issue_count": result.get("sanity_checks", {}).get("issue_count"),
            "stopped_reason": result.get("stopped_reason", ""),
        })

        if phase16["converged"] and wing_area_converged:
            outer_converged = True
            break
        previous_wing_area_m2 = selected_wing_area_m2
        MTOW_current = phase16["MTOW_next_kg"]

    final_result = dict(final_result)
    inner_converged = bool(final_result.get("converged", False))
    final_result["inner_converged"] = inner_converged
    final_result["outer_converged"] = bool(outer_converged)
    final_result["wing_area_converged"] = bool(wing_area_converged)
    final_result["outer_failure_reason"] = outer_failure_reason
    final_result["converged"] = bool(inner_converged and outer_converged)
    final_result["outer_history"] = outer_history
    final_result["outer_iterations"] = len(outer_history)
    final_result["MTOW_seed_kg"] = float(MTOW_seed_kg)
    final_result["MTOW_converged_kg"] = (
        final_result["phase16"]["MTOW_new_kg"]
        if outer_converged
        else final_result["phase16"]["MTOW_next_kg"]
    )
    final_result["notes"] = list(final_result.get("notes", [])) + [
        "Phase 16 closes the MTOW loop by feeding the Phase 15 mass estimate back into the fixed-MTOW sizing pass.",
        "MTOW convergence also requires the selected wing area to stop changing between outer iterations.",
    ]
    final_result["warnings"] = list(final_result.get("warnings", []))
    if not outer_converged:
        final_result["warnings"].append(
            "Phase 16 did not meet the requested MTOW and wing-area convergence tolerances before max_iter."
        )
    if outer_failure_reason is not None:
        final_result["warnings"].append(
            f"Phase 16 stopped when the next MTOW iterate became infeasible: {outer_failure_reason}"
        )

    effective_assumptions = Assumptions(
        **final_result.get("effective_assumptions", asdict(assumptions))
    )
    final_result["sanity_checks"] = _design_sanity_checks(
        final_result,
        mission,
        effective_assumptions,
    )
    if not final_result["sanity_checks"]["all_passed"]:
        existing = set(final_result["warnings"])
        for issue in final_result["sanity_checks"]["issues"]:
            warning = f"sanity: {issue}"
            if warning not in existing:
                final_result["warnings"].append(warning)

    if outer_failure_reason is not None:
        final_result["stopped_reason"] = (
            "Phase 16 MTOW iteration became mission-infeasible before convergence: "
            + outer_failure_reason
        )
    elif inner_converged and outer_converged:
        open_issues = []
        if final_result.get("phase11", {}) and not final_result["phase11"].get("feasible_preliminary_elevon", True):
            open_issues.append("Phase 11 fixed-wing elevon")
        if final_result.get("phase12", {}) and not final_result["phase12"].get("feasible_preliminary_hover_control", True):
            open_issues.append("Phase 12 hover control")
        if final_result.get("phase14", {}) and not final_result["phase14"].get("level_meets_8785C_preliminary", True):
            open_issues.append("Phase 14 dynamic stability")
        if final_result.get("sanity_checks", {}).get("issue_count", 0):
            open_issues.append("design sanity checks")

        if open_issues:
            issue_verb = (
                "remain"
                if len(open_issues) > 1 or open_issues[0].endswith("checks")
                else "remains"
            )
            final_result["stopped_reason"] = (
                "MTOW loop converged, but "
                + ", ".join(open_issues)
                + f" {issue_verb} infeasible/preliminary-failed"
            )
        else:
            final_result["stopped_reason"] = "full Phase 1-16 MTOW loop converged"
    elif inner_converged:
        final_result["stopped_reason"] = "Phase 16 MTOW loop did not converge before max_iter"
    else:
        final_result["stopped_reason"] = (
            "Phase 1-15 inner sizing did not converge during the Phase 16 MTOW loop"
        )
    return final_result
