"""Wing-area search around the coupled fixed-MTOW evaluator."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


def _candidate_failure_reasons(result: Dict,
                               hover_control_margin_min: float) -> list[str]:
    """Return binding reasons that make a wing-area candidate infeasible."""
    reasons = []
    if not result.get("converged", False):
        reasons.append("fixed-area mission/aerodynamic loop did not converge")
    if not result.get("phase3", {}).get("carry_forward", {}).get(
            "all_required_cases_feasible", False):
        reasons.append("required mission-distance cases are infeasible")
    if not result.get("phase4", {}).get("feasible", False):
        reasons.append("mission power validation failed")
    cf3 = result.get("phase3", {}).get("carry_forward", {})
    transition_ref = cf3.get("transition_reference", {})
    transition_margin = transition_ref.get("transition_complete_speed_margin_m_s")
    if transition_margin is not None and transition_margin < -1e-9:
        reasons.append("transition-complete speed limit is exceeded")
    stall_eas_margin = transition_ref.get("stall_EAS_margin_m_s")
    if stall_eas_margin is not None and stall_eas_margin < -1e-9:
        reasons.append("stall EAS limit is exceeded")
    power_reserve_margin = cf3.get("minimum_constraint_margins", {}).get(
        "power_margin_over_required_W"
    )
    if power_reserve_margin is not None and power_reserve_margin < -1e-6:
        reasons.append("mission power reserve margin is not met")
    if not result.get("phase10", {}).get("operational_CG_feasible", False):
        reasons.append("operational CG envelope is infeasible")
    if not result.get("phase11", {}).get(
            "feasible_preliminary_elevon", False):
        reasons.append("fixed-wing control is infeasible")
    if not result.get("phase12", {}).get(
            "feasible_preliminary_hover_control", False):
        reasons.append("hover control is infeasible")
    else:
        phase12 = result.get("phase12", {})
        hover_margins = [
            phase12.get("pitch_margin", np.inf),
            phase12.get("roll_margin", np.inf),
            phase12.get("yaw_margin", np.inf),
        ]
        if min(hover_margins) < hover_control_margin_min - 1e-9:
            reasons.append("hover-control margin target is not met")

    phase13 = result.get("phase13", {})
    if phase13:
        cruise_speed = phase13.get("V_cruise")
        required_speed = phase13.get("V_cruise_required_for_margin")
        if (
            cruise_speed is not None
            and required_speed is not None
            and cruise_speed < required_speed - 1e-9
        ):
            reasons.append("transition completion margin is infeasible")
    return reasons


def _candidate_summary(area_m2: float, result: Optional[Dict],
                       error: Optional[str], evaluation_index: int,
                       hover_control_margin_min: float) -> Dict:
    """Create a compact optimization record without embedding a full result."""
    if result is None:
        return {
            "evaluation_index": int(evaluation_index),
            "area_m2": float(area_m2),
            "feasible": False,
            "objective_MTOW_estimate_kg": None,
            "failure_reasons": [str(error or "candidate evaluation failed")],
        }

    reasons = _candidate_failure_reasons(result, hover_control_margin_min)
    phase3 = result.get("phase3", {})
    phase8 = result.get("phase8", {})
    phase13 = result.get("phase13", {})
    phase15 = result.get("phase15", {})
    cf3 = phase3.get("carry_forward", {})
    transition_ref = cf3.get("transition_reference", {})
    margins = cf3.get("minimum_constraint_margins", {})
    return {
        "evaluation_index": int(evaluation_index),
        "area_m2": float(area_m2),
        "feasible": not reasons,
        "objective_MTOW_estimate_kg": (
            None
            if "MTOW_estimate_kg" not in phase15
            else float(phase15["MTOW_estimate_kg"])
        ),
        "mission_energy_Wh": (
            None if "E_total_Wh" not in phase3 else float(phase3["E_total_Wh"])
        ),
        "stall_EAS_m_s": phase8.get("stall_EAS_m_s"),
        "stall_TAS_mission_m_s": phase8.get("stall_TAS_mission_m_s"),
        "stall_TAS_transition_m_s": phase8.get("stall_TAS_transition_m_s"),
        "reference_level_flight_TAS_m_s": phase3.get(
            "reference_level_flight_TAS_m_s"
        ),
        "transition_blend_end_m_s": phase13.get("V_blend_end"),
        "transition_complete_speed_m_s": transition_ref.get(
            "minimum_transition_complete_TAS_m_s"
        ),
        "transition_complete_speed_margin_m_s": transition_ref.get(
            "transition_complete_speed_margin_m_s"
        ),
        "max_transition_complete_speed_m_s": transition_ref.get(
            "max_transition_complete_speed_m_s"
        ),
        "power_margin_over_required_W": margins.get(
            "power_margin_over_required_W"
        ),
        "failure_reasons": reasons,
    }


def optimize_wing_area(
        evaluator: Callable[[float], Dict],
        seed_area_m2: float,
        area_tol: float = 0.01,
        fixed_area_m2: Optional[float] = None,
        initial_points: int = 7,
        lower_factor: float = 0.6,
        upper_factor: float = 1.8,
        expansion_factor: float = 1.5,
        max_expansions: int = 4,
        max_refinements: int = 12,
        hover_control_margin_min: float = 1.05) -> Tuple[Dict, Dict]:
    """Minimize coupled Phase 15 MTOW estimate over feasible wing areas."""
    if seed_area_m2 <= 0.0:
        raise ValueError("seed_area_m2 must be positive.")
    if fixed_area_m2 is not None and fixed_area_m2 <= 0.0:
        raise ValueError("fixed_area_m2 must be positive when provided.")
    if area_tol <= 0.0:
        raise ValueError("area_tol must be positive.")
    if initial_points < 3:
        raise ValueError("initial_points must be at least 3.")
    if not 0.0 < lower_factor < upper_factor:
        raise ValueError("wing-area search factors are invalid.")
    if expansion_factor <= 1.0:
        raise ValueError("expansion_factor must exceed 1.")
    if hover_control_margin_min <= 0.0:
        raise ValueError("hover_control_margin_min must be positive.")

    cache: Dict[float, Dict] = {}
    summaries: Dict[float, Dict] = {}
    evaluation_count = 0

    def _key(area: float) -> float:
        return round(float(area), 10)

    def _evaluate(area: float) -> Optional[Dict]:
        nonlocal evaluation_count
        key = _key(area)
        if key in summaries:
            return cache.get(key)
        evaluation_count += 1
        result = None
        error = None
        try:
            result = evaluator(float(area))
        except ValueError as exc:
            error = str(exc)
        if result is not None:
            cache[key] = result
        summaries[key] = _candidate_summary(
            area, result, error, evaluation_count, hover_control_margin_min
        )
        return result

    if fixed_area_m2 is not None:
        result = _evaluate(fixed_area_m2)
        if result is None:
            reason = summaries[_key(fixed_area_m2)]["failure_reasons"][0]
            raise ValueError(f"Fixed wing-area evaluation failed: {reason}")
        report = {
            "mode": "fixed_area_override",
            "objective": "minimum_converged_MTOW",
            "seed_area_m2": float(seed_area_m2),
            "selected_area_m2": float(fixed_area_m2),
            "selected_objective_MTOW_estimate_kg": float(
                result["phase15"]["MTOW_estimate_kg"]
            ),
            "converged": True,
            "local_minimum_verified": None,
            "area_tolerance": float(area_tol),
            "evaluation_count": int(evaluation_count),
            "candidates": list(summaries.values()),
            "notes": [
                "Wing-area optimization was bypassed by an explicit fixed-area override."
            ],
        }
        return result, report

    initial_areas = np.geomspace(
        lower_factor * seed_area_m2,
        upper_factor * seed_area_m2,
        initial_points,
    )
    for area in initial_areas:
        _evaluate(float(area))

    def _feasible_summaries() -> list[Dict]:
        return [
            summary
            for summary in summaries.values()
            if summary["feasible"]
            and summary["objective_MTOW_estimate_kg"] is not None
        ]

    def _best_summary() -> Optional[Dict]:
        feasible = _feasible_summaries()
        return (
            None
            if not feasible
            else min(
                feasible,
                key=lambda item: item["objective_MTOW_estimate_kg"],
            )
        )

    expansion_history = []
    for expansion in range(max_expansions + 1):
        best = _best_summary()
        areas = sorted(summary["area_m2"] for summary in summaries.values())
        if best is None:
            if expansion >= max_expansions:
                break
            lower = areas[0] / expansion_factor
            upper = areas[-1] * expansion_factor
            _evaluate(lower)
            _evaluate(upper)
            expansion_history.append({
                "expansion": expansion + 1,
                "direction": "both_no_feasible_candidate",
                "new_areas_m2": [float(lower), float(upper)],
            })
            continue

        best_area = best["area_m2"]
        at_lower = np.isclose(best_area, areas[0], rtol=0.0, atol=1e-9)
        at_upper = np.isclose(best_area, areas[-1], rtol=0.0, atol=1e-9)
        if expansion >= max_expansions or not (at_lower or at_upper):
            break
        new_area = (
            areas[0] / expansion_factor
            if at_lower
            else areas[-1] * expansion_factor
        )
        _evaluate(new_area)
        expansion_history.append({
            "expansion": expansion + 1,
            "direction": "lower" if at_lower else "upper",
            "new_areas_m2": [float(new_area)],
        })

    best = _best_summary()
    if best is None:
        failures = sorted(
            summaries.values(), key=lambda item: item["evaluation_index"]
        )
        common = failures[0]["failure_reasons"][0] if failures else "unknown"
        raise ValueError(
            "No feasible wing-area candidate was found. "
            f"First failure: {common}"
        )

    refinement_history = []
    bracket_converged = False
    for refinement in range(max_refinements):
        best = _best_summary()
        best_area = best["area_m2"]
        areas = sorted(summary["area_m2"] for summary in summaries.values())
        lower_neighbors = [area for area in areas if area < best_area]
        upper_neighbors = [area for area in areas if area > best_area]
        if not lower_neighbors or not upper_neighbors:
            break
        lower = lower_neighbors[-1]
        upper = upper_neighbors[0]
        relative_width = (upper - lower) / best_area
        if relative_width <= area_tol:
            bracket_converged = True
            break
        new_areas = [
            float(np.sqrt(lower * best_area)),
            float(np.sqrt(best_area * upper)),
        ]
        for area in new_areas:
            _evaluate(area)
        refinement_history.append({
            "refinement": refinement + 1,
            "bracket_m2": [float(lower), float(upper)],
            "relative_width": float(relative_width),
            "new_areas_m2": new_areas,
        })

    for _ in range(3):
        best = _best_summary()
        best_area = best["area_m2"]
        neighbor_areas = [
            best_area * (1.0 - area_tol),
            best_area * (1.0 + area_tol),
        ]
        for area in neighbor_areas:
            _evaluate(area)
        updated_best = _best_summary()
        if np.isclose(
            updated_best["area_m2"], best_area, rtol=0.0, atol=1e-9
        ):
            break

    best = _best_summary()
    best_area = best["area_m2"]
    for area in (
        best_area * (1.0 - area_tol),
        best_area * (1.0 + area_tol),
    ):
        _evaluate(area)
    best = _best_summary()
    best_area = best["area_m2"]
    for area in (
        best_area * (1.0 - area_tol),
        best_area * (1.0 + area_tol),
    ):
        _evaluate(area)
    best_objective = best["objective_MTOW_estimate_kg"]
    neighbor_summaries = [
        summaries[_key(best_area * (1.0 - area_tol))],
        summaries[_key(best_area * (1.0 + area_tol))],
    ]
    local_minimum_verified = all(
        not neighbor["feasible"]
        or neighbor["objective_MTOW_estimate_kg"] >= best_objective - 1e-9
        for neighbor in neighbor_summaries
    )
    selected_result = cache[_key(best_area)]
    report = {
        "mode": "optimized",
        "objective": "minimum_converged_MTOW",
        "seed_area_m2": float(seed_area_m2),
        "selected_area_m2": float(best_area),
        "selected_objective_MTOW_estimate_kg": float(best_objective),
        "converged": bool(bracket_converged and local_minimum_verified),
        "bracket_converged": bool(bracket_converged),
        "local_minimum_verified": bool(local_minimum_verified),
        "area_tolerance": float(area_tol),
        "initial_search_factors": [float(lower_factor), float(upper_factor)],
        "initial_points": int(initial_points),
        "expansion_factor": float(expansion_factor),
        "max_expansions": int(max_expansions),
        "expansion_history": expansion_history,
        "refinement_history": refinement_history,
        "verification_neighbors": neighbor_summaries,
        "evaluation_count": int(evaluation_count),
        "candidates": sorted(
            summaries.values(), key=lambda item: item["area_m2"]
        ),
        "notes": [
            "Feasible candidates must pass mission, transition-speed, power-reserve, operational CG, fixed-wing control, and hover-control checks.",
            "Preliminary dynamic-stability checks are reported but do not reject wing-area candidates.",
        ],
    }
    return selected_result, report
