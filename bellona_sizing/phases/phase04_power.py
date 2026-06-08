"""Phase 4: validate the preliminary mission electrical-power allowance."""
from __future__ import annotations

from typing import Dict


def phase4_mission_power_validation(phase3_result: Dict) -> Dict:
    """Summarize the Phase 3 power-cap check without component-level claims.

    Reads from the carry_forward.power_sizing_case block so the check
    reflects the globally highest peak power across all sweep distances,
    not just the energy-critical profile.
    """
    cf = phase3_result.get("carry_forward", {})
    psc = cf.get("power_sizing_case", {})

    # Prefer carry_forward; fall back to flat keys for backward compat.
    peak_power_W = float(
        psc.get("peak_electrical_power_W",
                phase3_result.get("peak_electrical_power_W", 0.0))
    )
    affordable_power_W = float(
        psc.get("max_affordable_electrical_power_W",
                phase3_result.get("max_affordable_electrical_power_W", 0.0))
    )
    min_margin_W = float(
        cf.get("minimum_constraint_margins", {}).get(
            "power_margin_W",
            phase3_result.get("power_margin_W", affordable_power_W - peak_power_W),
        )
    )
    required_margin_W = float(
        psc.get(
            "required_power_margin_W",
            cf.get("minimum_constraint_margins", {}).get(
                "required_power_margin_W",
                phase3_result.get("required_power_margin_W", 0.0),
            ),
        )
    )

    power_d = psc.get("target_distance_m")
    margin_W = affordable_power_W - peak_power_W
    reserve_margin_W = min_margin_W - required_margin_W

    return {
        "feasible": bool(reserve_margin_W >= -1e-6),
        "peak_electrical_power_W":          peak_power_W,
        "maximum_wingborne_electrical_power_W": float(
            psc.get("maximum_wingborne_electrical_power_W", peak_power_W)
        ),
        "power_critical_target_distance_m": power_d,
        "power_critical_state":             psc.get("critical_state", {}),
        "max_affordable_electrical_power_W": affordable_power_W,
        "power_margin_W":                   float(margin_W),
        "minimum_power_margin_W":           float(min_margin_W),
        "required_power_margin_W":          float(required_margin_W),
        "power_margin_over_required_W":     float(reserve_margin_W),
        "power_margin_fraction":            float(margin_W / affordable_power_W)
                                            if affordable_power_W > 0.0 else 0.0,
        "notes": [
            "This phase checks only the preliminary total-aircraft electrical-power allowance and selected reserve margin.",
            "peak_electrical_power_W is from the power-critical sweep case (worst-case distance).",
            "Motor, ESC, battery, and propeller selection require a later component-level study.",
        ],
        "warnings": (
            []
            if reserve_margin_W >= 0.0
            else ["The optimized mission does not meet the required electrical-power reserve margin."]
        ),
    }
