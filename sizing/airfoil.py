"""Optional XFOIL Reynolds-feedback refinement of the section coefficients.

XFOIL runs in an OUTER loop around the whole wing-area sweep: each pass runs one
full sweep at the current section coefficients, then a single XFOIL airfoil-pair
run at the best design's representative condition refreshes those coefficients
(lift slope, CL0, Cm0, DATCOM eta, and the finite-wing CL_max). The loop repeats
until XFOIL stops changing them, so XFOIL is launched only a handful of times per
sizing run. With use_xfoil_airfoil_updates off, the baseline airfoil assumptions
in the inputs are used unchanged.
"""

from __future__ import annotations

import math

from scissor_plot import datcom_lift_slope, finite_wing_clmax, leading_edge_sweep_deg
from xfoil_wrapper import (
    analyze_airfoil_pair,
    datcom_efficiency_from_section_slope,
    mach_number as xfoil_mach_number,
    reynolds_number,
)

from sizing.inputs import AIRCRAFT, MISSION
from sizing.atmosphere import isa_density
from sizing.loop import sweep_wing_area
from sizing.report import progress


def representative_xfoil_condition(mission):
    """Mission state used for Reynolds-dependent 2D airfoil analysis."""
    candidates = []
    for state in mission.get("states", []):
        if state.get("segment") != "wing_borne_climb":
            continue
        speed = state.get("speed_m_s")
        if speed is None or speed <= 0.0:
            continue
        altitude = state.get("altitude_mid_m", state.get("altitude_m", MISSION["altitude_m"]))
        unit_re = reynolds_number(altitude, speed, 1.0)
        candidates.append({
            "source": "minimum_Re_wing_borne_climb",
            "altitude_m": altitude,
            "true_speed_m_s": speed,
            "mach": xfoil_mach_number(altitude, speed),
            "unit_re_per_m": unit_re,
            "CL": state.get("CL"),
        })

    if candidates:
        return min(candidates, key=lambda item: item["unit_re_per_m"])

    altitude = MISSION["altitude_m"]
    speed = mission.get("cruise_true_speed_m_s", AIRCRAFT["cruise_true_speed_m_s"])
    return {
        "source": "cruise_fallback",
        "altitude_m": altitude,
        "true_speed_m_s": speed,
        "mach": xfoil_mach_number(altitude, speed),
        "unit_re_per_m": reynolds_number(altitude, speed, 1.0),
        "CL": mission.get("CL_cruise"),
    }


def representative_stall_xfoil_condition(wing):
    """Low-speed stall condition for the finite-wing CL_max section c_lmax.

    AD2 slide 22 requires the section c_lmax at the stall Reynolds number, which is
    lower (and gives a lower c_lmax) than the climb condition used for the
    linear-range coefficients. The stall speed comes from the current CL_max and
    MTOW at sea level; it converges in the XFOIL outer feedback loop.
    """
    altitude = AIRCRAFT["clmax_stall_xfoil_altitude_m"]
    # wing["stall_EAS_m_s"] is the sea-level stall EAS for the current CL_max.
    speed = AIRCRAFT["clmax_stall_speed_margin"] * wing["stall_EAS_m_s"]
    if altitude > 0.0:
        speed *= math.sqrt(isa_density(0.0) / isa_density(altitude))
    return {
        "source": "stall_low_speed",
        "altitude_m": altitude,
        "true_speed_m_s": speed,
        "mach": xfoil_mach_number(altitude, speed),
        "unit_re_per_m": reynolds_number(altitude, speed, 1.0),
        "CL": AIRCRAFT["wing_CL_max"],
    }


def _run_xfoil_pair(wing_reynolds, canard_reynolds, mach):
    """Run one XFOIL polar pair (wing SD7037 + canard NACA0012)."""
    return analyze_airfoil_pair(
        xfoil_path=AIRCRAFT["xfoil_path"],
        sd7037_file=AIRCRAFT["xfoil_sd7037_file"],
        wing_reynolds=wing_reynolds,
        canard_reynolds=canard_reynolds,
        mach=mach,
        x_transition=AIRCRAFT["xfoil_transition_x_c"],
        reynolds_rounding=AIRCRAFT["xfoil_reynolds_rounding"],
        reynolds_update_threshold=AIRCRAFT["xfoil_reynolds_update_threshold"],
        mach_rounding=AIRCRAFT["xfoil_mach_rounding"],
        alpha_start_deg=AIRCRAFT["xfoil_alpha_start_deg"],
        alpha_end_deg=AIRCRAFT["xfoil_alpha_end_deg"],
        alpha_step_deg=AIRCRAFT["xfoil_alpha_step_deg"],
        mach_command_min=AIRCRAFT["xfoil_mach_command_min"],
        timeout_s=AIRCRAFT["xfoil_timeout_s"],
    )


def update_airfoil_aerodynamics_from_xfoil(
    wing, mission, selected, show_progress=False, progress_indent=0
):
    """Refresh the section-derived aero inputs from XFOIL at the current geometry."""
    enabled = bool(AIRCRAFT.get("use_xfoil_airfoil_updates", False))
    update = {
        "enabled": enabled,
        "changed": False,
        "condition": None,
        "updates": {},
        "wing": None,
        "canard": None,
        "warnings": [],
    }
    if not enabled:
        return update

    canard = selected["canard"]
    condition = representative_xfoil_condition(mission)
    wing_re = condition["unit_re_per_m"] * wing["chord_m"]
    canard_re = condition["unit_re_per_m"] * canard["chord_m"]
    condition = dict(condition)
    condition["wing_reynolds"] = wing_re
    condition["canard_reynolds"] = canard_re
    condition["wing_chord_m"] = wing["chord_m"]
    condition["canard_chord_m"] = canard["chord_m"]
    update["condition"] = condition

    progress(
        (
            "XFOIL airfoils: "
            f"Re_w={wing_re / 1e6:.3f}e6, "
            f"Re_c={canard_re / 1e6:.3f}e6, "
            f"M={condition['mach']:.3f}"
        ),
        show_progress,
        progress_indent,
    )

    analysis = _run_xfoil_pair(wing_re, canard_re, condition["mach"])
    update["wing"] = analysis["wing"]
    update["canard"] = analysis["canard"]
    update["warnings"] = analysis["warnings"]

    # --- Finite-wing CL_max (AD2 slides 16-24) ---
    # The aircraft CL_max comes from the section c_lmax at the low-speed stall
    # Reynolds number (a second XFOIL condition), corrected to a finite wing by the
    # DATCOM high-AR method. The climb condition above supplies the linear-range
    # coefficients (slope, CL0, Cm0).
    stall_analysis = {"wing": None, "canard": None}
    stall_condition = None
    have_section = analysis["wing"] is not None or analysis["canard"] is not None
    if have_section:
        stall_condition = representative_stall_xfoil_condition(wing)
        stall_wing_re = stall_condition["unit_re_per_m"] * wing["chord_m"]
        stall_canard_re = stall_condition["unit_re_per_m"] * canard["chord_m"]
        stall_condition = dict(stall_condition)
        stall_condition["wing_reynolds"] = stall_wing_re
        stall_condition["canard_reynolds"] = stall_canard_re
        progress(
            (
                "XFOIL CL_max (stall): "
                f"Re_w={stall_wing_re / 1e6:.3f}e6, "
                f"Re_c={stall_canard_re / 1e6:.3f}e6, "
                f"V={stall_condition['true_speed_m_s']:.1f} m/s, "
                f"M={stall_condition['mach']:.3f}"
            ),
            show_progress,
            progress_indent,
        )
        stall_pair = _run_xfoil_pair(stall_wing_re, stall_canard_re, stall_condition["mach"])
        stall_analysis = {"wing": stall_pair["wing"], "canard": stall_pair["canard"]}
        update["warnings"] = update["warnings"] + stall_pair["warnings"]
    update["clmax_stall_condition"] = stall_condition

    clmax_diag = {}

    def aircraft_clmax(name, surface_geometry):
        """DATCOM finite-wing CL_max for one surface (stall-Re section, climb fallback)."""
        aspect_ratio, taper, thickness, sweep_c4_deg, dy_override = surface_geometry
        climb_section = analysis[name]
        stall_section = stall_analysis[name]
        if stall_section is not None:
            section_clmax = stall_section["cl_max"]
            section_mach = stall_section["mach"]
            section_source = "stall_reynolds"
        elif climb_section is not None:
            section_clmax = climb_section["cl_max"]
            section_mach = climb_section["mach"]
            section_source = "climb_reynolds_fallback"
            update["warnings"].append(
                f"{name} stall-Re XFOIL failed; CL_max uses climb-Re c_lmax"
            )
        else:
            return None
        sweep_le_deg = leading_edge_sweep_deg(sweep_c4_deg, aspect_ratio, taper)
        result = finite_wing_clmax(
            section_clmax=section_clmax,
            thickness_ratio=thickness,
            sweep_le_deg=sweep_le_deg,
            mach=section_mach,
            delta_y_pct=dy_override,
            dy_per_thickness=AIRCRAFT["le_sharpness_dy_per_tc"],
        )
        clmax_diag[name] = {
            "section_source": section_source,
            "CL_max": result["CL_max"],
            **result,
        }
        return result["CL_max"]

    values = {}
    if analysis["wing"] is not None:
        wing_eta = datcom_efficiency_from_section_slope(
            analysis["wing"]["cl_alpha_per_rad"]
        )
        values.update({
            "wing_CL0": analysis["wing"]["cl_at_alpha0"],
            "wing_airfoil_cm0": analysis["wing"]["cm_at_zero_lift"],
            "wing_datcom_eta": wing_eta,
            "wing_CL_alpha_per_rad": datcom_lift_slope(
                AIRCRAFT["wing_aspect_ratio"],
                condition["mach"],
                math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]),
                wing_eta,
            ),
        })

    if analysis["canard"] is not None:
        canard_eta = datcom_efficiency_from_section_slope(
            analysis["canard"]["cl_alpha_per_rad"]
        )
        values.update({
            "canard_datcom_eta": canard_eta,
            "canard_CL_alpha_per_rad": datcom_lift_slope(
                AIRCRAFT["canard_aspect_ratio"],
                condition["mach"],
                math.radians(AIRCRAFT["canard_sweep_half_chord_deg"]),
                canard_eta,
            ),
        })

    wing_clmax = aircraft_clmax(
        "wing",
        (
            AIRCRAFT["wing_aspect_ratio"],
            AIRCRAFT["wing_taper"],
            AIRCRAFT["wing_thickness_ratio"],
            AIRCRAFT["wing_sweep_quarter_chord_deg"],
            AIRCRAFT["wing_le_sharpness_dY_pct"],
        ),
    )
    if wing_clmax is not None:
        values["wing_CL_max"] = wing_clmax
    canard_clmax = aircraft_clmax(
        "canard",
        (
            AIRCRAFT["canard_aspect_ratio"],
            AIRCRAFT["canard_taper"],
            AIRCRAFT["canard_thickness_ratio"],
            AIRCRAFT["canard_sweep_quarter_chord_deg"],
            AIRCRAFT["canard_le_sharpness_dY_pct"],
        ),
    )
    if canard_clmax is not None:
        values["canard_CL_max"] = canard_clmax
    update["clmax"] = clmax_diag

    tolerance = AIRCRAFT["xfoil_update_tolerance_fraction"]
    relaxation = AIRCRAFT["xfoil_update_relaxation"]
    for key, target_value in values.items():
        old_value = AIRCRAFT[key]
        new_value = old_value + relaxation * (target_value - old_value)
        AIRCRAFT[key] = new_value
        limit = tolerance * max(1.0, abs(old_value))
        changed = abs(new_value - old_value) > limit
        update["changed"] = update["changed"] or changed
        update["updates"][key] = {
            "old": old_value,
            "target": target_value,
            "new": new_value,
            "changed": changed,
        }

    if update["updates"]:
        progress(
            (
                "XFOIL update: "
                f"CLmax_w={AIRCRAFT['wing_CL_max']:.3f}, "
                f"CLmax_c={AIRCRAFT['canard_CL_max']:.3f}, "
                f"eta_w={AIRCRAFT['wing_datcom_eta']:.3f}, "
                f"eta_c={AIRCRAFT['canard_datcom_eta']:.3f}, "
                f"changed={update['changed']}"
            ),
            show_progress,
            progress_indent,
        )
    if clmax_diag.get("wing"):
        wd = clmax_diag["wing"]
        progress(
            (
                "Finite-wing CLmax: "
                f"cl_max_w={wd['section_clmax']:.3f} ({wd['section_source']}), "
                f"dY={wd['delta_y_pct']:.2f}%, "
                f"CLmax/cl_max={wd['clmax_ratio']:.3f}"
            ),
            show_progress,
            progress_indent,
        )
    if update["warnings"]:
        progress(
            "XFOIL warning: " + " | ".join(update["warnings"]),
            show_progress,
            progress_indent,
        )

    return update


def sweep_with_xfoil_feedback(show_progress=True):
    """Outer Reynolds-feedback loop around the wing-area sweep.

    Each iteration runs one full wing-area sweep at the section coefficients
    currently in AIRCRAFT, then refreshes those coefficients with a single XFOIL
    airfoil-pair run at the best design's representative condition. It repeats
    until XFOIL stops changing the coefficients.
    """
    xfoil_on = bool(AIRCRAFT.get("use_xfoil_airfoil_updates", False))
    outer_count = max(1, int(AIRCRAFT.get("xfoil_outer_iteration_count", 1))) if xfoil_on else 1
    last_update = {"enabled": xfoil_on, "changed": False, "condition": None}
    result = sweep_rows = None

    for outer in range(1, outer_count + 1):
        if xfoil_on:
            progress(f"XFOIL outer iteration {outer}/{outer_count}", show_progress)
        result, _, sweep_rows = sweep_wing_area(show_progress=show_progress)
        if not xfoil_on:
            break
        last_update = update_airfoil_aerodynamics_from_xfoil(
            result["wing"], result["mission"], result["selected"],
            show_progress=show_progress, progress_indent=1,
        )
        if not last_update["changed"]:
            progress(f"Airfoil aero converged after {outer}/{outer_count} XFOIL passes", show_progress)
            break
        progress("Section coefficients changed; re-running the wing-area sweep", show_progress)

    # The reported design must be sized at the final coefficients. If the loop hit
    # its cap while still changing, run one more sweep to stay consistent.
    if xfoil_on and last_update.get("changed"):
        progress("Final wing-area sweep at the latest section coefficients", show_progress)
        result, _, sweep_rows = sweep_wing_area(show_progress=show_progress)
        last_update = dict(last_update, changed=False)

    if result is not None:
        result["xfoil_airfoil_update"] = last_update
    return result, sweep_rows
