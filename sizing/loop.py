"""The sizing loops.

Two nested fixed-point loops sit between the physical models and the report:

  coupled_sizing_iteration  iterates mass <-> mission energy <-> canard sizing at
                            one wing area until the mass estimate closes.
  sweep_wing_area           runs that closure across a wing-area grid and keeps
                            the lightest feasible design (feasible = the scissor
                            CG band closes *and* the stall speed meets the
                            transition cap).
"""

from __future__ import annotations

import time
import traceback

from sizing.inputs import (
    AIRCRAFT,
    MISSION,
    snapshot_aircraft,
    restore_aircraft_snapshot,
)
from sizing.atmosphere import isa_density
from sizing.geometry import propeller_disk_estimate, wing_geometry
from sizing.mission import course_method_mission_energy, trim_drag_descriptor
from sizing.scissor import canard_and_wing_iteration
from sizing.report import make_summary, progress, format_duration


def _run_sizing_pass_once(weight_N, wing_area_m2, trim_drag=None):
    """One pass through mission, canard, wing position, and mass."""
    rho_cruise = isa_density(MISSION["altitude_m"])
    propeller = propeller_disk_estimate(weight_N)
    wing = wing_geometry(weight_N, rho_cruise, wing_area_m2)
    mission = course_method_mission_energy(weight_N, wing, trim_drag)
    wing["cruise_true_speed_m_s"] = mission["cruise_true_speed_m_s"]
    wing["CL_trim"] = mission["CL_cruise"]
    selected, candidates = canard_and_wing_iteration(wing, mission, propeller)

    return {
        "wing": wing,
        "mission": mission,
        "propeller": propeller,
        "selected": selected,
        "candidates": candidates,
        "stall_limit": selected["stall_limit"],
    }


def run_sizing_pass(weight_N, wing_area_m2, trim_drag=None):
    """One sizing pass at the section coefficients currently held in AIRCRAFT.

    XFOIL is not called here: the airfoil section data is refreshed once per outer
    Reynolds-feedback iteration (sizing.airfoil), so every pass inside the
    wing-area sweep and mass loop is pure Python and fast. trim_drag carries the
    previous pass's converged canard/arm descriptor so the two-surface drag model
    can split lift and drag; None falls back to wing-only.
    """
    result = _run_sizing_pass_once(weight_N, wing_area_m2, trim_drag)
    result["aircraft"] = snapshot_aircraft()
    return result


def coupled_sizing_iteration(wing_area_m2, show_progress=False, progress_indent=0):
    """Iterate mass, mission energy, and canard sizing for one wing area."""
    mass_kg = AIRCRAFT["MTOW_kg"]
    history = []
    result = None
    # Two-surface drag couples the mission to the canard layout, which is solved
    # only after the mission runs. We break the loop by feeding each pass the
    # previous pass's converged canard/arm descriptor (None on the first pass =
    # wing-only drag); the mass fixed point converges it alongside the mass.
    trim_drag = None

    for iteration in range(1, AIRCRAFT["sizing_iteration_count"] + 1):
        weight_N = mass_kg * AIRCRAFT["g_m_s2"]
        progress(
            f"Mass iteration {iteration}/{AIRCRAFT['sizing_iteration_count']}: "
            f"input mass={mass_kg:.2f} kg",
            show_progress, progress_indent,
        )
        result = run_sizing_pass(weight_N, wing_area_m2, trim_drag=trim_drag)
        trim_drag = trim_drag_descriptor(result)
        estimated_mass_kg = result["selected"]["mass"]["total_mass_kg"]
        mass_change_kg = estimated_mass_kg - mass_kg

        history.append({
            "iteration": iteration,
            "mass_used_kg": mass_kg,
            "wing_area_m2": result["wing"]["area_m2"],
            "stall_EAS_m_s": result["wing"]["stall_EAS_m_s"],
            "max_stall_EAS_m_s": result["stall_limit"]["stall_EAS_max_m_s"],
            "stall_margin_m_s": result["selected"]["stall_margin_m_s"],
            "climb_EAS_m_s": result["mission"]["optimized_climb_EAS_m_s"],
            "mission_energy_Wh": result["mission"]["total_energy_Wh"],
            "battery_mass_kg": result["mission"]["battery_mass_kg"],
            "canard_area_ratio": result["selected"]["canard"]["area_ratio"],
            "estimated_mass_kg": estimated_mass_kg,
            "mass_change_kg": mass_change_kg,
        })

        progress(
            f"Mass estimate={estimated_mass_kg:.2f} kg (delta {mass_change_kg:+.2f} kg); "
            f"stall={result['wing']['stall_EAS_m_s']:.2f} m/s, "
            f"cap={result['stall_limit']['stall_EAS_max_m_s']:.2f} m/s, "
            f"climb EAS={result['mission']['optimized_climb_EAS_m_s']:.2f} m/s, "
            f"Sc/Sw={result['selected']['canard']['area_ratio']:.3f}",
            show_progress, progress_indent,
        )

        if abs(estimated_mass_kg - mass_kg) <= AIRCRAFT["sizing_mass_tolerance_kg"]:
            progress(
                f"Mass converged within {AIRCRAFT['sizing_mass_tolerance_kg']:.2f} kg",
                show_progress, progress_indent,
            )
            mass_kg = estimated_mass_kg
            break

        mass_kg = estimated_mass_kg

    weight_N = mass_kg * AIRCRAFT["g_m_s2"]
    progress("Final coupled pass at converged mass", show_progress, progress_indent)
    result = run_sizing_pass(weight_N, wing_area_m2, trim_drag=trim_drag)
    result["iteration_history"] = history
    result["final_mass_used_kg"] = mass_kg
    result["selected_wing_area_m2"] = wing_area_m2
    return result


def wing_area_sweep_values():
    values = []
    area = AIRCRAFT["wing_area_sweep_min_m2"]
    maximum = AIRCRAFT["wing_area_sweep_max_m2"]
    step = AIRCRAFT["wing_area_sweep_step_m2"]
    while area <= maximum + 1e-9:
        values.append(round(area, 6))
        area += step
    return values


# Wing-area trade-table columns (all but the first three come from make_summary).
_SWEEP_COLUMNS = (
    "MTOW_mass_estimate_kg", "mass_closure_error_kg", "wing_span_m",
    "wing_stall_EAS_m_s", "max_stall_EAS_m_s", "stall_margin_m_s",
    "stall_limit_source", "minimum_climb_EAS_m_s", "optimized_climb_EAS_m_s",
    "max_climb_CL", "climb_CL_limit", "cruise_true_speed_m_s",
    "mission_energy_Wh", "battery_mass_kg", "canard_area_ratio",
    "canard_area_m2", "canard_arm_m", "scissor_clearance_over_c",
    "outbound_time_s", "peak_electrical_power_W", "propeller_diameter_m",
    "sizing_iterations_used",
)


def _sweep_row(wing_area, feasible, failure_reason, summary=None):
    """One wing-area trade-table row; blank value columns when the point failed."""
    row = {
        "wing_area_m2": wing_area,
        "feasible": feasible,
        "failure_reason": failure_reason,
    }
    for column in _SWEEP_COLUMNS:
        row[column] = summary[column] if summary is not None else ""
    return row


def sweep_wing_area(show_progress=False):
    rows = []
    feasible_results = []
    wing_areas = wing_area_sweep_values()
    sweep_start = time.perf_counter()

    progress(
        f"Wing-area sweep: {len(wing_areas)} candidates from "
        f"{wing_areas[0]:.2f} to {wing_areas[-1]:.2f} m^2",
        show_progress,
    )

    for index, wing_area in enumerate(wing_areas, start=1):
        progress(
            f"[{index}/{len(wing_areas)} | {100.0 * (index - 1) / len(wing_areas):5.1f}% done] "
            f"Wing area {wing_area:.2f} m^2",
            show_progress,
        )
        area_start = time.perf_counter()
        try:
            result = coupled_sizing_iteration(
                wing_area, show_progress=show_progress, progress_indent=1
            )
            summary = make_summary(result)
            scissor_ok = result["selected"]["scissor_feasible"]
            stall_ok = result["selected"]["stall_feasible"]
            feasible = scissor_ok and stall_ok

            print(
                f"S={wing_area:.2f} m²  "
                f"stall={summary['wing_stall_EAS_m_s']:.1f}/{summary['max_stall_EAS_m_s']:.1f} m/s  "
                f"scissor={'OK' if scissor_ok else 'FAIL'}  "
                f"stall={'OK' if stall_ok else 'FAIL'}  "
                f"canard_ratio={result['selected']['canard']['area_ratio']:.3f}  "
                f"arm={result['selected']['arm_m']:.2f} m  "
                f"mass={result['selected']['mass']['total_mass_kg']:.1f} kg"
            )

            if not stall_ok:
                failure_reason = f"Stall speed above {summary['stall_limit_source']} limit."
            elif not scissor_ok:
                failure_reason = "Scissor constraints not feasible."
            else:
                failure_reason = ""

            rows.append(_sweep_row(wing_area, feasible, failure_reason, summary))
            if feasible:
                feasible_results.append((summary["MTOW_mass_estimate_kg"], result, summary))

            area_elapsed = time.perf_counter() - area_start
            remaining = (time.perf_counter() - sweep_start) / index * (len(wing_areas) - index)
            status = "feasible" if feasible else f"rejected: {failure_reason}"
            progress(
                f"Result {status}; mass={summary['MTOW_mass_estimate_kg']:.2f} kg, "
                f"stall={summary['wing_stall_EAS_m_s']:.2f}/{summary['max_stall_EAS_m_s']:.2f} m/s, "
                f"Sc/Sw={summary['canard_area_ratio']:.3f}, arm={summary['canard_arm_m']:.2f} m, "
                f"time={format_duration(area_elapsed)}, remaining ETA={format_duration(remaining)}",
                show_progress, 1,
            )

        except RuntimeError as error:
            print(f"S={wing_area:.2f} m²  RUNTIME ERROR: {error}")
            rows.append(_sweep_row(wing_area, False, str(error)))

        except Exception as error:
            print(f"S={wing_area:.2f} m²  UNEXPECTED ERROR: {type(error).__name__}: {error}")
            traceback.print_exc()
            rows.append(_sweep_row(wing_area, False, f"{type(error).__name__}: {error}"))

    if not feasible_results:
        raise RuntimeError("No feasible wing-area point was found in the sweep.")

    feasible_results.sort(key=lambda item: item[0])
    _, result, _ = feasible_results[0]
    restore_aircraft_snapshot(result["aircraft"])
    summary = make_summary(result)
    progress(
        f"Selected feasible point: S={summary['wing_area_m2']:.2f} m^2, "
        f"mass={summary['MTOW_mass_estimate_kg']:.2f} kg, "
        f"Sc/Sw={summary['canard_area_ratio']:.3f}, arm={summary['canard_arm_m']:.2f} m",
        show_progress,
    )
    return result, summary, rows
