"""Fixed-MTOW inner loop for Phases 1-9: propeller, mission, J, Reynolds, wing, and canard."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..common import isa, transition_cruise_speed_requirement
from ..models import Assumptions, Mission
from ..phases.phase01_propeller import phase1_propeller
from ..phases.phase02_hover import phase2_hover_climb_power
from ..phases.phase03_mission import phase3_mission_optimise
from ..phases.phase04_advance_ratio import phase4_J_coupling
from ..phases.phase05_energy import phase5_energy_battery
from ..phases.phase06_constraints import phase6_constraint_diagram
from ..phases.phase07_airfoil import phase7_airfoil_xfoil
from ..phases.phase08_wing import phase8_wing_planform
from ..phases.phase09_canard import phase9_canard


def _apply_prop_diameter_limit(phase1: Dict, D_prop_max_m: Optional[float]) -> Dict:
    """Apply an optional CAD diameter cap and recompute derived prop values."""
    if D_prop_max_m is None:
        phase1["disc_loading_target"] = float(phase1["disc_loading"])
        phase1["prop_diameter_max_m"] = None
        phase1["prop_diameter_limited"] = False
        return phase1
    if D_prop_max_m <= 0.0:
        raise ValueError("prop_diameter_max_m must be positive when provided.")
    if phase1["D_prop"] <= D_prop_max_m:
        phase1["prop_diameter_limited"] = False
        phase1["disc_loading_target"] = float(phase1["disc_loading"])
        phase1["prop_diameter_max_m"] = float(D_prop_max_m)
        return phase1

    limited = dict(phase1)
    D_prop = float(D_prop_max_m)
    A_disc = np.pi * (D_prop / 2.0) ** 2
    DL_actual = limited["T_per_rotor"] / A_disc
    n_max = limited["V_tip"] / (np.pi * D_prop)

    limited.update({
        "D_prop": float(D_prop),
        "A_disc": float(A_disc),
        "n_max": float(n_max),
        "rpm_max": float(60.0 * n_max),
        "disc_loading_target": float(phase1["disc_loading"]),
        "disc_loading": float(DL_actual),
        "prop_diameter_max_m": float(D_prop_max_m),
        "prop_diameter_limited": True,
    })
    limited.setdefault("warnings", [])
    limited["warnings"] = list(limited["warnings"]) + [
        "Propeller diameter was capped by prop_diameter_max_m; actual disc loading is higher than the target."
    ]
    return limited


def iterate_phases_1_to_9(MTOW_kg: float = 50.0,
                          mission: Optional[Mission] = None,
                          assumptions: Optional[Assumptions] = None,
                          max_inner_iter: int = 15,
                          j_tol: float = 0.01,
                          area_tol: float = 0.01,
                          re_tol: float = 0.02,
                          constraint_plot_path=None,
                          airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Close the propulsion/aero inner loop for Phases 1-9 at fixed MTOW."""
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()
    if MTOW_kg <= 0.0:
        raise ValueError("MTOW_kg must be positive.")
    if max_inner_iter <= 0:
        raise ValueError("max_inner_iter must be positive.")
    if area_tol <= 0.0:
        raise ValueError("area_tol must be positive.")
    if re_tol <= 0.0:
        raise ValueError("re_tol must be positive.")
    if assumptions.preliminary_wing_area_m2 <= 0.0:
        raise ValueError("preliminary_wing_area_m2 must be positive.")
    if not assumptions.cruise_J_min <= assumptions.cruise_J_target <= assumptions.cruise_J_max:
        raise ValueError("cruise_J_target must lie inside the selected J band.")

    MTOW_N = MTOW_kg * assumptions.g
    eta_fw = assumptions.eta_prop * assumptions.eta_motor * assumptions.eta_ESC
    rho_target, mu_target, a_sound_target, _ = isa(mission.altitude_m)
    S_guess = assumptions.preliminary_wing_area_m2
    cruise_n_rev_s = None
    Re_main_previous = None
    Re_canard_guess = None
    history = []
    converged = False
    stopped_reason = "maximum iterations reached"

    phase1 = phase2_hover = phase3 = phase4 = None
    phase5 = phase6 = phase7 = phase8 = phase9 = None

    for it in range(max_inner_iter):
        phase1 = phase1_propeller(
            MTOW_N,
            assumptions.thrust_to_weight,
            assumptions.n_rotors,
            assumptions.disc_loading_target_N_m2,
            a_sound_target,
            assumptions.tip_mach_max,
        )
        phase1 = _apply_prop_diameter_limit(phase1, assumptions.prop_diameter_max_m)

        def hover_power_total(h_m: float) -> float:
            rho_h, _, _, _ = isa(h_m)
            per_rotor = phase2_hover_climb_power(
                phase1["T_per_rotor"],
                phase1["A_disc"],
                rho_h,
                0.0,
                assumptions.figure_of_merit,
                assumptions.eta_motor,
                assumptions.eta_ESC,
            )["P_elec"]
            return assumptions.n_rotors * per_rotor

        phase2_hover = phase2_hover_climb_power(
            phase1["T_per_rotor"],
            phase1["A_disc"],
            rho_target,
            assumptions.vtol_climb_rate_m_s,
            assumptions.figure_of_merit,
            assumptions.eta_motor,
            assumptions.eta_ESC,
        )

        phase3 = phase3_mission_optimise(
            MTOW_N,
            S_guess,
            assumptions.CD0,
            assumptions.AR,
            assumptions.oswald_e,
            mission.altitude_m,
            mission.range_m,
            mission.time_budget_s,
            mission.hover_time_s,
            hover_power_total,
            eta_fw=eta_fw,
            V_min_required=transition_cruise_speed_requirement(assumptions)["V_cruise_min_m_s"],
        )

        if cruise_n_rev_s is None:
            cruise_n_current = phase1["n_max"]
        else:
            cruise_n_current = cruise_n_rev_s

        phase4_before = phase4_J_coupling(
            phase3["V_cruise"],
            cruise_n_current,
            phase1["D_prop"],
            assumptions.cruise_J_min,
            assumptions.cruise_J_max,
        )
        n_target = phase3["V_cruise"] / (assumptions.cruise_J_target * phase1["D_prop"])
        tip_limited = False
        if n_target > phase1["n_max"]:
            cruise_n_next = phase1["n_max"]
            tip_limited = True
        else:
            cruise_n_next = n_target

        phase4 = phase4_J_coupling(
            phase3["V_cruise"],
            cruise_n_next,
            phase1["D_prop"],
            assumptions.cruise_J_min,
            assumptions.cruise_J_max,
        )

        Re_main = phase3["Re_estimate"]
        Re_canard_for_airfoil = Re_canard_guess if Re_canard_guess is not None else Re_main
        phase7 = phase7_airfoil_xfoil(
            Re_main,
            Re_canard_for_airfoil,
            use_xfoil=assumptions.use_xfoil,
            xfoil_path=assumptions.xfoil_path,
            airfoil_files=airfoil_files,
        )

        mach_cruise = phase3["V_cruise"] / a_sound_target
        phase8 = phase8_wing_planform(
            MTOW_N,
            phase7["main"]["cl_max"],
            phase7["main"]["cl_a"],
            rho_target,
            assumptions.stall_speed_target_max_m_s,
            assumptions.AR,
            taper=assumptions.wing_taper,
            sweep_c4=assumptions.wing_sweep_c4_rad,
            e=assumptions.oswald_e,
            stall_margin=assumptions.wing_stall_margin,
            mach=mach_cruise,
        )

        l_c = assumptions.canard_arm_chord_ratio * phase8["c_bar"]
        phase9 = phase9_canard(
            phase8["S"],
            phase8["c_bar"],
            phase8["x_ac_w"],
            l_c,
            assumptions.canard_volume_coeff,
            assumptions.canard_AR,
            assumptions.canard_taper,
            assumptions.canard_sweep_c4_rad,
            phase7["canard"]["cl_max"],
            mach_cruise,
        )

        Re_canard_new = rho_target * phase3["V_cruise"] * phase9["c_bar_c"] / mu_target
        area_change = abs(phase8["S"] - S_guess) / S_guess
        if Re_main_previous is None:
            Re_main_change = np.inf
        else:
            Re_main_change = abs(Re_main - Re_main_previous) / Re_main_previous
        if Re_canard_guess is None:
            Re_canard_change = np.inf
        else:
            Re_canard_change = abs(Re_canard_new - Re_canard_guess) / Re_canard_guess
        j_error = abs(phase4["J"] - assumptions.cruise_J_target)

        history.append({
            "iteration": it + 1,
            "S_guess": float(S_guess),
            "S_wing": float(phase8["S"]),
            "S_canard": float(phase9["S_c"]),
            "area_change": float(area_change),
            "Re_main": float(Re_main),
            "Re_main_change": None if not np.isfinite(Re_main_change) else float(Re_main_change),
            "Re_canard_input": float(Re_canard_for_airfoil),
            "Re_canard_new": float(Re_canard_new),
            "Re_canard_change": None if not np.isfinite(Re_canard_change) else float(Re_canard_change),
            "V_cruise": float(phase3["V_cruise"]),
            "D_prop": float(phase1["D_prop"]),
            "disc_loading": float(phase1["disc_loading"]),
            "J_before_update": float(phase4_before["J"]),
            "J_after_update": float(phase4["J"]),
            "cruise_rpm": float(60.0 * cruise_n_next),
            "tip_limited": bool(tip_limited),
            "CL_max_3D_w": float(phase8["CL_max_3D"]),
            "CL_max_3D_c": float(phase9["CL_max_3D_c"]),
            "V_stall": float(phase8["V_stall"]),
        })

        S_guess = phase8["S"]
        cruise_n_rev_s = cruise_n_next
        Re_main_previous = Re_main
        Re_canard_guess = Re_canard_new

        re_converged = (
            np.isfinite(Re_main_change)
            and np.isfinite(Re_canard_change)
            and Re_main_change <= re_tol
            and Re_canard_change <= re_tol
        )
        if area_change <= area_tol and phase4["in_band"] and j_error <= j_tol and re_converged:
            converged = True
            stopped_reason = "propulsion, wing, airfoil Reynolds, and canard geometry converged"
            break
        if tip_limited and not phase4["in_band"]:
            stopped_reason = "tip-Mach RPM limit prevents reaching the target advance ratio"
            break

    segments = {
        "transition": (phase3["P_transition"], phase3["t_transition"]),
        "fixed_wing_climb": (phase3["P_fw"], phase3["t_fw"]),
        "mission_hover": (phase3["P_hover_target"], mission.hover_time_s),
    }
    phase5 = phase5_energy_battery(
        segments,
        assumptions.eta_batt,
        assumptions.usable_battery_fraction,
        assumptions.battery_specific_energy_Wh_kg,
    )

    W_S_range = np.linspace(0.35 * phase8["W_S_design"], 1.80 * phase8["W_S_design"], 90)
    phase6 = phase6_constraint_diagram(
        W_S_range,
        np.array([]),
        phase3["V_cruise"],
        phase8["V_stall"],
        phase3["ROC"],
        phase3["gamma"],
        rho_target,
        rho_target,
        phase8["CL_max_3D"],
        assumptions.CD0,
        assumptions.AR,
        assumptions.oswald_e,
        eta_fw,
        assumptions.thrust_to_weight,
        plot_path=constraint_plot_path,
    )

    warnings = []
    if phase7["warnings"]:
        warnings.extend(f"phase7: {warning}" for warning in phase7["warnings"])
    if phase8["warnings"]:
        warnings.extend(f"phase8: {warning}" for warning in phase8["warnings"])
    if phase9["warnings"]:
        warnings.extend(f"phase9: {warning}" for warning in phase9["warnings"])
    if mission.external_tow_load_N:
        warnings.append(
            "external_tow_load_N is not included in UAV MTOW or Phase 5 energy in this propulsion/aero loop."
        )
    if not converged:
        warnings.append(
            "Phase 1-9 inner loop did not meet the requested tolerances before stopping."
        )

    return {
        "MTOW_kg": float(MTOW_kg),
        "MTOW_N": float(MTOW_N),
        "converged": bool(converged),
        "stopped_reason": stopped_reason,
        "history": history,
        "cruise_n_rev_s": float(cruise_n_rev_s),
        "cruise_rpm": float(60.0 * cruise_n_rev_s),
        "Re_main": float(phase3["Re_estimate"]),
        "Re_canard": float(Re_canard_guess),
        "phase1": phase1,
        "phase2_hover": phase2_hover,
        "phase3": phase3,
        "phase4": phase4,
        "phase5": phase5,
        "phase6": phase6,
        "phase7": phase7,
        "phase8": phase8,
        "phase9": phase9,
        "notes": [
            "This closes the fixed-MTOW propulsion/aerodynamics inner loop for Phases 1-9.",
            "The loop updates cruise RPM for J, wing area for stall sizing, and canard Reynolds number from the canard chord.",
            "Battery mass is reported but does not feed back into MTOW until Phases 15-16.",
            "The canard moment arm still comes from canard_arm_chord_ratio until CAD and CG layout are available.",
        ],
        "warnings": warnings,
    }
