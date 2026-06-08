"""Fixed-MTOW inner loop for preliminary mission, wing, and canard sizing."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..common import isa
from ..models import Assumptions, Mission
from ..phases.phase01_propeller import phase1_propeller
from ..phases.phase02_hover import phase2_hover_climb_power
from ..phases.phase03_mission import phase3_mission_optimise
from ..phases.phase04_power import phase4_mission_power_validation
from ..phases.phase05_energy import phase5_energy_battery
from ..phases.phase06_constraints import phase6_constraint_diagram
from ..phases.phase07_airfoil import phase7_airfoil_xfoil
from ..phases.phase08_wing import _datcom_lift_curve_slope, phase8_wing_planform
from ..phases.phase09_canard import phase9_canard


def _apply_prop_diameter_limit(phase1: Dict, D_prop_max_m: Optional[float]) -> Dict:
    """Apply an optional CAD diameter cap to the preliminary disk estimate."""
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
    limited.update({
        "D_prop": D_prop,
        "A_disc": float(A_disc),
        "disc_loading_target": float(phase1["disc_loading"]),
        "disc_loading": float(DL_actual),
        "prop_diameter_max_m": float(D_prop_max_m),
        "prop_diameter_limited": True,
    })
    limited["warnings"] = list(limited.get("warnings", [])) + [
        "Propeller diameter was capped; actual disk loading is above the preliminary target."
    ]
    return limited


def iterate_phases_1_to_9(
        MTOW_kg: float = 50.0,
        mission: Optional[Mission] = None,
        assumptions: Optional[Assumptions] = None,
        max_inner_iter: int = 15,
        area_tol: float = 0.01,
        re_tol: float = 0.02,
        constraint_plot_path=None,
        airfoil_files: Optional[Dict[str, str]] = None,
        wing_area_m2: Optional[float] = None) -> Dict:
    """Close the preliminary mission/aerodynamic loop at fixed MTOW and area."""
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()
    if MTOW_kg <= 0.0:
        raise ValueError("MTOW_kg must be positive.")
    if max_inner_iter <= 0 or area_tol <= 0.0 or re_tol <= 0.0:
        raise ValueError("Iteration count and convergence tolerances must be positive.")
    selected_wing_area_m2 = (
        assumptions.preliminary_wing_area_m2
        if wing_area_m2 is None
        else float(wing_area_m2)
    )
    if selected_wing_area_m2 <= 0.0:
        raise ValueError("wing_area_m2 must be positive.")

    MTOW_N = MTOW_kg * assumptions.g
    eta_fw = assumptions.eta_prop * assumptions.eta_motor * assumptions.eta_ESC
    rho_target, mu_target, a_sound_target, _ = isa(mission.altitude_m)
    rho_transition, _, _, _ = isa(assumptions.vertical_takeoff_height_m)
    rho_reference, _, _, _ = isa(0.0)
    S_guess = selected_wing_area_m2
    Re_main_previous = None
    Re_canard_guess = None
    CL_max_mission = assumptions.CL_max_guess
    CL_max_previous = None
    mission_energy_previous = None
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
        )
        phase1 = _apply_prop_diameter_limit(
            phase1, assumptions.prop_diameter_max_m
        )
        phase2_hover = phase2_hover_climb_power(
            phase1["T_per_rotor"],
            phase1["A_disc"],
            rho_target,
            assumptions.vtol_climb_rate_m_s,
            assumptions.figure_of_merit,
            assumptions.eta_motor,
            assumptions.eta_ESC,
        )
        CL_a_for_stall_margin = (
            phase8["CL_a"]
            if phase8 is not None
            else _datcom_lift_curve_slope(
                assumptions.AR,
                assumptions.wing_sweep_c4_rad,
                assumptions.wing_taper,
                mach=0.0,
            )
        )
        stall_margin_CL = (
            CL_a_for_stall_margin
            * np.deg2rad(assumptions.cruise_stall_margin_deg)
        )
        CL_allowed_from_fraction = (
            assumptions.mission_CL_limit_fraction * CL_max_mission
        )
        CL_allowed_from_stall_margin = CL_max_mission - stall_margin_CL
        CL_allowed_override = min(
            CL_allowed_from_fraction,
            CL_allowed_from_stall_margin,
        )
        if CL_allowed_override <= 0.0:
            raise ValueError(
                "Cruise stall margin leaves no usable lift coefficient; "
                "reduce cruise_stall_margin_deg or revise CL_max."
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
            eta_prop=assumptions.eta_prop,
            eta_motor=assumptions.eta_motor,
            eta_ESC=assumptions.eta_ESC,
            max_affordable_electrical_power_W=assumptions.max_affordable_electrical_power_W,
            preliminary_hover_power_W=assumptions.preliminary_hover_power_W,
            preliminary_transition_power_W=assumptions.preliminary_transition_power_W,
            V_min_required=None,
            CL_max=CL_max_mission,
            CL_limit_fraction=assumptions.mission_CL_limit_fraction,
            CL_allowed_override=CL_allowed_override,
            altitude_step_m=assumptions.mission_altitude_step_m,
            vertical_takeoff_height_m=assumptions.vertical_takeoff_height_m,
            vertical_takeoff_rate_m_s=assumptions.vertical_takeoff_rate_m_s,
            transition_accel_m_s2=assumptions.transition_accel_m_s2,
            transition_blend_start_fraction=assumptions.transition_blend_start_frac,
            transition_blend_end_fraction=assumptions.transition_blend_end_frac,
            transition_cruise_margin_fraction=assumptions.transition_cruise_margin_frac,
            transition_sample_count=assumptions.transition_sample_count,
            max_transition_complete_speed_m_s=assumptions.max_transition_complete_speed_m_s,
            max_stall_EAS_m_s=assumptions.max_stall_EAS_m_s,
            minimum_power_margin_fraction=assumptions.minimum_power_margin_fraction,
            max_fixed_wing_climb_angle_deg=assumptions.max_fixed_wing_climb_angle_deg,
            allow_spiral_climb=assumptions.allow_spiral_climb,
            run_multistage_extension=assumptions.run_multistage_mission_extension,
            run_minimum_time_reference=assumptions.run_minimum_time_reference,
        )
        phase4 = phase4_mission_power_validation(phase3)

        Re_main = phase3["Re_estimate"]
        Re_canard_for_airfoil = (
            Re_canard_guess if Re_canard_guess is not None else Re_main
        )
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
            selected_wing_area_m2,
            phase7["main"]["cl_max"],
            phase7["main"]["cl_a"],
            rho_target,
            rho_transition,
            rho_reference,
            assumptions.AR,
            taper=assumptions.wing_taper,
            sweep_c4=assumptions.wing_sweep_c4_rad,
            e=assumptions.oswald_e,
            mach=mach_cruise,
        )
        CL_max_mission = phase8["CL_max_3D"]
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

        # Use unit_Re from Phase 3 so canard Re is consistent with the
        # reference level-flight speed rather than re-derived here.
        Re_canard_new = (
            phase3["unit_Reynolds_number_per_m"] * phase9["c_bar_c"]
        )
        area_change = abs(phase8["S"] - selected_wing_area_m2) / selected_wing_area_m2
        CL_max_change = (
            np.inf
            if CL_max_previous is None
            else abs(phase8["CL_max_3D"] - CL_max_previous) / CL_max_previous
        )
        mission_energy_change = (
            np.inf
            if mission_energy_previous is None
            else abs(phase3["E_total_Wh"] - mission_energy_previous)
            / mission_energy_previous
        )
        Re_main_change = (
            np.inf
            if Re_main_previous is None
            else abs(Re_main - Re_main_previous) / Re_main_previous
        )
        Re_canard_change = (
            np.inf
            if Re_canard_guess is None
            else abs(Re_canard_new - Re_canard_guess) / Re_canard_guess
        )
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
            "D_prop_preliminary": float(phase1["D_prop"]),
            "disc_loading": float(phase1["disc_loading"]),
            "peak_electrical_power_W": float(phase3["peak_electrical_power_W"]),
            "power_margin_W": float(phase4["power_margin_W"]),
            "power_margin_over_required_W": float(
                phase4.get("power_margin_over_required_W", phase4["power_margin_W"])
            ),
            "transition_complete_speed_m_s": float(
                phase3.get("transition_complete_speed_m_s", np.nan)
            ),
            "optimized_climb_angle_deg": float(
                phase3.get("optimized_climb_angle_deg", np.nan)
            ),
            "CL_max_3D_w": float(phase8["CL_max_3D"]),
            "CL_allowed_override": float(CL_allowed_override),
            "CL_allowed_from_fraction": float(CL_allowed_from_fraction),
            "CL_allowed_from_stall_margin": float(
                CL_allowed_from_stall_margin
            ),
            "CL_max_change": None if not np.isfinite(CL_max_change) else float(CL_max_change),
            "CL_max_3D_c": float(phase9["CL_max_3D_c"]),
            "mission_energy_Wh": float(phase3["E_total_Wh"]),
            "mission_energy_change": None if not np.isfinite(mission_energy_change) else float(mission_energy_change),
            "V_stall": float(phase8["V_stall"]),
        })
        Re_main_previous = Re_main
        Re_canard_guess = Re_canard_new
        CL_max_previous = phase8["CL_max_3D"]
        mission_energy_previous = phase3["E_total_Wh"]
        re_converged = (
            np.isfinite(Re_main_change)
            and np.isfinite(Re_canard_change)
            and Re_main_change <= re_tol
            and Re_canard_change <= re_tol
        )
        aero_mission_converged = (
            np.isfinite(CL_max_change)
            and np.isfinite(mission_energy_change)
            and CL_max_change <= area_tol
            and mission_energy_change <= area_tol
        )
        if phase4["feasible"] and re_converged and aero_mission_converged:
            converged = True
            stopped_reason = (
                "mission profile, airfoil Reynolds, finite-wing aerodynamics, and canard geometry converged at fixed wing area"
            )
            break

    # Battery energy: sized from the energy-critical sweep case (maximum range).
    energy_segs = phase3["carry_forward"]["energy_sizing_case"]["segment_summaries"]
    segments = {
        name: (segment["average_electrical_power_W"], segment["time_s"])
        for name, segment in energy_segs.items()
    }
    phase5 = phase5_energy_battery(
        segments,
        assumptions.eta_batt,
        assumptions.usable_battery_fraction,
        assumptions.battery_specific_energy_Wh_kg,
    )
    # Battery peak-power rating: sized from the power-critical sweep case.
    phase5["P_motor_cont"] = max(
        phase5["P_motor_cont"],
        phase3["carry_forward"]["power_sizing_case"]["peak_electrical_power_W"],
    )
    phase5["P_motor_peak"] = phase5["P_motor_cont"]

    W_S_range = np.linspace(
        0.35 * phase8["W_S_design"], 1.80 * phase8["W_S_design"], 90
    )
    cf3 = phase3["carry_forward"]
    phase2_hover_power_total_W = phase2_hover["P_elec"] * assumptions.n_rotors
    phase2_hover_power_loading_W_N = phase2_hover_power_total_W / MTOW_N
    # Constraint diagram receives mission-requirement quantities, not results
    # from a single optimized profile.  ROC is the minimum average climb rate
    # needed to reach h_target within the outbound time budget; V_ref is the
    # reference level-flight design speed; gamma is derived from both.  The
    # hover line uses Phase 2 at the installed thrust-to-weight ratio.
    phase6 = phase6_constraint_diagram(
        W_S_range,
        np.array([]),
        cf3["reference_level_flight"]["TAS_m_s"],
        cf3["minimum_required_average_ROC_m_s"],
        cf3["gamma_reference_rad"],
        rho_target,
        assumptions.CD0,
        assumptions.AR,
        assumptions.oswald_e,
        eta_fw,
        assumptions.thrust_to_weight,
        selected_W_S=phase8["W_S_design"],
        hover_power_loading_W_N=phase2_hover_power_loading_W_N,
        plot_path=constraint_plot_path,
    )

    warnings = []
    for name, phase in (
            ("phase7", phase7), ("phase8", phase8), ("phase9", phase9)):
        warnings.extend(f"{name}: {warning}" for warning in phase["warnings"])
    if mission.external_tow_load_N:
        warnings.append(
            "external_tow_load_N is not included in UAV MTOW or Phase 5 energy."
        )
    if not converged:
        warnings.append(
            "Phase 1-9 inner loop did not meet the requested tolerances before stopping."
        )

    return {
        "MTOW_kg": float(MTOW_kg),
        "MTOW_N": float(MTOW_N),
        "fixed_wing_area_m2": float(selected_wing_area_m2),
        "converged": bool(converged),
        "stopped_reason": stopped_reason,
        "history": history,
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
            "Phase 1 is only a preliminary disk-loading diameter estimate.",
            "Mission optimization uses assumed efficiencies and a total-aircraft electrical-power cap.",
            "Detailed propeller, motor, ESC, and battery selection is deferred until suitable component data exists.",
        ],
        "warnings": warnings,
    }
