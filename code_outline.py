"""
Bellona DSE Group 22 — Detailed-Design Sizing Skeleton.
Dependencies: numpy, matplotlib only.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass, replace
from typing import Tuple, Dict, Callable, Optional

# ---------- DEFAULT INPUTS ----------

@dataclass
class Mission:
    """Mission inputs that should be easy to override from the CLI later."""

    altitude_m: float = 6000.0
    range_m: float = 6000.0
    time_budget_s: float = 600.0
    hover_time_s: float = 300.0
    mission_equipment_mass_kg: float = 7.3
    external_tow_load_N: float = 0.0


@dataclass
class Assumptions:
    """First-cut sizing assumptions shared across the phase functions."""

    g: float = 9.80665
    n_rotors: int = 4
    thrust_to_weight: float = 1.30
    vtol_climb_rate_m_s: float = 0.0
    disc_loading_target_N_m2: float = 170.0
    prop_diameter_max_m: Optional[float] = None
    tip_mach_max: float = 0.72
    cruise_J_target: float = 0.60
    cruise_J_min: float = 0.40
    cruise_J_max: float = 0.80
    CD0: float = 0.040
    AR: float = 7.0
    oswald_e: float = 0.78
    CL_max_guess: float = 1.30
    preliminary_wing_area_m2: float = 5.0
    stall_speed_target_max_m_s: float = 15.0
    wing_stall_margin: float = 0.80
    wing_taper: float = 0.40
    wing_sweep_c4_rad: float = 0.0
    canard_volume_coeff: float = 0.375
    canard_arm_chord_ratio: float = 2.50
    canard_AR: float = 5.0
    canard_taper: float = 0.50
    canard_sweep_c4_rad: float = 0.0
    canard_eps_alpha_c: float = 0.0
    wing_eps_alpha_w: float = 0.0
    wing_mac_le_x_m: Optional[float] = None
    solve_wing_position_for_cg: bool = True
    use_xfoil: bool = False
    xfoil_path: Optional[str] = None
    figure_of_merit: float = 0.70
    eta_motor: float = 0.90
    eta_ESC: float = 0.95
    eta_prop: float = 0.75
    eta_batt: float = 0.95
    usable_battery_fraction: float = 0.85
    battery_specific_energy_Wh_kg: float = 200.0
    fuselage_length_m: float = 2.2
    static_margin_min: float = 0.10
    elevon_q_slipstream_ratio: float = 1.50
    elevon_max_deflection_deg: float = 25.0
    elevon_tau: float = 0.50
    elevon_eta: float = 0.90
    elevon_l_e_over_c: float = 0.75
    elevon_pitch_trim_margin: float = 1.20
    elevon_chord_fraction_min: float = 0.12
    elevon_chord_fraction_max: float = 0.35
    elevon_span_fraction_min: float = 0.20
    elevon_span_fraction_max: float = 0.80
    elevon_grid_points: int = 49
    roll_rate_required_deg_s: float = 60.0
    hover_pitch_arm_m: Optional[float] = None
    hover_roll_arm_m: Optional[float] = None
    hover_yaw_arm_m: Optional[float] = None
    hover_pitch_arm_fraction_fuselage: float = 0.37
    hover_roll_arm_fraction_span: float = 0.375
    hover_yaw_arm_fraction_chord: float = 0.75
    hover_Ixx_kg_m2: Optional[float] = None
    hover_Iyy_kg_m2: Optional[float] = None
    hover_Izz_kg_m2: Optional[float] = None
    hover_Ixx_radius_fraction_span: float = 0.20
    hover_Iyy_radius_fraction_fuselage: float = 0.35
    hover_Izz_radius_fraction_span: float = 0.22
    hover_angular_accel_required_rad_s2: float = 2.0
    hover_yaw_rate_required_deg_s: float = 30.0
    hover_yaw_response_time_s: float = 1.0
    hover_control_CL_de: Optional[float] = None
    transition_blend_start_frac: float = 0.50
    transition_blend_end_frac: float = 1.20
    transition_cruise_margin_frac: float = 0.05
    transition_accel_m_s2: float = 1.0
    transition_sample_count: int = 9
    dynamic_Cm_q: Optional[float] = None
    dynamic_Cm_alpha_dot: float = 0.0
    dynamic_Cl_beta: float = -0.08
    dynamic_Cl_p: Optional[float] = None
    dynamic_Cl_r: float = 0.02
    dynamic_Cn_beta: float = 0.06
    dynamic_Cn_p: float = -0.02
    dynamic_Cn_r: float = -0.20
    short_period_zeta_min: float = 0.30
    short_period_zeta_max: float = 2.00
    short_period_omega_min_rad_s: float = 1.00
    phugoid_zeta_min: float = 0.04
    dutch_roll_zeta_min: float = 0.08
    dutch_roll_omega_min_rad_s: float = 0.40
    spiral_time_to_double_min_s: float = 12.0
    wing_areal_density_kg_m2: float = 2.00
    canard_areal_density_kg_m2: float = 1.70
    fuselage_linear_density_kg_m: float = 2.20
    boom_landing_gear_mass_kg: float = 2.50
    motor_specific_mass_kg_W: float = 0.00027
    esc_specific_mass_kg_W: float = 0.00008
    prop_mass_coeff_kg_m2: float = 0.10
    avionics_mass_kg: float = 1.50
    wiring_fraction: float = 0.06
    mass_contingency_fraction: float = 0.08
    cg_envelope_half_width_over_mac: float = 0.05
    cg_required_margin_over_mac: float = 0.02
    canard_volume_grid_min: float = 0.05
    canard_volume_grid_max: float = 0.45
    canard_volume_grid_step: float = 0.01
    hover_control_margin_min: float = 1.05
    hover_pitch_arm_fraction_fuselage_max: float = 0.45
    hover_roll_arm_fraction_span_max: float = 0.45
    thrust_to_weight_min: float = 1.20
    thrust_to_weight_max: float = 1.60
    control_closure_max_iter: int = 4

# ---------- HELPERS ----------

def transition_cruise_speed_requirement(assumptions: Assumptions) -> Dict:
    """Estimate the minimum cruise speed needed to complete transition blending."""
    if not 0.0 < assumptions.wing_stall_margin <= 1.0:
        raise ValueError("wing_stall_margin should be in the range 0 < margin <= 1.")
    if assumptions.stall_speed_target_max_m_s <= 0.0:
        raise ValueError("stall_speed_target_max_m_s must be positive.")
    if assumptions.transition_blend_end_frac <= 0.0:
        raise ValueError("transition_blend_end_frac must be positive.")
    if assumptions.transition_cruise_margin_frac < 0.0:
        raise ValueError("transition_cruise_margin_frac must be non-negative.")

    V_stall_design = np.sqrt(
        assumptions.wing_stall_margin
    ) * assumptions.stall_speed_target_max_m_s
    V_blend_end = assumptions.transition_blend_end_frac * V_stall_design
    V_cruise_min = (1.0 + assumptions.transition_cruise_margin_frac) * V_blend_end
    return {
        "V_stall_design_estimate_m_s": float(V_stall_design),
        "V_blend_end_estimate_m_s": float(V_blend_end),
        "V_cruise_min_m_s": float(V_cruise_min),
        "margin_frac": float(assumptions.transition_cruise_margin_frac),
    }

def isa(h: float) -> Tuple[float, float, float, float]:
    """ISA 1976 atmosphere (troposphere only, 0–11 km).
    In:  h [m]
    Out: rho [kg/m^3], mu [Pa·s], a [m/s], T [K]
    Ref: U.S. Standard Atmosphere 1976."""
    if h < 0.0:
        raise ValueError("Altitude must be non-negative.")
    if h > 11000.0:
        raise ValueError("This simple ISA helper is limited to the troposphere.")

    g0 = 9.80665
    R = 287.05287
    gamma = 1.4
    T0 = 288.15
    p0 = 101325.0
    lapse = 0.0065
    sutherland_beta = 1.458e-6
    sutherland_C = 110.4

    T = T0 - lapse * h
    p = p0 * (T / T0) ** (g0 / (R * lapse))
    rho = p / (R * T)
    mu = sutherland_beta * T**1.5 / (T + sutherland_C)
    a = np.sqrt(gamma * R * T)
    return float(rho), float(mu), float(a), float(T)

def disc_loading_regression(MTOW_N: float, n_rotors: int, DL_target: float
                            ) -> float:
    """First-cut prop diameter from disc-loading regression.
    In:  MTOW_N [N] (Ph16), n_rotors, DL_target [N/m^2] (Ph1, Tab 7.4 Gundlach)
    Out: D_prop [m]
    Eq:  D = sqrt(4*T_rotor / (pi*DL))   — Gundlach 2014 Eq. 7.18 form
    Loop: MTOW outer."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if DL_target <= 0.0:
        raise ValueError("DL_target must be positive.")

    T_per_rotor = MTOW_N / n_rotors
    return float(np.sqrt(4.0 * T_per_rotor / (np.pi * DL_target)))

# ---------- PHASE 1 ----------

def phase1_propeller(MTOW_N: float, T_over_W: float, n_rotors: int,
                     DL_target: float, a_sound: float, M_tip_max: float = 0.72
                     ) -> Dict:
    """Physical propeller sizing.
    In:  MTOW [N], T/W, n_rotors, DL_target [N/m^2], a [m/s] (Ph0 ISA), M_tip_max
    Out: {'D_prop' [m], 'A_disc' [m^2], 'n_max' [rev/s], 'V_tip' [m/s]}
    Eq:  D=sqrt(4T/(pi·DL)); n_max = M_tip_max·a/(pi·D)
    Ref: Gudmundsson 2022 Ch.14; Hibbs Kitplanes "Prop Blade Mach"
    Loop: MTOW outer; J-feedback inner (Ph4)."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if T_over_W <= 0.0:
        raise ValueError("T_over_W must be positive.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if DL_target <= 0.0:
        raise ValueError("DL_target must be positive.")
    if a_sound <= 0.0:
        raise ValueError("a_sound must be positive.")
    if not 0.0 < M_tip_max < 1.0:
        raise ValueError("M_tip_max should be between 0 and 1 for first-cut sizing.")

    T_total = T_over_W * MTOW_N
    T_per_rotor = T_total / n_rotors
    A_disc = T_per_rotor / DL_target
    D_prop = np.sqrt(4.0 * A_disc / np.pi)
    V_tip = M_tip_max * a_sound
    n_max = V_tip / (np.pi * D_prop)

    warnings = []
    if M_tip_max > 0.72:
        warnings.append(
            "Tip Mach limit is above the conservative 0.72 value used in the legacy Bellona stage-1 model."
        )
    if DL_target > 250.0:
        warnings.append(
            "Disc loading is high for efficient hover; verify against propeller clearance and hover power."
        )

    return {
        "T_total": float(T_total),
        "T_per_rotor": float(T_per_rotor),
        "D_prop": float(D_prop),
        "A_disc": float(A_disc),
        "n_max": float(n_max),
        "rpm_max": float(60.0 * n_max),
        "V_tip": float(V_tip),
        "disc_loading": float(DL_target),
        "notes": [
            "The external balloon payload is not included in MTOW; this phase sizes only UAV lift/thrust.",
            "Propeller diameter is based on target disc loading and must be checked against CAD clearance.",
        ],
        "warnings": warnings,
    }

# ---------- PHASE 2 ----------

def phase2_hover_climb_power(T_per_rotor: float, A_disc: float, rho: float,
                             V_climb: float, FoM: float, eta_motor: float,
                             eta_ESC: float) -> Dict:
    """Actuator-disc induced velocity, hover & axial-climb power.
    In:  T [N], A_disc [m^2], rho [kg/m^3] (Ph0), V_c [m/s], FoM, eta_motor, eta_ESC
    Out: {'v_i', 'P_shaft', 'P_elec'}  [m/s, W, W]
    Eq:  v_i = -V_c/2 + sqrt((V_c/2)^2 + T/(2·rho·A))  — Leishman 2006 Eq. 2.91
         P_shaft = T(V_c + v_i)/FoM                   — Leishman Eq. 2.96
    Loop: MTOW outer."""
    if T_per_rotor <= 0.0:
        raise ValueError("T_per_rotor must be positive.")
    if A_disc <= 0.0:
        raise ValueError("A_disc must be positive.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if V_climb < 0.0:
        raise ValueError("V_climb must be non-negative for this hover/climb model.")
    if not 0.0 < FoM <= 1.0:
        raise ValueError("FoM should be in the range 0 < FoM <= 1.")
    if not 0.0 < eta_motor <= 1.0:
        raise ValueError("eta_motor should be in the range 0 < eta_motor <= 1.")
    if not 0.0 < eta_ESC <= 1.0:
        raise ValueError("eta_ESC should be in the range 0 < eta_ESC <= 1.")

    v_induced = -0.5 * V_climb + np.sqrt(
        (0.5 * V_climb) ** 2 + T_per_rotor / (2.0 * rho * A_disc)
    )
    P_ideal = T_per_rotor * (V_climb + v_induced)
    P_shaft = P_ideal / FoM
    eta_electric = eta_motor * eta_ESC
    P_elec = P_shaft / eta_electric

    return {
        "v_i": float(v_induced),
        "P_ideal": float(P_ideal),
        "P_shaft": float(P_shaft),
        "P_elec": float(P_elec),
        "T_per_rotor": float(T_per_rotor),
        "A_disc": float(A_disc),
        "rho": float(rho),
        "V_climb": float(V_climb),
        "FoM": float(FoM),
        "eta_motor": float(eta_motor),
        "eta_ESC": float(eta_ESC),
        "eta_electric": float(eta_electric),
        "power_loading_elec_W_N": float(P_elec / T_per_rotor),
        "notes": [
            "Power values are per rotor because the input thrust is per rotor.",
            "Use V_climb = 0 for hover; use a positive axial climb rate for vertical climb.",
        ],
        "warnings": [
            "Momentum theory is a first-cut model and does not include blade profile power, nonuniform inflow, installation losses, or descent/vortex-ring effects."
        ],
    }

# ---------- PHASE 3 ----------

def phase3_mission_optimise(MTOW_N: float, S_guess: float, CD0: float, AR: float,
                            e: float, h_target: float, range_target: float,
                            t_budget: float, t_hover: float,
                            P_hover_fn: Callable, isa_fn: Callable = isa,
                            eta_fw: float = 0.75 * 0.90 * 0.95,
                            V_min_required: Optional[float] = None,
                            V_grid=None, h_transition_grid=None
                            ) -> Dict:
    """Derive V_cruise, gamma, ROC, h_transition by minimising total energy.
    In:  MTOW [N], S [m^2] (Ph8 estimate), CD0, AR, e (Ph8 estimate),
         h_target=6000 m, range_target=6000 m, t_budget=600 s, t_hover=300 s,
         P_hover_fn(h)->W (Ph2)
    Out: {'V_cruise', 'gamma', 'ROC', 'h_tr', 'Re_estimate'}
    Eq:  V_Pmin = sqrt(2(W/S)/rho · sqrt(K/(3·CD0)))  — Raymer 2018 Ch.17
         ROC = (P_avail - P_req)/W                     — Raymer Eq. 17.22
    Method: grid-search h_tr in [50, 2000] m, inner 1-D minimise V over P_req(V).
    Loop: MTOW outer; V-J feedback from Ph4."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if S_guess <= 0.0:
        raise ValueError("S_guess must be positive.")
    if CD0 <= 0.0:
        raise ValueError("CD0 must be positive.")
    if AR <= 0.0:
        raise ValueError("AR must be positive.")
    if e <= 0.0:
        raise ValueError("Oswald efficiency e must be positive.")
    if h_target <= 0.0:
        raise ValueError("h_target must be positive.")
    if range_target <= 0.0:
        raise ValueError("range_target must be positive.")
    if t_budget <= 0.0:
        raise ValueError("t_budget must be positive.")
    if t_hover < 0.0:
        raise ValueError("t_hover must be non-negative.")
    if eta_fw <= 0.0 or eta_fw > 1.0:
        raise ValueError("eta_fw should be in the range 0 < eta_fw <= 1.")
    if V_min_required is not None and V_min_required <= 0.0:
        raise ValueError("V_min_required must be positive when provided.")

    def _power_w(value):
        if isinstance(value, dict):
            for key in ("P_total", "P_elec_total", "P_elec"):
                if key in value:
                    return float(value[key])
            raise ValueError("Power dictionaries must contain P_total, P_elec_total, or P_elec.")
        return float(value)

    K = 1.0 / (np.pi * AR * e)
    W_S = MTOW_N / S_guess
    c_guess = np.sqrt(S_guess / AR)
    mission_ROC = h_target / t_budget

    if V_grid is None:
        path_speed = np.hypot(range_target, h_target) / t_budget
        V_min = max(8.0, 0.70 * path_speed)
        V_max = max(45.0, 2.50 * path_speed)
        if V_min_required is not None:
            V_min = max(V_min, V_min_required)
        if V_max <= V_min:
            V_max = 1.25 * V_min
        V_grid = np.linspace(V_min, V_max, 180)
    else:
        V_grid = np.asarray(V_grid, dtype=float)
        if V_min_required is not None:
            V_grid = V_grid[V_grid >= V_min_required]
            if V_grid.size == 0:
                raise ValueError("V_grid has no points above V_min_required.")

    if h_transition_grid is None:
        h_low = min(50.0, 0.25 * h_target)
        h_high = min(2000.0, 0.95 * h_target)
        h_transition_grid = np.linspace(h_low, h_high, 40)
    else:
        h_transition_grid = np.asarray(h_transition_grid, dtype=float)

    P_hover_ground = _power_w(P_hover_fn(0.0))
    P_hover_target = _power_w(P_hover_fn(h_target))
    best = None
    feasible_count = 0

    for h_tr in h_transition_grid:
        if h_tr < 0.0 or h_tr >= h_target:
            continue

        h_fw = h_target - h_tr
        gamma = np.arctan2(h_fw, range_target)
        path_fw = np.hypot(range_target, h_fw)
        h_avg = h_tr + 0.5 * h_fw
        rho_avg, mu_avg, _, _ = isa_fn(h_avg)

        t_transition = h_tr / mission_ROC
        t_fw_available = t_budget - t_transition
        if t_fw_available <= 0.0:
            continue

        P_hover_transition_top = _power_w(P_hover_fn(h_tr))
        P_transition = 0.5 * (P_hover_ground + P_hover_transition_top)
        E_transition_J = P_transition * t_transition

        for V in V_grid:
            if V <= 0.0:
                continue

            t_fw = path_fw / V
            if t_fw > t_fw_available:
                continue

            q = 0.5 * rho_avg * V**2
            CL = MTOW_N / (q * S_guess)
            CD = CD0 + K * CL**2
            drag = q * S_guess * CD
            P_fw = (drag + MTOW_N * np.sin(gamma)) * V / eta_fw
            E_fw_J = P_fw * t_fw
            E_hover_J = P_hover_target * t_hover
            E_total_J = E_transition_J + E_fw_J + E_hover_J
            ROC = V * np.sin(gamma)

            feasible_count += 1
            candidate = {
                "E_total_J": E_total_J,
                "V_cruise": V,
                "gamma": gamma,
                "ROC": ROC,
                "h_tr": h_tr,
                "t_transition": t_transition,
                "t_fw": t_fw,
                "P_fw": P_fw,
                "P_transition": P_transition,
                "P_hover_target": P_hover_target,
                "E_transition_J": E_transition_J,
                "E_fw_J": E_fw_J,
                "E_hover_J": E_hover_J,
                "rho_avg": rho_avg,
                "mu_avg": mu_avg,
                "CL": CL,
                "CD": CD,
                "drag": drag,
                "path_fw": path_fw,
            }
            if best is None or candidate["E_total_J"] < best["E_total_J"]:
                best = candidate

    if best is None:
        raise ValueError(
            "No feasible Phase 3 mission candidate found. Check the speed grid, transition grid, and time budget."
        )

    rho_best = best["rho_avg"]
    mu_best = best["mu_avg"]
    V_power_min = np.sqrt((2.0 * W_S / rho_best) * np.sqrt(K / (3.0 * CD0)))
    Re_estimate = rho_best * best["V_cruise"] * c_guess / mu_best
    t_total_outbound = best["t_transition"] + best["t_fw"]
    notes = [
        "The speed and transition-altitude grids are intentionally simple so the sizing trend is inspectable.",
        "P_hover_fn should return total aircraft electric power in W if the energy terms are compared directly.",
        "The 10 m/s default mission climb reference comes from h_target / t_budget for the 6000 m in 600 s mission.",
    ]
    if V_min_required is not None:
        notes.append(
            "The cruise-speed grid lower bound was raised to clear the transition blend-end speed plus margin."
        )

    return {
        "V_cruise": float(best["V_cruise"]),
        "V_min_required": None if V_min_required is None else float(V_min_required),
        "V_min_requirement_active": bool(
            V_min_required is not None and best["V_cruise"] >= V_min_required
        ),
        "gamma": float(best["gamma"]),
        "gamma_deg": float(np.rad2deg(best["gamma"])),
        "ROC": float(best["ROC"]),
        "h_tr": float(best["h_tr"]),
        "Re_estimate": float(Re_estimate),
        "V_power_min": float(V_power_min),
        "t_transition": float(best["t_transition"]),
        "t_fw": float(best["t_fw"]),
        "t_total_outbound": float(t_total_outbound),
        "time_margin": float(t_budget - t_total_outbound),
        "E_total_Wh": float(best["E_total_J"] / 3600.0),
        "E_transition_Wh": float(best["E_transition_J"] / 3600.0),
        "E_fw_Wh": float(best["E_fw_J"] / 3600.0),
        "E_hover_Wh": float(best["E_hover_J"] / 3600.0),
        "P_fw": float(best["P_fw"]),
        "P_transition": float(best["P_transition"]),
        "P_hover_target": float(best["P_hover_target"]),
        "CL_cruise": float(best["CL"]),
        "CD_cruise": float(best["CD"]),
        "drag_N": float(best["drag"]),
        "path_fw_m": float(best["path_fw"]),
        "mission_ROC_reference": float(mission_ROC),
        "eta_fw": float(eta_fw),
        "feasible_candidates": int(feasible_count),
        "notes": notes,
        "warnings": [
            "Transition energy uses hover-like power and a mission-average vertical-rate time estimate; replace with a tail-sitter transition simulation or test data.",
            "Fixed-wing power uses a parabolic drag polar and a single average climb altitude; verify with a detailed mission simulation.",
        ],
    }

# ---------- PHASE 4 ----------

def phase4_J_coupling(V_cruise: float, n_rev_s: float, D_prop: float,
                      J_min: float = 0.4, J_max: float = 0.8) -> Dict:
    """Verify advance ratio in efficient band; signal D-revise if not.
    In:  V_cruise [m/s] (Ph3), n [rev/s] (Ph1), D [m] (Ph1)
    Out: {'J', 'in_band' (bool), 'D_recommend'}
    Ref: Gudmundsson 2022 Ch.14 §14.6; UIUC propeller DB.
    Loop: V_cruise–J inner loop with Ph3 (or Ph1 D-revise)."""
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if n_rev_s <= 0.0:
        raise ValueError("n_rev_s must be positive.")
    if D_prop <= 0.0:
        raise ValueError("D_prop must be positive.")
    if J_min <= 0.0:
        raise ValueError("J_min must be positive.")
    if J_max <= J_min:
        raise ValueError("J_max must be larger than J_min.")

    J = V_cruise / (n_rev_s * D_prop)
    D_for_J_max = V_cruise / (n_rev_s * J_max)
    D_for_J_min = V_cruise / (n_rev_s * J_min)
    n_for_J_max = V_cruise / (J_max * D_prop)
    n_for_J_min = V_cruise / (J_min * D_prop)

    if J < J_min:
        in_band = False
        D_recommend = D_for_J_min
        n_recommend = n_for_J_min
        recommendation = "Increase advance ratio by reducing diameter, reducing RPM, or increasing cruise speed."
    elif J > J_max:
        in_band = False
        D_recommend = D_for_J_max
        n_recommend = n_for_J_max
        recommendation = "Decrease advance ratio by increasing diameter, increasing RPM, or reducing cruise speed."
    else:
        in_band = True
        D_recommend = D_prop
        n_recommend = n_rev_s
        recommendation = "Advance ratio is within the selected first-cut band."

    warnings = []
    if not in_band:
        warnings.append(
            "Advance ratio is outside the selected band; check propeller diameter, cruise RPM, and Phase 3 cruise speed together."
        )

    return {
        "J": float(J),
        "in_band": bool(in_band),
        "D_recommend": float(D_recommend),
        "n_recommend": float(n_recommend),
        "rpm_current": float(60.0 * n_rev_s),
        "rpm_recommend": float(60.0 * n_recommend),
        "D_band_min": float(D_for_J_max),
        "D_band_max": float(D_for_J_min),
        "n_band_min": float(n_for_J_max),
        "n_band_max": float(n_for_J_min),
        "J_min": float(J_min),
        "J_max": float(J_max),
        "recommendation": recommendation,
        "notes": [
            "Use cruise shaft speed here. Phase 1 n_max is only a tip-Mach limit and may overstate actual cruise RPM.",
            "D_recommend holds cruise speed and shaft speed fixed, so it is a coupling signal rather than a final propeller selection.",
        ],
        "warnings": warnings,
    }

# ---------- PHASE 5 ----------

def phase5_energy_battery(segments: Dict[str, Tuple[float, float]],
                          eta_batt: float = 0.95, f_usable: float = 0.85,
                          e_batt_Wh_kg: float = 200.0) -> Dict:
    """Mission energy timeline, battery mass.
    In:  segments={'name': (P[W], dt[s])} — VTOL climb, transition, FW climb,
         hover, tow, return cruise, descent, reserve.  (Ph2, Ph3, Ph13)
    Out: {'E_total_Wh', 'm_batt_kg', 'P_motor_cont', 'P_motor_peak'}
    Eq:  E = sum(P·dt)/(eta_batt·f_usable)  — Gundlach 2014 Eq. 8.4
    Loop: MTOW outer."""
    if not segments:
        raise ValueError("segments must contain at least one mission segment.")
    if not 0.0 < eta_batt <= 1.0:
        raise ValueError("eta_batt should be in the range 0 < eta_batt <= 1.")
    if not 0.0 < f_usable <= 1.0:
        raise ValueError("f_usable should be in the range 0 < f_usable <= 1.")
    if e_batt_Wh_kg <= 0.0:
        raise ValueError("e_batt_Wh_kg must be positive.")

    segment_breakdown = {}
    E_load_Wh = 0.0
    P_values = []
    total_time_s = 0.0

    for name, values in segments.items():
        if len(values) != 2:
            raise ValueError(f"Segment '{name}' must be a (P_W, dt_s) pair.")
        P_W, dt_s = values
        P_W = float(P_W)
        dt_s = float(dt_s)
        if P_W < 0.0:
            raise ValueError(f"Segment '{name}' has negative power; regeneration is not modeled.")
        if dt_s < 0.0:
            raise ValueError(f"Segment '{name}' has negative duration.")

        E_segment_Wh = P_W * dt_s / 3600.0
        E_load_Wh += E_segment_Wh
        total_time_s += dt_s
        P_values.append(P_W)
        segment_breakdown[name] = {
            "P_W": P_W,
            "dt_s": dt_s,
            "E_Wh": float(E_segment_Wh),
        }

    E_total_Wh = E_load_Wh / (eta_batt * f_usable)
    m_batt_kg = E_total_Wh / e_batt_Wh_kg
    P_peak = max(P_values)
    P_average_load = 0.0
    if total_time_s > 0.0:
        P_average_load = E_load_Wh * 3600.0 / total_time_s

    warnings = []
    segment_names = " ".join(str(name).lower() for name in segments)
    if "reserve" not in segment_names:
        warnings.append(
            "No explicit reserve segment was found; only usable-capacity margin is applied."
        )
    if e_batt_Wh_kg > 265.0:
        warnings.append(
            "Battery specific energy is optimistic for high-power UAV packs; verify against pack-level datasheets."
        )

    return {
        "segments": segment_breakdown,
        "E_load_Wh": float(E_load_Wh),
        "E_total_Wh": float(E_total_Wh),
        "E_total_kWh": float(E_total_Wh / 1000.0),
        "m_batt_kg": float(m_batt_kg),
        "P_motor_cont": float(P_peak),
        "P_motor_peak": float(P_peak),
        "P_average_load": float(P_average_load),
        "eta_batt": float(eta_batt),
        "f_usable": float(f_usable),
        "reserve_fraction": float(1.0 - f_usable),
        "e_batt_Wh_kg": float(e_batt_Wh_kg),
        "notes": [
            "Segment powers are total aircraft electrical powers in W.",
            "E_load_Wh is the mission energy before battery-discharge efficiency and usable-capacity margin.",
            "E_total_Wh is the installed pack energy required by this first-cut model.",
            "Continuous and peak motor powers are both set to the maximum segment power until short-duration transient limits are modeled.",
        ],
        "warnings": warnings,
    }

# ---------- PHASE 6 ----------

def phase6_constraint_diagram(W_S_range: np.ndarray, W_P_range: np.ndarray,
                              V_cruise, V_stall, ROC, gamma, rho_SL, rho_6km,
                              CL_max, CD0, AR, e, eta_prop, T_W_floor,
                              plot_path=None) -> Dict:
    """Plot W/S vs W/P with stall, cruise, climb, hover, transition-floor lines.
    Verification step (post Ph3, Ph5, Ph8). Not a sizing driver.
    Ref: Raymer 2018 Ch.5 §5.3."""
    W_S = np.asarray(W_S_range, dtype=float)
    if W_S.ndim != 1 or W_S.size == 0:
        raise ValueError("W_S_range must be a non-empty 1-D array.")
    if np.any(W_S <= 0.0):
        raise ValueError("W_S_range values must be positive.")
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if ROC < 0.0:
        raise ValueError("ROC must be non-negative.")
    if rho_SL <= 0.0 or rho_6km <= 0.0:
        raise ValueError("Atmospheric densities must be positive.")
    if CL_max <= 0.0:
        raise ValueError("CL_max must be positive.")
    if CD0 <= 0.0:
        raise ValueError("CD0 must be positive.")
    if AR <= 0.0 or e <= 0.0:
        raise ValueError("AR and e must be positive.")
    if eta_prop <= 0.0 or eta_prop > 1.0:
        raise ValueError("eta_prop should be in the range 0 < eta_prop <= 1.")
    if T_W_floor <= 0.0:
        raise ValueError("T_W_floor must be positive.")

    q = 0.5 * rho_6km * V_cruise**2
    CL = W_S / q
    K = 1.0 / (np.pi * AR * e)
    CD = CD0 + K * CL**2
    P_W_cruise = V_cruise * (CD / CL) / eta_prop

    climb_rate_from_gamma = V_cruise * np.sin(gamma)
    climb_rate_used = ROC
    if climb_rate_used == 0.0 and climb_rate_from_gamma > 0.0:
        climb_rate_used = climb_rate_from_gamma
    P_W_climb = P_W_cruise + climb_rate_used / eta_prop

    W_S_stall_max = 0.5 * rho_SL * V_stall**2 * CL_max
    P_W_envelope = np.maximum(P_W_cruise, P_W_climb)

    with np.errstate(divide="ignore", invalid="ignore"):
        W_P_cruise = 1.0 / P_W_cruise
        W_P_climb = 1.0 / P_W_climb
        W_P_envelope = 1.0 / P_W_envelope

    warnings = [
        "Hover power cannot be plotted from this Phase 6 signature alone; pass Phase 2/5 hover power into a later integrated plot if needed.",
        "T_W_floor is reported as a thrust margin check, not converted to a power-loading line.",
    ]
    if abs(climb_rate_from_gamma - ROC) > max(0.5, 0.1 * max(ROC, 1.0)):
        warnings.append(
            "ROC and V_cruise*sin(gamma) differ noticeably; check Phase 3 consistency."
        )
    if np.any(W_S > W_S_stall_max):
        warnings.append(
            "Part of W_S_range is above the stall wing-loading limit."
        )

    saved_plot = None
    if plot_path:
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        ax.plot(W_S, P_W_cruise, color="#378ADD", lw=2.0, label="Cruise")
        ax.plot(W_S, P_W_climb, color="#1D9E75", lw=2.0, label=f"Climb ROC={climb_rate_used:.1f} m/s")
        ax.axvline(W_S_stall_max, color="#888780", lw=1.4, ls="--", label="Stall limit")
        ax.fill_between(W_S, 0.0, P_W_envelope, where=(W_S <= W_S_stall_max),
                        color="#1D9E75", alpha=0.08)
        ax.set_xlabel("Wing loading W/S [N/m^2]")
        ax.set_ylabel("Power-to-weight P/W [W/N]")
        ax.set_title("Phase 6 Constraint Diagram")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_range = np.asarray(W_P_range, dtype=float)
        if y_range.size >= 2 and np.all(y_range > 0.0):
            ax.set_ylim(float(np.min(y_range)), float(np.max(y_range)))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        saved_plot = str(plot_path)

    return {
        "W_S_range": W_S.tolist(),
        "P_W_cruise": P_W_cruise.tolist(),
        "P_W_climb": P_W_climb.tolist(),
        "P_W_envelope": P_W_envelope.tolist(),
        "W_P_cruise": W_P_cruise.tolist(),
        "W_P_climb": W_P_climb.tolist(),
        "W_P_envelope": W_P_envelope.tolist(),
        "W_S_stall_max": float(W_S_stall_max),
        "climb_rate_used": float(climb_rate_used),
        "climb_rate_from_gamma": float(climb_rate_from_gamma),
        "T_W_floor": float(T_W_floor),
        "plot_path": saved_plot,
        "notes": [
            "The plotted convention is P/W in W/N, matching the legacy Bellona stage-1 script.",
            "Inverse W/P arrays are returned for users who prefer the classical power-loading axis.",
            "Phase 6 is verification-only and does not alter the sizing loop.",
        ],
        "warnings": warnings,
    }

# ---------- EARLY INNER LOOP ----------

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


def iterate_phases_1_to_6(MTOW_kg: float = 50.0,
                          mission: Optional[Mission] = None,
                          assumptions: Optional[Assumptions] = None,
                          max_inner_iter: int = 8,
                          j_tol: float = 0.01,
                          constraint_plot_path=None) -> Dict:
    """Run the reviewed Phase 1-6 coupling without full MTOW convergence."""
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()
    if MTOW_kg <= 0.0:
        raise ValueError("MTOW_kg must be positive.")
    if max_inner_iter <= 0:
        raise ValueError("max_inner_iter must be positive.")
    if assumptions.cruise_J_min <= 0.0:
        raise ValueError("cruise_J_min must be positive.")
    if assumptions.cruise_J_max <= assumptions.cruise_J_min:
        raise ValueError("cruise_J_max must be larger than cruise_J_min.")
    if not assumptions.cruise_J_min <= assumptions.cruise_J_target <= assumptions.cruise_J_max:
        raise ValueError("cruise_J_target must lie inside the selected J band.")

    MTOW_N = MTOW_kg * assumptions.g
    eta_fw = assumptions.eta_prop * assumptions.eta_motor * assumptions.eta_ESC
    rho_target, _, a_sound_target, _ = isa(mission.altitude_m)
    rho_SL, _, _, _ = isa(0.0)
    cruise_n_rev_s = None
    history = []
    converged = False
    stopped_reason = "maximum iterations reached"

    phase1 = phase2_hover = phase3 = phase4 = None

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
            assumptions.preliminary_wing_area_m2,
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

        phase4 = phase4_J_coupling(
            phase3["V_cruise"],
            cruise_n_current,
            phase1["D_prop"],
            assumptions.cruise_J_min,
            assumptions.cruise_J_max,
        )

        n_target = phase3["V_cruise"] / (assumptions.cruise_J_target * phase1["D_prop"])
        tip_limited = False
        if n_target > phase1["n_max"]:
            n_next = phase1["n_max"]
            tip_limited = True
        else:
            n_next = n_target

        relative_n_change = abs(n_next - cruise_n_current) / cruise_n_current
        j_error = abs(phase4["J"] - assumptions.cruise_J_target)
        history.append({
            "iteration": it + 1,
            "V_cruise": float(phase3["V_cruise"]),
            "D_prop": float(phase1["D_prop"]),
            "disc_loading": float(phase1["disc_loading"]),
            "cruise_n_current": float(cruise_n_current),
            "cruise_rpm_current": float(60.0 * cruise_n_current),
            "J_current": float(phase4["J"]),
            "cruise_n_next": float(n_next),
            "cruise_rpm_next": float(60.0 * n_next),
            "relative_n_change": float(relative_n_change),
            "tip_limited": bool(tip_limited),
        })

        cruise_n_rev_s = n_next
        if phase4["in_band"] and j_error <= j_tol:
            converged = True
            stopped_reason = "advance-ratio target reached"
            break
        if tip_limited and relative_n_change < 1e-6:
            stopped_reason = "tip-Mach RPM limit prevents reaching the target advance ratio"
            break

    phase4 = phase4_J_coupling(
        phase3["V_cruise"],
        cruise_n_rev_s,
        phase1["D_prop"],
        assumptions.cruise_J_min,
        assumptions.cruise_J_max,
    )

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

    W_S_design = MTOW_N / assumptions.preliminary_wing_area_m2
    V_stall_guess = np.sqrt(2.0 * W_S_design / (rho_SL * assumptions.CL_max_guess))
    W_S_range = np.linspace(0.35 * W_S_design, 1.80 * W_S_design, 90)
    phase6 = phase6_constraint_diagram(
        W_S_range,
        np.array([]),
        phase3["V_cruise"],
        V_stall_guess,
        phase3["ROC"],
        phase3["gamma"],
        rho_SL,
        rho_target,
        assumptions.CL_max_guess,
        assumptions.CD0,
        assumptions.AR,
        assumptions.oswald_e,
        eta_fw,
        assumptions.thrust_to_weight,
        plot_path=constraint_plot_path,
    )

    warnings = []
    if mission.external_tow_load_N:
        warnings.append(
            "external_tow_load_N is not included in UAV MTOW or Phase 5 energy in this early loop."
        )
    if not phase4["in_band"]:
        warnings.append(
            "Final cruise advance ratio remains outside the selected band."
        )

    return {
        "MTOW_kg": float(MTOW_kg),
        "MTOW_N": float(MTOW_N),
        "converged": bool(converged),
        "stopped_reason": stopped_reason,
        "history": history,
        "cruise_n_rev_s": float(cruise_n_rev_s),
        "cruise_rpm": float(60.0 * cruise_n_rev_s),
        "phase1": phase1,
        "phase2_hover": phase2_hover,
        "phase3": phase3,
        "phase4": phase4,
        "phase5": phase5,
        "phase6": phase6,
        "notes": [
            "This is an inner coupling loop for Phases 1-6 only.",
            "It adjusts cruise RPM to meet the target advance ratio before the MTOW loop exists.",
            "Battery mass from Phase 5 is reported but does not feed back into MTOW until Phases 15-16.",
        ],
        "warnings": warnings,
    }


def iterate_phases_1_to_8(MTOW_kg: float = 50.0,
                          mission: Optional[Mission] = None,
                          assumptions: Optional[Assumptions] = None,
                          max_inner_iter: int = 12,
                          j_tol: float = 0.01,
                          area_tol: float = 0.01,
                          constraint_plot_path=None,
                          airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run the pre-canard prop/RPM/Re/wing-area coupling before MTOW convergence."""
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
    if assumptions.preliminary_wing_area_m2 <= 0.0:
        raise ValueError("preliminary_wing_area_m2 must be positive.")

    MTOW_N = MTOW_kg * assumptions.g
    eta_fw = assumptions.eta_prop * assumptions.eta_motor * assumptions.eta_ESC
    rho_target, _, a_sound_target, _ = isa(mission.altitude_m)
    S_guess = assumptions.preliminary_wing_area_m2
    cruise_n_rev_s = None
    history = []
    converged = False
    stopped_reason = "maximum iterations reached"

    phase1 = phase2_hover = phase3 = phase4 = phase5 = phase6 = phase7 = phase8 = None

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

        phase4_current = phase4_J_coupling(
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

        phase7 = phase7_airfoil_xfoil(
            phase3["Re_estimate"],
            phase3["Re_estimate"],
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

        area_change = abs(phase8["S"] - S_guess) / S_guess
        j_error = abs(phase4["J"] - assumptions.cruise_J_target)
        history.append({
            "iteration": it + 1,
            "S_guess": float(S_guess),
            "S_new": float(phase8["S"]),
            "area_change": float(area_change),
            "Re_main": float(phase3["Re_estimate"]),
            "V_cruise": float(phase3["V_cruise"]),
            "D_prop": float(phase1["D_prop"]),
            "disc_loading": float(phase1["disc_loading"]),
            "J_before_update": float(phase4_current["J"]),
            "J_after_update": float(phase4["J"]),
            "cruise_rpm": float(60.0 * cruise_n_next),
            "tip_limited": bool(tip_limited),
            "CL_max_3D": float(phase8["CL_max_3D"]),
            "V_stall": float(phase8["V_stall"]),
        })

        S_guess = phase8["S"]
        cruise_n_rev_s = cruise_n_next
        if area_change <= area_tol and phase4["in_band"] and j_error <= j_tol:
            converged = True
            stopped_reason = "wing area and advance-ratio coupling converged"
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
    if mission.external_tow_load_N:
        warnings.append(
            "external_tow_load_N is not included in UAV MTOW or Phase 5 energy in this early loop."
        )
    if not converged:
        warnings.append(
            "Phase 1-8 inner loop did not meet the requested tolerances before stopping."
        )

    return {
        "MTOW_kg": float(MTOW_kg),
        "MTOW_N": float(MTOW_N),
        "converged": bool(converged),
        "stopped_reason": stopped_reason,
        "history": history,
        "cruise_n_rev_s": float(cruise_n_rev_s),
        "cruise_rpm": float(60.0 * cruise_n_rev_s),
        "phase1": phase1,
        "phase2_hover": phase2_hover,
        "phase3": phase3,
        "phase4": phase4,
        "phase5": phase5,
        "phase6": phase6,
        "phase7": phase7,
        "phase8": phase8,
        "notes": [
            "This pre-canard loop replaces the preliminary S_guess in Phase 3 with the Phase 8 wing area.",
            "Use iterate_phases_1_to_9 for the complete propulsion/aerodynamics loop including canard Reynolds number.",
            "MTOW is still fixed; battery and structure mass feedback waits for Phases 15-16.",
        ],
        "warnings": warnings,
    }

# ---------- AIRFOIL HELPERS ----------

def _parse_xfoil_polar(polar_path: Path) -> Dict[str, np.ndarray]:
    """Read the numeric part of an XFOIL polar file."""
    rows = []
    with open(polar_path, "r", encoding="utf-8", errors="ignore") as polar_file:
        for line in polar_file:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                values = [float(value) for value in parts[:5]]
            except ValueError:
                continue
            rows.append(values)

    if not rows:
        raise ValueError("No numeric polar rows were found.")

    data = np.asarray(rows, dtype=float)
    return {
        "alpha_deg": data[:, 0],
        "cl": data[:, 1],
        "cd": data[:, 2],
        "cdp": data[:, 3],
        "cm": data[:, 4],
    }


def _summarise_airfoil_polar(polar: Dict[str, np.ndarray],
                             airfoil: str,
                             reynolds: float,
                             x_transition: float,
                             design_cl: float,
                             source: str) -> Dict:
    """Convert XFOIL polar arrays into the compact Phase 7 output shape."""
    alpha = polar["alpha_deg"]
    cl = polar["cl"]
    cd = polar["cd"]
    cm = polar["cm"]

    finite = np.isfinite(alpha) & np.isfinite(cl) & np.isfinite(cd) & np.isfinite(cm) & (cd > 0.0)
    alpha = alpha[finite]
    cl = cl[finite]
    cd = cd[finite]
    cm = cm[finite]
    if alpha.size < 3:
        raise ValueError("Too few converged polar points to summarize.")

    linear_mask = (alpha >= -2.0) & (alpha <= 6.0)
    if np.count_nonzero(linear_mask) < 3:
        linear_mask = np.ones_like(alpha, dtype=bool)
    slope_per_deg, intercept = np.polyfit(alpha[linear_mask], cl[linear_mask], 1)
    cl_a = slope_per_deg * 180.0 / np.pi

    idx_cl0 = int(np.argmin(np.abs(cl)))
    idx_design = int(np.argmin(np.abs(cl - design_cl)))
    idx_clmax = int(np.argmax(cl))
    cl_cd = np.where(cd > 0.0, cl / cd, np.nan)
    idx_best_ld = int(np.nanargmax(cl_cd))

    stall_char = "xfoil polar; inspect post-stall behavior before final use"
    if idx_clmax < alpha.size - 2 and cl[idx_clmax + 1] < cl[idx_clmax] - 0.05:
        stall_char = "xfoil polar suggests a defined peak; verify stall convergence"

    return {
        "airfoil": airfoil,
        "Re": float(reynolds),
        "x_transition": float(x_transition),
        "cl_a": float(cl_a),
        "cl_alpha_per_rad": float(cl_a),
        "cl_max": float(cl[idx_clmax]),
        "alpha_stall_deg": float(alpha[idx_clmax]),
        "cd0": float(cd[idx_cl0]),
        "cd_at_design": float(cd[idx_design]),
        "cl_design": float(design_cl),
        "cl_cd_at_design": float(cl[idx_design] / cd[idx_design]),
        "cl_cd_max": float(cl_cd[idx_best_ld]),
        "cm0": float(cm[idx_cl0]),
        "cm_at_design": float(cm[idx_design]),
        "stall_char": stall_char,
        "source": source,
        "polar_points": int(alpha.size),
    }


def _run_xfoil_airfoil(xfoil_exe: str,
                       airfoil: str,
                       reynolds: float,
                       x_transition: float,
                       design_cl: float,
                       coordinate_file: Optional[str] = None,
                       alpha_start: float = -4.0,
                       alpha_end: float = 14.0,
                       alpha_step: float = 1.0,
                       timeout_s: float = 30.0) -> Tuple[Optional[Dict], Optional[str]]:
    """Run XFOIL for one airfoil and return (result, warning)."""
    if reynolds <= 0.0:
        raise ValueError("reynolds must be positive.")

    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            polar_path = tmp_dir / "polar.txt"

            if coordinate_file:
                source_path = Path(coordinate_file)
                if not source_path.exists():
                    return None, f"Coordinate file not found for {airfoil}: {coordinate_file}"
                local_coord = tmp_dir / source_path.name
                shutil.copy2(source_path, local_coord)
                airfoil_command = f"LOAD {local_coord.name}"
            elif airfoil.lower().replace(" ", "") == "naca0012":
                airfoil_command = "NACA 0012"
            else:
                return None, f"No coordinate file was provided for {airfoil}."

            commands = "\n".join([
                "PLOP",
                "G F",
                "",
                airfoil_command,
                "PANE",
                "OPER",
                f"VISC {reynolds:.0f}",
                "ITER 150",
                "VPAR",
                f"XTR {x_transition:.3f} {x_transition:.3f}",
                "",
                "PACC",
                str(polar_path),
                "",
                f"ASEQ {alpha_start:.2f} {alpha_end:.2f} {alpha_step:.2f}",
                "PACC",
                "",
                "QUIT",
                "",
            ])

            completed = subprocess.run(
                [xfoil_exe],
                input=commands,
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                timeout=timeout_s,
            )
            if not polar_path.exists():
                tail = completed.stderr.strip() or completed.stdout.strip()[-300:]
                return None, f"XFOIL did not produce a polar for {airfoil}. {tail}"

            polar = _parse_xfoil_polar(polar_path)
            return _summarise_airfoil_polar(
                polar,
                airfoil,
                reynolds,
                x_transition,
                design_cl,
                "xfoil",
            ), None
    except subprocess.TimeoutExpired:
        return None, f"XFOIL timed out while analyzing {airfoil}."
    except OSError as exc:
        return None, f"XFOIL could not be started for {airfoil}: {exc}"
    except ValueError as exc:
        return None, f"XFOIL polar could not be parsed for {airfoil}: {exc}"


def _fallback_airfoil_result(role: str, reynolds: float, x_transition: float) -> Dict:
    """Project-table fallback values for SD7037 and NACA 0012."""
    reference_re = 788199.3377311177
    if role == "main":
        data = {
            "airfoil": "SD7037",
            "cl_design": 0.60,
            "cl_max": 1.4257,
            "cd_at_design": 0.00724,
            "cl_cd_at_design": 81.51933701657458,
            "cm0": -0.07579230769230769,
            "cm_at_design": -0.0760173144876325,
            "alpha_stall_deg": 12.0,
            "cl_a": 5.90,
            "stall_char": "low-Re airfoil; verify stall softness with measured or XFOIL polar data",
        }
    elif role == "canard":
        data = {
            "airfoil": "NACA 0012",
            "cl_design": 0.30,
            "cl_max": 1.3177,
            "cd_at_design": 0.00775,
            "cl_cd_at_design": 37.2,
            "cm0": 0.0,
            "cm_at_design": -0.0005575317604355717,
            "alpha_stall_deg": 12.0,
            "cl_a": 6.10,
            "stall_char": "symmetric reference section; verify control-surface performance with hinge and deflection data",
        }
    else:
        raise ValueError("role must be 'main' or 'canard'.")

    reynolds_ratio = reynolds / reference_re
    warnings = []
    if reynolds_ratio < 0.75 or reynolds_ratio > 1.25:
        warnings.append(
            "Requested Reynolds number differs substantially from the project-table reference value."
        )

    result = {
        "airfoil": data["airfoil"],
        "Re": float(reynolds),
        "Re_reference": float(reference_re),
        "x_transition": float(x_transition),
        "cl_a": float(data["cl_a"]),
        "cl_alpha_per_rad": float(data["cl_a"]),
        "cl_max": float(data["cl_max"]),
        "alpha_stall_deg": float(data["alpha_stall_deg"]),
        "cd0": float(data["cd_at_design"]),
        "cd_at_design": float(data["cd_at_design"]),
        "cl_design": float(data["cl_design"]),
        "cl_cd_at_design": float(data["cl_cd_at_design"]),
        "cl_cd_max": float(data["cl_cd_at_design"]),
        "cm0": float(data["cm0"]),
        "cm_at_design": float(data["cm_at_design"]),
        "stall_char": data["stall_char"],
        "source": "fallback_project_xfoil_table",
        "warnings": warnings,
    }
    return result

# ---------- PHASE 7 ----------

def phase7_airfoil_xfoil(Re_main: float, Re_canard: float,
                         x_transition: float = 0.50,
                         use_xfoil: bool = False,
                         xfoil_path: Optional[str] = None,
                         airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """XFOIL wrapper: evaluate SD7037 + NACA 0012 + candidates.
    In:  Re_main, Re_canard (Ph3+Ph8 estimate), forced transition x/c (Selig 2003).
    Out: {'main': {'cl_a', 'cl_max', 'cd0', 'cm0', 'stall_char'},
          'canard': {...}}
    Ref: Drela XFOIL User Guide; Selig 2003 VKI LRN lecture notes.
         SD7037 Cl/Cd_max ≈ 75 @ Re=2e5, ≈ 99 @ Re=5e5 (UIUC database).
    Loop: Re inner loop with Ph8."""
    if Re_main <= 0.0:
        raise ValueError("Re_main must be positive.")
    if Re_canard <= 0.0:
        raise ValueError("Re_canard must be positive.")
    if not 0.0 < x_transition < 1.0:
        raise ValueError("x_transition should be between 0 and 1.")

    airfoil_files = {
        str(key): str(value)
        for key, value in (airfoil_files or {}).items()
    }
    warnings = []
    notes = [
        "Fallback values are anchored to the project XFOIL comparison table at Re=7.88e5 and x/c=0.5.",
        "The fallback cd0 value is a profile-drag proxy taken from cd at the design lift coefficient.",
        "XFOIL near stall can be convergence-sensitive; measured polars or wind-tunnel data should replace these values when available.",
    ]

    xfoil_exe = None
    if xfoil_path:
        xfoil_exe = xfoil_path
        use_xfoil = True
    elif use_xfoil:
        xfoil_exe = shutil.which("xfoil") or shutil.which("xfoil.exe")

    xfoil_used = False
    main = None
    canard = None

    if use_xfoil and xfoil_exe:
        main, warning = _run_xfoil_airfoil(
            xfoil_exe,
            "SD7037",
            Re_main,
            x_transition,
            0.60,
            coordinate_file=airfoil_files.get("main") or airfoil_files.get("sd7037"),
        )
        if warning:
            warnings.append(warning)

        canard, warning = _run_xfoil_airfoil(
            xfoil_exe,
            "NACA 0012",
            Re_canard,
            x_transition,
            0.30,
            coordinate_file=airfoil_files.get("canard") or airfoil_files.get("naca0012"),
        )
        if warning:
            warnings.append(warning)

        xfoil_used = main is not None or canard is not None
    elif use_xfoil and not xfoil_exe:
        warnings.append("XFOIL was requested, but no executable was found on PATH and no xfoil_path was provided.")

    if main is None:
        main = _fallback_airfoil_result("main", Re_main, x_transition)
        warnings.extend(f"main: {warning}" for warning in main.get("warnings", []))
    if canard is None:
        canard = _fallback_airfoil_result("canard", Re_canard, x_transition)
        warnings.extend(f"canard: {warning}" for warning in canard.get("warnings", []))

    if abs(x_transition - 0.50) > 0.05 and (main["source"].startswith("fallback") or canard["source"].startswith("fallback")):
        warnings.append(
            "Fallback values were generated for x/c=0.5 forced transition; rerun XFOIL for other transition locations."
        )

    return {
        "main": main,
        "canard": canard,
        "xfoil": {
            "requested": bool(use_xfoil),
            "used": bool(xfoil_used),
            "executable": xfoil_exe,
            "airfoil_files": dict(airfoil_files),
        },
        "notes": notes,
        "warnings": warnings,
    }

# ---------- PLANFORM HELPERS ----------

def _half_chord_sweep_from_quarter_chord(sweep_c4: float, AR: float, taper: float) -> float:
    """Convert quarter-chord sweep to half-chord sweep for a straight tapered wing."""
    tan_half = np.tan(sweep_c4) - (1.0 / AR) * (1.0 - taper) / (1.0 + taper)
    return float(np.arctan(tan_half))


def _datcom_lift_curve_slope(AR: float, sweep_c4: float, taper: float,
                             mach: float = 0.0, k: float = 1.0) -> float:
    """DATCOM/Polhamus finite-wing lift-curve slope in 1/rad."""
    if AR <= 0.0:
        raise ValueError("AR must be positive.")
    if not 0.0 <= mach < 1.0:
        raise ValueError("mach must be subsonic and non-negative.")
    if k <= 0.0:
        raise ValueError("k must be positive.")
    beta_sq = 1.0 - mach**2
    sweep_c2 = _half_chord_sweep_from_quarter_chord(sweep_c4, AR, taper)
    term = (AR**2 * beta_sq / k**2) * (1.0 + np.tan(sweep_c2) ** 2 / beta_sq) + 4.0
    return float(2.0 * np.pi * AR / (2.0 + np.sqrt(term)))

# ---------- PHASE 8 ----------

def phase8_wing_planform(MTOW_N: float, cl_max_2D: float, cl_a_2D: float,
                         rho_SL: float, V_stall_target_max: float,
                         AR_guess: float = 7.0, taper: float = 0.4,
                         sweep_c4: float = 0.0, e: float = 0.78,
                         stall_margin: float = 0.80,
                         mach: float = 0.0) -> Dict:
    """Wing planform; DERIVE V_stall from CL_max,3D and chosen W/S.
    In:  MTOW [N], cl_max,2D, cl_a,2D (Ph7), rho_SL,
         V_stall_target_max [m/s] (handling-quality + Stone transition floor),
         AR, taper, sweep (design choices).
    Out: {'S', 'b', 'c_bar', 'AR', 'CL_a', 'CL_max_3D', 'V_stall', 'x_ac_w'}
    Eq:  CL_a = 2*pi*AR/(2 + sqrt(AR^2(1-M^2)(1+tan^2 Lambda_c2/(1-M^2))/k^2+4))
                                                  — Polhamus / DATCOM §4.1.3.2
         CL_max_3D = 0.9·cl_max·cos(Lambda_c4)    — Raymer 2018 Eq. 12.16
         V_stall  = sqrt(2(W/S)/(rho_SL·CL_max_3D))
    Loop: Re inner (Ph7); MTOW outer."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if cl_max_2D <= 0.0:
        raise ValueError("cl_max_2D must be positive.")
    if cl_a_2D <= 0.0:
        raise ValueError("cl_a_2D must be positive.")
    if rho_SL <= 0.0:
        raise ValueError("rho_SL must be positive.")
    if V_stall_target_max <= 0.0:
        raise ValueError("V_stall_target_max must be positive.")
    if AR_guess <= 0.0:
        raise ValueError("AR_guess must be positive.")
    if not 0.0 < taper <= 1.0:
        raise ValueError("taper should be in the range 0 < taper <= 1.")
    if e <= 0.0:
        raise ValueError("Oswald efficiency e must be positive.")
    if not 0.0 < stall_margin <= 1.0:
        raise ValueError("stall_margin should be in the range 0 < stall_margin <= 1.")
    if not 0.0 <= mach < 1.0:
        raise ValueError("mach must be subsonic and non-negative.")

    AR = AR_guess
    CL_a = _datcom_lift_curve_slope(AR, sweep_c4, taper, mach=mach, k=1.0)
    sweep_c2 = _half_chord_sweep_from_quarter_chord(sweep_c4, AR, taper)
    CL_max_3D = 0.9 * cl_max_2D * np.cos(sweep_c4)

    W_S_stall_max = 0.5 * rho_SL * V_stall_target_max**2 * CL_max_3D
    W_S_design = stall_margin * W_S_stall_max
    S = MTOW_N / W_S_design
    b = np.sqrt(S * AR)
    c_bar = S / b
    c_root = 2.0 * S / (b * (1.0 + taper))
    c_tip = taper * c_root
    c_mac = (2.0 / 3.0) * c_root * (1.0 + taper + taper**2) / (1.0 + taper)
    y_mac = (b / 6.0) * (1.0 + 2.0 * taper) / (1.0 + taper)
    x_ac_w = 0.25 * c_mac
    V_stall = np.sqrt(2.0 * W_S_design / (rho_SL * CL_max_3D))

    warnings = []
    if abs(cl_a_2D - 2.0 * np.pi) > 0.75:
        warnings.append(
            "Input 2-D lift-curve slope differs noticeably from 2*pi; DATCOM finite-wing slope still uses k=1.0 as in the project table."
        )
    if stall_margin == 1.0:
        warnings.append(
            "No stall wing-loading margin is applied; consider a margin below 1.0 for transition robustness."
        )

    return {
        "S": float(S),
        "b": float(b),
        "c_bar": float(c_bar),
        "c_root": float(c_root),
        "c_tip": float(c_tip),
        "c_mac": float(c_mac),
        "y_mac": float(y_mac),
        "AR": float(AR),
        "taper": float(taper),
        "sweep_c4": float(sweep_c4),
        "sweep_c4_deg": float(np.rad2deg(sweep_c4)),
        "sweep_c2": float(sweep_c2),
        "sweep_c2_deg": float(np.rad2deg(sweep_c2)),
        "CL_a": float(CL_a),
        "cl_a_2D": float(cl_a_2D),
        "CL_max_3D": float(CL_max_3D),
        "cl_max_2D": float(cl_max_2D),
        "W_S_stall_max": float(W_S_stall_max),
        "W_S_design": float(W_S_design),
        "stall_margin": float(stall_margin),
        "V_stall": float(V_stall),
        "V_stall_target_max": float(V_stall_target_max),
        "x_ac_w": float(x_ac_w),
        "x_ac_w_over_mac": 0.25,
        "mach": float(mach),
        "e": float(e),
        "notes": [
            "rho_SL is treated as the density at the stall-sizing condition; pass the 6000 m density if sizing for ceiling stall.",
            "Wing area is selected from the stall wing-loading limit multiplied by stall_margin.",
            "x_ac_w is measured from the leading edge of the mean aerodynamic chord; a fuselage reference needs CAD layout.",
        ],
        "warnings": warnings,
    }

# ---------- PHASE 9 ----------

def phase9_canard(S_w: float, c_bar_w: float, x_ac_w: float, l_c: float,
                  V_bar_c: float = 0.375, AR_c: float = 5.0,
                  taper_c: float = 0.5, sweep_c4_c: float = 0.0,
                  cl_max_2D_c: float = 1.3177, mach: float = 0.0) -> Dict:
    """Canard planform from volume coefficient.
    In:  S_w, c_bar_w, x_ac_w (Ph8), l_c, V̄_c target (Nicolai §11.6).
    Out: {'S_c', 'b_c', 'c_bar_c', 'AR_c', 'l_c', 'CL_a_c'}
    Eq:  S_c = V̄_c · S_w · c_bar_w / l_c       — Nicolai 2010 Eq. 11.7
         CL_a_c via Polhamus (same as Ph8).
    Loop: CG–canard inner loop with Ph10."""
    if S_w <= 0.0:
        raise ValueError("S_w must be positive.")
    if c_bar_w <= 0.0:
        raise ValueError("c_bar_w must be positive.")
    if l_c <= 0.0:
        raise ValueError("l_c must be positive.")
    if V_bar_c <= 0.0:
        raise ValueError("V_bar_c must be positive.")
    if AR_c <= 0.0:
        raise ValueError("AR_c must be positive.")
    if not 0.0 < taper_c <= 1.0:
        raise ValueError("taper_c should be in the range 0 < taper_c <= 1.")
    if cl_max_2D_c <= 0.0:
        raise ValueError("cl_max_2D_c must be positive.")
    if not 0.0 <= mach < 1.0:
        raise ValueError("mach must be subsonic and non-negative.")

    S_c = V_bar_c * S_w * c_bar_w / l_c
    b_c = np.sqrt(S_c * AR_c)
    c_bar_c = S_c / b_c
    c_root_c = 2.0 * S_c / (b_c * (1.0 + taper_c))
    c_tip_c = taper_c * c_root_c
    c_mac_c = (2.0 / 3.0) * c_root_c * (1.0 + taper_c + taper_c**2) / (1.0 + taper_c)
    y_mac_c = (b_c / 6.0) * (1.0 + 2.0 * taper_c) / (1.0 + taper_c)
    CL_a_c = _datcom_lift_curve_slope(AR_c, sweep_c4_c, taper_c, mach=mach, k=1.0)
    CL_max_3D_c = 0.9 * cl_max_2D_c * np.cos(sweep_c4_c)
    x_ac_c = x_ac_w - l_c
    area_ratio = S_c / S_w

    warnings = []
    if V_bar_c < 0.3 or V_bar_c > 0.6:
        warnings.append(
            "Canard volume coefficient is outside the 0.3-0.6 preliminary range cited in the project report."
        )
    if area_ratio < 0.10 or area_ratio > 0.25:
        warnings.append(
            "Canard area ratio is outside a typical first-cut range; check trim and transition-control authority."
        )

    return {
        "S_c": float(S_c),
        "b_c": float(b_c),
        "c_bar_c": float(c_bar_c),
        "c_root_c": float(c_root_c),
        "c_tip_c": float(c_tip_c),
        "c_mac_c": float(c_mac_c),
        "y_mac_c": float(y_mac_c),
        "AR_c": float(AR_c),
        "taper_c": float(taper_c),
        "sweep_c4_c": float(sweep_c4_c),
        "sweep_c4_c_deg": float(np.rad2deg(sweep_c4_c)),
        "l_c": float(l_c),
        "V_bar_c": float(V_bar_c),
        "area_ratio": float(area_ratio),
        "CL_a_c": float(CL_a_c),
        "CL_max_3D_c": float(CL_max_3D_c),
        "cl_max_2D_c": float(cl_max_2D_c),
        "x_ac_w": float(x_ac_w),
        "x_ac_c": float(x_ac_c),
        "x_ac_c_over_wing_mac": float(x_ac_c / c_bar_w),
        "mach": float(mach),
        "notes": [
            "The canard aerodynamic center is placed l_c ahead of the wing aerodynamic center in the same longitudinal datum.",
            "The default volume coefficient 0.375 matches the Bellona preliminary canard table rounded to 0.38.",
            "Incidence, elevator sizing, and load sharing are deferred to the stability/control phases.",
        ],
        "warnings": warnings,
    }


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

# ---------- PHASE 10 ----------

def phase10_scissor_canard(S_w, c_bar_w, x_ac_w, CL_a_w,
                           S_c, l_c, CL_a_c, eps_alpha_c, eps_alpha_w,
                           CL_c_max, CL_trim, SM_min: float = 0.10) -> Dict:
    """Longitudinal scissor check using a canard neutral-point formulation.
    In:  wing & canard geometry/derivs (Ph8, Ph9), downwash/upwash gradients,
         max canard lift coefficient, design SM_min.
    Out: {'x_np_over_c', 'x_cg_fwd_over_c', 'x_cg_aft_over_c', 'CG_range_pct'}
    Eq:  x_np/ell = [1 + a_w(1-eps_c)·S_w / (a_c(1+eps_w)·S_c)]^-1
                                                  — Phillips NASA TM-86694 Eq.16
         Equivalent canonical (Nelson-style canard):
         x_np/c̄ = x_ac_w/c̄ − (C_La_c/C_La_AC)·eta_c·V̄_c·(1+eps_w)
         Caughey MAE5070 Eq.3.29: canard contribution is NEGATIVE (forward).
    Loop: CG–canard inner (revises S_c, Ph9)."""
    if S_w <= 0.0:
        raise ValueError("S_w must be positive.")
    if c_bar_w <= 0.0:
        raise ValueError("c_bar_w must be positive.")
    if CL_a_w <= 0.0:
        raise ValueError("CL_a_w must be positive.")
    if S_c <= 0.0:
        raise ValueError("S_c must be positive.")
    if l_c <= 0.0:
        raise ValueError("l_c must be positive.")
    if CL_a_c <= 0.0:
        raise ValueError("CL_a_c must be positive.")
    if CL_c_max <= 0.0:
        raise ValueError("CL_c_max must be positive.")
    if CL_trim <= 0.0:
        raise ValueError("CL_trim must be positive.")
    if SM_min < 0.0:
        raise ValueError("SM_min must be non-negative.")
    if eps_alpha_c >= 1.0:
        raise ValueError("eps_alpha_c must be below 1.0 for this simplified model.")
    if eps_alpha_w <= -1.0:
        raise ValueError("eps_alpha_w must be above -1.0 for this simplified model.")

    x_ac_c = x_ac_w - l_c
    V_bar_c = S_c * l_c / (S_w * c_bar_w)
    area_ratio = S_c / S_w
    a_w_eff = CL_a_w * (1.0 - eps_alpha_c)
    a_c_eff = CL_a_c * (1.0 + eps_alpha_w)
    if a_w_eff <= 0.0 or a_c_eff <= 0.0:
        raise ValueError("Effective lift-curve slopes must remain positive.")

    x_np_forward_fraction = 1.0 / (1.0 + (a_w_eff * S_w) / (a_c_eff * S_c))
    x_np = x_ac_w - x_np_forward_fraction * l_c
    x_np_over_c = x_np / c_bar_w

    x_cg_aft = x_np - SM_min * c_bar_w
    x_cg_aft_over_c = x_cg_aft / c_bar_w

    canard_lift_fraction_max = CL_c_max * S_c / (CL_trim * S_w)
    x_cg_fwd = x_ac_w - canard_lift_fraction_max * l_c
    x_cg_fwd_over_c = x_cg_fwd / c_bar_w
    CG_range = x_cg_aft - x_cg_fwd
    CG_range_pct = 100.0 * CG_range / c_bar_w
    feasible = CG_range > 0.0

    CL_c_required_at_aft = (
        (x_ac_w - x_cg_aft) / l_c
        * CL_trim
        * S_w / S_c
    )
    CL_c_required_at_neutral = (
        (x_ac_w - x_np) / l_c
        * CL_trim
        * S_w / S_c
    )

    warnings = [
        "Phase 10 is a simplified verification model; it does not replace a full scissor plot with measured derivatives.",
        "CG limits are referenced to the wing mean aerodynamic chord datum, not a CAD fuselage datum.",
        "Forward CG limit assumes the canard can reach CL_c_max with no incidence, elevator, propwash, or stall-margin correction.",
    ]
    if not feasible:
        warnings.append(
            "The preliminary forward and aft CG limits do not overlap; revise canard volume, moment arm, or trim assumptions."
        )
    if x_cg_aft_over_c < -0.25 or x_cg_aft_over_c > 0.50:
        warnings.append(
            "Aft CG limit lies far from the wing MAC reference; check the longitudinal datum and canard layout."
        )
    if canard_lift_fraction_max > 0.6:
        warnings.append(
            "Canard maximum lift fraction is high; include canard stall margin and trim drag before accepting the forward CG limit."
        )
    if CL_c_required_at_aft > 0.7 * CL_c_max:
        warnings.append(
            "Canard lift required at the aft CG limit uses much of the available canard CL margin."
        )

    return {
        "x_np": float(x_np),
        "x_np_over_c": float(x_np_over_c),
        "x_np_forward_of_wing_ac": float(x_ac_w - x_np),
        "x_np_forward_fraction_l_c": float(x_np_forward_fraction),
        "x_cg_fwd": float(x_cg_fwd),
        "x_cg_fwd_over_c": float(x_cg_fwd_over_c),
        "x_cg_aft": float(x_cg_aft),
        "x_cg_aft_over_c": float(x_cg_aft_over_c),
        "CG_range_m": float(CG_range),
        "CG_range_pct": float(CG_range_pct),
        "feasible_preliminary_CG_range": bool(feasible),
        "SM_min": float(SM_min),
        "V_bar_c": float(V_bar_c),
        "area_ratio": float(area_ratio),
        "x_ac_w": float(x_ac_w),
        "x_ac_c": float(x_ac_c),
        "CL_trim": float(CL_trim),
        "CL_c_max": float(CL_c_max),
        "CL_c_required_at_aft_cg": float(CL_c_required_at_aft),
        "CL_c_required_at_neutral_point": float(CL_c_required_at_neutral),
        "canard_lift_fraction_max": float(canard_lift_fraction_max),
        "a_w_eff": float(a_w_eff),
        "a_c_eff": float(a_c_eff),
        "eps_alpha_c": float(eps_alpha_c),
        "eps_alpha_w": float(eps_alpha_w),
        "notes": [
            "x/c values use the wing MAC leading-edge datum from Phase 8.",
            "The neutral point is estimated as the weighted aerodynamic-center location of the wing and forward canard.",
            "The aft CG limit is x_np minus SM_min*c_bar_w for positive static margin.",
            "The forward CG limit is set by the canard CL_c_max trim capability.",
        ],
        "warnings": warnings,
    }


def iterate_phases_1_to_10(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           j_tol: float = 0.01,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-9 convergence, then Phase 10 canard scissor verification."""
    result = iterate_phases_1_to_9(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if assumptions is None:
        assumptions = Assumptions()

    phase8 = result["phase8"]
    phase9 = result["phase9"]
    phase10 = phase10_scissor_canard(
        phase8["S"],
        phase8["c_bar"],
        phase8["x_ac_w"],
        phase8["CL_a"],
        phase9["S_c"],
        phase9["l_c"],
        phase9["CL_a_c"],
        assumptions.canard_eps_alpha_c,
        assumptions.wing_eps_alpha_w,
        phase9["CL_max_3D_c"],
        result["phase3"]["CL_cruise"],
        assumptions.static_margin_min,
    )

    result = dict(result)
    result["phase10"] = phase10
    if result.get("converged"):
        if phase10["feasible_preliminary_CG_range"]:
            result["stopped_reason"] = (
                "propulsion, wing, airfoil Reynolds, canard geometry, and Phase 10 CG check completed"
            )
        else:
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 10 did not find a preliminary feasible CG range"
            )
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 10 is appended as a verification block after the fixed-MTOW propulsion/aero loop."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase10["warnings"]:
        result["warnings"].extend(f"phase10: {warning}" for warning in phase10["warnings"])
    return result

# ---------- PHASE 11 ----------

def _phase11_elevon_FW_outline(S_w, b_w, c_bar_w, CL_a_w, V_cruise, V_stall,
                      CG_range, q_slipstream_ratio: float = 1.5,
                      p_required_rad_s: float = np.deg2rad(60),
                      Cm_de_required: float = 0.5) -> Dict:
    """FW-mode elevon: pitch trim + roll rate.
    In:  wing geom (Ph8), CG range (Ph10), q_slip/q_inf factor in cruise (Stone 2008).
    Out: {'c_e_over_c', 'b_e_over_b', 'tau_e', 'Cm_de', 'Cl_da', 'p_achievable'}
    Eq:  Cm_de = -C_La · eta_e · (c_e/c) · (S_e/S) · (l_e/c̄) · tau_e
         p_ss  = -C_l_da · delta_a · V / (C_l_p · b/2)         — Nelson Ch.3
         q_local on aft-wing elevons = q_inf · (1 + 8·C_T/(pi·J^2)) — Stone 2008
         Floor: MIL-F-8785C §3.3.4 Class I Level 1 → p ≥ 60°/s
    Loop: max-of with Ph12."""
    ...


def phase11_elevon_FW(S_w, b_w, c_bar_w, CL_a_w, V_cruise, V_stall,
                      CG_range, q_slipstream_ratio: float = 1.5,
                      p_required_rad_s: float = np.deg2rad(60),
                      Cm_de_required: Optional[float] = None,
                      delta_e_max_rad: float = np.deg2rad(25),
                      tau_e: float = 0.50,
                      eta_e: float = 0.90,
                      l_e_over_c: float = 0.75,
                      CL_trim: Optional[float] = None,
                      pitch_trim_margin: float = 1.20,
                      control_margin_min: float = 1.0,
                      chord_fraction_bounds: Tuple[float, float] = (0.12, 0.35),
                      span_fraction_bounds: Tuple[float, float] = (0.20, 0.80),
                      grid_points: int = 49) -> Dict:
    """Fixed-wing elevon sizing for pitch trim and roll-rate authority."""
    if S_w <= 0.0:
        raise ValueError("S_w must be positive.")
    if b_w <= 0.0:
        raise ValueError("b_w must be positive.")
    if c_bar_w <= 0.0:
        raise ValueError("c_bar_w must be positive.")
    if CL_a_w <= 0.0:
        raise ValueError("CL_a_w must be positive.")
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if CG_range < 0.0:
        raise ValueError("CG_range must be non-negative.")
    if q_slipstream_ratio <= 0.0:
        raise ValueError("q_slipstream_ratio must be positive.")
    if p_required_rad_s <= 0.0:
        raise ValueError("p_required_rad_s must be positive.")
    if delta_e_max_rad <= 0.0:
        raise ValueError("delta_e_max_rad must be positive.")
    if tau_e <= 0.0:
        raise ValueError("tau_e must be positive.")
    if eta_e <= 0.0:
        raise ValueError("eta_e must be positive.")
    if l_e_over_c <= 0.0:
        raise ValueError("l_e_over_c must be positive.")
    if pitch_trim_margin <= 0.0:
        raise ValueError("pitch_trim_margin must be positive.")
    if control_margin_min <= 0.0:
        raise ValueError("control_margin_min must be positive.")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")

    chord_min, chord_max = chord_fraction_bounds
    span_min, span_max = span_fraction_bounds
    if not 0.0 < chord_min <= chord_max < 1.0:
        raise ValueError("chord_fraction_bounds must lie inside 0 < min <= max < 1.")
    if not 0.0 < span_min <= span_max <= 1.0:
        raise ValueError("span_fraction_bounds must lie inside 0 < min <= max <= 1.")

    if CL_trim is None:
        CL_trim_used = 1.0
        CL_trim_source = "fallback"
    else:
        if CL_trim <= 0.0:
            raise ValueError("CL_trim must be positive when provided.")
        CL_trim_used = float(CL_trim)
        CL_trim_source = "phase3"

    if Cm_de_required is None:
        pitch_moment_required = (
            pitch_trim_margin
            * CL_trim_used
            * 0.5
            * CG_range / c_bar_w
        )
        Cm_de_required_used = pitch_moment_required / delta_e_max_rad
        Cm_de_source = "computed_from_phase10_CG_range"
    else:
        if Cm_de_required <= 0.0:
            raise ValueError("Cm_de_required must be positive when provided.")
        Cm_de_required_used = float(Cm_de_required)
        pitch_moment_required = Cm_de_required_used * delta_e_max_rad
        Cm_de_source = "user_supplied"

    C_l_p = -CL_a_w / 12.0
    Cl_da_required = (
        p_required_rad_s
        * abs(C_l_p)
        * b_w
        / (2.0 * delta_e_max_rad * V_cruise)
    )

    def _surface_metrics(c_e_over_c: float, b_e_over_b: float) -> Dict:
        S_e_over_S = c_e_over_c * b_e_over_b
        S_e_total = S_w * S_e_over_S
        Cm_de = (
            -q_slipstream_ratio
            * CL_a_w
            * eta_e
            * tau_e
            * c_e_over_c
            * S_e_over_S
            * l_e_over_c
        )

        y2 = 0.5 * b_w
        y1 = y2 * (1.0 - b_e_over_b)
        c_rect = S_w / b_w
        strip_integral = 0.5 * c_rect * (y2**2 - y1**2)
        Cl_da = (
            2.0
            * q_slipstream_ratio
            * CL_a_w
            * tau_e
            * c_e_over_c
            * strip_integral
            / (S_w * b_w)
        )
        p_achievable = (
            2.0
            * abs(Cl_da)
            * delta_e_max_rad
            * V_cruise
            / (abs(C_l_p) * b_w)
        )
        return {
            "c_e_over_c": float(c_e_over_c),
            "b_e_over_b": float(b_e_over_b),
            "S_e_over_S": float(S_e_over_S),
            "S_e_total": float(S_e_total),
            "S_e_each": float(0.5 * S_e_total),
            "Cm_de": float(Cm_de),
            "Cl_da": float(Cl_da),
            "p_achievable_rad_s": float(p_achievable),
            "p_achievable_deg_s": float(np.rad2deg(p_achievable)),
            "strip_integral": float(strip_integral),
        }

    chord_grid = np.linspace(chord_min, chord_max, grid_points)
    span_grid = np.linspace(span_min, span_max, grid_points)
    feasible_candidates = []
    checked = 0
    for c_e_over_c in chord_grid:
        for b_e_over_b in span_grid:
            checked += 1
            metrics = _surface_metrics(c_e_over_c, b_e_over_b)
            pitch_ok = abs(metrics["Cm_de"]) >= control_margin_min * Cm_de_required_used
            roll_ok = metrics["p_achievable_rad_s"] >= control_margin_min * p_required_rad_s
            if pitch_ok and roll_ok:
                feasible_candidates.append(metrics)

    if feasible_candidates:
        selected = min(
            feasible_candidates,
            key=lambda item: (item["S_e_over_S"], item["c_e_over_c"], item["b_e_over_b"]),
        )
        feasible = True
    else:
        selected = _surface_metrics(chord_max, span_max)
        feasible = False

    pitch_margin = abs(selected["Cm_de"]) / Cm_de_required_used
    roll_margin = selected["p_achievable_rad_s"] / p_required_rad_s
    pitch_ok = pitch_margin >= 1.0
    roll_ok = roll_margin >= 1.0
    if pitch_ok and roll_ok:
        binding_case = "pitch" if pitch_margin <= roll_margin else "roll"
    elif pitch_ok:
        binding_case = "roll"
    elif roll_ok:
        binding_case = "pitch"
    else:
        binding_case = "pitch_and_roll"

    warnings = [
        "Phase 11 is a simplified verification model; it does not replace hinge-moment, aeroelastic, or control-derivative analysis.",
        "Roll authority uses a rectangular-wing strip approximation and Cl_p = -CL_a/12.",
        "Pitch authority uses the Phase 10 CG range and an assumed elevon moment arm; replace with CAD CG and aerodynamic derivative data.",
        "The slipstream factor is a scalar multiplier; replace with a propeller-wing interaction model or wind-tunnel data.",
    ]
    if not feasible:
        warnings.append(
            "No elevon geometry inside the selected chord/span bounds met both pitch and roll requirements."
        )
    if selected["c_e_over_c"] >= 0.30:
        warnings.append(
            "Selected elevon chord fraction is large; check hinge moment, stiffness, and packaging."
        )
    if selected["b_e_over_b"] >= 0.70:
        warnings.append(
            "Selected elevon span fraction is large; check flap-tip clearance and structural integration."
        )
    if V_cruise < 1.1 * V_stall:
        warnings.append(
            "Cruise speed is close to stall speed; fixed-wing control authority should also be checked at transition speed."
        )

    selected.update({
        "feasible_preliminary_elevon": bool(feasible),
        "pitch_meets_requirement": bool(pitch_ok),
        "roll_meets_requirement": bool(roll_ok),
        "binding_case": binding_case,
        "Cm_de_required": float(Cm_de_required_used),
        "Cm_de_abs": float(abs(selected["Cm_de"])),
        "pitch_moment_required": float(pitch_moment_required),
        "pitch_margin": float(pitch_margin),
        "Cl_da_required": float(Cl_da_required),
        "roll_margin": float(roll_margin),
        "C_l_p": float(C_l_p),
        "V_control_roll": float(V_cruise),
        "V_stall": float(V_stall),
        "delta_e_max_rad": float(delta_e_max_rad),
        "delta_e_max_deg": float(np.rad2deg(delta_e_max_rad)),
        "p_required_rad_s": float(p_required_rad_s),
        "p_required_deg_s": float(np.rad2deg(p_required_rad_s)),
        "q_slipstream_ratio": float(q_slipstream_ratio),
        "tau_e": float(tau_e),
        "eta_e": float(eta_e),
        "l_e_over_c": float(l_e_over_c),
        "CL_trim": float(CL_trim_used),
        "CL_trim_source": CL_trim_source,
        "Cm_de_source": Cm_de_source,
        "CG_range_m": float(CG_range),
        "CG_range_over_c": float(CG_range / c_bar_w),
        "pitch_trim_margin": float(pitch_trim_margin),
        "control_margin_min": float(control_margin_min),
        "chord_fraction_bounds": [float(chord_min), float(chord_max)],
        "span_fraction_bounds": [float(span_min), float(span_max)],
        "grid_points": int(grid_points),
        "checked_candidates": int(checked),
        "feasible_candidates": int(len(feasible_candidates)),
        "notes": [
            "b_e_over_b is the per-side elevon span divided by semispan.",
            "S_e_total is the combined left-plus-right elevon planform area.",
            "Cm_de is reported per radian of symmetric elevon deflection.",
            "Cl_da and p_achievable are reported per radian of differential elevon deflection.",
        ],
        "warnings": warnings,
    })
    return selected


def iterate_phases_1_to_11(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           j_tol: float = 0.01,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-10, then append Phase 11 fixed-wing elevon verification."""
    result = iterate_phases_1_to_10(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if assumptions is None:
        assumptions = Assumptions()

    phase8 = result["phase8"]
    phase10 = result["phase10"]
    phase11 = phase11_elevon_FW(
        phase8["S"],
        phase8["b"],
        phase8["c_bar"],
        phase8["CL_a"],
        result["phase3"]["V_cruise"],
        phase8["V_stall"],
        phase10["CG_range_m"],
        q_slipstream_ratio=assumptions.elevon_q_slipstream_ratio,
        p_required_rad_s=np.deg2rad(assumptions.roll_rate_required_deg_s),
        Cm_de_required=None,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
        tau_e=assumptions.elevon_tau,
        eta_e=assumptions.elevon_eta,
        l_e_over_c=assumptions.elevon_l_e_over_c,
        CL_trim=phase10["CL_trim"],
        pitch_trim_margin=assumptions.elevon_pitch_trim_margin,
        control_margin_min=1.05,
        chord_fraction_bounds=(
            assumptions.elevon_chord_fraction_min,
            assumptions.elevon_chord_fraction_max,
        ),
        span_fraction_bounds=(
            assumptions.elevon_span_fraction_min,
            assumptions.elevon_span_fraction_max,
        ),
        grid_points=assumptions.elevon_grid_points,
    )

    result = dict(result)
    result["phase11"] = phase11
    if result.get("converged"):
        if phase11["feasible_preliminary_elevon"]:
            result["stopped_reason"] = (
                "propulsion, aerodynamics, canard CG, and Phase 11 fixed-wing elevon check completed"
            )
        else:
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 11 did not find a preliminary feasible elevon"
            )
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 11 is appended as a fixed-wing control verification block after the canard CG check."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase11["warnings"]:
        result["warnings"].extend(f"phase11: {warning}" for warning in phase11["warnings"])
    return result

# ---------- PHASE 12 ----------

def _phase12_hover_control_outline(T_hover_per_rotor: float, A_disc: float, rho: float,
                          d_x_rotor: float, d_y_rotor: float,
                          I_xx: float, I_yy: float, I_zz: float,
                          S_e_FW: float, b_e_FW: float, l_z_elev: float,
                          omega_dot_required: float = 2.0,
                          yaw_rate_required: float = np.deg2rad(30)) -> Dict:
    """Hover pitch/roll via differential thrust; yaw via elevon-in-propwash.
    In:  rotor thrust & geometry (Ph1), I-tensor (Ph15 prelim),
         FW elevon size (Ph11), required hover-rate specs (ADS-33E Level 1).
    Out: {'dT_over_T', 'b_e_yaw_required', 'S_e_yaw_required', 'binding_case'}
    Eq:  M_y = dT · d_x;  omega_dot = M/I
         q_slip(hover) = T/A_disc                  — Stone 2008 (V_inf=0)
         V_slip(hover) = sqrt(2T/(rho·A_disc))     — Leishman Eq.2.42 → Stone
         M_z_yaw = q_slip · S_e · CL_de · delta_e · l_z
    Decision: take MAX of (Ph11.b_e, Ph12.b_e_yaw_required)."""
    ...


def phase12_hover_control(T_hover_per_rotor: float, A_disc: float, rho: float,
                          d_x_rotor: float, d_y_rotor: float,
                          I_xx: float, I_yy: float, I_zz: float,
                          S_e_FW: float, b_e_FW: float, l_z_elev: float,
                          omega_dot_required: float = 2.0,
                          yaw_rate_required: float = np.deg2rad(30),
                          T_available_per_rotor: Optional[float] = None,
                          roll_omega_dot_required: Optional[float] = None,
                          yaw_response_time_s: float = 1.0,
                          CL_de: float = 2.5,
                          delta_e_max_rad: float = np.deg2rad(25)) -> Dict:
    """Hover pitch/roll via differential thrust and yaw via elevon propwash."""
    if T_hover_per_rotor <= 0.0:
        raise ValueError("T_hover_per_rotor must be positive.")
    if A_disc <= 0.0:
        raise ValueError("A_disc must be positive.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if d_x_rotor <= 0.0:
        raise ValueError("d_x_rotor must be positive.")
    if d_y_rotor <= 0.0:
        raise ValueError("d_y_rotor must be positive.")
    if I_xx <= 0.0 or I_yy <= 0.0 or I_zz <= 0.0:
        raise ValueError("Moments of inertia must be positive.")
    if S_e_FW <= 0.0:
        raise ValueError("S_e_FW must be positive.")
    if b_e_FW <= 0.0:
        raise ValueError("b_e_FW must be positive.")
    if l_z_elev <= 0.0:
        raise ValueError("l_z_elev must be positive.")
    if omega_dot_required <= 0.0:
        raise ValueError("omega_dot_required must be positive.")
    if yaw_rate_required <= 0.0:
        raise ValueError("yaw_rate_required must be positive.")
    if yaw_response_time_s <= 0.0:
        raise ValueError("yaw_response_time_s must be positive.")
    if CL_de <= 0.0:
        raise ValueError("CL_de must be positive.")
    if delta_e_max_rad <= 0.0:
        raise ValueError("delta_e_max_rad must be positive.")

    if roll_omega_dot_required is None:
        roll_omega_dot = omega_dot_required
    else:
        if roll_omega_dot_required <= 0.0:
            raise ValueError("roll_omega_dot_required must be positive when provided.")
        roll_omega_dot = roll_omega_dot_required

    warnings = [
        "Phase 12 is a simplified verification model; replace arm lengths, inertias, and control derivatives with CAD and test data.",
        "Differential-thrust moments assume opposite rotor pairs with moment M = 2*dT*d_arm.",
        "Yaw authority assumes the fixed-wing elevon area is immersed in hover propwash.",
    ]

    if T_available_per_rotor is None:
        T_available = T_hover_per_rotor
        warnings.append(
            "T_available_per_rotor was not provided; no upward thrust headroom is assumed."
        )
    else:
        if T_available_per_rotor <= 0.0:
            raise ValueError("T_available_per_rotor must be positive when provided.")
        T_available = float(T_available_per_rotor)

    delta_T_headroom = max(0.0, T_available - T_hover_per_rotor)
    delta_T_downroom = T_hover_per_rotor
    delta_T_available = min(delta_T_headroom, delta_T_downroom)

    M_pitch_required = I_yy * omega_dot_required
    M_roll_required = I_xx * roll_omega_dot
    delta_T_pitch = M_pitch_required / (2.0 * d_x_rotor)
    delta_T_roll = M_roll_required / (2.0 * d_y_rotor)

    pitch_margin = delta_T_available / delta_T_pitch if delta_T_pitch > 0.0 else np.inf
    roll_margin = delta_T_available / delta_T_roll if delta_T_roll > 0.0 else np.inf
    d_x_required_for_pitch = (
        M_pitch_required / (2.0 * delta_T_available)
        if delta_T_available > 0.0
        else np.inf
    )
    d_y_required_for_roll = (
        M_roll_required / (2.0 * delta_T_available)
        if delta_T_available > 0.0
        else np.inf
    )
    thrust_to_weight_required_pitch = 1.0 + delta_T_pitch / T_hover_per_rotor
    thrust_to_weight_required_roll = 1.0 + delta_T_roll / T_hover_per_rotor
    pitch_angular_accel_available = (
        2.0 * delta_T_available * d_x_rotor / I_yy
        if I_yy > 0.0
        else np.inf
    )
    roll_angular_accel_available = (
        2.0 * delta_T_available * d_y_rotor / I_xx
        if I_xx > 0.0
        else np.inf
    )
    I_yy_max_for_pitch = (
        2.0 * delta_T_available * d_x_rotor / omega_dot_required
        if omega_dot_required > 0.0
        else np.inf
    )
    I_xx_max_for_roll = (
        2.0 * delta_T_available * d_y_rotor / roll_omega_dot
        if roll_omega_dot > 0.0
        else np.inf
    )
    pitch_meets = pitch_margin >= 1.0
    roll_meets = roll_margin >= 1.0

    q_slip_hover = T_hover_per_rotor / A_disc
    V_slip_hover = np.sqrt(2.0 * T_hover_per_rotor / (rho * A_disc))
    yaw_accel_required = yaw_rate_required / yaw_response_time_s
    M_yaw_required = I_zz * yaw_accel_required
    M_yaw_available = q_slip_hover * S_e_FW * CL_de * delta_e_max_rad * l_z_elev
    S_e_yaw_required = M_yaw_required / (q_slip_hover * CL_de * delta_e_max_rad * l_z_elev)
    b_e_yaw_required = b_e_FW * S_e_yaw_required / S_e_FW
    yaw_margin = M_yaw_available / M_yaw_required if M_yaw_required > 0.0 else np.inf
    yaw_meets = yaw_margin >= 1.0

    margins = {
        "pitch": pitch_margin,
        "roll": roll_margin,
        "yaw": yaw_margin,
    }
    binding_case = min(margins, key=margins.get)
    feasible = pitch_meets and roll_meets and yaw_meets
    if not feasible:
        failed = [name for name, margin in margins.items() if margin < 1.0]
        warnings.append(
            "Hover-control margin is below unity for: " + ", ".join(failed) + "."
        )
    if delta_T_available <= 0.0:
        warnings.append(
            "Available thrust headroom is zero or negative; increase T/W or reduce hover trim thrust."
        )
    if delta_T_pitch > delta_T_available or delta_T_roll > delta_T_available:
        warnings.append(
            "Differential-thrust authority is limited by motor thrust headroom."
        )
    if S_e_yaw_required > S_e_FW:
        warnings.append(
            "Hover-yaw elevon area requirement exceeds the Phase 11 fixed-wing elevon area."
        )

    final_elevon_area_required = max(S_e_FW, S_e_yaw_required)
    final_elevon_span_required = max(b_e_FW, b_e_yaw_required)

    return {
        "feasible_preliminary_hover_control": bool(feasible),
        "binding_case": binding_case,
        "pitch_meets_requirement": bool(pitch_meets),
        "roll_meets_requirement": bool(roll_meets),
        "yaw_meets_requirement": bool(yaw_meets),
        "T_hover_per_rotor": float(T_hover_per_rotor),
        "T_available_per_rotor": float(T_available),
        "delta_T_available": float(delta_T_available),
        "delta_T_headroom": float(delta_T_headroom),
        "delta_T_downroom": float(delta_T_downroom),
        "delta_T_pitch_required": float(delta_T_pitch),
        "delta_T_roll_required": float(delta_T_roll),
        "delta_T_pitch_over_hover": float(delta_T_pitch / T_hover_per_rotor),
        "delta_T_roll_over_hover": float(delta_T_roll / T_hover_per_rotor),
        "delta_T_pitch_over_available": float(delta_T_pitch / delta_T_available) if delta_T_available > 0.0 else np.inf,
        "delta_T_roll_over_available": float(delta_T_roll / delta_T_available) if delta_T_available > 0.0 else np.inf,
        "pitch_arm_required_m": float(d_x_required_for_pitch),
        "roll_arm_required_m": float(d_y_required_for_roll),
        "thrust_to_weight_required_pitch": float(thrust_to_weight_required_pitch),
        "thrust_to_weight_required_roll": float(thrust_to_weight_required_roll),
        "pitch_angular_accel_available_rad_s2": float(pitch_angular_accel_available),
        "roll_angular_accel_available_rad_s2": float(roll_angular_accel_available),
        "Iyy_max_for_pitch_kg_m2": float(I_yy_max_for_pitch),
        "Ixx_max_for_roll_kg_m2": float(I_xx_max_for_roll),
        "pitch_margin": float(pitch_margin),
        "roll_margin": float(roll_margin),
        "yaw_margin": float(yaw_margin),
        "M_pitch_required": float(M_pitch_required),
        "M_roll_required": float(M_roll_required),
        "M_yaw_required": float(M_yaw_required),
        "M_yaw_available": float(M_yaw_available),
        "d_x_rotor": float(d_x_rotor),
        "d_y_rotor": float(d_y_rotor),
        "l_z_elev": float(l_z_elev),
        "I_xx": float(I_xx),
        "I_yy": float(I_yy),
        "I_zz": float(I_zz),
        "omega_dot_pitch_required": float(omega_dot_required),
        "omega_dot_roll_required": float(roll_omega_dot),
        "yaw_rate_required_rad_s": float(yaw_rate_required),
        "yaw_rate_required_deg_s": float(np.rad2deg(yaw_rate_required)),
        "yaw_response_time_s": float(yaw_response_time_s),
        "yaw_accel_required": float(yaw_accel_required),
        "q_slip_hover": float(q_slip_hover),
        "V_slip_hover": float(V_slip_hover),
        "CL_de": float(CL_de),
        "delta_e_max_rad": float(delta_e_max_rad),
        "delta_e_max_deg": float(np.rad2deg(delta_e_max_rad)),
        "S_e_FW": float(S_e_FW),
        "b_e_FW": float(b_e_FW),
        "S_e_yaw_required": float(S_e_yaw_required),
        "b_e_yaw_required": float(b_e_yaw_required),
        "final_elevon_area_required": float(final_elevon_area_required),
        "final_elevon_span_required": float(final_elevon_span_required),
        "A_disc": float(A_disc),
        "rho": float(rho),
        "notes": [
            "T_hover_per_rotor is the trim thrust per rotor, while T_available_per_rotor is the maximum available thrust per rotor.",
            "delta_T_available is limited by the smaller of upward headroom and downward thrust reduction.",
            "Pitch and roll differential thrust are per-rotor increments for opposite rotor pairs.",
            "S_e_yaw_required is the total left-plus-right elevon area required for hover yaw if chord effectiveness is unchanged.",
        ],
        "warnings": warnings,
    }


def iterate_phases_1_to_12(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           j_tol: float = 0.01,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-11, then append Phase 12 hover-control verification."""
    result = iterate_phases_1_to_11(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if assumptions is None:
        assumptions = Assumptions()

    MTOW_N = result["MTOW_N"]
    phase1 = result["phase1"]
    phase8 = result["phase8"]
    phase11 = result["phase11"]
    rho_target, _, _, _ = isa(mission.altitude_m if mission else Mission.altitude_m)

    d_x_rotor = (
        assumptions.hover_pitch_arm_m
        if assumptions.hover_pitch_arm_m is not None
        else assumptions.hover_pitch_arm_fraction_fuselage * assumptions.fuselage_length_m
    )
    d_y_rotor = (
        assumptions.hover_roll_arm_m
        if assumptions.hover_roll_arm_m is not None
        else assumptions.hover_roll_arm_fraction_span * phase8["b"]
    )
    l_z_elev = (
        assumptions.hover_yaw_arm_m
        if assumptions.hover_yaw_arm_m is not None
        else assumptions.hover_yaw_arm_fraction_chord * phase8["c_bar"]
    )

    I_xx = (
        assumptions.hover_Ixx_kg_m2
        if assumptions.hover_Ixx_kg_m2 is not None
        else MTOW_kg * (assumptions.hover_Ixx_radius_fraction_span * phase8["b"]) ** 2
    )
    I_yy = (
        assumptions.hover_Iyy_kg_m2
        if assumptions.hover_Iyy_kg_m2 is not None
        else MTOW_kg * (assumptions.hover_Iyy_radius_fraction_fuselage * assumptions.fuselage_length_m) ** 2
    )
    I_zz = (
        assumptions.hover_Izz_kg_m2
        if assumptions.hover_Izz_kg_m2 is not None
        else MTOW_kg * (assumptions.hover_Izz_radius_fraction_span * phase8["b"]) ** 2
    )

    CL_de = (
        assumptions.hover_control_CL_de
        if assumptions.hover_control_CL_de is not None
        else phase8["CL_a"] * assumptions.elevon_tau
    )

    S_e_FW = phase11["S_e_total"]
    b_e_FW = phase11["b_e_over_b"] * phase8["b"]
    T_hover_trim = MTOW_N / assumptions.n_rotors
    T_available = phase1["T_per_rotor"]

    phase12 = phase12_hover_control(
        T_hover_trim,
        phase1["A_disc"],
        rho_target,
        d_x_rotor,
        d_y_rotor,
        I_xx,
        I_yy,
        I_zz,
        S_e_FW,
        b_e_FW,
        l_z_elev,
        omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_rate_required=np.deg2rad(assumptions.hover_yaw_rate_required_deg_s),
        T_available_per_rotor=T_available,
        roll_omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_response_time_s=assumptions.hover_yaw_response_time_s,
        CL_de=CL_de,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
    )

    phase12["inertia_source"] = {
        "I_xx": "assumption" if assumptions.hover_Ixx_kg_m2 is not None else "radius_of_gyration_estimate",
        "I_yy": "assumption" if assumptions.hover_Iyy_kg_m2 is not None else "radius_of_gyration_estimate",
        "I_zz": "assumption" if assumptions.hover_Izz_kg_m2 is not None else "radius_of_gyration_estimate",
    }
    phase12["arm_source"] = {
        "d_x_rotor": "assumption" if assumptions.hover_pitch_arm_m is not None else "fuselage_fraction_estimate",
        "d_y_rotor": "assumption" if assumptions.hover_roll_arm_m is not None else "span_fraction_estimate",
        "l_z_elev": "assumption" if assumptions.hover_yaw_arm_m is not None else "chord_fraction_estimate",
    }

    result = dict(result)
    result["phase12"] = phase12
    if result.get("converged"):
        if not phase11["feasible_preliminary_elevon"]:
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 11 fixed-wing elevon remains infeasible; Phase 12 hover-control check appended"
            )
        elif phase12["feasible_preliminary_hover_control"]:
            result["stopped_reason"] = (
                "propulsion, aerodynamics, fixed-wing control, and Phase 12 hover-control check completed"
            )
        else:
            result["stopped_reason"] = (
                "propulsion/aero/control loop completed, but Phase 12 hover control is not feasible with current assumptions"
            )
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 12 is appended as a hover-control verification block before the detailed mass/inertia phase exists."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase12["warnings"]:
        result["warnings"].extend(f"phase12: {warning}" for warning in phase12["warnings"])
    return result

# ---------- PHASE 13 ----------

def _phase13_transition_blending_outline(V_stall: float,
                                V_blend_start_frac: float = 0.5,
                                V_blend_end_frac: float = 1.2) -> Dict:
    """Control mixer schedule from hover (diff thrust) to FW (elevon).
    In:  V_stall (Ph8); blending fractions (Stone & Clarke 2001).
    Out: {'alpha_blend_fn', 'V_blend_start', 'V_blend_end',
          'E_transition_estimate', 't_transition'}
    Ref: Stone & Clarke AIAC 2001 Paper 105; Stone & Wong 2004; Bapst IROS 2015;
         Li et al. IEEE Access 2020 (collocation-method optimisation)."""
    ...


def phase13_transition_blending(V_stall: float,
                                V_blend_start_frac: float = 0.5,
                                V_blend_end_frac: float = 1.2,
                                V_entry: float = 0.0,
                                V_exit: Optional[float] = None,
                                transition_accel_m_s2: float = 1.0,
                                V_cruise: Optional[float] = None,
                                cruise_speed_margin_frac: float = 0.0,
                                P_hover_W: Optional[float] = None,
                                P_fw_W: Optional[float] = None,
                                sample_count: int = 9) -> Dict:
    """Speed-based hover-to-fixed-wing control blending schedule."""
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if V_blend_start_frac <= 0.0:
        raise ValueError("V_blend_start_frac must be positive.")
    if V_blend_end_frac <= V_blend_start_frac:
        raise ValueError("V_blend_end_frac must be larger than V_blend_start_frac.")
    if V_entry < 0.0:
        raise ValueError("V_entry must be non-negative.")
    if transition_accel_m_s2 <= 0.0:
        raise ValueError("transition_accel_m_s2 must be positive.")
    if cruise_speed_margin_frac < 0.0:
        raise ValueError("cruise_speed_margin_frac must be non-negative.")
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3.")

    V_blend_start = V_blend_start_frac * V_stall
    V_blend_end = V_blend_end_frac * V_stall
    if V_exit is None:
        V_exit_used = V_blend_end
        V_exit_source = "blend_end"
    else:
        if V_exit <= V_entry:
            raise ValueError("V_exit must be greater than V_entry.")
        V_exit_used = float(V_exit)
        V_exit_source = "user_supplied"
    if V_exit_used <= V_entry:
        raise ValueError("V_exit must be greater than V_entry.")

    def _alpha_fw(speed: float) -> float:
        xi = (speed - V_blend_start) / (V_blend_end - V_blend_start)
        xi = float(np.clip(xi, 0.0, 1.0))
        return float(3.0 * xi**2 - 2.0 * xi**3)

    def _sample(speed: float) -> Dict:
        xi = (speed - V_blend_start) / (V_blend_end - V_blend_start)
        xi_clipped = float(np.clip(xi, 0.0, 1.0))
        alpha_fw = float(3.0 * xi_clipped**2 - 2.0 * xi_clipped**3)
        if speed <= V_blend_start:
            mode = "hover"
        elif speed >= V_blend_end:
            mode = "fixed_wing"
        else:
            mode = "blend"
        return {
            "V_m_s": float(speed),
            "xi": float(xi_clipped),
            "alpha_hover": float(1.0 - alpha_fw),
            "alpha_fw": alpha_fw,
            "mode": mode,
        }

    V_samples = np.linspace(V_entry, V_exit_used, sample_count)
    schedule = [_sample(speed) for speed in V_samples]

    preblend_delta_V = max(0.0, min(V_blend_start, V_exit_used) - V_entry)
    blend_delta_V = max(
        0.0,
        min(V_blend_end, V_exit_used) - max(V_blend_start, V_entry),
    )
    postblend_delta_V = max(0.0, V_exit_used - max(V_blend_end, V_entry))

    t_preblend = preblend_delta_V / transition_accel_m_s2
    t_blend = blend_delta_V / transition_accel_m_s2
    t_postblend = postblend_delta_V / transition_accel_m_s2
    t_transition = (V_exit_used - V_entry) / transition_accel_m_s2
    distance_transition = (
        V_exit_used**2 - V_entry**2
    ) / (2.0 * transition_accel_m_s2)

    alpha_fw_at_cruise = None
    alpha_hover_at_cruise = None
    V_cruise_required_for_margin = None
    cruise_speed_margin_over_blend_end = None
    if V_cruise is not None:
        if V_cruise <= 0.0:
            raise ValueError("V_cruise must be positive when provided.")
        alpha_fw_at_cruise = _alpha_fw(V_cruise)
        alpha_hover_at_cruise = 1.0 - alpha_fw_at_cruise
        V_cruise_required_for_margin = (
            (1.0 + cruise_speed_margin_frac) * V_blend_end
        )
        cruise_speed_margin_over_blend_end = V_cruise / V_blend_end - 1.0

    E_transition_Wh = None
    P_transition_average_W = None
    if P_hover_W is not None or P_fw_W is not None:
        if P_hover_W is None or P_fw_W is None:
            raise ValueError("P_hover_W and P_fw_W must be provided together.")
        if P_hover_W < 0.0 or P_fw_W < 0.0:
            raise ValueError("Transition powers must be non-negative.")
        energy_speeds = np.linspace(V_entry, V_exit_used, max(80, sample_count * 10))
        alpha_values = np.asarray([_alpha_fw(speed) for speed in energy_speeds])
        powers = (1.0 - alpha_values) * P_hover_W + alpha_values * P_fw_W
        E_transition_Wh = float(np.trapz(powers, energy_speeds) / transition_accel_m_s2 / 3600.0)
        P_transition_average_W = float(E_transition_Wh * 3600.0 / t_transition)

    warnings = [
        "Phase 13 is a simplified speed-based mixer; it does not model tail-sitter attitude dynamics, angle of attack, or propeller-wing interaction during transition.",
        "Transition time assumes constant acceleration in airspeed.",
        "Transition energy is a first-cut interpolation between hover and fixed-wing power and is not yet fed back into Phase 5.",
    ]
    if V_cruise is not None and V_cruise < V_blend_end:
        warnings.append(
            "Cruise speed is below the blend-end speed; the nominal mission cruise point remains partly in hover-control blending."
        )
    elif (
        V_cruise is not None
        and V_cruise_required_for_margin is not None
        and V_cruise < V_cruise_required_for_margin
    ):
        warnings.append(
            "Cruise speed clears the blend-end speed but does not meet the selected transition margin."
        )
    if V_entry < V_blend_start and V_exit_used < V_blend_end:
        warnings.append(
            "The requested transition speed range ends before full fixed-wing control authority is scheduled."
        )
    if V_blend_start < 0.5 * V_stall:
        warnings.append(
            "Blend start is below 0.5 V_stall; verify that aerodynamic control surfaces have useful authority."
        )

    return {
        "V_stall": float(V_stall),
        "V_blend_start": float(V_blend_start),
        "V_blend_end": float(V_blend_end),
        "V_blend_start_frac": float(V_blend_start_frac),
        "V_blend_end_frac": float(V_blend_end_frac),
        "V_entry": float(V_entry),
        "V_exit": float(V_exit_used),
        "V_exit_source": V_exit_source,
        "transition_accel_m_s2": float(transition_accel_m_s2),
        "t_transition": float(t_transition),
        "t_preblend": float(t_preblend),
        "t_blend": float(t_blend),
        "t_postblend": float(t_postblend),
        "distance_transition_m": float(distance_transition),
        "E_transition_estimate_Wh": E_transition_Wh,
        "P_transition_average_W": P_transition_average_W,
        "P_hover_W": None if P_hover_W is None else float(P_hover_W),
        "P_fw_W": None if P_fw_W is None else float(P_fw_W),
        "V_cruise": None if V_cruise is None else float(V_cruise),
        "cruise_speed_margin_frac": float(cruise_speed_margin_frac),
        "V_cruise_required_for_margin": (
            None
            if V_cruise_required_for_margin is None
            else float(V_cruise_required_for_margin)
        ),
        "cruise_speed_margin_over_blend_end": (
            None
            if cruise_speed_margin_over_blend_end is None
            else float(cruise_speed_margin_over_blend_end)
        ),
        "alpha_fw_at_cruise": alpha_fw_at_cruise,
        "alpha_hover_at_cruise": alpha_hover_at_cruise,
        "alpha_blend_fn": "smoothstep: alpha_fw = 3*xi^2 - 2*xi^3, xi = clip((V - V_start)/(V_end - V_start), 0, 1)",
        "schedule_samples": schedule,
        "notes": [
            "alpha_hover multiplies hover/differential-thrust control commands.",
            "alpha_fw multiplies fixed-wing/elevon control commands.",
            "The schedule is speed based so it can be inspected without a full transition simulation.",
        ],
        "warnings": warnings,
    }


def iterate_phases_1_to_13(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           j_tol: float = 0.01,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-12, then append Phase 13 transition blending."""
    result = iterate_phases_1_to_12(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if assumptions is None:
        assumptions = Assumptions()

    phase13 = phase13_transition_blending(
        result["phase8"]["V_stall"],
        assumptions.transition_blend_start_frac,
        assumptions.transition_blend_end_frac,
        V_entry=0.0,
        V_exit=None,
        transition_accel_m_s2=assumptions.transition_accel_m_s2,
        V_cruise=result["phase3"]["V_cruise"],
        cruise_speed_margin_frac=assumptions.transition_cruise_margin_frac,
        P_hover_W=result["phase3"]["P_hover_target"],
        P_fw_W=result["phase3"]["P_fw"],
        sample_count=assumptions.transition_sample_count,
    )

    result = dict(result)
    result["phase13"] = phase13
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 13 is appended as a transition mixer schedule; its energy estimate is reported but Phase 5 still uses the Phase 3 transition estimate."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase13["warnings"]:
        result["warnings"].extend(f"phase13: {warning}" for warning in phase13["warnings"])
    if result.get("converged"):
        phase11 = result.get("phase11", {})
        phase12 = result.get("phase12", {})
        if phase11 and not phase11.get("feasible_preliminary_elevon", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 11 fixed-wing elevon remains infeasible; Phase 13 schedule appended"
            )
        elif phase12 and not phase12.get("feasible_preliminary_hover_control", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 12 hover control remains infeasible; Phase 13 schedule appended"
            )
        else:
            result["stopped_reason"] = (
                "propulsion, aerodynamics, controls, and Phase 13 transition blending completed"
            )
    return result

# ---------- PHASE 14 ----------

def _phase14_dynamic_stability_outline(Cm_a, Cm_q, Cm_a_dot, CL_a, I_yy, I_xx, I_zz,
                                       S_w, c_bar_w, b_w, V_cruise, rho_cruise) -> Dict:
    """Short-period, phugoid, Dutch-roll, spiral — VERIFICATION ONLY.
    In:  all stability derivatives & inertias (Ph10, 11, 15); flight cond (Ph3).
    Out: {'omega_sp', 'zeta_sp', 'omega_ph', 'zeta_ph', 'omega_dr', 'zeta_dr',
          'T_spiral', 'level_meets_8785C'}
    Ref: Etkin & Reid 1996 Ch.6; MIL-F-8785C §3.2, §3.3."""
    ...

def phase14_dynamic_stability(Cm_a, Cm_q, Cm_a_dot, CL_a, I_yy, I_xx, I_zz,
                              S_w, c_bar_w, b_w, V_cruise, rho_cruise,
                              CL_trim: float = 0.60,
                              CD_trim: float = 0.05,
                              C_l_beta: float = -0.08,
                              C_l_p: Optional[float] = None,
                              C_l_r: float = 0.02,
                              C_n_beta: float = 0.06,
                              C_n_p: float = -0.02,
                              C_n_r: float = -0.20,
                              g: float = 9.80665,
                              short_period_zeta_min: float = 0.30,
                              short_period_zeta_max: float = 2.00,
                              short_period_omega_min_rad_s: float = 1.00,
                              phugoid_zeta_min: float = 0.04,
                              dutch_roll_zeta_min: float = 0.08,
                              dutch_roll_omega_min_rad_s: float = 0.40,
                              spiral_time_to_double_min_s: float = 12.0) -> Dict:
    """Simplified dynamic-stability verification."""
    if I_yy <= 0.0 or I_xx <= 0.0 or I_zz <= 0.0:
        raise ValueError("Moments of inertia must be positive.")
    if S_w <= 0.0 or c_bar_w <= 0.0 or b_w <= 0.0:
        raise ValueError("Wing reference geometry must be positive.")
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if rho_cruise <= 0.0:
        raise ValueError("rho_cruise must be positive.")
    if CL_a <= 0.0:
        raise ValueError("CL_a must be positive.")
    if CL_trim <= 0.0:
        raise ValueError("CL_trim must be positive.")
    if CD_trim <= 0.0:
        raise ValueError("CD_trim must be positive.")
    if g <= 0.0:
        raise ValueError("g must be positive.")
    if short_period_zeta_min < 0.0 or short_period_zeta_max <= short_period_zeta_min:
        raise ValueError("Short-period damping limits must be ordered and non-negative.")
    if short_period_omega_min_rad_s < 0.0:
        raise ValueError("short_period_omega_min_rad_s must be non-negative.")
    if phugoid_zeta_min < 0.0:
        raise ValueError("phugoid_zeta_min must be non-negative.")
    if dutch_roll_zeta_min < 0.0 or dutch_roll_omega_min_rad_s < 0.0:
        raise ValueError("Dutch-roll criteria must be non-negative.")
    if spiral_time_to_double_min_s <= 0.0:
        raise ValueError("spiral_time_to_double_min_s must be positive.")

    if C_l_p is None:
        C_l_p_used = -CL_a / 12.0
        C_l_p_source = "estimated_from_CL_a"
    else:
        C_l_p_used = float(C_l_p)
        C_l_p_source = "input"

    qbar = 0.5 * rho_cruise * V_cruise**2
    Cm_q_eff = Cm_q + Cm_a_dot

    M_alpha = qbar * S_w * c_bar_w * Cm_a
    M_q = qbar * S_w * c_bar_w**2 / (2.0 * V_cruise) * Cm_q_eff

    short_period_stable = Cm_a < 0.0 and Cm_q_eff < 0.0 and M_alpha < 0.0
    if short_period_stable:
        omega_sp = np.sqrt(-M_alpha / I_yy)
        zeta_sp = -M_q / (2.0 * np.sqrt(I_yy * (-M_alpha)))
    else:
        omega_sp = np.nan
        zeta_sp = np.nan

    short_period_meets = bool(
        short_period_stable
        and short_period_zeta_min <= zeta_sp <= short_period_zeta_max
        and omega_sp >= short_period_omega_min_rad_s
    )

    omega_ph = np.sqrt(2.0) * g / V_cruise
    zeta_ph = CD_trim / (np.sqrt(2.0) * CL_trim)
    phugoid_meets = bool(zeta_ph >= phugoid_zeta_min)

    L_p = qbar * S_w * b_w**2 / (2.0 * V_cruise) * C_l_p_used
    N_beta = qbar * S_w * b_w * C_n_beta
    N_r = qbar * S_w * b_w**2 / (2.0 * V_cruise) * C_n_r

    roll_subsidence_tau = -I_xx / L_p if L_p < 0.0 else np.inf
    dutch_roll_stable = C_n_beta > 0.0 and C_n_r < 0.0 and N_beta > 0.0
    if dutch_roll_stable:
        omega_dr = np.sqrt(N_beta / I_zz)
        zeta_dr = -N_r / (2.0 * np.sqrt(I_zz * N_beta))
    else:
        omega_dr = np.nan
        zeta_dr = np.nan

    dutch_roll_meets = bool(
        dutch_roll_stable
        and zeta_dr >= dutch_roll_zeta_min
        and omega_dr >= dutch_roll_omega_min_rad_s
    )

    spiral_numerator = C_l_beta * C_n_r - C_n_beta * C_l_r
    spiral_denominator = C_l_p_used * C_n_beta - C_n_p * C_l_beta
    if abs(spiral_denominator) > 1e-12:
        spiral_root = (g / V_cruise) * spiral_numerator / spiral_denominator
    else:
        spiral_root = np.nan

    if np.isfinite(spiral_root) and spiral_root < 0.0:
        spiral_stable = True
        spiral_time_constant_s = -1.0 / spiral_root
        spiral_time_to_double_s = np.inf
    elif np.isfinite(spiral_root) and spiral_root > 0.0:
        spiral_stable = False
        spiral_time_constant_s = np.inf
        spiral_time_to_double_s = np.log(2.0) / spiral_root
    else:
        spiral_stable = False
        spiral_time_constant_s = np.inf
        spiral_time_to_double_s = np.nan

    spiral_meets = bool(
        spiral_stable
        or (
            np.isfinite(spiral_time_to_double_s)
            and spiral_time_to_double_s >= spiral_time_to_double_min_s
        )
    )

    level_meets_8785C = bool(
        short_period_meets
        and phugoid_meets
        and dutch_roll_meets
        and spiral_meets
    )

    warnings = [
        "Phase 14 is a simplified verification model; replace with a full linearized 6-DOF state-space analysis before accepting stability margins.",
        "Lateral-directional derivatives are placeholders unless supplied from DATCOM, AVL, CFD, wind tunnel, or flight-test identification.",
        "The spiral-mode result is a reduced derivative proxy and should not be treated as a certified time-to-double calculation.",
    ]
    if not short_period_meets:
        warnings.append(
            "Short-period proxy does not meet the selected preliminary damping/frequency criteria."
        )
    if not phugoid_meets:
        warnings.append(
            "Phugoid damping proxy is below the selected preliminary criterion."
        )
    if not dutch_roll_meets:
        warnings.append(
            "Dutch-roll proxy does not meet the selected preliminary damping/frequency criteria."
        )
    if not spiral_meets:
        warnings.append(
            "Spiral proxy is divergent faster than the selected preliminary time-to-double criterion."
        )

    return {
        "level_meets_8785C_preliminary": level_meets_8785C,
        "short_period_meets": short_period_meets,
        "short_period_stable": bool(short_period_stable),
        "omega_sp_rad_s": None if not np.isfinite(omega_sp) else float(omega_sp),
        "zeta_sp": None if not np.isfinite(zeta_sp) else float(zeta_sp),
        "phugoid_meets": phugoid_meets,
        "omega_ph_rad_s": float(omega_ph),
        "zeta_ph": float(zeta_ph),
        "dutch_roll_meets": dutch_roll_meets,
        "dutch_roll_stable": bool(dutch_roll_stable),
        "omega_dr_rad_s": None if not np.isfinite(omega_dr) else float(omega_dr),
        "zeta_dr": None if not np.isfinite(zeta_dr) else float(zeta_dr),
        "spiral_meets": spiral_meets,
        "spiral_stable": bool(spiral_stable),
        "spiral_root_1_s": None if not np.isfinite(spiral_root) else float(spiral_root),
        "T_spiral_s": None if not np.isfinite(spiral_time_constant_s) else float(spiral_time_constant_s),
        "spiral_time_to_double_s": None if not np.isfinite(spiral_time_to_double_s) else float(spiral_time_to_double_s),
        "roll_subsidence_tau_s": None if not np.isfinite(roll_subsidence_tau) else float(roll_subsidence_tau),
        "qbar_Pa": float(qbar),
        "M_alpha_Nm_per_rad": float(M_alpha),
        "M_q_Nm_per_rad_s": float(M_q),
        "L_p_Nm_per_rad_s": float(L_p),
        "N_beta_Nm_per_rad": float(N_beta),
        "N_r_Nm_per_rad_s": float(N_r),
        "Cm_alpha": float(Cm_a),
        "Cm_q": float(Cm_q),
        "Cm_alpha_dot": float(Cm_a_dot),
        "Cm_q_effective": float(Cm_q_eff),
        "CL_alpha_total": float(CL_a),
        "CL_trim": float(CL_trim),
        "CD_trim": float(CD_trim),
        "C_l_beta": float(C_l_beta),
        "C_l_p": float(C_l_p_used),
        "C_l_p_source": C_l_p_source,
        "C_l_r": float(C_l_r),
        "C_n_beta": float(C_n_beta),
        "C_n_p": float(C_n_p),
        "C_n_r": float(C_n_r),
        "I_xx": float(I_xx),
        "I_yy": float(I_yy),
        "I_zz": float(I_zz),
        "S_w": float(S_w),
        "c_bar_w": float(c_bar_w),
        "b_w": float(b_w),
        "V_cruise": float(V_cruise),
        "rho_cruise": float(rho_cruise),
        "criteria": {
            "short_period_zeta_min": float(short_period_zeta_min),
            "short_period_zeta_max": float(short_period_zeta_max),
            "short_period_omega_min_rad_s": float(short_period_omega_min_rad_s),
            "phugoid_zeta_min": float(phugoid_zeta_min),
            "dutch_roll_zeta_min": float(dutch_roll_zeta_min),
            "dutch_roll_omega_min_rad_s": float(dutch_roll_omega_min_rad_s),
            "spiral_time_to_double_min_s": float(spiral_time_to_double_min_s),
        },
        "notes": [
            "Short-period proxy uses Iyy*theta_ddot + damping*theta_dot + stiffness*theta = 0 with M_alpha = q*S*c*Cm_alpha.",
            "Phugoid proxy uses omega_ph = sqrt(2)*g/V and zeta_ph = CD/(sqrt(2)*CL).",
            "Dutch-roll proxy uses yaw stiffness N_beta and yaw damping N_r only.",
            "Spiral proxy uses lambda = (g/V)*(Cl_beta*Cn_r - Cn_beta*Cl_r)/(Cl_p*Cn_beta - Cn_p*Cl_beta).",
        ],
        "warnings": warnings,
    }


def iterate_phases_1_to_14(MTOW_kg: float = 50.0,
                           mission: Optional[Mission] = None,
                           assumptions: Optional[Assumptions] = None,
                           max_inner_iter: int = 15,
                           j_tol: float = 0.01,
                           area_tol: float = 0.01,
                           re_tol: float = 0.02,
                           constraint_plot_path=None,
                           airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-13, then append Phase 14 dynamic-stability verification."""
    result = iterate_phases_1_to_13(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()

    phase3 = result["phase3"]
    phase8 = result["phase8"]
    phase9 = result["phase9"]
    phase10 = result["phase10"]
    phase11 = result["phase11"]
    phase12 = result["phase12"]

    rho_cruise, _, _, _ = isa(mission.altitude_m)
    CL_alpha_total = phase10["a_w_eff"] + phase10["a_c_eff"] * phase10["area_ratio"]
    static_margin_used = phase10["SM_min"]
    Cm_alpha = -static_margin_used * CL_alpha_total

    if assumptions.dynamic_Cm_q is None:
        Cm_q = -2.0 * phase9["CL_a_c"] * phase10["V_bar_c"] * (
            phase9["l_c"] / phase8["c_bar"]
        )
        Cm_q_source = "canard_volume_estimate"
    else:
        Cm_q = assumptions.dynamic_Cm_q
        Cm_q_source = "assumption"

    C_l_p_input = (
        assumptions.dynamic_Cl_p
        if assumptions.dynamic_Cl_p is not None
        else phase11.get("C_l_p")
    )

    phase14 = phase14_dynamic_stability(
        Cm_alpha,
        Cm_q,
        assumptions.dynamic_Cm_alpha_dot,
        CL_alpha_total,
        phase12["I_yy"],
        phase12["I_xx"],
        phase12["I_zz"],
        phase8["S"],
        phase8["c_bar"],
        phase8["b"],
        phase3["V_cruise"],
        rho_cruise,
        CL_trim=phase10["CL_trim"],
        CD_trim=phase3["CD_cruise"],
        C_l_beta=assumptions.dynamic_Cl_beta,
        C_l_p=C_l_p_input,
        C_l_r=assumptions.dynamic_Cl_r,
        C_n_beta=assumptions.dynamic_Cn_beta,
        C_n_p=assumptions.dynamic_Cn_p,
        C_n_r=assumptions.dynamic_Cn_r,
        g=assumptions.g,
        short_period_zeta_min=assumptions.short_period_zeta_min,
        short_period_zeta_max=assumptions.short_period_zeta_max,
        short_period_omega_min_rad_s=assumptions.short_period_omega_min_rad_s,
        phugoid_zeta_min=assumptions.phugoid_zeta_min,
        dutch_roll_zeta_min=assumptions.dutch_roll_zeta_min,
        dutch_roll_omega_min_rad_s=assumptions.dutch_roll_omega_min_rad_s,
        spiral_time_to_double_min_s=assumptions.spiral_time_to_double_min_s,
    )
    phase14["derivative_sources"] = {
        "Cm_alpha": "static_margin_times_total_lift_curve_slope",
        "Cm_q": Cm_q_source,
        "Cm_alpha_dot": "assumption",
        "CL_alpha_total": "wing_plus_canard_area_weighted_lift_curve_slope",
        "C_l_p": "assumption" if assumptions.dynamic_Cl_p is not None else "phase11_or_CL_alpha_estimate",
        "C_l_beta": "assumption",
        "C_l_r": "assumption",
        "C_n_beta": "assumption",
        "C_n_p": "assumption",
        "C_n_r": "assumption",
        "inertias": "phase12_preliminary_radius_of_gyration_or_user_assumption",
    }

    result = dict(result)
    result["phase14"] = phase14
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 14 is appended as a dynamic-stability verification block using placeholder derivatives where measured data are unavailable."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase14["warnings"]:
        result["warnings"].extend(f"phase14: {warning}" for warning in phase14["warnings"])

    if result.get("converged"):
        if phase11 and not phase11.get("feasible_preliminary_elevon", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 11 fixed-wing elevon remains infeasible; Phase 14 dynamic-stability check appended"
            )
        elif phase12 and not phase12.get("feasible_preliminary_hover_control", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 12 hover control remains infeasible; Phase 14 dynamic-stability check appended"
            )
        elif not phase14["level_meets_8785C_preliminary"]:
            result["stopped_reason"] = (
                "propulsion/aero/control loop converged, but Phase 14 has preliminary dynamic-stability failures"
            )
        else:
            result["stopped_reason"] = (
                "propulsion, aerodynamics, controls, transition, and Phase 14 dynamic-stability verification completed"
            )
    return result

# ---------- PHASE 15 ----------

def _phase15_mass_outline(S_w, S_c, fuselage_length, P_motor_cont_W, m_battery_kg,
                          mission_equipment_mass_kg: float = 7.3,
                 UAV_correction_wing: float = 0.70,
                 UAV_correction_fuse: float = 0.60) -> Dict:
    """Class II mass breakdown with UAV correction factors.
    In:  geometric outputs, P_cont (Ph5), m_batt (Ph5).
    Out: {'m_wing', 'm_canard', 'm_fuse', 'm_prop', 'm_motor', 'm_ESC',
          'm_batt', 'm_avionics', 'm_mission_equipment',
          'MTOW_kg', 'x_CG', 'I_tensor'}
    Eq:  m_wing  = K_w · f(S,W) [Roskam Part V Tab. 5.1] · UAV_correction
         m_fuse  = Raymer Eq.15.49 · UAV_correction
         m_motor = 0.27 g/W · P_cont                  — Gundlach 2014 Tab. 9.3
    Ref: Gundlach 2014 Ch.19 cautions 30–50 % overestimate without UAV correction."""
    ...

def phase15_mass(S_w, S_c, fuselage_length, P_motor_cont_W, m_battery_kg,
                 mission_equipment_mass_kg: float,
                 n_rotors: int = 4,
                 prop_diameter_m: Optional[float] = None,
                 b_w: Optional[float] = None,
                 c_bar_w: Optional[float] = None,
                 x_ac_w: Optional[float] = None,
                 x_ac_c: Optional[float] = None,
                 wing_mac_le_x_m: float = 0.0,
                 external_tow_load_N: float = 0.0,
                 g: float = 9.80665,
                 wing_areal_density_kg_m2: float = 2.00,
                 canard_areal_density_kg_m2: float = 1.70,
                 fuselage_linear_density_kg_m: float = 2.20,
                 boom_landing_gear_mass_kg: float = 2.50,
                 motor_specific_mass_kg_W: float = 0.00027,
                 esc_specific_mass_kg_W: float = 0.00008,
                 prop_mass_coeff_kg_m2: float = 0.10,
                 avionics_mass_kg: float = 1.50,
                 wiring_fraction: float = 0.06,
                 mass_contingency_fraction: float = 0.08,
                 Ixx_radius_fraction_span: float = 0.20,
                 Iyy_radius_fraction_fuselage: float = 0.35,
                 Izz_radius_fraction_span: float = 0.22) -> Dict:
    """Preliminary mass breakdown using project-level first-cut coefficients."""
    if S_w <= 0.0 or S_c <= 0.0:
        raise ValueError("Wing and canard areas must be positive.")
    if fuselage_length <= 0.0:
        raise ValueError("fuselage_length must be positive.")
    if P_motor_cont_W <= 0.0:
        raise ValueError("P_motor_cont_W must be positive.")
    if m_battery_kg <= 0.0:
        raise ValueError("m_battery_kg must be positive.")
    if mission_equipment_mass_kg < 0.0:
        raise ValueError("mission_equipment_mass_kg must be non-negative.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if external_tow_load_N < 0.0:
        raise ValueError("external_tow_load_N must be non-negative.")
    if g <= 0.0:
        raise ValueError("g must be positive.")

    coefficients = [
        wing_areal_density_kg_m2,
        canard_areal_density_kg_m2,
        fuselage_linear_density_kg_m,
        boom_landing_gear_mass_kg,
        motor_specific_mass_kg_W,
        esc_specific_mass_kg_W,
        prop_mass_coeff_kg_m2,
        avionics_mass_kg,
        wiring_fraction,
        mass_contingency_fraction,
        Ixx_radius_fraction_span,
        Iyy_radius_fraction_fuselage,
        Izz_radius_fraction_span,
    ]
    if any(value < 0.0 for value in coefficients):
        raise ValueError("Mass and inertia coefficients must be non-negative.")

    prop_diameter_used = 0.0 if prop_diameter_m is None else float(prop_diameter_m)
    if prop_diameter_used < 0.0:
        raise ValueError("prop_diameter_m must be non-negative when provided.")

    b_w_used = np.sqrt(S_w * 7.0) if b_w is None else float(b_w)
    c_bar_used = S_w / b_w_used if c_bar_w is None else float(c_bar_w)
    if b_w_used <= 0.0 or c_bar_used <= 0.0:
        raise ValueError("Wing span and mean chord must be positive.")

    wing_mac_le_x_m = float(wing_mac_le_x_m)
    x_wing_local = 0.25 * c_bar_used if x_ac_w is None else float(x_ac_w)
    x_canard_local = (
        x_wing_local - 2.5 * c_bar_used
        if x_ac_c is None
        else float(x_ac_c)
    )
    x_reference_fuselage = 0.0
    x_wing_fuselage = wing_mac_le_x_m + x_wing_local
    x_canard_fuselage = wing_mac_le_x_m + x_canard_local

    m_wing = wing_areal_density_kg_m2 * S_w
    m_canard = canard_areal_density_kg_m2 * S_c
    m_fuselage = fuselage_linear_density_kg_m * fuselage_length
    m_boom_landing_gear = boom_landing_gear_mass_kg
    m_motor = motor_specific_mass_kg_W * P_motor_cont_W
    m_ESC = esc_specific_mass_kg_W * P_motor_cont_W
    m_propeller = n_rotors * prop_mass_coeff_kg_m2 * prop_diameter_used**2
    m_avionics = avionics_mass_kg
    m_mission_equipment = mission_equipment_mass_kg

    base_masses = {
        "wing": m_wing,
        "canard": m_canard,
        "fuselage": m_fuselage,
        "boom_landing_gear": m_boom_landing_gear,
        "motors": m_motor,
        "ESCs": m_ESC,
        "propellers": m_propeller,
        "battery": m_battery_kg,
        "avionics": m_avionics,
        "mission_equipment": m_mission_equipment,
    }
    base_mass = sum(base_masses.values())
    m_wiring = wiring_fraction * (m_motor + m_ESC + m_avionics)
    contingency_base = base_mass + m_wiring
    m_contingency = mass_contingency_fraction * contingency_base
    MTOW_estimate_kg = contingency_base + m_contingency

    component_masses = dict(base_masses)
    component_masses["wiring"] = m_wiring
    component_masses["contingency"] = m_contingency

    x_locations_fuselage = {
        "wing": x_wing_fuselage,
        "canard": x_canard_fuselage,
        "fuselage": x_reference_fuselage,
        "boom_landing_gear": x_reference_fuselage,
        "motors": x_reference_fuselage,
        "ESCs": x_reference_fuselage,
        "propellers": x_reference_fuselage,
        "battery": x_reference_fuselage,
        "avionics": x_reference_fuselage,
        "mission_equipment": x_reference_fuselage,
        "wiring": x_reference_fuselage,
        "contingency": x_reference_fuselage,
    }
    x_locations = {
        name: value - wing_mac_le_x_m
        for name, value in x_locations_fuselage.items()
    }
    x_CG_fuselage = (
        sum(
            component_masses[name] * x_locations_fuselage[name]
            for name in component_masses
        )
        / MTOW_estimate_kg
    )
    x_CG = x_CG_fuselage - wing_mac_le_x_m

    Ixx = MTOW_estimate_kg * (Ixx_radius_fraction_span * b_w_used) ** 2
    Iyy = MTOW_estimate_kg * (Iyy_radius_fraction_fuselage * fuselage_length) ** 2
    Izz = MTOW_estimate_kg * (Izz_radius_fraction_span * b_w_used) ** 2

    structure_mass = m_wing + m_canard + m_fuselage + m_boom_landing_gear
    propulsion_mass = m_motor + m_ESC + m_propeller
    fixed_equipment_mass = m_avionics + m_mission_equipment
    external_tow_load_equivalent_kg = external_tow_load_N / g

    warnings = [
        "Phase 15 is a first-cut mass model; replace areal densities, propulsion masses, and CG locations with CAD and datasheet values.",
        "All nonlifting components are placed at a provisional fuselage/equipment datum for the preliminary CG estimate.",
        "wing_mac_le_x_m shifts the wing MAC leading edge relative to that provisional mass datum; the canard arm remains fixed in this first-cut layout solve.",
        "The contingency mass is included in MTOW but has no independent CAD location yet.",
    ]
    if prop_diameter_m is None:
        warnings.append(
            "No propeller diameter was provided; propeller mass was set to zero."
        )
    if external_tow_load_N > 0.0:
        warnings.append(
            "external_tow_load_N is reported as an external load and is not included in UAV MTOW."
        )

    return {
        "MTOW_estimate_kg": float(MTOW_estimate_kg),
        "empty_mass_no_battery_no_mission_kg": float(
            MTOW_estimate_kg - m_battery_kg - m_mission_equipment
        ),
        "structure_mass_kg": float(structure_mass),
        "propulsion_mass_kg": float(propulsion_mass),
        "fixed_equipment_mass_kg": float(fixed_equipment_mass),
        "component_masses_kg": {name: float(value) for name, value in component_masses.items()},
        "component_x_locations_m": {name: float(value) for name, value in x_locations.items()},
        "component_x_locations_fuselage_m": {
            name: float(value) for name, value in x_locations_fuselage.items()
        },
        "wing_mac_le_x_m": float(wing_mac_le_x_m),
        "x_CG_fuselage_m": float(x_CG_fuselage),
        "x_CG_m": float(x_CG),
        "x_CG_over_wing_mac": float(x_CG / c_bar_used),
        "I_tensor_kg_m2": {
            "Ixx": float(Ixx),
            "Iyy": float(Iyy),
            "Izz": float(Izz),
            "Ixy": 0.0,
            "Ixz": 0.0,
            "Iyz": 0.0,
        },
        "m_wing_kg": float(m_wing),
        "m_canard_kg": float(m_canard),
        "m_fuselage_kg": float(m_fuselage),
        "m_boom_landing_gear_kg": float(m_boom_landing_gear),
        "m_motor_kg": float(m_motor),
        "m_ESC_kg": float(m_ESC),
        "m_propeller_kg": float(m_propeller),
        "m_battery_kg": float(m_battery_kg),
        "m_avionics_kg": float(m_avionics),
        "m_mission_equipment_kg": float(m_mission_equipment),
        "m_wiring_kg": float(m_wiring),
        "m_contingency_kg": float(m_contingency),
        "external_tow_load_N": float(external_tow_load_N),
        "external_tow_load_equivalent_kg": float(external_tow_load_equivalent_kg),
        "external_tow_load_included_in_MTOW": False,
        "mass_coefficients": {
            "wing_areal_density_kg_m2": float(wing_areal_density_kg_m2),
            "canard_areal_density_kg_m2": float(canard_areal_density_kg_m2),
            "fuselage_linear_density_kg_m": float(fuselage_linear_density_kg_m),
            "motor_specific_mass_kg_W": float(motor_specific_mass_kg_W),
            "esc_specific_mass_kg_W": float(esc_specific_mass_kg_W),
            "prop_mass_coeff_kg_m2": float(prop_mass_coeff_kg_m2),
            "wiring_fraction": float(wiring_fraction),
            "mass_contingency_fraction": float(mass_contingency_fraction),
        },
        "notes": [
            "m_mission_equipment_kg contains onboard netgun, sensor, and mission hardware.",
            "No separate capture mass is included in this model.",
            "m_motor_kg = motor_specific_mass_kg_W * P_motor_cont_W.",
            "m_ESC_kg = esc_specific_mass_kg_W * P_motor_cont_W.",
            "m_propeller_kg = n_rotors * prop_mass_coeff_kg_m2 * D_prop^2.",
            "Phase 15 reports MTOW_estimate_kg; Phase 16 will decide whether to iterate the sizing MTOW.",
        ],
        "warnings": warnings,
    }


def _estimate_phase15_from_result(result: Dict, phase5: Dict, phase9: Dict,
                                  mission: Mission, assumptions: Assumptions,
                                  wing_mac_le_x_m: Optional[float] = None) -> Dict:
    """Run Phase 15 from an existing sizing result and a selected canard."""
    wing_x = (
        0.0
        if wing_mac_le_x_m is None
        else float(wing_mac_le_x_m)
    )
    return phase15_mass(
        result["phase8"]["S"],
        phase9["S_c"],
        assumptions.fuselage_length_m,
        phase5["P_motor_cont"],
        phase5["m_batt_kg"],
        mission.mission_equipment_mass_kg,
        n_rotors=assumptions.n_rotors,
        prop_diameter_m=result["phase1"]["D_prop"],
        b_w=result["phase8"]["b"],
        c_bar_w=result["phase8"]["c_bar"],
        x_ac_w=result["phase8"]["x_ac_w"],
        x_ac_c=phase9["x_ac_c"],
        wing_mac_le_x_m=wing_x,
        external_tow_load_N=mission.external_tow_load_N,
        g=assumptions.g,
        wing_areal_density_kg_m2=assumptions.wing_areal_density_kg_m2,
        canard_areal_density_kg_m2=assumptions.canard_areal_density_kg_m2,
        fuselage_linear_density_kg_m=assumptions.fuselage_linear_density_kg_m,
        boom_landing_gear_mass_kg=assumptions.boom_landing_gear_mass_kg,
        motor_specific_mass_kg_W=assumptions.motor_specific_mass_kg_W,
        esc_specific_mass_kg_W=assumptions.esc_specific_mass_kg_W,
        prop_mass_coeff_kg_m2=assumptions.prop_mass_coeff_kg_m2,
        avionics_mass_kg=assumptions.avionics_mass_kg,
        wiring_fraction=assumptions.wiring_fraction,
        mass_contingency_fraction=assumptions.mass_contingency_fraction,
        Ixx_radius_fraction_span=assumptions.hover_Ixx_radius_fraction_span,
        Iyy_radius_fraction_fuselage=assumptions.hover_Iyy_radius_fraction_fuselage,
        Izz_radius_fraction_span=assumptions.hover_Izz_radius_fraction_span,
    )


def _operational_cg_envelope(phase15: Dict, phase8: Dict,
                             assumptions: Assumptions) -> Dict:
    """Build a simple operational CG envelope around the Phase 15 mass CG."""
    c_bar = phase8["c_bar"]
    half_width_m = assumptions.cg_envelope_half_width_over_mac * c_bar
    margin_m = assumptions.cg_required_margin_over_mac * c_bar
    x_cg = phase15["x_CG_m"]
    x_fwd = x_cg - half_width_m
    x_aft = x_cg + half_width_m
    return {
        "x_cg_mass_m": float(x_cg),
        "x_cg_mass_over_mac": float(x_cg / c_bar),
        "x_cg_fwd_operational_m": float(x_fwd),
        "x_cg_aft_operational_m": float(x_aft),
        "x_cg_fwd_operational_over_mac": float(x_fwd / c_bar),
        "x_cg_aft_operational_over_mac": float(x_aft / c_bar),
        "cg_envelope_half_width_m": float(half_width_m),
        "cg_envelope_half_width_over_mac": float(assumptions.cg_envelope_half_width_over_mac),
        "cg_required_margin_m": float(margin_m),
        "cg_required_margin_over_mac": float(assumptions.cg_required_margin_over_mac),
        "operational_CG_range_m": float(2.0 * half_width_m),
        "operational_CG_range_over_mac": float(2.0 * assumptions.cg_envelope_half_width_over_mac),
    }


def _attach_operational_cg_to_phase10(phase10: Dict, cg_envelope: Dict,
                                      phase8: Dict) -> Dict:
    """Add operational CG-envelope checks to a Phase 10 scissor result."""
    c_bar = phase8["c_bar"]
    margin_m = cg_envelope["cg_required_margin_m"]
    half_width_m = cg_envelope["cg_envelope_half_width_m"]

    fwd_margin_m = cg_envelope["x_cg_fwd_operational_m"] - phase10["x_cg_fwd"]
    aft_margin_m = phase10["x_cg_aft"] - cg_envelope["x_cg_aft_operational_m"]
    mass_fwd_margin_m = cg_envelope["x_cg_mass_m"] - phase10["x_cg_fwd"]
    mass_aft_margin_m = phase10["x_cg_aft"] - cg_envelope["x_cg_mass_m"]
    feasible = fwd_margin_m >= margin_m and aft_margin_m >= margin_m
    mass_inside = mass_fwd_margin_m >= 0.0 and mass_aft_margin_m >= 0.0

    center_min = phase10["x_cg_fwd"] + margin_m + half_width_m
    center_max = phase10["x_cg_aft"] - margin_m - half_width_m
    if center_min <= center_max:
        target_center = float(np.clip(cg_envelope["x_cg_mass_m"], center_min, center_max))
        required_shift_m = target_center - cg_envelope["x_cg_mass_m"]
        envelope_width_excess_m = 0.0
    else:
        required_shift_m = np.nan
        envelope_width_excess_m = center_min - center_max

    result = dict(phase10)
    result.update(cg_envelope)
    result.update({
        "operational_CG_feasible": bool(feasible),
        "mass_CG_inside_theoretical": bool(mass_inside),
        "operational_fwd_margin_m": float(fwd_margin_m),
        "operational_aft_margin_m": float(aft_margin_m),
        "operational_fwd_margin_over_mac": float(fwd_margin_m / c_bar),
        "operational_aft_margin_over_mac": float(aft_margin_m / c_bar),
        "mass_CG_fwd_margin_m": float(mass_fwd_margin_m),
        "mass_CG_aft_margin_m": float(mass_aft_margin_m),
        "cg_center_min_with_margin_m": float(center_min),
        "cg_center_max_with_margin_m": float(center_max),
        "required_CG_shift_m": None if not np.isfinite(required_shift_m) else float(required_shift_m),
        "required_CG_shift_over_mac": (
            None if not np.isfinite(required_shift_m) else float(required_shift_m / c_bar)
        ),
        "cg_envelope_width_excess_m": float(envelope_width_excess_m),
        "cg_envelope_width_excess_over_mac": float(envelope_width_excess_m / c_bar),
    })
    result["warnings"] = list(result.get("warnings", []))
    if not feasible:
        result["warnings"].append(
            "Operational CG envelope does not fit inside the theoretical Phase 10 CG limits with the required margin."
        )
    if not mass_inside:
        result["warnings"].append(
            "Mass-model CG lies outside the theoretical Phase 10 CG range."
        )
    return result


def _run_phase10_for_phase9(result: Dict, phase9: Dict, phase15: Dict,
                            assumptions: Assumptions) -> Dict:
    """Run Phase 10 and attach operational CG-envelope data."""
    phase8 = result["phase8"]
    phase10 = phase10_scissor_canard(
        phase8["S"],
        phase8["c_bar"],
        phase8["x_ac_w"],
        phase8["CL_a"],
        phase9["S_c"],
        phase9["l_c"],
        phase9["CL_a_c"],
        assumptions.canard_eps_alpha_c,
        assumptions.wing_eps_alpha_w,
        phase9["CL_max_3D_c"],
        result["phase3"]["CL_cruise"],
        assumptions.static_margin_min,
    )
    return _attach_operational_cg_to_phase10(
        phase10,
        _operational_cg_envelope(phase15, phase8, assumptions),
        phase8,
    )


def _target_cg_center_from_phase10(phase10: Dict, phase8: Dict,
                                   assumptions: Assumptions) -> Dict:
    """Find the wing-local CG target that centers the operational envelope in Phase 10."""
    c_bar = phase8["c_bar"]
    half_width_m = assumptions.cg_envelope_half_width_over_mac * c_bar
    margin_m = assumptions.cg_required_margin_over_mac * c_bar
    center_min = phase10["x_cg_fwd"] + margin_m + half_width_m
    center_max = phase10["x_cg_aft"] - margin_m - half_width_m
    feasible_width = center_min <= center_max
    if feasible_width:
        target_center = 0.5 * (center_min + center_max)
    else:
        target_center = 0.5 * (phase10["x_cg_fwd"] + phase10["x_cg_aft"])

    return {
        "target_x_cg_m": float(target_center),
        "target_x_cg_over_mac": float(target_center / c_bar),
        "target_center_min_m": float(center_min),
        "target_center_max_m": float(center_max),
        "target_center_min_over_mac": float(center_min / c_bar),
        "target_center_max_over_mac": float(center_max / c_bar),
        "target_width_feasible": bool(feasible_width),
        "required_theoretical_cg_range_over_mac": float(
            2.0 * assumptions.cg_envelope_half_width_over_mac
            + 2.0 * assumptions.cg_required_margin_over_mac
        ),
        "theoretical_cg_range_over_mac": float(phase10["CG_range_m"] / c_bar),
    }


def _phase15_phase10_with_wing_layout(result: Dict, phase5: Dict, phase9: Dict,
                                      mission: Mission,
                                      assumptions: Assumptions
                                      ) -> Tuple[Dict, Dict, Dict]:
    """Solve the wing MAC station needed to place mass CG inside the scissor range."""
    initial_wing_x = (
        0.0
        if assumptions.wing_mac_le_x_m is None
        else float(assumptions.wing_mac_le_x_m)
    )
    phase15_initial = _estimate_phase15_from_result(
        result,
        phase5,
        phase9,
        mission,
        assumptions,
        wing_mac_le_x_m=initial_wing_x,
    )
    phase10_initial = _run_phase10_for_phase9(
        result,
        phase9,
        phase15_initial,
        assumptions,
    )
    target = _target_cg_center_from_phase10(
        phase10_initial,
        result["phase8"],
        assumptions,
    )

    phase15 = phase15_initial
    phase10 = phase10_initial
    solved_wing_x = initial_wing_x
    solved = False
    warnings = []

    lift_mass = phase15_initial["m_wing_kg"] + phase15_initial["m_canard_kg"]
    total_mass = phase15_initial["MTOW_estimate_kg"]
    slope = lift_mass / total_mass - 1.0
    required_wing_shift_m = None

    if assumptions.solve_wing_position_for_cg:
        if not target["target_width_feasible"]:
            warnings.append(
                "The Phase 10 scissor range is too narrow for the requested operational CG envelope and margin; wing station alone cannot make it feasible."
            )
        elif abs(slope) < 1e-9:
            warnings.append(
                "Wing station solve is singular because nearly all mass moves with the lifting group."
            )
        else:
            required_wing_shift_m = (
                target["target_x_cg_m"] - phase15_initial["x_CG_m"]
            ) / slope
            solved_wing_x = initial_wing_x + required_wing_shift_m
            if np.isfinite(solved_wing_x):
                phase15 = _estimate_phase15_from_result(
                    result,
                    phase5,
                    phase9,
                    mission,
                    assumptions,
                    wing_mac_le_x_m=solved_wing_x,
                )
                phase10 = _run_phase10_for_phase9(
                    result,
                    phase9,
                    phase15,
                    assumptions,
                )
                solved = True
            else:
                solved_wing_x = initial_wing_x
                warnings.append(
                    "Wing station solve produced a non-finite location; keeping the initial wing station."
                )

    c_bar = result["phase8"]["c_bar"]
    layout_report = {
        "enabled": bool(assumptions.solve_wing_position_for_cg),
        "solved": bool(solved),
        "initial_wing_mac_le_x_m": float(initial_wing_x),
        "solved_wing_mac_le_x_m": float(solved_wing_x),
        "required_wing_shift_m": (
            None if required_wing_shift_m is None else float(required_wing_shift_m)
        ),
        "initial_x_CG_over_mac": float(phase15_initial["x_CG_over_wing_mac"]),
        "final_x_CG_over_mac": float(phase15["x_CG_over_wing_mac"]),
        "initial_x_CG_fuselage_m": float(phase15_initial["x_CG_fuselage_m"]),
        "final_x_CG_fuselage_m": float(phase15["x_CG_fuselage_m"]),
        "target_x_CG_over_mac": target["target_x_cg_over_mac"],
        "target_x_CG_m": target["target_x_cg_m"],
        "target_center_min_over_mac": target["target_center_min_over_mac"],
        "target_center_max_over_mac": target["target_center_max_over_mac"],
        "target_width_feasible": target["target_width_feasible"],
        "cg_change_per_wing_shift": float(slope),
        "lifting_group_mass_fraction": float(lift_mass / total_mass),
        "operational_CG_feasible_initial": phase10_initial["operational_CG_feasible"],
        "operational_CG_feasible_final": phase10["operational_CG_feasible"],
        "final_operational_fwd_margin_over_mac": phase10["operational_fwd_margin_over_mac"],
        "final_operational_aft_margin_over_mac": phase10["operational_aft_margin_over_mac"],
        "wing_station_over_mac": float(solved_wing_x / c_bar),
        "notes": [
            "This solve shifts the wing MAC leading edge relative to a provisional fuselage/equipment mass datum.",
            "The canard arm ratio is held fixed, so the first-cut solve moves the lifting group as a block.",
            "CAD component locations should replace the provisional mass datum before final acceptance.",
        ],
        "warnings": warnings,
    }

    phase15 = dict(phase15)
    phase15["wing_layout_solver"] = layout_report
    phase10 = dict(phase10)
    phase10["wing_layout_solver"] = layout_report
    return phase15, phase10, layout_report


def _canard_cg_grid_search(result: Dict, phase5: Dict, mission: Mission,
                           assumptions: Assumptions) -> Tuple[Dict, Dict, Dict, Dict]:
    """Select the smallest canard volume that fits the operational CG envelope."""
    phase8 = result["phase8"]
    phase7 = result["phase7"]
    rho_target, mu_target, a_sound_target, _ = isa(mission.altitude_m)
    mach_cruise = result["phase3"]["V_cruise"] / a_sound_target
    l_c = assumptions.canard_arm_chord_ratio * phase8["c_bar"]
    default_phase9 = result["phase9"]
    default_phase15, default_phase10, default_layout = _phase15_phase10_with_wing_layout(
        result,
        phase5,
        default_phase9,
        mission,
        assumptions,
    )

    start = assumptions.canard_volume_grid_min
    stop = assumptions.canard_volume_grid_max
    step = assumptions.canard_volume_grid_step
    if step <= 0.0 or stop < start:
        raise ValueError("Canard volume grid bounds are invalid.")

    candidates = []
    selected = None
    for V_bar_c in np.arange(start, stop + 0.5 * step, step):
        phase9_candidate = phase9_canard(
            phase8["S"],
            phase8["c_bar"],
            phase8["x_ac_w"],
            l_c,
            float(V_bar_c),
            assumptions.canard_AR,
            assumptions.canard_taper,
            assumptions.canard_sweep_c4_rad,
            phase7["canard"]["cl_max"],
            mach_cruise,
        )
        phase15_candidate, phase10_candidate, layout_candidate = _phase15_phase10_with_wing_layout(
            result,
            phase5,
            phase9_candidate,
            mission,
            assumptions,
        )
        min_margin = min(
            phase10_candidate["operational_fwd_margin_over_mac"],
            phase10_candidate["operational_aft_margin_over_mac"],
        )
        candidate_record = {
            "V_bar_c": float(V_bar_c),
            "S_c": phase9_candidate["S_c"],
            "MTOW_estimate_kg": phase15_candidate["MTOW_estimate_kg"],
            "wing_mac_le_x_m": phase15_candidate["wing_mac_le_x_m"],
            "x_CG_over_mac": phase15_candidate["x_CG_over_wing_mac"],
            "target_x_CG_over_mac": layout_candidate["target_x_CG_over_mac"],
            "operational_CG_feasible": phase10_candidate["operational_CG_feasible"],
            "min_operational_margin_over_mac": float(min_margin),
            "required_CG_shift_over_mac": phase10_candidate["required_CG_shift_over_mac"],
            "cg_envelope_width_excess_over_mac": phase10_candidate["cg_envelope_width_excess_over_mac"],
            "wing_layout_solved": layout_candidate["solved"],
        }
        candidates.append(candidate_record)
        if selected is None and phase10_candidate["operational_CG_feasible"]:
            selected = (phase9_candidate, phase15_candidate, phase10_candidate, layout_candidate)

    warnings = []
    if selected is None:
        selected = (default_phase9, default_phase15, default_phase10, default_layout)
        warnings.append(
            "No canard-volume candidate fit the operational CG envelope; keeping the current canard and reporting required CG shift."
        )

    selected_phase9, selected_phase15, selected_phase10, selected_layout = selected
    Re_canard = (
        rho_target
        * result["phase3"]["V_cruise"]
        * selected_phase9["c_bar_c"]
        / mu_target
    )
    search = {
        "selected_V_bar_c": selected_phase9["V_bar_c"],
        "selected_S_c": selected_phase9["S_c"],
        "selected_Re_canard": float(Re_canard),
        "selected_wing_layout": selected_layout,
        "default_V_bar_c": default_phase9["V_bar_c"],
        "default_operational_CG_feasible": default_phase10["operational_CG_feasible"],
        "default_wing_layout": default_layout,
        "wing_position_solve_enabled": bool(assumptions.solve_wing_position_for_cg),
        "grid_min": float(start),
        "grid_max": float(stop),
        "grid_step": float(step),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warnings": warnings,
        "notes": [
            "The canard arm ratio is held fixed during this grid search.",
            "The first feasible candidate is selected so canard area is kept as small as possible.",
            "The selected canard Reynolds number is reported; final XFOIL or measured-polar checks should use that selected value.",
        ],
    }
    return selected_phase9, selected_phase15, selected_phase10, search


def _phase5_with_transition_energy(result: Dict, phase13: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Recompute Phase 5 using Phase 13 transition energy when available."""
    if phase13.get("P_transition_average_W") is not None:
        transition_segment = (
            phase13["P_transition_average_W"],
            phase13["t_transition"],
        )
        source = "phase13_transition_blending"
    else:
        transition_segment = (
            result["phase3"]["P_transition"],
            result["phase3"]["t_transition"],
        )
        source = "phase3_mission_optimise_fallback"

    phase5 = phase5_energy_battery(
        {
            "transition": transition_segment,
            "fixed_wing_climb": (result["phase3"]["P_fw"], result["phase3"]["t_fw"]),
            "mission_hover": (result["phase3"]["P_hover_target"], mission.hover_time_s),
        },
        assumptions.eta_batt,
        assumptions.usable_battery_fraction,
        assumptions.battery_specific_energy_Wh_kg,
    )
    phase5["transition_energy_source"] = source
    phase5["phase3_transition_energy_Wh"] = result["phase3"]["E_transition_Wh"]
    phase5["phase13_transition_energy_Wh"] = phase13.get("E_transition_estimate_Wh")
    return phase5


def _run_phase11_with_operational_cg(result: Dict, phase10: Dict,
                                     assumptions: Assumptions) -> Dict:
    """Run Phase 11 with operational CG travel instead of full theoretical range."""
    phase8 = result["phase8"]
    return phase11_elevon_FW(
        phase8["S"],
        phase8["b"],
        phase8["c_bar"],
        phase8["CL_a"],
        result["phase3"]["V_cruise"],
        phase8["V_stall"],
        phase10.get("operational_CG_range_m", phase10["CG_range_m"]),
        q_slipstream_ratio=assumptions.elevon_q_slipstream_ratio,
        p_required_rad_s=np.deg2rad(assumptions.roll_rate_required_deg_s),
        Cm_de_required=None,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
        tau_e=assumptions.elevon_tau,
        eta_e=assumptions.elevon_eta,
        l_e_over_c=assumptions.elevon_l_e_over_c,
        CL_trim=phase10["CL_trim"],
        pitch_trim_margin=assumptions.elevon_pitch_trim_margin,
        control_margin_min=1.05,
        chord_fraction_bounds=(
            assumptions.elevon_chord_fraction_min,
            assumptions.elevon_chord_fraction_max,
        ),
        span_fraction_bounds=(
            assumptions.elevon_span_fraction_min,
            assumptions.elevon_span_fraction_max,
        ),
        grid_points=assumptions.elevon_grid_points,
    )


def _run_phase12_with_mass_inertia(result: Dict, phase11: Dict, phase15: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Run Phase 12 using Phase 15 inertias."""
    phase1 = result["phase1"]
    phase8 = result["phase8"]
    rho_target, _, _, _ = isa(mission.altitude_m)
    d_x_rotor = (
        assumptions.hover_pitch_arm_m
        if assumptions.hover_pitch_arm_m is not None
        else assumptions.hover_pitch_arm_fraction_fuselage * assumptions.fuselage_length_m
    )
    d_y_rotor = (
        assumptions.hover_roll_arm_m
        if assumptions.hover_roll_arm_m is not None
        else assumptions.hover_roll_arm_fraction_span * phase8["b"]
    )
    l_z_elev = (
        assumptions.hover_yaw_arm_m
        if assumptions.hover_yaw_arm_m is not None
        else assumptions.hover_yaw_arm_fraction_chord * phase8["c_bar"]
    )
    CL_de = (
        assumptions.hover_control_CL_de
        if assumptions.hover_control_CL_de is not None
        else phase8["CL_a"] * assumptions.elevon_tau
    )
    phase12 = phase12_hover_control(
        result["MTOW_N"] / assumptions.n_rotors,
        phase1["A_disc"],
        rho_target,
        d_x_rotor,
        d_y_rotor,
        phase15["I_tensor_kg_m2"]["Ixx"],
        phase15["I_tensor_kg_m2"]["Iyy"],
        phase15["I_tensor_kg_m2"]["Izz"],
        phase11["S_e_total"],
        phase11["b_e_over_b"] * phase8["b"],
        l_z_elev,
        omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_rate_required=np.deg2rad(assumptions.hover_yaw_rate_required_deg_s),
        T_available_per_rotor=phase1["T_per_rotor"],
        roll_omega_dot_required=assumptions.hover_angular_accel_required_rad_s2,
        yaw_response_time_s=assumptions.hover_yaw_response_time_s,
        CL_de=CL_de,
        delta_e_max_rad=np.deg2rad(assumptions.elevon_max_deflection_deg),
    )
    phase12["inertia_source"] = {
        "I_xx": "phase15_mass_model",
        "I_yy": "phase15_mass_model",
        "I_zz": "phase15_mass_model",
    }
    phase12["arm_source"] = {
        "d_x_rotor": "assumption" if assumptions.hover_pitch_arm_m is not None else "fuselage_fraction_estimate",
        "d_y_rotor": "assumption" if assumptions.hover_roll_arm_m is not None else "span_fraction_estimate",
        "l_z_elev": "assumption" if assumptions.hover_yaw_arm_m is not None else "chord_fraction_estimate",
    }
    return phase12


def _run_phase13_from_result(result: Dict, assumptions: Assumptions) -> Dict:
    """Run Phase 13 transition blending."""
    return phase13_transition_blending(
        result["phase8"]["V_stall"],
        assumptions.transition_blend_start_frac,
        assumptions.transition_blend_end_frac,
        V_entry=0.0,
        V_exit=None,
        transition_accel_m_s2=assumptions.transition_accel_m_s2,
        V_cruise=result["phase3"]["V_cruise"],
        cruise_speed_margin_frac=assumptions.transition_cruise_margin_frac,
        P_hover_W=result["phase3"]["P_hover_target"],
        P_fw_W=result["phase3"]["P_fw"],
        sample_count=assumptions.transition_sample_count,
    )


def _run_phase14_with_mass_inertia(result: Dict, phase10: Dict, phase15: Dict,
                                   assumptions: Assumptions,
                                   mission: Mission) -> Dict:
    """Run Phase 14 using Phase 15 inertias."""
    rho_cruise, _, _, _ = isa(mission.altitude_m)
    phase8 = result["phase8"]
    phase9 = result["phase9"]
    CL_alpha_total = phase10["a_w_eff"] + phase10["a_c_eff"] * phase10["area_ratio"]
    Cm_alpha = -phase10["SM_min"] * CL_alpha_total
    if assumptions.dynamic_Cm_q is None:
        Cm_q = -2.0 * phase9["CL_a_c"] * phase10["V_bar_c"] * (
            phase9["l_c"] / phase8["c_bar"]
        )
    else:
        Cm_q = assumptions.dynamic_Cm_q
    phase14 = phase14_dynamic_stability(
        Cm_alpha,
        Cm_q,
        assumptions.dynamic_Cm_alpha_dot,
        CL_alpha_total,
        phase15["I_tensor_kg_m2"]["Iyy"],
        phase15["I_tensor_kg_m2"]["Ixx"],
        phase15["I_tensor_kg_m2"]["Izz"],
        phase8["S"],
        phase8["c_bar"],
        phase8["b"],
        result["phase3"]["V_cruise"],
        rho_cruise,
        CL_trim=phase10["CL_trim"],
        CD_trim=result["phase3"]["CD_cruise"],
        C_l_beta=assumptions.dynamic_Cl_beta,
        C_l_p=assumptions.dynamic_Cl_p or result["phase11"].get("C_l_p"),
        C_l_r=assumptions.dynamic_Cl_r,
        C_n_beta=assumptions.dynamic_Cn_beta,
        C_n_p=assumptions.dynamic_Cn_p,
        C_n_r=assumptions.dynamic_Cn_r,
        g=assumptions.g,
        short_period_zeta_min=assumptions.short_period_zeta_min,
        short_period_zeta_max=assumptions.short_period_zeta_max,
        short_period_omega_min_rad_s=assumptions.short_period_omega_min_rad_s,
        phugoid_zeta_min=assumptions.phugoid_zeta_min,
        dutch_roll_zeta_min=assumptions.dutch_roll_zeta_min,
        dutch_roll_omega_min_rad_s=assumptions.dutch_roll_omega_min_rad_s,
        spiral_time_to_double_min_s=assumptions.spiral_time_to_double_min_s,
    )
    phase14["derivative_sources"] = {
        "Cm_alpha": "static_margin_times_total_lift_curve_slope",
        "Cm_q": "assumption" if assumptions.dynamic_Cm_q is not None else "canard_volume_estimate",
        "inertias": "phase15_mass_model",
    }
    return phase14


def _hover_control_closure_update(result: Dict, assumptions: Assumptions) -> Tuple[Assumptions, Dict]:
    """Compute geometry-first hover-control updates for the next closure pass."""
    phase12 = result["phase12"]
    phase8 = result["phase8"]
    margin_min = assumptions.hover_control_margin_min
    update_margin = margin_min + 0.005
    updates = {}
    messages = []

    if phase12["pitch_margin"] < margin_min:
        required_fraction = (
            phase12["pitch_arm_required_m"] * update_margin / assumptions.fuselage_length_m
        )
        new_fraction = min(
            assumptions.hover_pitch_arm_fraction_fuselage_max,
            max(assumptions.hover_pitch_arm_fraction_fuselage, required_fraction),
        )
        if new_fraction > assumptions.hover_pitch_arm_fraction_fuselage + 1e-9:
            updates["hover_pitch_arm_fraction_fuselage"] = float(new_fraction)
            messages.append(f"pitch arm fraction -> {new_fraction:.3f}")

    if phase12["roll_margin"] < margin_min:
        required_fraction = phase12["roll_arm_required_m"] * update_margin / phase8["b"]
        new_fraction = min(
            assumptions.hover_roll_arm_fraction_span_max,
            max(assumptions.hover_roll_arm_fraction_span, required_fraction),
        )
        if new_fraction > assumptions.hover_roll_arm_fraction_span + 1e-9:
            updates["hover_roll_arm_fraction_span"] = float(new_fraction)
            messages.append(f"roll arm fraction -> {new_fraction:.3f}")

    pitch_at_cap = (
        assumptions.hover_pitch_arm_fraction_fuselage
        >= assumptions.hover_pitch_arm_fraction_fuselage_max - 1e-9
    )
    roll_at_cap = (
        assumptions.hover_roll_arm_fraction_span
        >= assumptions.hover_roll_arm_fraction_span_max - 1e-9
    )
    if (
        (phase12["pitch_margin"] < margin_min and pitch_at_cap)
        or (phase12["roll_margin"] < margin_min and roll_at_cap)
    ):
        T_hover = phase12["T_hover_per_rotor"]
        T_W_pitch = 1.0 + update_margin * phase12["delta_T_pitch_required"] / T_hover
        T_W_roll = 1.0 + update_margin * phase12["delta_T_roll_required"] / T_hover
        T_W_required = max(T_W_pitch, T_W_roll, assumptions.thrust_to_weight)
        new_T_W = min(assumptions.thrust_to_weight_max, T_W_required)
        if new_T_W > assumptions.thrust_to_weight + 1e-9:
            updates["thrust_to_weight"] = float(new_T_W)
            messages.append(f"T/W -> {new_T_W:.3f}")

    updated = replace(assumptions, **updates) if updates else assumptions
    return updated, {
        "updated": bool(updates),
        "updates": updates,
        "messages": messages,
        "pitch_margin": phase12["pitch_margin"],
        "roll_margin": phase12["roll_margin"],
        "yaw_margin": phase12["yaw_margin"],
        "target_margin": margin_min,
        "thrust_to_weight": assumptions.thrust_to_weight,
        "hover_pitch_arm_fraction_fuselage": assumptions.hover_pitch_arm_fraction_fuselage,
        "hover_roll_arm_fraction_span": assumptions.hover_roll_arm_fraction_span,
    }


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


def _iterate_phases_1_to_15_legacy(MTOW_kg: float = 50.0,
                                   mission: Optional[Mission] = None,
                                   assumptions: Optional[Assumptions] = None,
                                   max_inner_iter: int = 15,
                                   j_tol: float = 0.01,
                                   area_tol: float = 0.01,
                                   re_tol: float = 0.02,
                                   constraint_plot_path=None,
                                   airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run Phases 1-14, then append Phase 15 preliminary mass breakdown."""
    result = iterate_phases_1_to_14(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )
    if mission is None:
        mission = Mission()
    if assumptions is None:
        assumptions = Assumptions()

    phase15 = phase15_mass(
        result["phase8"]["S"],
        result["phase9"]["S_c"],
        assumptions.fuselage_length_m,
        result["phase5"]["P_motor_cont"],
        result["phase5"]["m_batt_kg"],
        mission.mission_equipment_mass_kg,
        n_rotors=assumptions.n_rotors,
        prop_diameter_m=result["phase1"]["D_prop"],
        b_w=result["phase8"]["b"],
        c_bar_w=result["phase8"]["c_bar"],
        x_ac_w=result["phase8"]["x_ac_w"],
        x_ac_c=result["phase9"]["x_ac_c"],
        wing_mac_le_x_m=(
            0.0
            if assumptions.wing_mac_le_x_m is None
            else assumptions.wing_mac_le_x_m
        ),
        external_tow_load_N=mission.external_tow_load_N,
        g=assumptions.g,
        wing_areal_density_kg_m2=assumptions.wing_areal_density_kg_m2,
        canard_areal_density_kg_m2=assumptions.canard_areal_density_kg_m2,
        fuselage_linear_density_kg_m=assumptions.fuselage_linear_density_kg_m,
        boom_landing_gear_mass_kg=assumptions.boom_landing_gear_mass_kg,
        motor_specific_mass_kg_W=assumptions.motor_specific_mass_kg_W,
        esc_specific_mass_kg_W=assumptions.esc_specific_mass_kg_W,
        prop_mass_coeff_kg_m2=assumptions.prop_mass_coeff_kg_m2,
        avionics_mass_kg=assumptions.avionics_mass_kg,
        wiring_fraction=assumptions.wiring_fraction,
        mass_contingency_fraction=assumptions.mass_contingency_fraction,
        Ixx_radius_fraction_span=assumptions.hover_Ixx_radius_fraction_span,
        Iyy_radius_fraction_fuselage=assumptions.hover_Iyy_radius_fraction_fuselage,
        Izz_radius_fraction_span=assumptions.hover_Izz_radius_fraction_span,
    )

    result = dict(result)
    result["phase15"] = phase15
    result["notes"] = list(result.get("notes", [])) + [
        "Phase 15 is appended as a mass breakdown; the sizing MTOW is still fixed until Phase 16."
    ]
    result["warnings"] = list(result.get("warnings", []))
    if phase15["warnings"]:
        result["warnings"].extend(f"phase15: {warning}" for warning in phase15["warnings"])

    if result.get("converged"):
        if result.get("phase11", {}) and not result["phase11"].get("feasible_preliminary_elevon", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 11 fixed-wing elevon remains infeasible; Phase 15 mass estimate appended"
            )
        elif result.get("phase12", {}) and not result["phase12"].get("feasible_preliminary_hover_control", True):
            result["stopped_reason"] = (
                "propulsion/aero loop converged, but Phase 12 hover control remains infeasible; Phase 15 mass estimate appended"
            )
        elif result.get("phase14", {}) and not result["phase14"].get("level_meets_8785C_preliminary", True):
            result["stopped_reason"] = (
                "propulsion/aero/control loop converged, but Phase 14 has preliminary dynamic-stability failures; Phase 15 mass estimate appended"
            )
        else:
            result["stopped_reason"] = (
                "propulsion, aerodynamics, controls, transition, stability, and Phase 15 mass estimate completed"
            )
    return result


def _run_coupled_fixed_mtow_once(MTOW_kg: float,
                                 mission: Mission,
                                 assumptions: Assumptions,
                                 max_inner_iter: int,
                                 j_tol: float,
                                 area_tol: float,
                                 re_tol: float,
                                 constraint_plot_path=None,
                                 airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Run one coupled fixed-MTOW pass with mass/CG/control feedback."""
    result = iterate_phases_1_to_9(
        MTOW_kg=MTOW_kg,
        mission=mission,
        assumptions=assumptions,
        max_inner_iter=max_inner_iter,
        j_tol=j_tol,
        area_tol=area_tol,
        re_tol=re_tol,
        constraint_plot_path=constraint_plot_path,
        airfoil_files=airfoil_files,
    )

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
                           j_tol: float = 0.01,
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
        result = _run_coupled_fixed_mtow_once(
            MTOW_kg,
            mission,
            effective,
            max_inner_iter,
            j_tol,
            area_tol,
            re_tol,
            constraint_plot_path=constraint_plot_path,
            airfoil_files=airfoil_files,
        )
        updated, closure_record = _hover_control_closure_update(result, effective)
        closure_record["closure_iteration"] = closure_it + 1
        closure_record["MTOW_kg"] = float(MTOW_kg)
        closure_record["phase15_MTOW_estimate_kg"] = result["phase15"]["MTOW_estimate_kg"]
        control_mass_history.append(closure_record)
        final_result = result
        if not closure_record["updated"]:
            break
        effective = updated

    final_result = dict(final_result)
    final_result["control_mass_history"] = control_mass_history
    final_result["effective_assumptions"] = asdict(effective)
    final_result["sanity_checks"] = _design_sanity_checks(final_result, mission, effective)
    if not final_result["sanity_checks"]["all_passed"]:
        final_result["warnings"] = list(final_result.get("warnings", [])) + [
            f"sanity: {issue}" for issue in final_result["sanity_checks"]["issues"]
        ]
    return final_result

# ---------- PHASE 16 ----------

def phase16_mtow_converge(MTOW_old: float, MTOW_new: float,
                          damping: float = 0.4, tol: float = 0.01
                          ) -> Dict:
    """Outer-loop MTOW convergence with damping (Gundlach 2014 Ch.19).
    Returns a dictionary so the iteration history is easy to inspect."""
    if MTOW_old <= 0.0:
        raise ValueError("MTOW_old must be positive.")
    if MTOW_new <= 0.0:
        raise ValueError("MTOW_new must be positive.")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in the range 0 < damping <= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    delta = MTOW_new - MTOW_old
    MTOW_next = MTOW_old + damping * delta
    relative_error = abs(delta) / MTOW_old
    converged = relative_error < tol
    return {
        "MTOW_old_kg": float(MTOW_old),
        "MTOW_new_kg": float(MTOW_new),
        "MTOW_next_kg": float(MTOW_next),
        "delta_kg": float(delta),
        "relative_error": float(relative_error),
        "damping": float(damping),
        "tol": float(tol),
        "converged": bool(converged),
        "notes": [
            "MTOW_next = MTOW_old + damping*(MTOW_new - MTOW_old).",
            "Convergence is based on abs(MTOW_new - MTOW_old)/MTOW_old < tol.",
        ],
        "warnings": [],
    }

# ---------- ORCHESTRATION ----------

def converge_design(MTOW_seed_kg: float = 50.0,
                    mission: Optional[Mission] = None,
                    assumptions: Optional[Assumptions] = None,
                    max_iter: int = 30,
                    tol: float = 0.01,
                    damping: float = 0.4,
                    max_inner_iter: int = 15,
                    j_tol: float = 0.01,
                    area_tol: float = 0.01,
                    re_tol: float = 0.02,
                    constraint_plot_path=None,
                    airfoil_files: Optional[Dict[str, str]] = None) -> Dict:
    """Outer MTOW loop orchestrating all 16 phases.
    Sequence (dependency-graph order, NOT constraint-diagram-first):
      0. ISA tables
      1. Phase 1  prop diameter
      2. Phase 2  hover/climb power
      3. Phase 3  mission optimise  (V_cruise, gamma, h_tr DERIVED here)
      4. Phase 4  J coupling — inner V_cruise–J loop with Ph3
      5. Phase 7  airfoil — inner Re loop seeded from Ph3 V_cruise
      6. Phase 8  wing — V_stall DERIVED here from CL,max,3D and W/S
      7. Phase 7' Re-loop refine (one iteration)
      8. Phase 9  canard
      9. Phase 10 scissor (canard form) — inner CG–canard loop with Ph9
     10. Phase 11 FW elevon
     11. Phase 12 hover elevon — MAX-of with Ph11 closes elevon spec
     12. Phase 13 transition blending
     13. Phase 5  energy / battery
     14. Phase 15 mass — produces new MTOW
     15. Phase 16 converge check (damping 0.4)
     16. Phase 6  constraint diagram VERIFY (final)
     17. Phase 14 dynamic stability VERIFY (final)
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

    for outer_it in range(max_iter):
        result = iterate_phases_1_to_15(
            MTOW_kg=MTOW_current,
            mission=mission,
            assumptions=assumptions,
            max_inner_iter=max_inner_iter,
            j_tol=j_tol,
            area_tol=area_tol,
            re_tol=re_tol,
            constraint_plot_path=constraint_plot_path,
            airfoil_files=airfoil_files,
        )
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

        outer_history.append({
            "outer_iteration": outer_it + 1,
            "MTOW_input_kg": float(MTOW_current),
            "MTOW_mass_estimate_kg": float(MTOW_estimate),
            "MTOW_next_kg": phase16["MTOW_next_kg"],
            "mass_delta_kg": phase16["delta_kg"],
            "relative_error": phase16["relative_error"],
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

        if phase16["converged"]:
            outer_converged = True
            break
        MTOW_current = phase16["MTOW_next_kg"]

    final_result = dict(final_result)
    inner_converged = bool(final_result.get("converged", False))
    final_result["inner_converged"] = inner_converged
    final_result["outer_converged"] = bool(outer_converged)
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
        "Phase 16 closes the MTOW loop by feeding the Phase 15 mass estimate back into the fixed-MTOW sizing pass."
    ]
    final_result["warnings"] = list(final_result.get("warnings", []))
    if not outer_converged:
        final_result["warnings"].append(
            "Phase 16 did not meet the requested MTOW convergence tolerance before max_iter."
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

    if inner_converged and outer_converged:
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


# ---------- CLI / JSON OUTPUT ----------

def _json_safe(value):
    """Convert numpy and Path values into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_summary(result: Dict) -> Dict:
    """Pull the most useful Phase 1-16 outputs into one compact dictionary."""
    phase1 = result["phase1"]
    phase2 = result["phase2_hover"]
    phase3 = result["phase3"]
    phase4 = result["phase4"]
    phase5 = result["phase5"]
    phase8 = result["phase8"]
    phase9 = result["phase9"]
    phase10 = result.get("phase10", {})
    phase11 = result.get("phase11", {})
    phase12 = result.get("phase12", {})
    phase13 = result.get("phase13", {})
    phase14 = result.get("phase14", {})
    phase15 = result.get("phase15", {})
    phase16 = result.get("phase16", {})
    canard_search = result.get("canard_cg_search", {})
    sanity_checks = result.get("sanity_checks", {})
    effective_assumptions = result.get("effective_assumptions", {})
    wing_layout = result.get("wing_layout_solver", phase15.get("wing_layout_solver", {}))

    n_rotors = phase1["T_total"] / phase1["T_per_rotor"]
    summary = {
        "converged": result["converged"],
        "inner_loop_converged": result.get("inner_converged", result["converged"]),
        "outer_loop_converged": result.get("outer_converged"),
        "stopped_reason": result["stopped_reason"],
        "iterations": len(result["history"]),
        "outer_iterations": result.get("outer_iterations"),
        "control_closure_iterations": len(result.get("control_mass_history", [])),
        "MTOW_kg": result["MTOW_kg"],
        "MTOW_seed_kg": result.get("MTOW_seed_kg"),
        "MTOW_converged_kg": result.get("MTOW_converged_kg"),
        "effective_thrust_to_weight": effective_assumptions.get("thrust_to_weight"),
        "effective_hover_pitch_arm_fraction_fuselage": effective_assumptions.get("hover_pitch_arm_fraction_fuselage"),
        "effective_hover_roll_arm_fraction_span": effective_assumptions.get("hover_roll_arm_fraction_span"),
        "propeller_diameter_m": phase1["D_prop"],
        "propeller_diameter_max_m": phase1.get("prop_diameter_max_m"),
        "propeller_diameter_limited": phase1.get("prop_diameter_limited", False),
        "disc_loading_target_N_m2": phase1.get("disc_loading_target", phase1["disc_loading"]),
        "disc_loading_actual_N_m2": phase1["disc_loading"],
        "cruise_speed_m_s": phase3["V_cruise"],
        "cruise_speed_min_required_m_s": phase3["V_min_required"],
        "cruise_speed_min_requirement_active": phase3["V_min_requirement_active"],
        "fixed_wing_ROC_m_s": phase3["ROC"],
        "cruise_advance_ratio": phase4["J"],
        "cruise_rpm": result["cruise_rpm"],
        "hover_power_total_W": phase2["P_elec"] * n_rotors,
        "battery_mass_kg": phase5["m_batt_kg"],
        "installed_battery_energy_Wh": phase5["E_total_Wh"],
        "phase5_transition_energy_source": phase5.get("transition_energy_source", "phase3_mission_optimise"),
        "wing_area_m2": phase8["S"],
        "wing_span_m": phase8["b"],
        "wing_mean_chord_m": phase8["c_bar"],
        "wing_stall_speed_m_s": phase8["V_stall"],
        "canard_area_m2": phase9["S_c"],
        "canard_span_m": phase9["b_c"],
        "canard_mean_chord_m": phase9["c_bar_c"],
        "Re_main": result["Re_main"],
        "Re_canard": result["Re_canard"],
    }
    if phase10:
        summary.update({
            "neutral_point_x_over_c": phase10["x_np_over_c"],
            "cg_forward_x_over_c": phase10["x_cg_fwd_over_c"],
            "cg_aft_x_over_c": phase10["x_cg_aft_over_c"],
            "cg_range_pct_mac": phase10["CG_range_pct"],
            "cg_range_feasible": phase10["feasible_preliminary_CG_range"],
            "operational_cg_feasible": phase10.get("operational_CG_feasible"),
            "operational_cg_forward_x_over_c": phase10.get("x_cg_fwd_operational_over_mac"),
            "operational_cg_aft_x_over_c": phase10.get("x_cg_aft_operational_over_mac"),
            "operational_cg_range_over_mac": phase10.get("operational_CG_range_over_mac"),
            "operational_cg_fwd_margin_over_mac": phase10.get("operational_fwd_margin_over_mac"),
            "operational_cg_aft_margin_over_mac": phase10.get("operational_aft_margin_over_mac"),
            "mass_cg_inside_theoretical": phase10.get("mass_CG_inside_theoretical"),
            "required_CG_shift_over_mac": phase10.get("required_CG_shift_over_mac"),
        })
    if phase11:
        summary.update({
            "elevon_chord_fraction": phase11["c_e_over_c"],
            "elevon_span_fraction": phase11["b_e_over_b"],
            "elevon_area_total_m2": phase11["S_e_total"],
            "elevon_area_ratio": phase11["S_e_over_S"],
            "elevon_binding_case": phase11["binding_case"],
            "elevon_feasible": phase11["feasible_preliminary_elevon"],
            "roll_rate_achievable_deg_s": phase11["p_achievable_deg_s"],
            "roll_rate_required_deg_s": phase11["p_required_deg_s"],
            "Cm_de_abs": phase11["Cm_de_abs"],
            "Cm_de_required": phase11["Cm_de_required"],
            "elevon_pitch_margin": phase11["pitch_margin"],
            "elevon_roll_margin": phase11["roll_margin"],
        })
    if phase12:
        summary.update({
            "hover_control_feasible": phase12["feasible_preliminary_hover_control"],
            "hover_control_binding_case": phase12["binding_case"],
            "hover_pitch_margin": phase12["pitch_margin"],
            "hover_roll_margin": phase12["roll_margin"],
            "hover_yaw_margin": phase12["yaw_margin"],
            "hover_delta_T_available_N": phase12["delta_T_available"],
            "hover_delta_T_pitch_required_N": phase12["delta_T_pitch_required"],
            "hover_delta_T_roll_required_N": phase12["delta_T_roll_required"],
            "hover_pitch_arm_m": phase12["d_x_rotor"],
            "hover_pitch_arm_required_m": phase12["pitch_arm_required_m"],
            "hover_roll_arm_m": phase12["d_y_rotor"],
            "hover_roll_arm_required_m": phase12["roll_arm_required_m"],
            "hover_thrust_to_weight_required_pitch": phase12["thrust_to_weight_required_pitch"],
            "hover_thrust_to_weight_required_roll": phase12["thrust_to_weight_required_roll"],
            "hover_pitch_angular_accel_available_rad_s2": phase12["pitch_angular_accel_available_rad_s2"],
            "hover_roll_angular_accel_available_rad_s2": phase12["roll_angular_accel_available_rad_s2"],
            "hover_Iyy_max_for_pitch_kg_m2": phase12["Iyy_max_for_pitch_kg_m2"],
            "hover_Ixx_max_for_roll_kg_m2": phase12["Ixx_max_for_roll_kg_m2"],
            "hover_yaw_elevon_area_required_m2": phase12["S_e_yaw_required"],
            "hover_slipstream_speed_m_s": phase12["V_slip_hover"],
        })
    if phase13:
        summary.update({
            "transition_blend_start_m_s": phase13["V_blend_start"],
            "transition_blend_end_m_s": phase13["V_blend_end"],
            "transition_time_s": phase13["t_transition"],
            "transition_distance_m": phase13["distance_transition_m"],
            "transition_energy_estimate_Wh": phase13["E_transition_estimate_Wh"],
            "transition_cruise_required_for_margin_m_s": phase13["V_cruise_required_for_margin"],
            "transition_cruise_margin_over_blend_end": phase13["cruise_speed_margin_over_blend_end"],
            "transition_alpha_fw_at_cruise": phase13["alpha_fw_at_cruise"],
            "transition_alpha_hover_at_cruise": phase13["alpha_hover_at_cruise"],
        })
    if phase14:
        summary.update({
            "dynamic_stability_preliminary": phase14["level_meets_8785C_preliminary"],
            "short_period_meets": phase14["short_period_meets"],
            "short_period_omega_rad_s": phase14["omega_sp_rad_s"],
            "short_period_zeta": phase14["zeta_sp"],
            "phugoid_meets": phase14["phugoid_meets"],
            "phugoid_omega_rad_s": phase14["omega_ph_rad_s"],
            "phugoid_zeta": phase14["zeta_ph"],
            "dutch_roll_meets": phase14["dutch_roll_meets"],
            "dutch_roll_omega_rad_s": phase14["omega_dr_rad_s"],
            "dutch_roll_zeta": phase14["zeta_dr"],
            "spiral_meets": phase14["spiral_meets"],
            "spiral_stable": phase14["spiral_stable"],
            "spiral_time_to_double_s": phase14["spiral_time_to_double_s"],
            "roll_subsidence_tau_s": phase14["roll_subsidence_tau_s"],
            "Cm_alpha": phase14["Cm_alpha"],
            "Cm_q_effective": phase14["Cm_q_effective"],
        })
    if phase15:
        summary.update({
            "MTOW_mass_estimate_kg": phase15["MTOW_estimate_kg"],
            "MTOW_mass_delta_kg": phase15["MTOW_estimate_kg"] - result["MTOW_kg"],
            "MTOW_mass_delta_fraction": (
                (phase15["MTOW_estimate_kg"] - result["MTOW_kg"]) / result["MTOW_kg"]
            ),
            "empty_mass_no_battery_no_mission_kg": phase15["empty_mass_no_battery_no_mission_kg"],
            "structure_mass_kg": phase15["structure_mass_kg"],
            "propulsion_mass_kg": phase15["propulsion_mass_kg"],
            "mission_equipment_mass_kg": phase15["m_mission_equipment_kg"],
            "wing_mac_le_x_m": phase15.get("wing_mac_le_x_m"),
            "mass_model_x_CG_m": phase15["x_CG_m"],
            "mass_model_x_CG_fuselage_m": phase15.get("x_CG_fuselage_m"),
            "mass_model_x_CG_over_wing_mac": phase15["x_CG_over_wing_mac"],
            "mass_model_Ixx_kg_m2": phase15["I_tensor_kg_m2"]["Ixx"],
            "mass_model_Iyy_kg_m2": phase15["I_tensor_kg_m2"]["Iyy"],
            "mass_model_Izz_kg_m2": phase15["I_tensor_kg_m2"]["Izz"],
            "external_tow_load_included_in_MTOW": phase15["external_tow_load_included_in_MTOW"],
        })
    if wing_layout:
        summary.update({
            "wing_layout_solve_enabled": wing_layout["enabled"],
            "wing_layout_solved": wing_layout["solved"],
            "wing_layout_required_shift_m": wing_layout["required_wing_shift_m"],
            "wing_layout_initial_x_CG_over_mac": wing_layout["initial_x_CG_over_mac"],
            "wing_layout_final_x_CG_over_mac": wing_layout["final_x_CG_over_mac"],
            "wing_layout_target_x_CG_over_mac": wing_layout["target_x_CG_over_mac"],
            "wing_layout_station_over_mac": wing_layout["wing_station_over_mac"],
        })
    if canard_search:
        summary.update({
            "canard_selected_volume_coeff": canard_search["selected_V_bar_c"],
            "canard_default_volume_coeff": canard_search["default_V_bar_c"],
            "canard_cg_search_candidate_count": canard_search["candidate_count"],
            "canard_default_operational_CG_feasible": canard_search["default_operational_CG_feasible"],
        })
    if phase16:
        summary.update({
            "phase16_MTOW_old_kg": phase16["MTOW_old_kg"],
            "phase16_MTOW_new_kg": phase16["MTOW_new_kg"],
            "phase16_MTOW_next_kg": phase16["MTOW_next_kg"],
            "phase16_delta_kg": phase16["delta_kg"],
            "phase16_relative_error": phase16["relative_error"],
            "phase16_damping": phase16["damping"],
            "phase16_tol": phase16["tol"],
            "phase16_converged": phase16["converged"],
        })
    if sanity_checks:
        summary.update({
            "design_sanity_all_passed": sanity_checks["all_passed"],
            "design_sanity_issue_count": sanity_checks["issue_count"],
            "design_sanity_issues": sanity_checks["issues"],
        })
    return summary


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the completed Bellona Phase 1-16 propulsion, aerodynamics, controls, transition, stability, mass, and MTOW convergence loop."
    )
    parser.add_argument("--mtow-kg", type=float, default=50.0,
                        help="MTOW seed for the Phase 1-16 loop, or fixed MTOW when --fixed-mtow is used.")
    parser.add_argument("--fixed-mtow", action="store_true",
                        help="Run one fixed-MTOW Phase 1-15 pass instead of the Phase 16 outer loop.")
    parser.add_argument("--altitude-m", type=float, default=Mission.altitude_m)
    parser.add_argument("--range-m", type=float, default=Mission.range_m)
    parser.add_argument("--time-budget-s", type=float, default=Mission.time_budget_s)
    parser.add_argument("--hover-time-s", type=float, default=Mission.hover_time_s)
    parser.add_argument("--mission-equipment-mass-kg", type=float,
                        default=Mission.mission_equipment_mass_kg)
    parser.add_argument("--external-tow-load-n", type=float,
                        default=Mission.external_tow_load_N)

    parser.add_argument("--disc-loading-target-n-m2", type=float,
                        default=Assumptions.disc_loading_target_N_m2)
    parser.add_argument("--thrust-to-weight", type=float,
                        default=Assumptions.thrust_to_weight)
    parser.add_argument("--prop-diameter-max-m", type=float, default=None,
                        help="Optional CAD/geometry propeller diameter cap.")
    parser.add_argument("--battery-specific-energy-wh-kg", type=float,
                        default=Assumptions.battery_specific_energy_Wh_kg)
    parser.add_argument("--stall-speed-target-max-m-s", type=float,
                        default=Assumptions.stall_speed_target_max_m_s)
    parser.add_argument("--wing-area-seed-m2", type=float,
                        default=Assumptions.preliminary_wing_area_m2)
    parser.add_argument("--cruise-j-target", type=float,
                        default=Assumptions.cruise_J_target)
    parser.add_argument("--canard-volume-coeff", type=float,
                        default=Assumptions.canard_volume_coeff)
    parser.add_argument("--canard-volume-grid-min", type=float,
                        default=Assumptions.canard_volume_grid_min)
    parser.add_argument("--canard-volume-grid-max", type=float,
                        default=Assumptions.canard_volume_grid_max)
    parser.add_argument("--canard-volume-grid-step", type=float,
                        default=Assumptions.canard_volume_grid_step)
    parser.add_argument("--cg-envelope-half-width-over-mac", type=float,
                        default=Assumptions.cg_envelope_half_width_over_mac)
    parser.add_argument("--cg-required-margin-over-mac", type=float,
                        default=Assumptions.cg_required_margin_over_mac)
    parser.add_argument("--wing-mac-le-x-m", type=float, default=None,
                        help="Optional initial/fixed wing MAC leading-edge station from the provisional fuselage/equipment datum.")
    parser.add_argument("--disable-wing-position-solve", action="store_true",
                        help="Keep the provided wing MAC station instead of solving it from the scissor CG target.")
    parser.add_argument("--roll-rate-required-deg-s", type=float,
                        default=Assumptions.roll_rate_required_deg_s)
    parser.add_argument("--elevon-max-deflection-deg", type=float,
                        default=Assumptions.elevon_max_deflection_deg)
    parser.add_argument("--elevon-q-slipstream-ratio", type=float,
                        default=Assumptions.elevon_q_slipstream_ratio)
    parser.add_argument("--hover-angular-accel-rad-s2", type=float,
                        default=Assumptions.hover_angular_accel_required_rad_s2)
    parser.add_argument("--hover-yaw-rate-required-deg-s", type=float,
                        default=Assumptions.hover_yaw_rate_required_deg_s)
    parser.add_argument("--hover-yaw-response-time-s", type=float,
                        default=Assumptions.hover_yaw_response_time_s)
    parser.add_argument("--hover-pitch-arm-m", type=float, default=None,
                        help="Optional pitch rotor moment arm from CG to rotor pair.")
    parser.add_argument("--hover-roll-arm-m", type=float, default=None,
                        help="Optional roll rotor moment arm from CG to rotor pair.")
    parser.add_argument("--hover-yaw-arm-m", type=float, default=None,
                        help="Optional effective elevon yaw moment arm in hover.")
    parser.add_argument("--hover-control-margin-min", type=float,
                        default=Assumptions.hover_control_margin_min)
    parser.add_argument("--hover-pitch-arm-fraction-fuselage-max", type=float,
                        default=Assumptions.hover_pitch_arm_fraction_fuselage_max)
    parser.add_argument("--hover-roll-arm-fraction-span-max", type=float,
                        default=Assumptions.hover_roll_arm_fraction_span_max)
    parser.add_argument("--thrust-to-weight-max", type=float,
                        default=Assumptions.thrust_to_weight_max)
    parser.add_argument("--control-closure-max-iter", type=int,
                        default=Assumptions.control_closure_max_iter)
    parser.add_argument("--transition-blend-start-frac", type=float,
                        default=Assumptions.transition_blend_start_frac)
    parser.add_argument("--transition-blend-end-frac", type=float,
                        default=Assumptions.transition_blend_end_frac)
    parser.add_argument("--transition-cruise-margin-frac", type=float,
                        default=Assumptions.transition_cruise_margin_frac)
    parser.add_argument("--transition-accel-m-s2", type=float,
                        default=Assumptions.transition_accel_m_s2)
    parser.add_argument("--dynamic-cm-q", type=float, default=None,
                        help="Optional pitch damping derivative for Phase 14.")
    parser.add_argument("--dynamic-cm-alpha-dot", type=float,
                        default=Assumptions.dynamic_Cm_alpha_dot)
    parser.add_argument("--dynamic-cl-beta", type=float,
                        default=Assumptions.dynamic_Cl_beta)
    parser.add_argument("--dynamic-cl-p", type=float, default=None,
                        help="Optional roll damping derivative for Phase 14.")
    parser.add_argument("--dynamic-cl-r", type=float,
                        default=Assumptions.dynamic_Cl_r)
    parser.add_argument("--dynamic-cn-beta", type=float,
                        default=Assumptions.dynamic_Cn_beta)
    parser.add_argument("--dynamic-cn-p", type=float,
                        default=Assumptions.dynamic_Cn_p)
    parser.add_argument("--dynamic-cn-r", type=float,
                        default=Assumptions.dynamic_Cn_r)

    parser.add_argument("--use-xfoil", action="store_true",
                        help="Try to run XFOIL; falls back to project table values if unavailable.")
    parser.add_argument("--xfoil-path", default=None,
                        help="Path to an XFOIL executable.")
    parser.add_argument("--airfoil-main-file", default=None,
                        help="Optional SD7037 coordinate file for XFOIL.")
    parser.add_argument("--airfoil-canard-file", default=None,
                        help="Optional NACA 0012 coordinate file for XFOIL.")

    parser.add_argument("--max-inner-iter", type=int, default=15)
    parser.add_argument("--outer-max-iter", type=int, default=30)
    parser.add_argument("--mtow-tol", type=float, default=0.01)
    parser.add_argument("--mtow-damping", type=float, default=0.4)
    parser.add_argument("--j-tol", type=float, default=0.01)
    parser.add_argument("--area-tol", type=float, default=0.01)
    parser.add_argument("--re-tol", type=float, default=0.02)
    parser.add_argument("--plot", default=None,
                        help="Optional path for the Phase 6 constraint diagram.")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Optional path to write full JSON results.")
    return parser


def _inputs_from_args(args) -> Tuple[Mission, Assumptions, Dict[str, str]]:
    mission = Mission(
        altitude_m=args.altitude_m,
        range_m=args.range_m,
        time_budget_s=args.time_budget_s,
        hover_time_s=args.hover_time_s,
        mission_equipment_mass_kg=args.mission_equipment_mass_kg,
        external_tow_load_N=args.external_tow_load_n,
    )
    assumptions = Assumptions(
        thrust_to_weight=args.thrust_to_weight,
        disc_loading_target_N_m2=args.disc_loading_target_n_m2,
        prop_diameter_max_m=args.prop_diameter_max_m,
        battery_specific_energy_Wh_kg=args.battery_specific_energy_wh_kg,
        preliminary_wing_area_m2=args.wing_area_seed_m2,
        stall_speed_target_max_m_s=args.stall_speed_target_max_m_s,
        cruise_J_target=args.cruise_j_target,
        canard_volume_coeff=args.canard_volume_coeff,
        canard_volume_grid_min=args.canard_volume_grid_min,
        canard_volume_grid_max=args.canard_volume_grid_max,
        canard_volume_grid_step=args.canard_volume_grid_step,
        cg_envelope_half_width_over_mac=args.cg_envelope_half_width_over_mac,
        cg_required_margin_over_mac=args.cg_required_margin_over_mac,
        wing_mac_le_x_m=args.wing_mac_le_x_m,
        solve_wing_position_for_cg=not args.disable_wing_position_solve,
        roll_rate_required_deg_s=args.roll_rate_required_deg_s,
        elevon_max_deflection_deg=args.elevon_max_deflection_deg,
        elevon_q_slipstream_ratio=args.elevon_q_slipstream_ratio,
        hover_angular_accel_required_rad_s2=args.hover_angular_accel_rad_s2,
        hover_yaw_rate_required_deg_s=args.hover_yaw_rate_required_deg_s,
        hover_yaw_response_time_s=args.hover_yaw_response_time_s,
        hover_pitch_arm_m=args.hover_pitch_arm_m,
        hover_roll_arm_m=args.hover_roll_arm_m,
        hover_yaw_arm_m=args.hover_yaw_arm_m,
        hover_control_margin_min=args.hover_control_margin_min,
        hover_pitch_arm_fraction_fuselage_max=args.hover_pitch_arm_fraction_fuselage_max,
        hover_roll_arm_fraction_span_max=args.hover_roll_arm_fraction_span_max,
        thrust_to_weight_max=args.thrust_to_weight_max,
        control_closure_max_iter=args.control_closure_max_iter,
        transition_blend_start_frac=args.transition_blend_start_frac,
        transition_blend_end_frac=args.transition_blend_end_frac,
        transition_cruise_margin_frac=args.transition_cruise_margin_frac,
        transition_accel_m_s2=args.transition_accel_m_s2,
        dynamic_Cm_q=args.dynamic_cm_q,
        dynamic_Cm_alpha_dot=args.dynamic_cm_alpha_dot,
        dynamic_Cl_beta=args.dynamic_cl_beta,
        dynamic_Cl_p=args.dynamic_cl_p,
        dynamic_Cl_r=args.dynamic_cl_r,
        dynamic_Cn_beta=args.dynamic_cn_beta,
        dynamic_Cn_p=args.dynamic_cn_p,
        dynamic_Cn_r=args.dynamic_cn_r,
        use_xfoil=args.use_xfoil,
        xfoil_path=args.xfoil_path,
    )

    airfoil_files = {}
    if args.airfoil_main_file:
        airfoil_files["main"] = args.airfoil_main_file
    if args.airfoil_canard_file:
        airfoil_files["canard"] = args.airfoil_canard_file
    return mission, assumptions, airfoil_files


def _print_summary(summary: Dict) -> None:
    if "phase16_relative_error" in summary:
        print("Bellona Phase 1-16 propulsion/aerodynamics/canard/control/stability/mass sizing")
    else:
        print("Bellona Phase 1-15 fixed-MTOW propulsion/aerodynamics/canard/control/stability/mass sizing")
    print(f"  Converged: {summary['converged']} ({summary['stopped_reason']})")
    if summary.get("outer_iterations") is None:
        print(f"  Iterations: {summary['iterations']}")
    else:
        print(
            "  Iterations: "
            f"outer={summary['outer_iterations']}, "
            f"inner-final={summary['iterations']}, "
            f"control-closure={summary['control_closure_iterations']}"
        )
    print(f"  Propeller diameter: {summary['propeller_diameter_m']:.3f} m")
    if summary["propeller_diameter_limited"]:
        print(
            "  Disc loading: "
            f"{summary['disc_loading_actual_N_m2']:.1f} N/m^2 "
            f"(target {summary['disc_loading_target_N_m2']:.1f}, diameter capped)"
        )
    else:
        print(
            "  Disc loading: "
            f"{summary['disc_loading_actual_N_m2']:.1f} N/m^2 "
            "(target met; no diameter cap active)"
        )
    print(
        "  Cruise speed: "
        f"{summary['cruise_speed_m_s']:.2f} m/s "
        f"(min {summary['cruise_speed_min_required_m_s']:.2f})"
    )
    print(f"  Cruise RPM / J: {summary['cruise_rpm']:.0f} rpm / {summary['cruise_advance_ratio']:.3f}")
    if summary.get("effective_thrust_to_weight") is not None:
        print(
            "  Effective T/W and arms: "
            f"T/W={summary['effective_thrust_to_weight']:.3f}, "
            f"pitch arm frac={summary['effective_hover_pitch_arm_fraction_fuselage']:.3f}, "
            f"roll arm frac={summary['effective_hover_roll_arm_fraction_span']:.3f}"
        )
    print(f"  Hover power: {summary['hover_power_total_W'] / 1000.0:.2f} kW")
    print(f"  Battery: {summary['battery_mass_kg']:.2f} kg, {summary['installed_battery_energy_Wh'] / 1000.0:.2f} kWh")
    print(f"  Wing: {summary['wing_area_m2']:.2f} m^2, span {summary['wing_span_m']:.2f} m")
    print(f"  Canard: {summary['canard_area_m2']:.2f} m^2, span {summary['canard_span_m']:.2f} m")
    if summary.get("wing_mac_le_x_m") is not None:
        print(
            "  Wing layout: "
            f"MAC LE x={summary['wing_mac_le_x_m']:.3f} m, "
            f"CG={summary['mass_model_x_CG_over_wing_mac']:.3f} x/c"
        )
    if summary.get("wing_layout_solve_enabled") is not None:
        print(
            "  Wing-CG solve: "
            f"enabled={summary['wing_layout_solve_enabled']}, "
            f"solved={summary['wing_layout_solved']}, "
            f"target={summary['wing_layout_target_x_CG_over_mac']:.3f} x/c"
        )
    if "canard_selected_volume_coeff" in summary:
        print(
            "  Canard search: "
            f"Vbar={summary['canard_selected_volume_coeff']:.3f} "
            f"(default {summary['canard_default_volume_coeff']:.3f})"
        )
    if "neutral_point_x_over_c" in summary:
        print(
            "  CG range: "
            f"{summary['cg_forward_x_over_c']:.3f} to {summary['cg_aft_x_over_c']:.3f} x/c "
            f"({summary['cg_range_pct_mac']:.1f}% MAC, feasible={summary['cg_range_feasible']})"
        )
        if summary.get("operational_cg_feasible") is not None:
            print(
                "  Operational CG: "
                f"{summary['operational_cg_forward_x_over_c']:.3f} to "
                f"{summary['operational_cg_aft_x_over_c']:.3f} x/c, "
                f"feasible={summary['operational_cg_feasible']}"
            )
        print(f"  Neutral point: {summary['neutral_point_x_over_c']:.3f} x/c")
    if "elevon_chord_fraction" in summary:
        print(
            "  Elevon: "
            f"c_e/c={summary['elevon_chord_fraction']:.3f}, "
            f"b_e/b={summary['elevon_span_fraction']:.3f}, "
            f"S_e={summary['elevon_area_total_m2']:.2f} m^2, "
            f"binding={summary['elevon_binding_case']}, feasible={summary['elevon_feasible']}"
        )
        print(
            "  Roll rate: "
            f"{summary['roll_rate_achievable_deg_s']:.1f} deg/s "
            f"(required {summary['roll_rate_required_deg_s']:.1f})"
        )
        print(
            "  Elevon margins: "
            f"pitch={summary['elevon_pitch_margin']:.2f}, "
            f"roll={summary['elevon_roll_margin']:.2f}"
        )
    if "hover_control_feasible" in summary:
        print(
            "  Hover control: "
            f"binding={summary['hover_control_binding_case']}, "
            f"feasible={summary['hover_control_feasible']}, "
            f"margins P/R/Y="
            f"{summary['hover_pitch_margin']:.2f}/"
            f"{summary['hover_roll_margin']:.2f}/"
            f"{summary['hover_yaw_margin']:.2f}"
        )
    if "transition_blend_start_m_s" in summary:
        print(
            "  Transition blend: "
            f"{summary['transition_blend_start_m_s']:.2f}-"
            f"{summary['transition_blend_end_m_s']:.2f} m/s, "
            f"t={summary['transition_time_s']:.1f} s, "
            f"alpha_fw@cruise={summary['transition_alpha_fw_at_cruise']:.2f}"
        )
    if "dynamic_stability_preliminary" in summary:
        print(
            "  Dynamic stability: "
            f"prelim={summary['dynamic_stability_preliminary']}, "
            f"SP zeta/omega={summary['short_period_zeta']:.2f}/"
            f"{summary['short_period_omega_rad_s']:.2f}, "
            f"DR zeta/omega={summary['dutch_roll_zeta']:.2f}/"
            f"{summary['dutch_roll_omega_rad_s']:.2f}"
        )
    if "MTOW_mass_estimate_kg" in summary:
        print(
            "  Mass estimate: "
            f"{summary['MTOW_mass_estimate_kg']:.2f} kg "
            f"(delta {summary['MTOW_mass_delta_kg']:+.2f} kg vs sizing MTOW)"
        )
    if "phase16_relative_error" in summary:
        print(
            "  MTOW loop: "
            f"old={summary['phase16_MTOW_old_kg']:.2f} kg, "
            f"new={summary['phase16_MTOW_new_kg']:.2f} kg, "
            f"rel err={summary['phase16_relative_error']:.4f}"
        )
    if "design_sanity_issue_count" in summary:
        print(
            "  Sanity checks: "
            f"passed={summary['design_sanity_all_passed']}, "
            f"issues={summary['design_sanity_issue_count']}"
        )
    print(f"  Reynolds main/canard: {summary['Re_main']:.0f} / {summary['Re_canard']:.0f}")


def main(argv=None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    mission, assumptions, airfoil_files = _inputs_from_args(args)

    if args.fixed_mtow:
        result = iterate_phases_1_to_15(
            MTOW_kg=args.mtow_kg,
            mission=mission,
            assumptions=assumptions,
            max_inner_iter=args.max_inner_iter,
            j_tol=args.j_tol,
            area_tol=args.area_tol,
            re_tol=args.re_tol,
            constraint_plot_path=args.plot,
            airfoil_files=airfoil_files,
        )
        description = "Bellona Phase 1-15 fixed-MTOW propulsion/aerodynamics/canard/control/transition/stability/mass sizing result."
    else:
        result = converge_design(
            MTOW_seed_kg=args.mtow_kg,
            mission=mission,
            assumptions=assumptions,
            max_iter=args.outer_max_iter,
            tol=args.mtow_tol,
            damping=args.mtow_damping,
            max_inner_iter=args.max_inner_iter,
            j_tol=args.j_tol,
            area_tol=args.area_tol,
            re_tol=args.re_tol,
            constraint_plot_path=args.plot,
            airfoil_files=airfoil_files,
        )
        description = "Bellona Phase 1-16 MTOW-converged propulsion/aerodynamics/canard/control/transition/stability/mass sizing result."
    summary = _build_summary(result)
    payload = {
        "description": description,
        "inputs": {
            "MTOW_seed_kg": args.mtow_kg,
            "fixed_mtow": args.fixed_mtow,
            "outer_max_iter": args.outer_max_iter,
            "mtow_tol": args.mtow_tol,
            "mtow_damping": args.mtow_damping,
            "mission": asdict(mission),
            "assumptions": asdict(assumptions),
            "airfoil_files": airfoil_files,
            "constraint_plot_path": args.plot,
        },
        "summary": summary,
        "result": result,
    }

    _print_summary(summary)
    if args.json_path:
        json_path = Path(args.json_path)
        if json_path.parent != Path("."):
            json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(_json_safe(payload), indent=2),
            encoding="utf-8",
        )
        print(f"  JSON written to: {json_path}")
    if args.plot:
        print(f"  Constraint diagram written to: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
