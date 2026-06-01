"""Phase 12: hover differential-thrust and yaw-control verification."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


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
