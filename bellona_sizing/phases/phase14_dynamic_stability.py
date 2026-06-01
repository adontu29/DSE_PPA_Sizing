"""Phase 14: simplified dynamic-stability verification."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


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
