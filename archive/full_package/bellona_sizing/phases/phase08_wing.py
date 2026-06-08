"""Phase 8: wing planform and finite-wing lift estimates."""
from __future__ import annotations

from typing import Dict

import numpy as np


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


def phase8_wing_planform(MTOW_N: float, S_wing: float,
                         cl_max_2D: float, cl_a_2D: float,
                         rho_mission: float, rho_transition: float,
                         rho_reference: float = 1.225,
                         AR_guess: float = 7.0, taper: float = 0.4,
                         sweep_c4: float = 0.0, e: float = 0.78,
                         mach: float = 0.0) -> Dict:
    """Wing planform and calculated stall speeds for a selected wing area.
    In:  MTOW [N], selected wing area, cl_max,2D, cl_a,2D (Ph7),
         mission/transition/reference densities, AR, taper, and sweep.
    Out: {'S', 'b', 'c_bar', 'AR', 'CL_a', 'CL_max_3D',
          'stall_EAS_m_s', 'stall_TAS_mission_m_s',
          'stall_TAS_transition_m_s', 'x_ac_w'}
    Eq:  CL_a = 2*pi*AR/(2 + sqrt(AR^2(1-M^2)(1+tan^2 Lambda_c2/(1-M^2))/k^2+4))
                                                   Polhamus / DATCOM 4.1.3.2
         CL_max_3D = 0.9cl_maxcos(Lambda_c4)     Raymer 2018 Eq. 12.16
         V_stall  = sqrt(2(W/S)/(rho CL_max_3D))
    Loop: Re inner (Ph7); MTOW outer."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if S_wing <= 0.0:
        raise ValueError("S_wing must be positive.")
    if cl_max_2D <= 0.0:
        raise ValueError("cl_max_2D must be positive.")
    if cl_a_2D <= 0.0:
        raise ValueError("cl_a_2D must be positive.")
    if rho_mission <= 0.0 or rho_transition <= 0.0 or rho_reference <= 0.0:
        raise ValueError("Stall-speed densities must be positive.")
    if AR_guess <= 0.0:
        raise ValueError("AR_guess must be positive.")
    if not 0.0 < taper <= 1.0:
        raise ValueError("taper should be in the range 0 < taper <= 1.")
    if e <= 0.0:
        raise ValueError("Oswald efficiency e must be positive.")
    if not 0.0 <= mach < 1.0:
        raise ValueError("mach must be subsonic and non-negative.")

    AR = AR_guess
    CL_a = _datcom_lift_curve_slope(AR, sweep_c4, taper, mach=mach, k=1.0)
    sweep_c2 = _half_chord_sweep_from_quarter_chord(sweep_c4, AR, taper)
    CL_max_3D = 0.9 * cl_max_2D * np.cos(sweep_c4)

    S = float(S_wing)
    W_S_design = MTOW_N / S
    b = np.sqrt(S * AR)
    c_bar = S / b
    c_root = 2.0 * S / (b * (1.0 + taper))
    c_tip = taper * c_root
    c_mac = (2.0 / 3.0) * c_root * (1.0 + taper + taper**2) / (1.0 + taper)
    y_mac = (b / 6.0) * (1.0 + 2.0 * taper) / (1.0 + taper)
    x_ac_w = 0.25 * c_mac
    stall_EAS = np.sqrt(
        2.0 * W_S_design / (rho_reference * CL_max_3D)
    )
    stall_TAS_mission = np.sqrt(
        2.0 * W_S_design / (rho_mission * CL_max_3D)
    )
    stall_TAS_transition = np.sqrt(
        2.0 * W_S_design / (rho_transition * CL_max_3D)
    )

    warnings = []
    if abs(cl_a_2D - 2.0 * np.pi) > 0.75:
        warnings.append(
            "Input 2-D lift-curve slope differs noticeably from 2*pi; DATCOM finite-wing slope still uses k=1.0 as in the project table."
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
        "W_S": float(W_S_design),
        "W_S_design": float(W_S_design),
        "stall_EAS_m_s": float(stall_EAS),
        "stall_TAS_mission_m_s": float(stall_TAS_mission),
        "stall_TAS_transition_m_s": float(stall_TAS_transition),
        "stall_density_reference_kg_m3": float(rho_reference),
        "stall_density_mission_kg_m3": float(rho_mission),
        "stall_density_transition_kg_m3": float(rho_transition),
        # Deprecated compatibility alias. Use the explicit EAS/TAS keys.
        "V_stall": float(stall_TAS_mission),
        "x_ac_w": float(x_ac_w),
        "x_ac_w_over_mac": 0.25,
        "mach": float(mach),
        "e": float(e),
        "notes": [
            "Wing area is an independent design input; stall speeds are calculated outputs.",
            "stall_EAS_m_s uses rho_reference; mission and transition stall speeds are true airspeeds at their stated densities.",
            "V_stall is a deprecated alias for stall_TAS_mission_m_s.",
            "x_ac_w is measured from the leading edge of the mean aerodynamic chord; a fuselage reference needs CAD layout.",
        ],
        "warnings": warnings,
    }
