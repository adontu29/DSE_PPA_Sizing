"""Phase 9: canard planform from volume coefficient."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .phase08_wing import _datcom_lift_curve_slope


def phase9_canard(S_w: float, c_bar_w: float, x_ac_w: float, l_c: float,
                  V_bar_c: float = 0.375, AR_c: float = 5.0,
                  taper_c: float = 0.5, sweep_c4_c: float = 0.0,
                  cl_max_2D_c: float = 1.3177, mach: float = 0.0) -> Dict:
    """Canard planform from volume coefficient.
    In:  S_w, c_bar_w, x_ac_w (Ph8), l_c, V_c target (Nicolai 11.6).
    Out: {'S_c', 'b_c', 'c_bar_c', 'AR_c', 'l_c', 'CL_a_c'}
    Eq:  S_c = V_c  S_w  c_bar_w / l_c        Nicolai 2010 Eq. 11.7
         CL_a_c via Polhamus (same as Ph8).
    Loop: CGcanard inner loop with Ph10."""
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
