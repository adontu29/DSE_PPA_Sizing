"""Shared atmosphere and simple sizing helper equations."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .models import Assumptions


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
    """ISA 1976 atmosphere (troposphere only, 011 km).
    In:  h [m]
    Out: rho [kg/m^3], mu [Pas], a [m/s], T [K]
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
    Eq:  D = sqrt(4*T_rotor / (pi*DL))    Gundlach 2014 Eq. 7.18 form
    Loop: MTOW outer."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if DL_target <= 0.0:
        raise ValueError("DL_target must be positive.")

    T_per_rotor = MTOW_N / n_rotors
    return float(np.sqrt(4.0 * T_per_rotor / (np.pi * DL_target)))
