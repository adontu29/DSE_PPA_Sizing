"""Shared atmosphere and simple sizing helper equations."""
from __future__ import annotations

from typing import Tuple

import numpy as np

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
