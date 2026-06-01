"""Phase 2: hover and vertical-climb power."""
from __future__ import annotations

from typing import Dict

import numpy as np


def phase2_hover_climb_power(T_per_rotor: float, A_disc: float, rho: float,
                             V_climb: float, FoM: float, eta_motor: float,
                             eta_ESC: float) -> Dict:
    """Actuator-disc induced velocity, hover & axial-climb power.
    In:  T [N], A_disc [m^2], rho [kg/m^3] (Ph0), V_c [m/s], FoM, eta_motor, eta_ESC
    Out: {'v_i', 'P_shaft', 'P_elec'}  [m/s, W, W]
    Eq:  v_i = -V_c/2 + sqrt((V_c/2)^2 + T/(2rhoA))   Leishman 2006 Eq. 2.91
         P_shaft = T(V_c + v_i)/FoM                    Leishman Eq. 2.96
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
