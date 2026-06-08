"""Phase 1: preliminary rotor diameter from hover disk loading."""
from __future__ import annotations

from typing import Dict

import numpy as np


def phase1_propeller(MTOW_N: float, T_over_W: float, n_rotors: int,
                     DL_target: float) -> Dict:
    """Estimate rotor disk area and diameter from a target disk loading.

    This preliminary geometry estimate does not select a propeller or predict
    RPM, thrust coefficient, efficiency, or installed propulsion capability.
    """
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if T_over_W <= 0.0:
        raise ValueError("T_over_W must be positive.")
    if n_rotors <= 0:
        raise ValueError("n_rotors must be positive.")
    if DL_target <= 0.0:
        raise ValueError("DL_target must be positive.")
    T_total = T_over_W * MTOW_N
    T_per_rotor = T_total / n_rotors
    A_disc = T_per_rotor / DL_target
    D_prop = np.sqrt(4.0 * A_disc / np.pi)

    warnings = []
    if DL_target > 250.0:
        warnings.append(
            "Disc loading is high for efficient hover; verify against propeller clearance and hover power."
        )

    return {
        "T_total": float(T_total),
        "T_per_rotor": float(T_per_rotor),
        "D_prop": float(D_prop),
        "A_disc": float(A_disc),
        "disc_loading": float(DL_target),
        "notes": [
            "The external balloon payload is not included in MTOW; this phase sizes only UAV lift/thrust.",
            "Rotor diameter is based only on target disk loading and must be checked against CAD clearance.",
            "No propeller operating point or component capability is predicted.",
        ],
        "warnings": warnings,
    }
