"""Phase 1: propeller diameter and hover disk loading."""
from __future__ import annotations

from typing import Dict

import numpy as np


def phase1_propeller(MTOW_N: float, T_over_W: float, n_rotors: int,
                     DL_target: float, a_sound: float, M_tip_max: float = 0.72
                     ) -> Dict:
    """Physical propeller sizing.
    In:  MTOW [N], T/W, n_rotors, DL_target [N/m^2], a [m/s] (Ph0 ISA), M_tip_max
    Out: {'D_prop' [m], 'A_disc' [m^2], 'n_max' [rev/s], 'V_tip' [m/s]}
    Eq:  D=sqrt(4T/(piDL)); n_max = M_tip_maxa/(piD)
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
