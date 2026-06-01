"""Phase 4: cruise advance-ratio check."""
from __future__ import annotations

from typing import Dict

import numpy as np


def phase4_J_coupling(V_cruise: float, n_rev_s: float, D_prop: float,
                      J_min: float = 0.4, J_max: float = 0.8) -> Dict:
    """Verify advance ratio in efficient band; signal D-revise if not.
    In:  V_cruise [m/s] (Ph3), n [rev/s] (Ph1), D [m] (Ph1)
    Out: {'J', 'in_band' (bool), 'D_recommend'}
    Ref: Gudmundsson 2022 Ch.14 14.6; UIUC propeller DB.
    Loop: V_cruiseJ inner loop with Ph3 (or Ph1 D-revise)."""
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
