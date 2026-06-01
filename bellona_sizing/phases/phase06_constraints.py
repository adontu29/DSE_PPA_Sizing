"""Phase 6: verification-only constraint diagram."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def phase6_constraint_diagram(W_S_range: np.ndarray, W_P_range: np.ndarray,
                              V_cruise, V_stall, ROC, gamma, rho_SL, rho_6km,
                              CL_max, CD0, AR, e, eta_prop, T_W_floor,
                              plot_path=None) -> Dict:
    """Plot W/S vs W/P with stall, cruise, climb, hover, transition-floor lines.
    Verification step (post Ph3, Ph5, Ph8). Not a sizing driver.
    Ref: Raymer 2018 Ch.5 5.3."""
    W_S = np.asarray(W_S_range, dtype=float)
    if W_S.ndim != 1 or W_S.size == 0:
        raise ValueError("W_S_range must be a non-empty 1-D array.")
    if np.any(W_S <= 0.0):
        raise ValueError("W_S_range values must be positive.")
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if ROC < 0.0:
        raise ValueError("ROC must be non-negative.")
    if rho_SL <= 0.0 or rho_6km <= 0.0:
        raise ValueError("Atmospheric densities must be positive.")
    if CL_max <= 0.0:
        raise ValueError("CL_max must be positive.")
    if CD0 <= 0.0:
        raise ValueError("CD0 must be positive.")
    if AR <= 0.0 or e <= 0.0:
        raise ValueError("AR and e must be positive.")
    if eta_prop <= 0.0 or eta_prop > 1.0:
        raise ValueError("eta_prop should be in the range 0 < eta_prop <= 1.")
    if T_W_floor <= 0.0:
        raise ValueError("T_W_floor must be positive.")

    q = 0.5 * rho_6km * V_cruise**2
    CL = W_S / q
    K = 1.0 / (np.pi * AR * e)
    CD = CD0 + K * CL**2
    P_W_cruise = V_cruise * (CD / CL) / eta_prop

    climb_rate_from_gamma = V_cruise * np.sin(gamma)
    climb_rate_used = ROC
    if climb_rate_used == 0.0 and climb_rate_from_gamma > 0.0:
        climb_rate_used = climb_rate_from_gamma
    P_W_climb = P_W_cruise + climb_rate_used / eta_prop

    W_S_stall_max = 0.5 * rho_SL * V_stall**2 * CL_max
    P_W_envelope = np.maximum(P_W_cruise, P_W_climb)

    with np.errstate(divide="ignore", invalid="ignore"):
        W_P_cruise = 1.0 / P_W_cruise
        W_P_climb = 1.0 / P_W_climb
        W_P_envelope = 1.0 / P_W_envelope

    warnings = [
        "Hover power cannot be plotted from this Phase 6 signature alone; pass Phase 2/5 hover power into a later integrated plot if needed.",
        "T_W_floor is reported as a thrust margin check, not converted to a power-loading line.",
    ]
    if abs(climb_rate_from_gamma - ROC) > max(0.5, 0.1 * max(ROC, 1.0)):
        warnings.append(
            "ROC and V_cruise*sin(gamma) differ noticeably; check Phase 3 consistency."
        )
    if np.any(W_S > W_S_stall_max):
        warnings.append(
            "Part of W_S_range is above the stall wing-loading limit."
        )

    saved_plot = None
    if plot_path:
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        ax.plot(W_S, P_W_cruise, color="#378ADD", lw=2.0, label="Cruise")
        ax.plot(W_S, P_W_climb, color="#1D9E75", lw=2.0, label=f"Climb ROC={climb_rate_used:.1f} m/s")
        ax.axvline(W_S_stall_max, color="#888780", lw=1.4, ls="--", label="Stall limit")
        ax.fill_between(W_S, 0.0, P_W_envelope, where=(W_S <= W_S_stall_max),
                        color="#1D9E75", alpha=0.08)
        ax.set_xlabel("Wing loading W/S [N/m^2]")
        ax.set_ylabel("Power-to-weight P/W [W/N]")
        ax.set_title("Phase 6 Constraint Diagram")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_range = np.asarray(W_P_range, dtype=float)
        if y_range.size >= 2 and np.all(y_range > 0.0):
            ax.set_ylim(float(np.min(y_range)), float(np.max(y_range)))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        saved_plot = str(plot_path)

    return {
        "W_S_range": W_S.tolist(),
        "P_W_cruise": P_W_cruise.tolist(),
        "P_W_climb": P_W_climb.tolist(),
        "P_W_envelope": P_W_envelope.tolist(),
        "W_P_cruise": W_P_cruise.tolist(),
        "W_P_climb": W_P_climb.tolist(),
        "W_P_envelope": W_P_envelope.tolist(),
        "W_S_stall_max": float(W_S_stall_max),
        "climb_rate_used": float(climb_rate_used),
        "climb_rate_from_gamma": float(climb_rate_from_gamma),
        "T_W_floor": float(T_W_floor),
        "plot_path": saved_plot,
        "notes": [
            "The plotted convention is P/W in W/N, matching the legacy Bellona stage-1 script.",
            "Inverse W/P arrays are returned for users who prefer the classical power-loading axis.",
            "Phase 6 is verification-only and does not alter the sizing loop.",
        ],
        "warnings": warnings,
    }
