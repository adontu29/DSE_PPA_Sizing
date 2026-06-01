"""Phase 3: cruise speed and transition altitude grid search."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..common import isa


def phase3_mission_optimise(MTOW_N: float, S_guess: float, CD0: float, AR: float,
                            e: float, h_target: float, range_target: float,
                            t_budget: float, t_hover: float,
                            P_hover_fn: Callable, isa_fn: Callable = isa,
                            eta_fw: float = 0.75 * 0.90 * 0.95,
                            V_min_required: Optional[float] = None,
                            V_grid=None, h_transition_grid=None
                            ) -> Dict:
    """Derive V_cruise, gamma, ROC, h_transition by minimising total energy.
    In:  MTOW [N], S [m^2] (Ph8 estimate), CD0, AR, e (Ph8 estimate),
         h_target=6000 m, range_target=6000 m, t_budget=600 s, t_hover=300 s,
         P_hover_fn(h)->W (Ph2)
    Out: {'V_cruise', 'gamma', 'ROC', 'h_tr', 'Re_estimate'}
    Eq:  V_Pmin = sqrt(2(W/S)/rho  sqrt(K/(3CD0)))   Raymer 2018 Ch.17
         ROC = (P_avail - P_req)/W                      Raymer Eq. 17.22
    Method: grid-search h_tr in [50, 2000] m, inner 1-D minimise V over P_req(V).
    Loop: MTOW outer; V-J feedback from Ph4."""
    if MTOW_N <= 0.0:
        raise ValueError("MTOW_N must be positive.")
    if S_guess <= 0.0:
        raise ValueError("S_guess must be positive.")
    if CD0 <= 0.0:
        raise ValueError("CD0 must be positive.")
    if AR <= 0.0:
        raise ValueError("AR must be positive.")
    if e <= 0.0:
        raise ValueError("Oswald efficiency e must be positive.")
    if h_target <= 0.0:
        raise ValueError("h_target must be positive.")
    if range_target <= 0.0:
        raise ValueError("range_target must be positive.")
    if t_budget <= 0.0:
        raise ValueError("t_budget must be positive.")
    if t_hover < 0.0:
        raise ValueError("t_hover must be non-negative.")
    if eta_fw <= 0.0 or eta_fw > 1.0:
        raise ValueError("eta_fw should be in the range 0 < eta_fw <= 1.")
    if V_min_required is not None and V_min_required <= 0.0:
        raise ValueError("V_min_required must be positive when provided.")

    def _power_w(value):
        if isinstance(value, dict):
            for key in ("P_total", "P_elec_total", "P_elec"):
                if key in value:
                    return float(value[key])
            raise ValueError("Power dictionaries must contain P_total, P_elec_total, or P_elec.")
        return float(value)

    K = 1.0 / (np.pi * AR * e)
    W_S = MTOW_N / S_guess
    c_guess = np.sqrt(S_guess / AR)
    mission_ROC = h_target / t_budget

    if V_grid is None:
        path_speed = np.hypot(range_target, h_target) / t_budget
        V_min = max(8.0, 0.70 * path_speed)
        V_max = max(45.0, 2.50 * path_speed)
        if V_min_required is not None:
            V_min = max(V_min, V_min_required)
        if V_max <= V_min:
            V_max = 1.25 * V_min
        V_grid = np.linspace(V_min, V_max, 180)
    else:
        V_grid = np.asarray(V_grid, dtype=float)
        if V_min_required is not None:
            V_grid = V_grid[V_grid >= V_min_required]
            if V_grid.size == 0:
                raise ValueError("V_grid has no points above V_min_required.")

    if h_transition_grid is None:
        h_low = min(50.0, 0.25 * h_target)
        h_high = min(2000.0, 0.95 * h_target)
        h_transition_grid = np.linspace(h_low, h_high, 40)
    else:
        h_transition_grid = np.asarray(h_transition_grid, dtype=float)

    P_hover_ground = _power_w(P_hover_fn(0.0))
    P_hover_target = _power_w(P_hover_fn(h_target))
    best = None
    feasible_count = 0

    for h_tr in h_transition_grid:
        if h_tr < 0.0 or h_tr >= h_target:
            continue

        h_fw = h_target - h_tr
        gamma = np.arctan2(h_fw, range_target)
        path_fw = np.hypot(range_target, h_fw)
        h_avg = h_tr + 0.5 * h_fw
        rho_avg, mu_avg, _, _ = isa_fn(h_avg)

        t_transition = h_tr / mission_ROC
        t_fw_available = t_budget - t_transition
        if t_fw_available <= 0.0:
            continue

        P_hover_transition_top = _power_w(P_hover_fn(h_tr))
        P_transition = 0.5 * (P_hover_ground + P_hover_transition_top)
        E_transition_J = P_transition * t_transition

        for V in V_grid:
            if V <= 0.0:
                continue

            t_fw = path_fw / V
            if t_fw > t_fw_available:
                continue

            q = 0.5 * rho_avg * V**2
            CL = MTOW_N / (q * S_guess)
            CD = CD0 + K * CL**2
            drag = q * S_guess * CD
            P_fw = (drag + MTOW_N * np.sin(gamma)) * V / eta_fw
            E_fw_J = P_fw * t_fw
            E_hover_J = P_hover_target * t_hover
            E_total_J = E_transition_J + E_fw_J + E_hover_J
            ROC = V * np.sin(gamma)

            feasible_count += 1
            candidate = {
                "E_total_J": E_total_J,
                "V_cruise": V,
                "gamma": gamma,
                "ROC": ROC,
                "h_tr": h_tr,
                "t_transition": t_transition,
                "t_fw": t_fw,
                "P_fw": P_fw,
                "P_transition": P_transition,
                "P_hover_target": P_hover_target,
                "E_transition_J": E_transition_J,
                "E_fw_J": E_fw_J,
                "E_hover_J": E_hover_J,
                "rho_avg": rho_avg,
                "mu_avg": mu_avg,
                "CL": CL,
                "CD": CD,
                "drag": drag,
                "path_fw": path_fw,
            }
            if best is None or candidate["E_total_J"] < best["E_total_J"]:
                best = candidate

    if best is None:
        raise ValueError(
            "No feasible Phase 3 mission candidate found. Check the speed grid, transition grid, and time budget."
        )

    rho_best = best["rho_avg"]
    mu_best = best["mu_avg"]
    V_power_min = np.sqrt((2.0 * W_S / rho_best) * np.sqrt(K / (3.0 * CD0)))
    Re_estimate = rho_best * best["V_cruise"] * c_guess / mu_best
    t_total_outbound = best["t_transition"] + best["t_fw"]
    notes = [
        "The speed and transition-altitude grids are intentionally simple so the sizing trend is inspectable.",
        "P_hover_fn should return total aircraft electric power in W if the energy terms are compared directly.",
        "The 10 m/s default mission climb reference comes from h_target / t_budget for the 6000 m in 600 s mission.",
    ]
    if V_min_required is not None:
        notes.append(
            "The cruise-speed grid lower bound was raised to clear the transition blend-end speed plus margin."
        )

    return {
        "V_cruise": float(best["V_cruise"]),
        "V_min_required": None if V_min_required is None else float(V_min_required),
        "V_min_requirement_active": bool(
            V_min_required is not None and best["V_cruise"] >= V_min_required
        ),
        "gamma": float(best["gamma"]),
        "gamma_deg": float(np.rad2deg(best["gamma"])),
        "ROC": float(best["ROC"]),
        "h_tr": float(best["h_tr"]),
        "Re_estimate": float(Re_estimate),
        "V_power_min": float(V_power_min),
        "t_transition": float(best["t_transition"]),
        "t_fw": float(best["t_fw"]),
        "t_total_outbound": float(t_total_outbound),
        "time_margin": float(t_budget - t_total_outbound),
        "E_total_Wh": float(best["E_total_J"] / 3600.0),
        "E_transition_Wh": float(best["E_transition_J"] / 3600.0),
        "E_fw_Wh": float(best["E_fw_J"] / 3600.0),
        "E_hover_Wh": float(best["E_hover_J"] / 3600.0),
        "P_fw": float(best["P_fw"]),
        "P_transition": float(best["P_transition"]),
        "P_hover_target": float(best["P_hover_target"]),
        "CL_cruise": float(best["CL"]),
        "CD_cruise": float(best["CD"]),
        "drag_N": float(best["drag"]),
        "path_fw_m": float(best["path_fw"]),
        "mission_ROC_reference": float(mission_ROC),
        "eta_fw": float(eta_fw),
        "feasible_candidates": int(feasible_count),
        "notes": notes,
        "warnings": [
            "Transition energy uses hover-like power and a mission-average vertical-rate time estimate; replace with a tail-sitter transition simulation or test data.",
            "Fixed-wing power uses a parabolic drag polar and a single average climb altitude; verify with a detailed mission simulation.",
        ],
    }
