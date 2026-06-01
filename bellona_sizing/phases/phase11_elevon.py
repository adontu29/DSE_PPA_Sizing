"""Phase 11: fixed-wing elevon verification and grid sizing."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def phase11_elevon_FW(S_w, b_w, c_bar_w, CL_a_w, V_cruise, V_stall,
                      CG_range, q_slipstream_ratio: float = 1.5,
                      p_required_rad_s: float = np.deg2rad(60),
                      Cm_de_required: Optional[float] = None,
                      delta_e_max_rad: float = np.deg2rad(25),
                      tau_e: float = 0.50,
                      eta_e: float = 0.90,
                      l_e_over_c: float = 0.75,
                      CL_trim: Optional[float] = None,
                      pitch_trim_margin: float = 1.20,
                      control_margin_min: float = 1.0,
                      chord_fraction_bounds: Tuple[float, float] = (0.12, 0.35),
                      span_fraction_bounds: Tuple[float, float] = (0.20, 0.80),
                      grid_points: int = 49) -> Dict:
    """Fixed-wing elevon sizing for pitch trim and roll-rate authority."""
    if S_w <= 0.0:
        raise ValueError("S_w must be positive.")
    if b_w <= 0.0:
        raise ValueError("b_w must be positive.")
    if c_bar_w <= 0.0:
        raise ValueError("c_bar_w must be positive.")
    if CL_a_w <= 0.0:
        raise ValueError("CL_a_w must be positive.")
    if V_cruise <= 0.0:
        raise ValueError("V_cruise must be positive.")
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if CG_range < 0.0:
        raise ValueError("CG_range must be non-negative.")
    if q_slipstream_ratio <= 0.0:
        raise ValueError("q_slipstream_ratio must be positive.")
    if p_required_rad_s <= 0.0:
        raise ValueError("p_required_rad_s must be positive.")
    if delta_e_max_rad <= 0.0:
        raise ValueError("delta_e_max_rad must be positive.")
    if tau_e <= 0.0:
        raise ValueError("tau_e must be positive.")
    if eta_e <= 0.0:
        raise ValueError("eta_e must be positive.")
    if l_e_over_c <= 0.0:
        raise ValueError("l_e_over_c must be positive.")
    if pitch_trim_margin <= 0.0:
        raise ValueError("pitch_trim_margin must be positive.")
    if control_margin_min <= 0.0:
        raise ValueError("control_margin_min must be positive.")
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")

    chord_min, chord_max = chord_fraction_bounds
    span_min, span_max = span_fraction_bounds
    if not 0.0 < chord_min <= chord_max < 1.0:
        raise ValueError("chord_fraction_bounds must lie inside 0 < min <= max < 1.")
    if not 0.0 < span_min <= span_max <= 1.0:
        raise ValueError("span_fraction_bounds must lie inside 0 < min <= max <= 1.")

    if CL_trim is None:
        CL_trim_used = 1.0
        CL_trim_source = "fallback"
    else:
        if CL_trim <= 0.0:
            raise ValueError("CL_trim must be positive when provided.")
        CL_trim_used = float(CL_trim)
        CL_trim_source = "phase3"

    if Cm_de_required is None:
        pitch_moment_required = (
            pitch_trim_margin
            * CL_trim_used
            * 0.5
            * CG_range / c_bar_w
        )
        Cm_de_required_used = pitch_moment_required / delta_e_max_rad
        Cm_de_source = "computed_from_phase10_CG_range"
    else:
        if Cm_de_required <= 0.0:
            raise ValueError("Cm_de_required must be positive when provided.")
        Cm_de_required_used = float(Cm_de_required)
        pitch_moment_required = Cm_de_required_used * delta_e_max_rad
        Cm_de_source = "user_supplied"

    C_l_p = -CL_a_w / 12.0
    Cl_da_required = (
        p_required_rad_s
        * abs(C_l_p)
        * b_w
        / (2.0 * delta_e_max_rad * V_cruise)
    )

    def _surface_metrics(c_e_over_c: float, b_e_over_b: float) -> Dict:
        S_e_over_S = c_e_over_c * b_e_over_b
        S_e_total = S_w * S_e_over_S
        Cm_de = (
            -q_slipstream_ratio
            * CL_a_w
            * eta_e
            * tau_e
            * c_e_over_c
            * S_e_over_S
            * l_e_over_c
        )

        y2 = 0.5 * b_w
        y1 = y2 * (1.0 - b_e_over_b)
        c_rect = S_w / b_w
        strip_integral = 0.5 * c_rect * (y2**2 - y1**2)
        Cl_da = (
            2.0
            * q_slipstream_ratio
            * CL_a_w
            * tau_e
            * c_e_over_c
            * strip_integral
            / (S_w * b_w)
        )
        p_achievable = (
            2.0
            * abs(Cl_da)
            * delta_e_max_rad
            * V_cruise
            / (abs(C_l_p) * b_w)
        )
        return {
            "c_e_over_c": float(c_e_over_c),
            "b_e_over_b": float(b_e_over_b),
            "S_e_over_S": float(S_e_over_S),
            "S_e_total": float(S_e_total),
            "S_e_each": float(0.5 * S_e_total),
            "Cm_de": float(Cm_de),
            "Cl_da": float(Cl_da),
            "p_achievable_rad_s": float(p_achievable),
            "p_achievable_deg_s": float(np.rad2deg(p_achievable)),
            "strip_integral": float(strip_integral),
        }

    chord_grid = np.linspace(chord_min, chord_max, grid_points)
    span_grid = np.linspace(span_min, span_max, grid_points)
    feasible_candidates = []
    checked = 0
    for c_e_over_c in chord_grid:
        for b_e_over_b in span_grid:
            checked += 1
            metrics = _surface_metrics(c_e_over_c, b_e_over_b)
            pitch_ok = abs(metrics["Cm_de"]) >= control_margin_min * Cm_de_required_used
            roll_ok = metrics["p_achievable_rad_s"] >= control_margin_min * p_required_rad_s
            if pitch_ok and roll_ok:
                feasible_candidates.append(metrics)

    if feasible_candidates:
        selected = min(
            feasible_candidates,
            key=lambda item: (item["S_e_over_S"], item["c_e_over_c"], item["b_e_over_b"]),
        )
        feasible = True
    else:
        selected = _surface_metrics(chord_max, span_max)
        feasible = False

    pitch_margin = abs(selected["Cm_de"]) / Cm_de_required_used
    roll_margin = selected["p_achievable_rad_s"] / p_required_rad_s
    pitch_ok = pitch_margin >= 1.0
    roll_ok = roll_margin >= 1.0
    if pitch_ok and roll_ok:
        binding_case = "pitch" if pitch_margin <= roll_margin else "roll"
    elif pitch_ok:
        binding_case = "roll"
    elif roll_ok:
        binding_case = "pitch"
    else:
        binding_case = "pitch_and_roll"

    warnings = [
        "Phase 11 is a simplified verification model; it does not replace hinge-moment, aeroelastic, or control-derivative analysis.",
        "Roll authority uses a rectangular-wing strip approximation and Cl_p = -CL_a/12.",
        "Pitch authority uses the Phase 10 CG range and an assumed elevon moment arm; replace with CAD CG and aerodynamic derivative data.",
        "The slipstream factor is a scalar multiplier; replace with a propeller-wing interaction model or wind-tunnel data.",
    ]
    if not feasible:
        warnings.append(
            "No elevon geometry inside the selected chord/span bounds met both pitch and roll requirements."
        )
    if selected["c_e_over_c"] >= 0.30:
        warnings.append(
            "Selected elevon chord fraction is large; check hinge moment, stiffness, and packaging."
        )
    if selected["b_e_over_b"] >= 0.70:
        warnings.append(
            "Selected elevon span fraction is large; check flap-tip clearance and structural integration."
        )
    if V_cruise < 1.1 * V_stall:
        warnings.append(
            "Cruise speed is close to stall speed; fixed-wing control authority should also be checked at transition speed."
        )

    selected.update({
        "feasible_preliminary_elevon": bool(feasible),
        "pitch_meets_requirement": bool(pitch_ok),
        "roll_meets_requirement": bool(roll_ok),
        "binding_case": binding_case,
        "Cm_de_required": float(Cm_de_required_used),
        "Cm_de_abs": float(abs(selected["Cm_de"])),
        "pitch_moment_required": float(pitch_moment_required),
        "pitch_margin": float(pitch_margin),
        "Cl_da_required": float(Cl_da_required),
        "roll_margin": float(roll_margin),
        "C_l_p": float(C_l_p),
        "V_control_roll": float(V_cruise),
        "V_stall": float(V_stall),
        "delta_e_max_rad": float(delta_e_max_rad),
        "delta_e_max_deg": float(np.rad2deg(delta_e_max_rad)),
        "p_required_rad_s": float(p_required_rad_s),
        "p_required_deg_s": float(np.rad2deg(p_required_rad_s)),
        "q_slipstream_ratio": float(q_slipstream_ratio),
        "tau_e": float(tau_e),
        "eta_e": float(eta_e),
        "l_e_over_c": float(l_e_over_c),
        "CL_trim": float(CL_trim_used),
        "CL_trim_source": CL_trim_source,
        "Cm_de_source": Cm_de_source,
        "CG_range_m": float(CG_range),
        "CG_range_over_c": float(CG_range / c_bar_w),
        "pitch_trim_margin": float(pitch_trim_margin),
        "control_margin_min": float(control_margin_min),
        "chord_fraction_bounds": [float(chord_min), float(chord_max)],
        "span_fraction_bounds": [float(span_min), float(span_max)],
        "grid_points": int(grid_points),
        "checked_candidates": int(checked),
        "feasible_candidates": int(len(feasible_candidates)),
        "notes": [
            "b_e_over_b is the per-side elevon span divided by semispan.",
            "S_e_total is the combined left-plus-right elevon planform area.",
            "Cm_de is reported per radian of symmetric elevon deflection.",
            "Cl_da and p_achievable are reported per radian of differential elevon deflection.",
        ],
        "warnings": warnings,
    })
    return selected
