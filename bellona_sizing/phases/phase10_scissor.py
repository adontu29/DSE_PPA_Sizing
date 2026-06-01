"""Phase 10: simplified canard scissor and CG range check."""
from __future__ import annotations

from typing import Dict


def phase10_scissor_canard(S_w, c_bar_w, x_ac_w, CL_a_w,
                           S_c, l_c, CL_a_c, eps_alpha_c, eps_alpha_w,
                           CL_c_max, CL_trim, SM_min: float = 0.10) -> Dict:
    """Longitudinal scissor check using a canard neutral-point formulation.
    In:  wing & canard geometry/derivs (Ph8, Ph9), downwash/upwash gradients,
         max canard lift coefficient, design SM_min.
    Out: {'x_np_over_c', 'x_cg_fwd_over_c', 'x_cg_aft_over_c', 'CG_range_pct'}
    Eq:  x_np/ell = [1 + a_w(1-eps_c)S_w / (a_c(1+eps_w)S_c)]^-1
                                                   Phillips NASA TM-86694 Eq.16
         Equivalent canonical (Nelson-style canard):
         x_np/c = x_ac_w/c  (C_La_c/C_La_AC)eta_cV_c(1+eps_w)
         Caughey MAE5070 Eq.3.29: canard contribution is NEGATIVE (forward).
    Loop: CGcanard inner (revises S_c, Ph9)."""
    if S_w <= 0.0:
        raise ValueError("S_w must be positive.")
    if c_bar_w <= 0.0:
        raise ValueError("c_bar_w must be positive.")
    if CL_a_w <= 0.0:
        raise ValueError("CL_a_w must be positive.")
    if S_c <= 0.0:
        raise ValueError("S_c must be positive.")
    if l_c <= 0.0:
        raise ValueError("l_c must be positive.")
    if CL_a_c <= 0.0:
        raise ValueError("CL_a_c must be positive.")
    if CL_c_max <= 0.0:
        raise ValueError("CL_c_max must be positive.")
    if CL_trim <= 0.0:
        raise ValueError("CL_trim must be positive.")
    if SM_min < 0.0:
        raise ValueError("SM_min must be non-negative.")
    if eps_alpha_c >= 1.0:
        raise ValueError("eps_alpha_c must be below 1.0 for this simplified model.")
    if eps_alpha_w <= -1.0:
        raise ValueError("eps_alpha_w must be above -1.0 for this simplified model.")

    x_ac_c = x_ac_w - l_c
    V_bar_c = S_c * l_c / (S_w * c_bar_w)
    area_ratio = S_c / S_w
    a_w_eff = CL_a_w * (1.0 - eps_alpha_c)
    a_c_eff = CL_a_c * (1.0 + eps_alpha_w)
    if a_w_eff <= 0.0 or a_c_eff <= 0.0:
        raise ValueError("Effective lift-curve slopes must remain positive.")

    x_np_forward_fraction = 1.0 / (1.0 + (a_w_eff * S_w) / (a_c_eff * S_c))
    x_np = x_ac_w - x_np_forward_fraction * l_c
    x_np_over_c = x_np / c_bar_w

    x_cg_aft = x_np - SM_min * c_bar_w
    x_cg_aft_over_c = x_cg_aft / c_bar_w

    canard_lift_fraction_max = CL_c_max * S_c / (CL_trim * S_w)
    x_cg_fwd = x_ac_w - canard_lift_fraction_max * l_c
    x_cg_fwd_over_c = x_cg_fwd / c_bar_w
    CG_range = x_cg_aft - x_cg_fwd
    CG_range_pct = 100.0 * CG_range / c_bar_w
    feasible = CG_range > 0.0

    CL_c_required_at_aft = (
        (x_ac_w - x_cg_aft) / l_c
        * CL_trim
        * S_w / S_c
    )
    CL_c_required_at_neutral = (
        (x_ac_w - x_np) / l_c
        * CL_trim
        * S_w / S_c
    )

    warnings = [
        "Phase 10 is a simplified verification model; it does not replace a full scissor plot with measured derivatives.",
        "CG limits are referenced to the wing mean aerodynamic chord datum, not a CAD fuselage datum.",
        "Forward CG limit assumes the canard can reach CL_c_max with no incidence, elevator, propwash, or stall-margin correction.",
    ]
    if not feasible:
        warnings.append(
            "The preliminary forward and aft CG limits do not overlap; revise canard volume, moment arm, or trim assumptions."
        )
    if x_cg_aft_over_c < -0.25 or x_cg_aft_over_c > 0.50:
        warnings.append(
            "Aft CG limit lies far from the wing MAC reference; check the longitudinal datum and canard layout."
        )
    if canard_lift_fraction_max > 0.6:
        warnings.append(
            "Canard maximum lift fraction is high; include canard stall margin and trim drag before accepting the forward CG limit."
        )
    if CL_c_required_at_aft > 0.7 * CL_c_max:
        warnings.append(
            "Canard lift required at the aft CG limit uses much of the available canard CL margin."
        )

    return {
        "x_np": float(x_np),
        "x_np_over_c": float(x_np_over_c),
        "x_np_forward_of_wing_ac": float(x_ac_w - x_np),
        "x_np_forward_fraction_l_c": float(x_np_forward_fraction),
        "x_cg_fwd": float(x_cg_fwd),
        "x_cg_fwd_over_c": float(x_cg_fwd_over_c),
        "x_cg_aft": float(x_cg_aft),
        "x_cg_aft_over_c": float(x_cg_aft_over_c),
        "CG_range_m": float(CG_range),
        "CG_range_pct": float(CG_range_pct),
        "feasible_preliminary_CG_range": bool(feasible),
        "SM_min": float(SM_min),
        "V_bar_c": float(V_bar_c),
        "area_ratio": float(area_ratio),
        "x_ac_w": float(x_ac_w),
        "x_ac_c": float(x_ac_c),
        "CL_trim": float(CL_trim),
        "CL_c_max": float(CL_c_max),
        "CL_c_required_at_aft_cg": float(CL_c_required_at_aft),
        "CL_c_required_at_neutral_point": float(CL_c_required_at_neutral),
        "canard_lift_fraction_max": float(canard_lift_fraction_max),
        "a_w_eff": float(a_w_eff),
        "a_c_eff": float(a_c_eff),
        "eps_alpha_c": float(eps_alpha_c),
        "eps_alpha_w": float(eps_alpha_w),
        "notes": [
            "x/c values use the wing MAC leading-edge datum from Phase 8.",
            "The neutral point is estimated as the weighted aerodynamic-center location of the wing and forward canard.",
            "The aft CG limit is x_np minus SM_min*c_bar_w for positive static margin.",
            "The forward CG limit is set by the canard CL_c_max trim capability.",
        ],
        "warnings": warnings,
    }
