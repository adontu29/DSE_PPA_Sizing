"""Phase 10: canard scissor plot equations and CG range check."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CanardScissorInputs:
    """Inputs for the lecture canard scissor equations.

    All longitudinal positions are normalized by the wing reference chord,
    with the wing leading edge as the datum. The canard moment arm is negative
    because the canard aerodynamic center is ahead of the wing aerodynamic
    center.
    """

    x_ac: float
    Cm_ac: float
    CL_Ah: float
    CLa_Ah: float
    CL_h: float
    CLa_h: float
    l_h: float
    c_bar: float
    SM: float = 0.05
    VhV2: float = 1.0
    deps_dalpha: float = 0.0
    x_cg_fixed: Optional[float] = None

    def validate(self) -> list[str]:
        """Validate inputs and return non-fatal topology warnings."""
        warnings: list[str] = []
        if self.l_h >= 0.0:
            warnings.append("Canard moment arm should be negative.")
        if self.c_bar <= 0.0:
            raise ValueError("c_bar must be positive.")
        if self.CL_Ah <= 0.0:
            raise ValueError("CL_Ah must be positive.")
        if self.CLa_Ah <= 0.0 or self.CLa_h <= 0.0:
            raise ValueError("Lift-curve slopes must be positive.")
        if self.CL_h <= 0.0:
            raise ValueError("CL_h must be positive.")
        if self.SM < 0.0:
            warnings.append("Static margin is negative.")
        if self.VhV2 <= 0.0:
            raise ValueError("VhV2 must be positive.")
        if self.deps_dalpha >= 1.0:
            raise ValueError("deps_dalpha must remain below 1.0.")

        ctrl_ratio = self.CL_h / self.CL_Ah
        stab_ratio = self.CLa_h / self.CLa_Ah
        if ctrl_ratio <= stab_ratio * (1.0 - self.deps_dalpha):
            warnings.append(
                "Canard controllability slope is not steeper than the "
                "stability slope; the feasible scissor band may not open."
            )
        return warnings


def _stability_slope(p: CanardScissorInputs) -> float:
    """d(x_cg/c)/d(S_c/S_w) for the stability line."""
    return (
        (p.CLa_h / p.CLa_Ah)
        * (1.0 - p.deps_dalpha)
        * (p.l_h / p.c_bar)
        * p.VhV2
    )


def _controllability_slope(p: CanardScissorInputs) -> float:
    """d(x_cg/c)/d(S_c/S_w) for the controllability line."""
    return (p.CL_h / p.CL_Ah) * (p.l_h / p.c_bar) * p.VhV2


def stability_x_cg(ShS, p: CanardScissorInputs):
    """Aft CG limit from the lecture stability equation."""
    return p.x_ac + _stability_slope(p) * ShS - p.SM


def controllability_x_cg(ShS, p: CanardScissorInputs):
    """Forward CG limit from the lecture controllability equation."""
    return p.x_ac - p.Cm_ac / p.CL_Ah + _controllability_slope(p) * ShS


def line_crossing(p: CanardScissorInputs) -> Optional[Tuple[float, float]]:
    """Return (x_cg/c, S_c/S_w) where the two scissor lines intersect."""
    ms = _stability_slope(p)
    mc = _controllability_slope(p)
    b_stab = p.x_ac - p.SM
    b_ctrl = p.x_ac - p.Cm_ac / p.CL_Ah
    if abs(ms - mc) < 1e-14:
        return None
    ShS = (b_ctrl - b_stab) / (ms - mc)
    x_cg = b_stab + ms * ShS
    return float(x_cg), float(ShS)


def fixed_cg_area_bounds(p: CanardScissorInputs) -> Dict:
    """Invert the scissor lines for a fixed CG.

    For a canard, both scissor slopes are normally negative. Controllability
    then gives a lower bound on canard area, while stability gives an upper
    bound because a larger forward lifting surface moves the neutral point
    forward.
    """
    if p.x_cg_fixed is None:
        raise ValueError("x_cg_fixed must be set before inverting the scissor plot.")

    ms = _stability_slope(p)
    mc = _controllability_slope(p)
    b_stab = p.x_ac - p.SM
    b_ctrl = p.x_ac - p.Cm_ac / p.CL_Ah
    x = p.x_cg_fixed

    lower_bounds = [(0.0, "nonnegative area")]
    upper_bounds = [(np.inf, "no active upper bound")]

    ctrl_boundary = (x - b_ctrl) / mc if abs(mc) > 1e-14 else np.nan
    stab_boundary = (x - b_stab) / ms if abs(ms) > 1e-14 else np.nan

    if abs(mc) < 1e-14:
        ctrl_feasible_without_area = b_ctrl <= x
        if not ctrl_feasible_without_area:
            lower_bounds.append((np.inf, "controllability"))
    elif mc < 0.0:
        lower_bounds.append((ctrl_boundary, "controllability"))
    else:
        upper_bounds.append((ctrl_boundary, "controllability"))

    if abs(ms) < 1e-14:
        stab_feasible_without_area = x <= b_stab
        if not stab_feasible_without_area:
            lower_bounds.append((np.inf, "stability"))
    elif ms < 0.0:
        upper_bounds.append((stab_boundary, "stability"))
    else:
        lower_bounds.append((stab_boundary, "stability"))

    ShS_min_raw, lower_binding = max(lower_bounds, key=lambda item: item[0])
    ShS_max_raw, upper_binding = min(upper_bounds, key=lambda item: item[0])
    ShS_min = max(0.0, float(ShS_min_raw))
    ShS_max = float(ShS_max_raw)
    feasible = np.isfinite(ShS_min) and ShS_min <= ShS_max and ShS_max >= 0.0

    return {
        "ShS_min": float(ShS_min),
        "ShS_max": None if np.isinf(ShS_max) else float(ShS_max),
        "ShS_from_controllability": float(ctrl_boundary),
        "ShS_from_stability": float(stab_boundary),
        "lower_bound_governed_by": lower_binding,
        "upper_bound_governed_by": upper_binding,
        "feasible_area_window": bool(feasible),
    }


def minimum_ShS_for_fixed_cg(p: CanardScissorInputs) -> Dict:
    """Compatibility wrapper using the lecture script's sizing terminology."""
    bounds = fixed_cg_area_bounds(p)
    return {
        "ShS_min": bounds["ShS_min"],
        "ShS_max": bounds["ShS_max"],
        "ShS_from_stability": bounds["ShS_from_stability"],
        "ShS_from_ctrl": bounds["ShS_from_controllability"],
        "governed_by": bounds["lower_bound_governed_by"],
        "upper_limited_by": bounds["upper_bound_governed_by"],
        "feasible": bounds["feasible_area_window"],
    }


def _required_canard_cl_for_xcg(x_cg_over_c: float, ShS: float,
                                p: CanardScissorInputs) -> float:
    """Canard CL needed to trim a selected x_cg/c at a selected area ratio."""
    denominator = ShS * (p.l_h / p.c_bar) * p.VhV2
    if abs(denominator) < 1e-14:
        return float("nan")
    return float(
        (x_cg_over_c - p.x_ac + p.Cm_ac / p.CL_Ah)
        * p.CL_Ah
        / denominator
    )


def phase10_scissor_canard(S_w, c_bar_w, x_ac_w, CL_a_w,
                           S_c, l_c, CL_a_c, eps_alpha_c, eps_alpha_w,
                           CL_c_max, CL_trim, SM_min: float = 0.10, *,
                           Cm_ac: float = 0.0,
                           Cm_ac_source: str = "assumed_zero",
                           VhV2: float = 1.0,
                           x_cg_fixed: Optional[float] = None,
                           CL_Ah_control: Optional[float] = None) -> Dict:
    """Run the lecture canard scissor check using current sizing estimates.

    Inputs come from the active sizing workflow:
    - Phase 8 supplies wing area, reference chord, wing aerodynamic center,
      and wing lift-curve slope.
    - Phase 9 supplies canard area, moment arm, and lift-curve slope.
    - Phase 3 supplies the reference total trim CL.
    - Phase 7 may supply the aircraft-minus-canard pitching-moment estimate.

    The returned legacy keys are preserved for the layout, control, dynamic
    stability, CLI, and sanity-check workflows.
    """
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
        raise ValueError("eps_alpha_c must be below 1.0.")
    if eps_alpha_w <= -1.0:
        raise ValueError("eps_alpha_w must be above -1.0.")

    area_ratio = S_c / S_w
    x_ac_over_c = x_ac_w / c_bar_w
    l_h = -float(l_c)
    a_w_eff = CL_a_w * (1.0 - eps_alpha_c)
    a_c_eff = CL_a_c * (1.0 + eps_alpha_w)
    if CL_Ah_control is None:
        CL_Ah = CL_trim - CL_c_max * area_ratio * VhV2
        CL_Ah_source = "CL_trim_minus_usable_canard_lift_at_current_area"
    else:
        CL_Ah = float(CL_Ah_control)
        CL_Ah_source = "input_override"
    if CL_Ah <= 0.0:
        raise ValueError(
            "Aircraft-minus-canard CL at the controllability condition must "
            "be positive; reduce canard usable CL or area ratio, or provide "
            "CL_Ah_control explicitly."
        )
    p = CanardScissorInputs(
        x_ac=x_ac_over_c,
        Cm_ac=float(Cm_ac),
        CL_Ah=float(CL_Ah),
        CLa_Ah=float(a_w_eff),
        CL_h=float(CL_c_max),
        CLa_h=float(a_c_eff),
        l_h=l_h,
        c_bar=float(c_bar_w),
        SM=float(SM_min),
        VhV2=float(VhV2),
        deps_dalpha=0.0,
        x_cg_fixed=x_cg_fixed,
    )
    warnings = p.validate()

    x_stab_over_c = float(stability_x_cg(area_ratio, p))
    x_ctrl_over_c = float(controllability_x_cg(area_ratio, p))
    x_np_over_c = x_stab_over_c + SM_min
    x_cg_aft = x_stab_over_c * c_bar_w
    x_cg_fwd = x_ctrl_over_c * c_bar_w
    CG_range = x_cg_aft - x_cg_fwd
    feasible = CG_range > 0.0
    V_bar_c = S_c * l_c / (S_w * c_bar_w)
    cross = line_crossing(p)

    fixed_cg_sizing = None
    if x_cg_fixed is not None:
        fixed_cg_sizing = fixed_cg_area_bounds(p)
        ShS_max = fixed_cg_sizing["ShS_max"]
        fixed_cg_sizing["current_area_ratio"] = float(area_ratio)
        fixed_cg_sizing["current_area_ratio_feasible"] = bool(
            fixed_cg_sizing["feasible_area_window"]
            and area_ratio >= fixed_cg_sizing["ShS_min"] - 1e-12
            and (ShS_max is None or area_ratio <= ShS_max + 1e-12)
        )

    CL_c_required_at_aft = _required_canard_cl_for_xcg(
        x_stab_over_c,
        area_ratio,
        p,
    )
    CL_c_required_at_neutral = _required_canard_cl_for_xcg(
        x_np_over_c,
        area_ratio,
        p,
    )

    notes = [
        "Phase 10 uses the canard scissor equations from the lecture script.",
        "The canard moment arm is stored as negative in the scissor inputs and positive in Phase 9 geometry.",
        "x/c values use the wing reference-chord leading-edge datum from Phase 8.",
        "The controllability line is evaluated with the usable canard CL limit supplied by the workflow.",
        "CL_Ah is estimated as total trim CL minus the usable canard lift contribution at the current area ratio.",
        "For the canard topology, stability normally sets an upper area bound for a fixed CG.",
    ]
    warnings.extend([
        "Phase 10 is a preliminary scissor check; replace first-cut derivatives and moment estimates with aircraft-level data.",
        "Forward CG limit assumes the canard can reach the selected usable CL with no incidence, elevator, or propwash correction.",
    ])
    if Cm_ac_source == "assumed_zero":
        warnings.append(
            "Cm_ac is assumed zero because no aircraft-minus-canard moment estimate was supplied."
        )
    if not feasible:
        warnings.append(
            "The preliminary forward and aft CG limits do not overlap; revise canard area, moment arm, or trim assumptions."
        )
    if x_cg_aft / c_bar_w < -0.50 or x_cg_aft / c_bar_w > 0.75:
        warnings.append(
            "Aft CG limit lies far from the wing reference chord; check the longitudinal datum and canard layout."
        )
    if CL_c_required_at_aft > 0.7 * CL_c_max:
        warnings.append(
            "Canard lift required at the aft CG limit uses much of the available canard CL margin."
        )

    result = {
        "x_np": float(x_np_over_c * c_bar_w),
        "x_np_over_c": float(x_np_over_c),
        "x_np_forward_of_wing_ac": float(x_ac_w - x_np_over_c * c_bar_w),
        "x_np_forward_fraction_l_c": float((x_ac_w - x_np_over_c * c_bar_w) / l_c),
        "x_cg_fwd": float(x_cg_fwd),
        "x_cg_fwd_over_c": float(x_ctrl_over_c),
        "x_cg_aft": float(x_cg_aft),
        "x_cg_aft_over_c": float(x_stab_over_c),
        "CG_range_m": float(CG_range),
        "CG_range_pct": float(100.0 * CG_range / c_bar_w),
        "feasible_preliminary_CG_range": bool(feasible),
        "SM_min": float(SM_min),
        "V_bar_c": float(V_bar_c),
        "area_ratio": float(area_ratio),
        "x_ac_w": float(x_ac_w),
        "x_ac_w_over_c": float(x_ac_over_c),
        "x_ac_c": float(x_ac_w - l_c),
        "CL_trim": float(CL_trim),
        "CL_Ah": float(CL_Ah),
        "CL_Ah_source": CL_Ah_source,
        "CL_c_max": float(CL_c_max),
        "CL_c_usable": float(CL_c_max),
        "CL_h": float(CL_c_max),
        "CL_c_required_at_aft_cg": float(CL_c_required_at_aft),
        "CL_c_required_at_neutral_point": float(CL_c_required_at_neutral),
        "canard_lift_fraction_max": float(CL_c_max * area_ratio / CL_trim),
        "a_w_eff": float(a_w_eff),
        "a_c_eff": float(a_c_eff),
        "eps_alpha_c": float(eps_alpha_c),
        "eps_alpha_w": float(eps_alpha_w),
        "Cm_ac": float(Cm_ac),
        "Cm_ac_source": Cm_ac_source,
        "VhV2": float(VhV2),
        "scissor_inputs": asdict(p),
        "scissor_stability_slope": float(_stability_slope(p)),
        "scissor_controllability_slope": float(_controllability_slope(p)),
        "scissor_line_crossing_x_over_c": None if cross is None else cross[0],
        "scissor_line_crossing_area_ratio": None if cross is None else cross[1],
        "scissor_band_open_at_current_area": bool(feasible),
        "fixed_cg_scissor_sizing": fixed_cg_sizing,
        "notes": notes,
        "warnings": warnings,
    }
    if fixed_cg_sizing is not None:
        result.update({
            "x_cg_fixed_over_c": float(x_cg_fixed),
            "ShS_min_for_fixed_cg": fixed_cg_sizing["ShS_min"],
            "ShS_max_for_fixed_cg": fixed_cg_sizing["ShS_max"],
            "fixed_cg_area_window_feasible": fixed_cg_sizing["feasible_area_window"],
            "current_area_ratio_feasible_for_fixed_cg": fixed_cg_sizing["current_area_ratio_feasible"],
            "fixed_cg_lower_bound_governed_by": fixed_cg_sizing["lower_bound_governed_by"],
            "fixed_cg_upper_bound_governed_by": fixed_cg_sizing["upper_bound_governed_by"],
        })
    return result


def _inputs_from_phase10_result(phase10: Dict) -> CanardScissorInputs:
    """Rebuild scissor inputs from a Phase 10 result dictionary."""
    inputs = dict(phase10["scissor_inputs"])
    return CanardScissorInputs(**inputs)


def plot_phase10_scissor(phase10: Dict, out_path,
                         ShS_max: Optional[float] = None,
                         sample_count: int = 500) -> str:
    """Create a canard scissor plot from a Phase 10 result dictionary."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    p = _inputs_from_phase10_result(phase10)
    current_area_ratio = float(phase10["area_ratio"])
    crossing = line_crossing(p)
    if ShS_max is None:
        candidates = [
            0.60,
            1.20 * current_area_ratio,
        ]
        if crossing is not None and np.isfinite(crossing[1]):
            candidates.append(1.25 * crossing[1])
        fixed = phase10.get("fixed_cg_scissor_sizing") or {}
        if fixed.get("ShS_max") is not None:
            candidates.append(1.15 * fixed["ShS_max"])
        ShS_max = max(candidates)
    if ShS_max <= 0.0:
        raise ValueError("ShS_max must be positive.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    ShS = np.linspace(0.0, ShS_max, sample_count)
    x_stab = stability_x_cg(ShS, p)
    x_ctrl = controllability_x_cg(ShS, p)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(x_stab, ShS, color="#c0392b", lw=2.0, label="Stability")
    ax.plot(x_ctrl, ShS, color="#2c3e50", lw=2.0, label="Controllability")
    ax.fill_betweenx(
        ShS,
        x_ctrl,
        x_stab,
        where=(x_stab >= x_ctrl),
        color="#2e8b57",
        alpha=0.14,
        label="Feasible CG band",
    )

    if crossing is not None and 0.0 <= crossing[1] <= ShS_max:
        ax.plot(crossing[0], crossing[1], "o", color="#7d3c98", ms=6)
        ax.annotate(
            f"band opens at S_c/S={crossing[1]:.3f}",
            xy=crossing,
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color="#7d3c98",
            arrowprops={"arrowstyle": "->", "color": "#7d3c98", "lw": 0.9},
        )

    if p.x_cg_fixed is not None:
        ax.axvline(p.x_cg_fixed, color="#1f77b4", ls="-.", lw=1.6)
        ax.text(
            p.x_cg_fixed + 0.008,
            ShS_max * 0.96,
            f"fixed CG\nx/c={p.x_cg_fixed:.3f}",
            color="#1f77b4",
            fontsize=8,
            va="top",
        )

    ax.axhline(
        current_area_ratio,
        color="#d35400",
        ls="--",
        lw=1.5,
        label=f"Current S_c/S={current_area_ratio:.3f}",
    )
    ax.plot(
        [phase10["x_cg_fwd_over_c"], phase10["x_cg_aft_over_c"]],
        [current_area_ratio, current_area_ratio],
        color="#d35400",
        lw=3.0,
        alpha=0.65,
    )

    fixed = phase10.get("fixed_cg_scissor_sizing") or {}
    if fixed:
        ShS_min = fixed["ShS_min"]
        ShS_max_fixed = fixed["ShS_max"]
        if np.isfinite(ShS_min) and 0.0 <= ShS_min <= ShS_max:
            ax.axhline(
                ShS_min,
                color="#16a085",
                ls=":",
                lw=1.3,
                label=f"Fixed-CG lower bound={ShS_min:.3f}",
            )
        if ShS_max_fixed is not None and 0.0 <= ShS_max_fixed <= ShS_max:
            ax.axhline(
                ShS_max_fixed,
                color="#8e44ad",
                ls=":",
                lw=1.3,
                label=f"Fixed-CG upper bound={ShS_max_fixed:.3f}",
            )

    x_min = min(float(np.nanmin(x_ctrl)), float(np.nanmin(x_stab)), -0.1)
    x_max = max(float(np.nanmax(x_ctrl)), float(np.nanmax(x_stab)), 0.4)
    padding = max(0.05, 0.06 * (x_max - x_min))
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(0.0, ShS_max)
    ax.set_xlabel("x_cg / c_bar from wing reference-chord leading edge")
    ax.set_ylabel("S_c / S_w")
    ax.set_title("Canard scissor plot - lecture equations")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)
