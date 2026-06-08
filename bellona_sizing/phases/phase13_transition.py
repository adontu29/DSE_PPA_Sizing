"""Phase 13: transition blending and transition energy estimate."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def phase13_transition_blending(V_stall: float,
                                V_blend_start_frac: float = 0.5,
                                V_blend_end_frac: float = 1.2,
                                V_entry: float = 0.0,
                                V_exit: Optional[float] = None,
                                transition_accel_m_s2: float = 1.0,
                                V_cruise: Optional[float] = None,
                                cruise_speed_margin_frac: float = 0.0,
                                P_hover_W: Optional[float] = None,
                                P_fw_W: Optional[float] = None,
                                sample_count: int = 9) -> Dict:
    """Speed-based hover-to-fixed-wing control blending schedule."""
    if V_stall <= 0.0:
        raise ValueError("V_stall must be positive.")
    if V_blend_start_frac <= 0.0:
        raise ValueError("V_blend_start_frac must be positive.")
    if V_blend_end_frac <= V_blend_start_frac:
        raise ValueError("V_blend_end_frac must be larger than V_blend_start_frac.")
    if V_entry < 0.0:
        raise ValueError("V_entry must be non-negative.")
    if transition_accel_m_s2 <= 0.0:
        raise ValueError("transition_accel_m_s2 must be positive.")
    if cruise_speed_margin_frac < 0.0:
        raise ValueError("cruise_speed_margin_frac must be non-negative.")
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3.")

    V_blend_start = V_blend_start_frac * V_stall
    V_blend_end = V_blend_end_frac * V_stall
    if V_exit is None:
        V_exit_used = V_blend_end
        V_exit_source = "blend_end"
    else:
        if V_exit <= V_entry:
            raise ValueError("V_exit must be greater than V_entry.")
        V_exit_used = float(V_exit)
        V_exit_source = "user_supplied"
    if V_exit_used <= V_entry:
        raise ValueError("V_exit must be greater than V_entry.")

    def _alpha_fw(speed: float) -> float:
        xi = (speed - V_blend_start) / (V_blend_end - V_blend_start)
        xi = float(np.clip(xi, 0.0, 1.0))
        return float(3.0 * xi**2 - 2.0 * xi**3)

    def _sample(speed: float) -> Dict:
        xi = (speed - V_blend_start) / (V_blend_end - V_blend_start)
        xi_clipped = float(np.clip(xi, 0.0, 1.0))
        alpha_fw = float(3.0 * xi_clipped**2 - 2.0 * xi_clipped**3)
        if speed <= V_blend_start:
            mode = "hover"
        elif speed >= V_blend_end:
            mode = "fixed_wing"
        else:
            mode = "blend"
        return {
            "V_m_s": float(speed),
            "xi": float(xi_clipped),
            "alpha_hover": float(1.0 - alpha_fw),
            "alpha_fw": alpha_fw,
            "mode": mode,
        }

    V_samples = np.linspace(V_entry, V_exit_used, sample_count)
    schedule = [_sample(speed) for speed in V_samples]

    preblend_delta_V = max(0.0, min(V_blend_start, V_exit_used) - V_entry)
    blend_delta_V = max(
        0.0,
        min(V_blend_end, V_exit_used) - max(V_blend_start, V_entry),
    )
    postblend_delta_V = max(0.0, V_exit_used - max(V_blend_end, V_entry))

    t_preblend = preblend_delta_V / transition_accel_m_s2
    t_blend = blend_delta_V / transition_accel_m_s2
    t_postblend = postblend_delta_V / transition_accel_m_s2
    t_transition = (V_exit_used - V_entry) / transition_accel_m_s2
    distance_transition = (
        V_exit_used**2 - V_entry**2
    ) / (2.0 * transition_accel_m_s2)

    alpha_fw_at_cruise = None
    alpha_hover_at_cruise = None
    V_cruise_required_for_margin = None
    cruise_speed_margin_over_blend_end = None
    if V_cruise is not None:
        if V_cruise <= 0.0:
            raise ValueError("V_cruise must be positive when provided.")
        alpha_fw_at_cruise = _alpha_fw(V_cruise)
        alpha_hover_at_cruise = 1.0 - alpha_fw_at_cruise
        V_cruise_required_for_margin = (
            (1.0 + cruise_speed_margin_frac) * V_blend_end
        )
        cruise_speed_margin_over_blend_end = V_cruise / V_blend_end - 1.0

    E_transition_Wh = None
    P_transition_average_W = None
    if P_hover_W is not None or P_fw_W is not None:
        if P_hover_W is None or P_fw_W is None:
            raise ValueError("P_hover_W and P_fw_W must be provided together.")
        if P_hover_W < 0.0 or P_fw_W < 0.0:
            raise ValueError("Transition powers must be non-negative.")
        energy_speeds = np.linspace(V_entry, V_exit_used, max(80, sample_count * 10))
        alpha_values = np.asarray([_alpha_fw(speed) for speed in energy_speeds])
        powers = (1.0 - alpha_values) * P_hover_W + alpha_values * P_fw_W
        E_transition_Wh = float(np.trapezoid(powers, energy_speeds) / transition_accel_m_s2 / 3600.0)
        P_transition_average_W = float(E_transition_Wh * 3600.0 / t_transition)

    warnings = [
        "Phase 13 is a simplified speed-based mixer; it does not model tail-sitter attitude dynamics, angle of attack, or propeller-wing interaction during transition.",
        "Transition time assumes constant acceleration in airspeed.",
    ]
    if P_hover_W is not None:
        warnings.append(
            "Transition energy is a first-cut interpolation between hover and fixed-wing power."
        )
    if V_cruise is not None and V_cruise < V_blend_end:
        warnings.append(
            "Cruise speed is below the blend-end speed; the nominal mission cruise point remains partly in hover-control blending."
        )
    elif (
        V_cruise is not None
        and V_cruise_required_for_margin is not None
        and V_cruise < V_cruise_required_for_margin
    ):
        warnings.append(
            "Cruise speed clears the blend-end speed but does not meet the selected transition margin."
        )
    if V_entry < V_blend_start and V_exit_used < V_blend_end:
        warnings.append(
            "The requested transition speed range ends before full fixed-wing control authority is scheduled."
        )
    if V_blend_start < 0.5 * V_stall:
        warnings.append(
            "Blend start is below 0.5 V_stall; verify that aerodynamic control surfaces have useful authority."
        )

    return {
        "V_stall": float(V_stall),
        "V_blend_start": float(V_blend_start),
        "V_blend_end": float(V_blend_end),
        "V_blend_start_frac": float(V_blend_start_frac),
        "V_blend_end_frac": float(V_blend_end_frac),
        "V_entry": float(V_entry),
        "V_exit": float(V_exit_used),
        "V_exit_source": V_exit_source,
        "transition_accel_m_s2": float(transition_accel_m_s2),
        "t_transition": float(t_transition),
        "t_preblend": float(t_preblend),
        "t_blend": float(t_blend),
        "t_postblend": float(t_postblend),
        "distance_transition_m": float(distance_transition),
        "E_transition_estimate_Wh": E_transition_Wh,
        "P_transition_average_W": P_transition_average_W,
        "P_hover_W": None if P_hover_W is None else float(P_hover_W),
        "P_fw_W": None if P_fw_W is None else float(P_fw_W),
        "V_cruise": None if V_cruise is None else float(V_cruise),
        "cruise_speed_margin_frac": float(cruise_speed_margin_frac),
        "V_cruise_required_for_margin": (
            None
            if V_cruise_required_for_margin is None
            else float(V_cruise_required_for_margin)
        ),
        "cruise_speed_margin_over_blend_end": (
            None
            if cruise_speed_margin_over_blend_end is None
            else float(cruise_speed_margin_over_blend_end)
        ),
        "alpha_fw_at_cruise": alpha_fw_at_cruise,
        "alpha_hover_at_cruise": alpha_hover_at_cruise,
        "alpha_blend_fn": "smoothstep: alpha_fw = 3*xi^2 - 2*xi^3, xi = clip((V - V_start)/(V_end - V_start), 0, 1)",
        "schedule_samples": schedule,
        "notes": [
            "alpha_hover multiplies hover/differential-thrust control commands.",
            "alpha_fw multiplies fixed-wing/elevon control commands.",
            "The schedule is speed based so it can be inspected without a full transition simulation.",
        ],
        "warnings": warnings,
    }
