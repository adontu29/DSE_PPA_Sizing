"""Phase 5: mission energy and battery mass."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def phase5_energy_battery(segments: Dict[str, Tuple[float, float]],
                          eta_batt: float = 0.95, f_usable: float = 0.85,
                          e_batt_Wh_kg: float = 200.0) -> Dict:
    """Mission energy timeline, battery mass.
    In:  segments={'name': (P[W], dt[s])}  VTOL climb, transition, FW climb,
         hover, tow, return cruise, descent, reserve.  (Ph2, Ph3, Ph13)
    Out: {'E_total_Wh', 'm_batt_kg', 'P_motor_cont', 'P_motor_peak'}
    Eq:  E = sum(Pdt)/(eta_battf_usable)   Gundlach 2014 Eq. 8.4
    Loop: MTOW outer."""
    if not segments:
        raise ValueError("segments must contain at least one mission segment.")
    if not 0.0 < eta_batt <= 1.0:
        raise ValueError("eta_batt should be in the range 0 < eta_batt <= 1.")
    if not 0.0 < f_usable <= 1.0:
        raise ValueError("f_usable should be in the range 0 < f_usable <= 1.")
    if e_batt_Wh_kg <= 0.0:
        raise ValueError("e_batt_Wh_kg must be positive.")

    segment_breakdown = {}
    E_load_Wh = 0.0
    P_values = []
    total_time_s = 0.0

    for name, values in segments.items():
        if len(values) != 2:
            raise ValueError(f"Segment '{name}' must be a (P_W, dt_s) pair.")
        P_W, dt_s = values
        P_W = float(P_W)
        dt_s = float(dt_s)
        if P_W < 0.0:
            raise ValueError(f"Segment '{name}' has negative power; regeneration is not modeled.")
        if dt_s < 0.0:
            raise ValueError(f"Segment '{name}' has negative duration.")

        E_segment_Wh = P_W * dt_s / 3600.0
        E_load_Wh += E_segment_Wh
        total_time_s += dt_s
        P_values.append(P_W)
        segment_breakdown[name] = {
            "P_W": P_W,
            "dt_s": dt_s,
            "E_Wh": float(E_segment_Wh),
        }

    E_total_Wh = E_load_Wh / (eta_batt * f_usable)
    m_batt_kg = E_total_Wh / e_batt_Wh_kg
    P_peak = max(P_values)
    P_average_load = 0.0
    if total_time_s > 0.0:
        P_average_load = E_load_Wh * 3600.0 / total_time_s

    warnings = []
    segment_names = " ".join(str(name).lower() for name in segments)
    if "reserve" not in segment_names:
        warnings.append(
            "No explicit reserve segment was found; only usable-capacity margin is applied."
        )
    if e_batt_Wh_kg > 265.0:
        warnings.append(
            "Battery specific energy is optimistic for high-power UAV packs; verify against pack-level datasheets."
        )

    return {
        "segments": segment_breakdown,
        "E_load_Wh": float(E_load_Wh),
        "E_total_Wh": float(E_total_Wh),
        "E_total_kWh": float(E_total_Wh / 1000.0),
        "m_batt_kg": float(m_batt_kg),
        "P_motor_cont": float(P_peak),
        "P_motor_peak": float(P_peak),
        "P_average_load": float(P_average_load),
        "eta_batt": float(eta_batt),
        "f_usable": float(f_usable),
        "reserve_fraction": float(1.0 - f_usable),
        "e_batt_Wh_kg": float(e_batt_Wh_kg),
        "notes": [
            "Segment powers are total aircraft electrical powers in W.",
            "E_load_Wh is the mission energy before battery-discharge efficiency and usable-capacity margin.",
            "E_total_Wh is the installed pack energy required by this first-cut model.",
            "Continuous and peak motor powers are both set to the maximum segment power until short-duration transient limits are modeled.",
        ],
        "warnings": warnings,
    }
