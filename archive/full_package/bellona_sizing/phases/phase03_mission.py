"""Phase 3: mission-envelope optimization via a target-distance sweep.

The sweep identifies separate energy-, power-, and time-critical cases so that
downstream phases are not inadvertently sized by whichever distance happened to
win the latest single-profile optimization.

A reference level-flight condition is computed independently of the optimized
climb schedule, so that CL_cruise and CD_cruise are genuine trim values rather
than final-climb-step values.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..common import isa
from ..mission_profile import (
    AircraftPerformance,
    MissionPowerAssumptions,
    MissionProfileConfig,
    minimum_time_reference_climb,
    optimize_constant_eas_mission,
    optimize_multistage_mission,
    simplified_mission_reference,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reference_level_flight(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        altitude_m: float,
        TAS_m_s: float,
        isa_fn: Callable = isa) -> Dict:
    """Compute steady level-flight aerodynamic state at altitude/TAS.

    Returns CL, CD, q, Mach, and required electrical power for level trim.
    """
    rho, mu, a_sound, _ = isa_fn(altitude_m)
    q_Pa = 0.5 * rho * TAS_m_s**2
    CL = aircraft.weight_N / (q_Pa * aircraft.wing_area_m2)
    K = aircraft.induced_drag_factor
    CD = aircraft.CD0 + K * CL**2
    drag_N = q_Pa * aircraft.wing_area_m2 * CD
    EAS_m_s = TAS_m_s * np.sqrt(rho / 1.225)
    Mach = TAS_m_s / a_sound
    P_elec_W = drag_N * TAS_m_s / power.total_forward_efficiency
    return {
        "altitude_m": float(altitude_m),
        "TAS_m_s": float(TAS_m_s),
        "EAS_m_s": float(EAS_m_s),
        "CL": float(CL),
        "CD": float(CD),
        "dynamic_pressure_Pa": float(q_Pa),
        "Mach": float(Mach),
        "required_thrust_N": float(drag_N),
        "electrical_power_W": float(P_elec_W),
        "density_kg_m3": float(rho),
        "dynamic_viscosity_Pa_s": float(mu),
        "speed_of_sound_m_s": float(a_sound),
    }


def _extract_peak_wingborne_state(mission_result: Dict) -> Dict:
    """Return the wing-borne state carrying maximum electrical power."""
    states = mission_result.get("states", [])
    if not states:
        return {}
    peak = max(states, key=lambda s: s.get("electrical_power_W", 0.0))
    return {
        "altitude_m": float(peak.get("altitude_m", peak.get("altitude_start_m", 0.0))),
        "TAS_m_s": float(peak.get("speed_m_s", 0.0)),
        "EAS_m_s": float(peak.get("EAS_m_s", 0.0)),
        "climb_angle_deg": float(peak.get("climb_angle_deg", 0.0)),
        "required_thrust_N": float(peak.get("required_thrust_N", 0.0)),
        "electrical_power_W": float(peak.get("electrical_power_W", 0.0)),
        "CL": float(peak.get("CL", 0.0)),
        "CD": float(peak.get("CD", 0.0)),
        "segment": str(peak.get("segment", "wing_borne_climb")),
    }


def _max_wingborne_power_W(mission_result: Dict) -> float:
    """Return the maximum electrical power across all wing-borne states."""
    states = mission_result.get("states", [])
    return float(max((s.get("electrical_power_W", 0.0) for s in states), default=0.0))


def _run_for_distance(
        aircraft: AircraftPerformance,
        power: MissionPowerAssumptions,
        config_template: MissionProfileConfig,
        target_distance_m: float,
        eas_grid_m_s, climb_angle_grid_rad, cruise_tas_grid_m_s,
        isa_fn: Callable) -> Dict:
    """Run `optimize_constant_eas_mission` for one horizontal range target."""
    cfg = _dc_replace(config_template, horizontal_range_m=float(target_distance_m))
    return optimize_constant_eas_mission(
        aircraft, power, cfg,
        eas_grid_m_s=eas_grid_m_s,
        climb_angle_grid_rad=climb_angle_grid_rad,
        cruise_tas_grid_m_s=cruise_tas_grid_m_s,
        isa_fn=isa_fn,
    )


# ---------------------------------------------------------------------------
# Public phase function
# ---------------------------------------------------------------------------

def phase3_mission_optimise(
        MTOW_N: float,
        S_guess: float,
        CD0: float,
        AR: float,
        e: float,
        h_target: float,
        range_target: float,
        t_budget: float,
        t_hover: float,
        isa_fn: Callable = isa,
        eta_prop: float = 0.75,
        eta_motor: float = 0.90,
        eta_ESC: float = 0.95,
        max_affordable_electrical_power_W: float = 20000.0,
        preliminary_hover_power_W: float = 14000.0,
        preliminary_transition_power_W: float = 8000.0,
        V_min_required: Optional[float] = None,
        V_grid=None,
        h_transition_grid=None,
        CL_max: float = 1.30,
        CL_limit_fraction: float = 0.90,
        CL_allowed_override: Optional[float] = None,
        altitude_step_m: float = 100.0,
        vertical_takeoff_height_m: float = 20.0,
        vertical_takeoff_rate_m_s: float = 2.0,
        transition_accel_m_s2: float = 1.0,
        transition_blend_start_fraction: float = 0.50,
        transition_blend_end_fraction: float = 1.20,
        transition_cruise_margin_fraction: float = 0.05,
        transition_sample_count: int = 9,
        max_transition_complete_speed_m_s: Optional[float] = 20.0,
        max_stall_EAS_m_s: Optional[float] = None,
        minimum_power_margin_fraction: float = 0.05,
        max_fixed_wing_climb_angle_deg: float = 30.0,
        allow_spiral_climb: bool = True,
        eas_grid_m_s: Optional[Sequence[float]] = None,
        climb_angle_grid_rad: Optional[Sequence[float]] = None,
        cruise_tas_grid_m_s: Optional[Sequence[float]] = None,
        run_multistage_extension: bool = False,
        run_minimum_time_reference: bool = False,
        target_distance_sweep_m: Optional[Sequence[float]] = None) -> Dict:
    """Minimize mission energy over a target-distance sweep and build a
    structured mission envelope with separate critical cases.

    The sweep always includes the overhead case (0 m) and the design-range
    case (range_target).  Additional distances can be requested via
    target_distance_sweep_m.

    Returns a dict with two top-level sections:

    carry_forward
        Structured data for downstream phases.  This is the authoritative
        interface between Phase 3 and later sizing phases.

    diagnostics
        Full mission profiles, sweep summaries, and comparison references.
        Later phases must NOT consume data from this section for sizing.

    All legacy flat keys are also returned at the top level for backward
    compatibility.  Where legacy values were incorrect (CL_cruise from a
    climb state, ROC/gamma from one optimized profile) they are now replaced
    by the correct carry_forward values.
    """
    positive = {
        "MTOW_N": MTOW_N, "S_guess": S_guess, "CD0": CD0, "AR": AR,
        "e": e, "h_target": h_target, "t_budget": t_budget,
        "CL_max": CL_max, "altitude_step_m": altitude_step_m,
    }
    if any(v <= 0.0 for v in positive.values()):
        raise ValueError("Phase 3 weight, geometry, mission, and step inputs must be positive.")
    if range_target < 0.0:
        raise ValueError("range_target must be non-negative.")
    if t_hover < 0.0:
        raise ValueError("t_hover must be non-negative.")
    if V_min_required is not None and V_min_required <= 0.0:
        raise ValueError("V_min_required must be positive when provided.")
    if cruise_tas_grid_m_s is None and V_grid is not None:
        cruise_tas_grid_m_s = np.asarray(V_grid, dtype=float)

    aircraft = AircraftPerformance(
        mass_kg=MTOW_N / 9.80665,
        wing_area_m2=S_guess,
        CD0=CD0,
        aspect_ratio=AR,
        oswald_efficiency=e,
        CL_max=CL_max,
        CL_limit_fraction=CL_limit_fraction,
        CL_allowed_override=CL_allowed_override,
    )
    power = MissionPowerAssumptions(
        propulsive_efficiency=eta_prop,
        motor_efficiency=eta_motor,
        esc_efficiency=eta_ESC,
        max_affordable_electrical_power_W=max_affordable_electrical_power_W,
        preliminary_hover_power_W=preliminary_hover_power_W,
        preliminary_transition_power_W=preliminary_transition_power_W,
    )
    # Use design range for the base config; individual sweep runs override range.
    config = MissionProfileConfig(
        target_altitude_m=h_target,
        horizontal_range_m=range_target,
        outbound_time_budget_s=t_budget,
        hover_duration_s=t_hover,
        altitude_step_m=altitude_step_m,
        vertical_takeoff_height_m=vertical_takeoff_height_m,
        vertical_takeoff_rate_m_s=vertical_takeoff_rate_m_s,
        transition_accel_m_s2=transition_accel_m_s2,
        transition_blend_start_fraction=transition_blend_start_fraction,
        transition_blend_end_fraction=transition_blend_end_fraction,
        transition_cruise_margin_fraction=transition_cruise_margin_fraction,
        transition_sample_count=transition_sample_count,
        minimum_cruise_true_airspeed_m_s=V_min_required,
        max_transition_complete_speed_m_s=max_transition_complete_speed_m_s,
        max_stall_EAS_m_s=max_stall_EAS_m_s,
        minimum_power_margin_fraction=minimum_power_margin_fraction,
        max_fixed_wing_climb_angle_deg=max_fixed_wing_climb_angle_deg,
        allow_spiral_climb=allow_spiral_climb,
    )

    # -------------------------------------------------------------------
    # Build the sweep distance list.
    # Always include the overhead case (0 m) and the design range.
    # -------------------------------------------------------------------
    base_distances: List[float] = [0.0, float(range_target)]
    if target_distance_sweep_m is not None:
        for d in target_distance_sweep_m:
            base_distances.append(float(d))
    sweep_distances = sorted(set(base_distances))

    # -------------------------------------------------------------------
    # Run the optimization for every distance in the sweep.
    # -------------------------------------------------------------------
    sweep_results: Dict[float, Dict] = {}
    for d in sweep_distances:
        sweep_results[d] = _run_for_distance(
            aircraft, power, config, d,
            eas_grid_m_s, climb_angle_grid_rad, cruise_tas_grid_m_s,
            isa_fn,
        )

    feasible = {d: r for d, r in sweep_results.items() if r["feasible"]}
    if not feasible:
        # Find the most common failure across all distances.
        all_failures: Dict[str, int] = {}
        for r in sweep_results.values():
            for reason, count in r.get("failure_counts", {}).items():
                all_failures[reason] = all_failures.get(reason, 0) + count
        top = max(all_failures, key=lambda k: all_failures[k]) if all_failures else "unknown"
        raise ValueError(f"No feasible Phase 3 mission profile for any target distance: {top}")

    # -------------------------------------------------------------------
    # Identify the three critical cases.
    # -------------------------------------------------------------------
    energy_d = max(feasible, key=lambda d: feasible[d]["total_electrical_energy_Wh"])
    power_d  = max(feasible, key=lambda d: feasible[d]["peak_electrical_power_W"])
    time_d   = max(feasible, key=lambda d: feasible[d]["outbound_time_s"])

    energy_cr = feasible[energy_d]
    power_cr  = feasible[power_d]
    time_cr   = feasible[time_d]

    # Check that the design-range case is feasible (most important constraint).
    if float(range_target) not in feasible:
        failure_summary = sorted(
            sweep_results[float(range_target)].get("failure_counts", {}).items(),
            key=lambda item: item[1], reverse=True,
        )
        detail = (
            failure_summary[0][0]
            if failure_summary
            else sweep_results[float(range_target)].get("failure_reason", "unknown")
        )
        raise ValueError(
            f"No feasible Phase 3 profile at the design range ({range_target} m): {detail}"
        )

    # -------------------------------------------------------------------
    # Reference level-flight condition (independent of optimized profile).
    # Use the cruise TAS from the energy-critical case as the reference
    # design speed.  This is a fixed design condition, not "optimized cruise".
    # -------------------------------------------------------------------
    ref_TAS = float(energy_cr["cruise_true_airspeed_m_s"])
    ref_lf = _reference_level_flight(aircraft, power, h_target, ref_TAS, isa_fn)

    # -------------------------------------------------------------------
    # Transition reference (from energy-critical case — most representative).
    # -------------------------------------------------------------------
    tr_data = energy_cr["takeoff_transition"]
    tr_inner = tr_data["transition"]
    transition_ref = {
        "transition_altitude_m": float(tr_data["transition_altitude_m"]),
        "stall_speed_m_s": float(tr_data["stall_speed_m_s"]),
        "stall_EAS_m_s": float(tr_data["stall_EAS_m_s"]),
        "stall_TAS_transition_m_s": float(
            tr_data["stall_TAS_transition_m_s"]
        ),
        "transition_density_kg_m3": float(
            tr_data["transition_density_kg_m3"]
        ),
        "blend_start_speed_m_s": float(tr_inner["V_blend_start"]),
        "blend_end_speed_m_s": float(tr_inner["V_blend_end"]),
        "minimum_transition_complete_TAS_m_s": float(
            (1.0 + transition_cruise_margin_fraction)
            * tr_inner["V_blend_end"]
        ),
        "max_transition_complete_speed_m_s": (
            None
            if config.max_transition_complete_speed_m_s is None
            else float(config.max_transition_complete_speed_m_s)
        ),
        "transition_complete_speed_margin_m_s": (
            None
            if tr_data.get("transition_complete_speed_margin_m_s") is None
            else float(tr_data["transition_complete_speed_margin_m_s"])
        ),
        "max_stall_EAS_m_s": (
            None
            if config.max_stall_EAS_m_s is None
            else float(config.max_stall_EAS_m_s)
        ),
        "stall_EAS_margin_m_s": (
            None
            if tr_data.get("stall_EAS_margin_m_s") is None
            else float(tr_data["stall_EAS_margin_m_s"])
        ),
        "transition_time_s": float(tr_inner["t_transition"]),
    }

    # -------------------------------------------------------------------
    # Unit Reynolds number (rho * V / mu) at reference conditions.
    # Phase 7 multiplies by the actual chord; Phase 3 no longer bakes in
    # a guessed chord.
    # -------------------------------------------------------------------
    rho_ref = ref_lf["density_kg_m3"]
    mu_ref  = ref_lf["dynamic_viscosity_Pa_s"]
    unit_Re = rho_ref * ref_TAS / mu_ref
    # Backward-compat Re_estimate still uses c_guess for the inner loop.
    c_guess = np.sqrt(S_guess / AR)
    Re_estimate = unit_Re * c_guess

    # -------------------------------------------------------------------
    # Minimum required average ROC derived from mission geometry.
    # This is the climb-rate requirement for the constraint diagram —
    # it does not vary with target distance and is not the optimized ROC.
    # -------------------------------------------------------------------
    t_vtol = vertical_takeoff_height_m / vertical_takeoff_rate_m_s
    t_tr   = float(tr_inner["t_transition"])
    climb_altitude_m = h_target - vertical_takeoff_height_m
    t_available_climb = t_budget - t_vtol - t_tr
    if t_available_climb > 1e-9:
        ROC_min = climb_altitude_m / t_available_climb
    else:
        ROC_min = float("inf")

    # Derive a reference climb angle from ROC_min and ref_TAS.
    sin_gamma = float(np.clip(ROC_min / ref_TAS, 0.0, 1.0))
    gamma_ref_rad = float(np.arcsin(sin_gamma))

    # -------------------------------------------------------------------
    # Minimum power margin across all feasible cases.
    # -------------------------------------------------------------------
    min_power_margin = min(
        r["constraint_margins"]["power_margin_W"] for r in feasible.values()
    )
    required_power_margin_W = (
        config.minimum_power_margin_fraction
        * max_affordable_electrical_power_W
    )
    cl_margins = [
        r["constraint_margins"]["CL_margin"]
        for r in feasible.values()
        if r["constraint_margins"]["CL_margin"] is not None
    ]
    min_CL_margin = float(min(cl_margins)) if cl_margins else float("inf")

    # -------------------------------------------------------------------
    # Operating envelope across all feasible sweep distances.
    # -------------------------------------------------------------------
    all_TAS = [s["speed_m_s"]          for r in feasible.values() for s in r["states"]]
    all_CL  = [s["CL"]                 for r in feasible.values() for s in r["states"]]
    all_q   = [s["dynamic_pressure_Pa"] for r in feasible.values() for s in r["states"]]
    operating_envelope = {
        "minimum_TAS_m_s":           float(min(all_TAS)) if all_TAS else None,
        "maximum_TAS_m_s":           float(max(all_TAS)) if all_TAS else None,
        "minimum_CL":                float(min(all_CL))  if all_CL  else None,
        "maximum_CL":                float(max(all_CL))  if all_CL  else None,
        "maximum_dynamic_pressure_Pa": float(max(all_q)) if all_q   else None,
    }

    # -------------------------------------------------------------------
    # carry_forward — authoritative interface for downstream phases.
    # -------------------------------------------------------------------
    carry_forward = {
        "all_required_cases_feasible": len(feasible) == len(sweep_distances),

        "energy_sizing_case": {
            "target_distance_m":      float(energy_d),
            "total_load_energy_Wh":   float(energy_cr["total_electrical_energy_Wh"]),
            "segment_summaries":      energy_cr["segment_summaries"],
            "peak_electrical_power_W": float(energy_cr["peak_electrical_power_W"]),
            "outbound_time_s":        float(energy_cr["outbound_time_s"]),
        },

        "power_sizing_case": {
            "target_distance_m":                    float(power_d),
            "peak_electrical_power_W":              float(power_cr["peak_electrical_power_W"]),
            "maximum_wingborne_electrical_power_W": float(_max_wingborne_power_W(power_cr)),
            "critical_state": _extract_peak_wingborne_state(power_cr),
            "max_affordable_electrical_power_W":    float(max_affordable_electrical_power_W),
            "minimum_power_margin_W":               float(min_power_margin),
            "minimum_power_margin_fraction":         float(minimum_power_margin_fraction),
            "required_power_margin_W":               float(required_power_margin_W),
            "power_margin_over_required_W":          float(
                min_power_margin - required_power_margin_W
            ),
        },

        "time_sizing_case": {
            "target_distance_m": float(time_d),
            "outbound_time_s":   float(time_cr["outbound_time_s"]),
        },

        "reference_level_flight": ref_lf,
        "transition_reference":   transition_ref,

        "minimum_constraint_margins": {
            "power_margin_W": float(min_power_margin),
            "required_power_margin_W": float(required_power_margin_W),
            "power_margin_over_required_W": float(
                min_power_margin - required_power_margin_W
            ),
            "CL_margin":      float(min_CL_margin),
        },

        "minimum_required_average_ROC_m_s": float(ROC_min),
        "gamma_reference_rad":              float(gamma_ref_rad),
        "gamma_reference_deg":              float(np.rad2deg(gamma_ref_rad)),
        "unit_Reynolds_number_per_m":       float(unit_Re),
        "operating_envelope":               operating_envelope,
        "CL_allowed":                       float(aircraft.permitted_CL),
        "CL_allowed_override": (
            None
            if CL_allowed_override is None
            else float(CL_allowed_override)
        ),
        "aerodynamic_speed_limits": dict(
            energy_cr.get("grid", {}).get("aerodynamic_speed_limits", {})
        ),
    }

    # -------------------------------------------------------------------
    # Optional extensions (run on the energy-critical profile).
    # -------------------------------------------------------------------
    extensions: Dict = {}
    config_energy = _dc_replace(config, horizontal_range_m=float(energy_d))
    if run_minimum_time_reference:
        extensions["minimum_time_reference_climb"] = minimum_time_reference_climb(
            aircraft, power, config_energy,
            eas_grid_m_s=eas_grid_m_s,
            climb_angle_grid_rad=climb_angle_grid_rad,
            isa_fn=isa_fn,
        )
    if run_multistage_extension:
        extensions["multi_stage_energy_profile"] = optimize_multistage_mission(
            aircraft, power, config_energy,
            initial_result=energy_cr,
            isa_fn=isa_fn,
        )

    eta_fw = power.total_forward_efficiency
    gamma_opt = float(energy_cr["optimized_climb_angle_rad"])
    simplified_reference = simplified_mission_reference(
        aircraft, config, eta_fw,
        energy_cr["cruise_true_airspeed_m_s"],
        gamma_opt,
        isa_fn=isa_fn,
    )

    # -------------------------------------------------------------------
    # Diagnostics sweep summary (one row per distance).
    # -------------------------------------------------------------------
    distance_sweep: List[Dict] = []
    for d in sweep_distances:
        r = sweep_results[d]
        row: Dict = {"target_distance_m": float(d), "feasible": r["feasible"]}
        if r["feasible"]:
            row.update({
                "total_electrical_energy_Wh": float(r["total_electrical_energy_Wh"]),
                "peak_electrical_power_W":    float(r["peak_electrical_power_W"]),
                "outbound_time_s":            float(r["outbound_time_s"]),
                "optimized_climb_EAS_m_s":    float(r["optimized_climb_EAS_m_s"]),
                "optimized_climb_angle_deg":  float(np.rad2deg(r["optimized_climb_angle_rad"])),
                "cruise_true_airspeed_m_s":   float(r["cruise_true_airspeed_m_s"]),
                "time_margin_s": float(
                    t_budget - r["outbound_time_s"]
                ),
            })
        else:
            row["failure_reason"] = r.get("failure_reason")
        distance_sweep.append(row)

    # -------------------------------------------------------------------
    # Backward-compat flat keys.
    # Values that were previously wrong are now replaced:
    #   CL_cruise / CD_cruise  <- reference level-flight (not climb state)
    #   ROC                    <- mission-geometry minimum (not optimized)
    #   gamma                  <- arcsin(ROC_min / V_ref) (not optimized)
    #   peak_electrical_power_W <- power-critical case (not energy-critical)
    # -------------------------------------------------------------------
    ec = energy_cr
    climb  = ec["climb"]
    cruise = ec["cruise"]
    tr_bc  = ec["takeoff_transition"]["transition"]  # backward-compat transition
    climb_states  = climb["states"]
    cruise_states = cruise["states"]

    fw_time   = climb["time_s"] + cruise["time_s"]
    fw_energy = climb["energy_Wh"] + cruise["energy_Wh"]
    P_fw_avg  = fw_energy * 3600.0 / fw_time if fw_time > 1e-9 else 0.0

    K = aircraft.induced_drag_factor
    rho_avg, mu_avg, _, _ = isa_fn(
        0.5 * (vertical_takeoff_height_m + h_target)
    )
    V_power_min = np.sqrt(
        (2.0 * (MTOW_N / S_guess) / rho_avg) * np.sqrt(K / (3.0 * CD0))
    )

    n_grid = (
        len(ec.get("grid", {}).get("climb_EAS_m_s", []))
        * len(ec.get("grid", {}).get("climb_angle_deg", []))
        * len(ec.get("grid", {}).get("cruise_TAS_m_s", []))
    )
    n_fail = sum(ec.get("failure_counts", {}).values())

    warnings_list = [
        "Phase 3 uses preliminary total-aircraft hover and transition powers; update when test data are available.",
        "The affordable electrical-power cap is a design input, not a validated propulsion-component limit.",
        "CL_cruise and CD_cruise are now from a reference level-flight calculation, not the final climb state.",
        "ROC and gamma are derived from mission geometry requirements, not an optimized single-profile result.",
        f"Power-critical case is at {power_d:.0f} m; energy-critical case is at {energy_d:.0f} m.",
    ]
    if h_transition_grid is not None:
        warnings_list.append("h_transition_grid was ignored; transition altitude is fixed low.")

    return {
        # ================================================================
        # STRUCTURED OUTPUT (authoritative interface for later phases)
        # ================================================================
        "carry_forward": carry_forward,
        "diagnostics": {
            "distance_sweep":           distance_sweep,
            "energy_critical_profile":  ec,
            "power_critical_profile":   power_cr,
            "simplified_mission_reference": simplified_reference,
            "extensions":               extensions,
        },

        # ================================================================
        # BACKWARD-COMPATIBLE FLAT KEYS
        # Downstream phases that have not yet been updated to read
        # carry_forward can still find the data here.
        # ================================================================
        "feasible":       True,
        "failure_reason": None,

        # Speed — reference_level_flight_TAS_m_s is the preferred name.
        "V_cruise":                      float(ref_TAS),
        "reference_level_flight_TAS_m_s": float(ref_TAS),
        "optimized_climb_EAS_m_s":       float(ec["optimized_climb_EAS_m_s"]),
        "optimized_climb_angle_deg":     float(
            np.rad2deg(ec["optimized_climb_angle_rad"])
        ),
        "max_fixed_wing_climb_angle_deg": float(
            max_fixed_wing_climb_angle_deg
        ),
        "TAS_schedule_m_s": [float(s["speed_m_s"]) for s in climb_states],
        "EAS_schedule_m_s": [float(s["EAS_m_s"])   for s in climb_states],
        "climb_angle_schedule_deg": [
            float(s["climb_angle_deg"]) for s in climb_states
        ],
        "rate_of_climb_schedule_m_s": [
            float(s["rate_of_climb_m_s"]) for s in climb_states
        ],
        "V_min_required": float(
            V_min_required
            if V_min_required is not None
            else (1.0 + transition_cruise_margin_fraction) * tr_bc["V_blend_end"]
        ),
        "V_min_requirement_active": True,

        # Climb angle and rate — now from mission requirements, not single profile.
        "gamma":     float(gamma_ref_rad),
        "gamma_deg": float(np.rad2deg(gamma_ref_rad)),
        "ROC":       float(ROC_min),

        # Transition geometry.
        "h_tr":                       float(tr_data["transition_altitude_m"]),
        "t_transition":               float(tr_inner["t_transition"]),
        "transition_stall_speed_m_s": float(tr_data["stall_speed_m_s"]),
        "transition_blend_start_m_s": float(tr_inner["V_blend_start"]),
        "transition_blend_end_m_s":   float(tr_inner["V_blend_end"]),
        "transition_complete_speed_m_s": float(
            tr_data["transition_complete_speed_m_s"]
        ),
        "transition_complete_speed_margin_m_s": (
            None
            if tr_data.get("transition_complete_speed_margin_m_s") is None
            else float(tr_data["transition_complete_speed_margin_m_s"])
        ),

        # Reynolds number — unit_Reynolds_number_per_m is the preferred form.
        "Re_estimate":                  float(Re_estimate),
        "unit_Reynolds_number_per_m":   float(unit_Re),
        "V_power_min":                  float(V_power_min),

        # Time — from energy-critical case.
        "t_fw":              float(fw_time),
        "t_climb":           float(climb["time_s"]),
        "t_cruise":          float(cruise["time_s"]),
        "t_total_outbound":  float(ec["outbound_time_s"]),
        "time_margin":       float(t_budget - ec["outbound_time_s"]),
        "total_mission_time_s": float(ec["total_mission_time_s"]),

        # Energy — from energy-critical case (correct for battery sizing).
        "E_total_Wh":      float(ec["total_electrical_energy_Wh"]),
        "E_transition_Wh": float(tr_data["transition_energy_Wh"]),
        "E_fw_Wh":         float(fw_energy),
        "E_climb_Wh":      float(climb["energy_Wh"]),
        "E_cruise_Wh":     float(cruise["energy_Wh"]),
        "E_hover_Wh":      float(ec["segment_summaries"]["mission_hover"]["energy_Wh"]),
        "P_fw":            float(P_fw_avg),
        "P_transition":    float(preliminary_transition_power_W),
        "P_hover_target":  float(preliminary_hover_power_W),

        # Power — from power-critical case (correct for power-system sizing).
        "peak_electrical_power_W":       float(power_cr["peak_electrical_power_W"]),
        "peak_shaft_power_W":            float(power_cr["peak_shaft_power_W"]),
        "max_affordable_electrical_power_W": float(max_affordable_electrical_power_W),
        "power_margin_W":                float(min_power_margin),
        "minimum_power_margin_fraction":  float(minimum_power_margin_fraction),
        "required_power_margin_W":        float(required_power_margin_W),
        "power_margin_over_required_W":   float(
            min_power_margin - required_power_margin_W
        ),

        # Aerodynamics — from reference level-flight (NOT climb state).
        "CL_cruise": float(ref_lf["CL"]),
        "CD_cruise": float(ref_lf["CD"]),
        "drag_N":    float(ref_lf["required_thrust_N"]),

        # Mission geometry — from energy-critical case.
        "path_fw_m": float(
            sum(s["delta_s_m"] for s in climb_states) + cruise["distance_m"]
        ),
        "climb_horizontal_distance_m": float(climb["distance_m"]),
        "level_cruise_distance_m":     float(cruise["distance_m"]),
        "total_ground_track_distance_m": float(ec["total_ground_track_distance_m"]),
        "spiral_excess_ground_track_distance_m": float(
            ec["spiral_excess_ground_track_distance_m"]
        ),
        "mission_ROC_reference": float(h_target / t_budget),

        # Efficiency and assumptions.
        "eta_fw":           float(eta_fw),
        "power_assumptions": ec["power_assumptions"],
        "feasible_candidates": int(n_grid - n_fail),

        # States and summaries — from energy-critical case.
        "states":             ec["states"],
        "segment_summaries":  ec["segment_summaries"],
        "constraint_margins": ec["constraint_margins"],
        "optimization_grid":  ec.get("grid", {}),
        "optimization_failure_counts": ec.get("failure_counts", {}),

        "simplified_mission_reference": simplified_reference,
        "extensions":   extensions,
        "notes": ec.get("assumptions", []) + [
            "CL_cruise and CD_cruise are from a steady level-flight reference, not the final climb state.",
            "ROC and gamma are mission-requirement quantities used for the constraint diagram.",
            "peak_electrical_power_W is from the power-critical sweep case, not the energy-critical case.",
        ],
        "warnings": warnings_list,
    }
