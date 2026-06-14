"""Canard sizing: the scissor plot and the wing-position solve.

The scissor plot gives, for each canard area ratio Sc/Sw, the forward
(controllability) and aft (static-stability) CG limits over the wing chord. The
wing longitudinal station is solved jointly with the band (the band moves with
the canard->wing arm), and the canard area ratio is swept to find the lightest
feasible design. Equations and sign conventions live in scissor_plot.py.
"""

from __future__ import annotations

import math

from mission_energy_course import permitted_lift_coefficient
from scissor_plot import (
    mach_number,
    datcom_lift_slope,
    aircraft_less_canard_lift_slope,
    aerodynamic_centre_over_mac,
    zero_lift_pitching_moment,
    wing_downwash_gradient,
    scissor_cg_limits,
)

from sizing.inputs import AIRCRAFT, MASS, MISSION
from sizing.geometry import canard_geometry, longitudinal_layout
from sizing.mass import mass_and_cg
from sizing.transition import stall_speed_limit


def scissor_control_lift_condition(mission=None):
    """Return the C_L_A-h value used for the controllability scissor line."""
    permitted_cl = permitted_lift_coefficient(AIRCRAFT)
    source = AIRCRAFT.get("scissor_control_CL_Ah_source", "mission_max")

    if source == "permitted_limit" or mission is None:
        return {
            "CL_Ah_control": permitted_cl,
            "source": "permitted_limit",
            "mission_max_CL": None,
            "margin_factor": None,
            "permitted_CL": permitted_cl,
        }
    if source != "mission_max":
        raise ValueError(
            "scissor_control_CL_Ah_source must be 'mission_max' or 'permitted_limit'"
        )

    mission_cls = [
        state["CL"]
        for state in mission.get("states", [])
        if state.get("segment") == "wing_borne_climb" and state.get("CL") is not None
    ]
    if mission.get("CL_cruise") is not None:
        mission_cls.append(mission["CL_cruise"])

    if not mission_cls:
        return {
            "CL_Ah_control": permitted_cl,
            "source": "permitted_limit",
            "mission_max_CL": None,
            "margin_factor": None,
            "permitted_CL": permitted_cl,
        }

    margin_factor = AIRCRAFT.get("scissor_control_CL_Ah_margin_factor", 1.0)
    mission_max_cl = max(mission_cls)
    control_cl = min(permitted_cl, margin_factor * mission_max_cl)
    return {
        "CL_Ah_control": control_cl,
        "source": "mission_max",
        "mission_max_CL": mission_max_cl,
        "margin_factor": margin_factor,
        "permitted_CL": permitted_cl,
    }


def scissor_coefficients(wing, canard, fuselage_length, lh_over_mac, mission=None):
    """Course-method aerodynamic inputs for the canard scissor plot.

    `fuselage_length` (= L_fus) and `lh_over_mac` (canard->wing arm / c_bar, < 0)
    come from the longitudinal layout and change as the wing moves, so this is
    re-evaluated per candidate wing station. `canard` carries the canard span at
    the area ratio being evaluated, needed for the canard-on-wing downwash.
    `mission`, when supplied, sets the controllability CL from the actual
    wing-borne mission envelope.
    """
    cruise_speed = wing.get("cruise_true_speed_m_s", AIRCRAFT["cruise_true_speed_m_s"])
    mach = mach_number(MISSION["altitude_m"], cruise_speed)
    wing_eta = AIRCRAFT.get("wing_datcom_eta", AIRCRAFT["datcom_eta"])
    canard_eta = AIRCRAFT.get("canard_datcom_eta", AIRCRAFT["datcom_eta"])
    fuselage_width = MASS["fuselage_width_m"]
    mac = wing["chord_m"]                          # geometric mean chord used as c_bar
    mean_geo_chord = wing["area_m2"] / wing["span_m"]

    cl_alpha_wing = datcom_lift_slope(
        AIRCRAFT["wing_aspect_ratio"], mach,
        math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]), wing_eta,
    )
    cl_alpha_canard = datcom_lift_slope(           # (Vh/V) = 1 for a canard
        AIRCRAFT["canard_aspect_ratio"], mach,
        math.radians(AIRCRAFT["canard_sweep_half_chord_deg"]), canard_eta,
    )
    cl_alpha_A_h = aircraft_less_canard_lift_slope(
        cl_alpha_wing, fuselage_width, wing["span_m"],
        wing["area_m2"], wing["root_chord_m"],
    )

    # l_fn = fuselage length ahead of the wing (nose to wing root LE). L_fus also
    # includes the aft body behind the wing root TE (wing_te_to_tail), so subtract
    # both the root chord and that aft-body margin, not the root chord alone.
    nose_length = (
        fuselage_length - wing["root_chord_m"] - MASS["wing_te_to_tail_m"]
    )
    x_ac_over_mac = aerodynamic_centre_over_mac(
        cl_alpha_A_h, fuselage_width, nose_length, wing["area_m2"], mac,
        wing["span_m"], AIRCRAFT["wing_taper"], mean_geo_chord,
        math.radians(AIRCRAFT["wing_sweep_quarter_chord_deg"]),
    )
    cmac = zero_lift_pitching_moment(
        AIRCRAFT["wing_airfoil_cm0"], AIRCRAFT["wing_aspect_ratio"],
        math.radians(AIRCRAFT["wing_sweep_half_chord_deg"]), cl_alpha_A_h,
        fuselage_width, fuselage_length, wing["area_m2"], mac, AIRCRAFT["wing_CL0"],
    )

    # Controllability is sized at the most demanding wing-borne lift condition the
    # mission actually flies, with a configurable margin and a cap at the permitted
    # CL limit. VTOL/transition handle the very-low-speed cases, so using the
    # stall-boundary CL here would force an unnecessarily large destabilising canard.
    control_condition = scissor_control_lift_condition(mission)
    cl_A_h_control = control_condition["CL_Ah_control"]

    # Canard-on-wing downwash gradient de/da (Slingerland). The canard is the
    # GENERATING surface (the wing sits in its wake), so use the canard's lift
    # slope, aspect ratio and span. r = l_h / (b_canard/2); lh_over_mac is
    # negative, so take the magnitude of the arm. m_tv = 0 (coplanar surfaces).
    lh_arm = lh_over_mac * mac                          # signed arm length [m]
    r = 2.0 * abs(lh_arm) / canard["span_m"]
    de_da = wing_downwash_gradient(
        cl_alpha_canard,
        AIRCRAFT["canard_aspect_ratio"],
        math.radians(AIRCRAFT["canard_sweep_deg"]),
        r,
        m_tv=0.0,
    )

    # Fraction of the wing AREA immersed in the canard wake. The wake spans the
    # canard span, so the wing is immersed inboard of +/- b_canard/2. For a
    # trapezoid (taper = ct/cr), the area inboard of span fraction eta is
    # (eta - 0.5*(1-taper)*eta^2) / (0.5*(1+taper)). Downwash and the wake
    # dynamic-pressure loss act only on this fraction.
    taper = AIRCRAFT["wing_taper"]
    eta_star = min(1.0, canard["span_m"] / wing["span_m"])
    wing_immersed_fraction = (
        (eta_star - 0.5 * (1.0 - taper) * eta_star**2) / (0.5 * (1.0 + taper))
    )
    wing_lift_slope_factor = (
        (1.0 - wing_immersed_fraction)
        + wing_immersed_fraction
        * (1.0 - de_da)
        * AIRCRAFT["wing_wake_dynamic_pressure_ratio"]
    )

    return {
        "mach": mach,
        "cl_alpha_wing": cl_alpha_wing,
        "cl_alpha_canard": cl_alpha_canard,
        "cl_alpha_A_h": cl_alpha_A_h,
        "x_ac_over_mac": x_ac_over_mac,
        "cmac": cmac,
        # Canard control authority: course-method full-moving value |C_Lh|=1
        # (positive, the canard lifts up), capped at the canard's real CL_max.
        "cl_h_control": min(
            AIRCRAFT["canard_control_CLh_full_moving"], AIRCRAFT["canard_CL_max"]
        ),
        "cl_A_h_control": cl_A_h_control,
        "cl_A_h_control_source": control_condition["source"],
        "cl_A_h_control_mission_max": control_condition["mission_max_CL"],
        "cl_A_h_control_margin_factor": control_condition["margin_factor"],
        "cl_A_h_control_permitted": control_condition["permitted_CL"],
        "lh_over_mac": lh_over_mac,                              # < 0 for a canard
        "de_da": de_da,                                         # canard-on-wing downwash
        "wing_immersed_fraction": wing_immersed_fraction,
        "wing_wake_dynamic_pressure_ratio": AIRCRAFT["wing_wake_dynamic_pressure_ratio"],
        "canard_speed_ratio_sq": AIRCRAFT["canard_speed_ratio_sq"],
        "wing_lift_slope_factor": wing_lift_slope_factor,
        "static_margin": AIRCRAFT["static_margin"],
    }


def scissor_limits(area_ratio, coeffs):
    """Forward (controllability) and aft (stability) CG limits at this Sc/Sw."""
    if coeffs["cl_A_h_control"] <= 0.0:
        return None
    x_forward, x_aft = scissor_cg_limits(
        area_ratio,
        x_ac_over_mac=coeffs["x_ac_over_mac"],
        static_margin=coeffs["static_margin"],
        cl_alpha_canard=coeffs["cl_alpha_canard"],
        cl_alpha_A_h=coeffs["cl_alpha_A_h"],
        cl_h_control=coeffs["cl_h_control"],
        cl_A_h_control=coeffs["cl_A_h_control"],
        cmac=coeffs["cmac"],
        lh_over_mac=coeffs["lh_over_mac"],
        de_da=coeffs["de_da"],
        wing_immersed_fraction=coeffs["wing_immersed_fraction"],
        wing_wake_dynamic_pressure_ratio=coeffs["wing_wake_dynamic_pressure_ratio"],
        canard_speed_ratio_sq=coeffs["canard_speed_ratio_sq"],
    )
    return {
        "x_forward_over_mac": x_forward,
        "x_aft_over_mac": x_aft,
        "cg_range_over_mac": x_aft - x_forward,
        "CL_Ah": coeffs["cl_A_h_control"],
        "wing_lift_slope_factor": coeffs["wing_lift_slope_factor"],
    }


def evaluate_wing_station(wing, canard, area_ratio, mission, propeller, wing_le_m):
    """Scissor band, mass CG, and CG-envelope fit clearance at one wing station.

    Both the band (via the arm-dependent lh and fuselage-length-dependent x_ac)
    and the CG depend on wing_le_m, so they are evaluated together. `clearance` is
    the worst-side gap between the operational CG envelope and the scissor band
    (>= 0 means the envelope fits); maximising it is the right objective for
    placing the wing, because the band *width* also changes with the arm.

    `lower_clearance` is the controllability (forward-CG) gap, `upper_clearance`
    the static-stability (aft-CG) gap. When require_static_stability is False the
    aft gap is dropped from `clearance`.
    """
    half_width = AIRCRAFT["cg_envelope_half_width_over_mac"]
    margin = AIRCRAFT["cg_margin_over_mac"]

    layout = longitudinal_layout(wing, canard, wing_le_m)
    coeffs = scissor_coefficients(
        wing, canard, layout["L_fus_m"], layout["lh_over_mac"], mission
    )
    scissor = scissor_limits(area_ratio, coeffs)
    mass = mass_and_cg(wing, canard, mission, propeller, wing_le_m)

    x_cg = mass["x_cg_over_mac"]
    lower_clear = (x_cg - half_width) - (scissor["x_forward_over_mac"] + margin)
    upper_clear = (scissor["x_aft_over_mac"] - margin) - (x_cg + half_width)
    if AIRCRAFT["require_static_stability"]:
        clearance = min(lower_clear, upper_clear)
    else:
        clearance = lower_clear
    # Achieved static margin = neutral point - CG (negative => statically unstable,
    # recovered by the autopilot). x_aft = NP - static_margin, so NP = x_aft + SM.
    neutral_point = scissor["x_aft_over_mac"] + AIRCRAFT["static_margin"]
    return {
        "wing_le_m": wing_le_m,
        "layout": layout,
        "coeffs": coeffs,
        "scissor": scissor,
        "mass": mass,
        "band_center": 0.5 * (scissor["x_forward_over_mac"] + scissor["x_aft_over_mac"]),
        "lower_clearance": lower_clear,
        "upper_clearance": upper_clear,
        "clearance": clearance,
        "achieved_static_margin_over_mac": neutral_point - x_cg,
    }


def best_wing_station_by_clearance(wing, canard, area_ratio, mission, propeller, lo, hi):
    """Return the bounded wing station with the largest CG-envelope clearance."""
    sample_count = 41
    best = None
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        wing_le = lo + fraction * (hi - lo)
        evaluation = evaluate_wing_station(
            wing, canard, area_ratio, mission, propeller, wing_le
        )
        if best is None or evaluation["clearance"] > best["clearance"]:
            best = evaluation

    sample_step = (hi - lo) / (sample_count - 1)
    left = max(lo, best["wing_le_m"] - sample_step)
    right = min(hi, best["wing_le_m"] + sample_step)
    for _ in range(24):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        e1 = evaluate_wing_station(wing, canard, area_ratio, mission, propeller, m1)
        e2 = evaluate_wing_station(wing, canard, area_ratio, mission, propeller, m2)
        for evaluation in (e1, e2):
            if evaluation["clearance"] > best["clearance"]:
                best = evaluation
        if e1["clearance"] < e2["clearance"]:
            left = m1
        else:
            right = m2
        if (right - left) < 1e-3:
            break
    return best


def solve_wing_station(wing, canard, area_ratio, mission, propeller):
    """Solve the smallest arm (shortest, lightest fuselage) that fits the envelope.

    With static stability required, moving the wing aft monotonically widens the
    scissor band and raises the fit clearance, so the lightest viable design is the
    smallest wing station whose clearance >= 0, found by bisecting the lower
    crossing. With stability waived the clearance can be non-monotonic in the arm,
    so when both ends are infeasible we fall back to a grid+golden-section search.
    """
    # Floor the wing station so the wing root LE sits at least canard_wing_min_gap_m
    # behind the canard root TE (no overlap / interference).
    canard_root_chord = (
        2.0 * canard["area_m2"]
        / (canard["span_m"] * (1.0 + AIRCRAFT["canard_taper"]))
    )
    lo_overlap = MASS["nose_to_canard_m"] + canard_root_chord + AIRCRAFT["canard_wing_min_gap_m"]
    lo = max(MASS["nose_to_canard_m"] + AIRCRAFT["canard_arm_min_m"], lo_overlap)
    hi = max(lo, MASS["nose_to_canard_m"] + AIRCRAFT["canard_arm_max_m"])

    def evaluate(wing_le):
        return evaluate_wing_station(wing, canard, area_ratio, mission, propeller, wing_le)

    eval_lo, eval_hi = evaluate(lo), evaluate(hi)
    if eval_hi["clearance"] < 0.0:
        return best_wing_station_by_clearance(
            wing, canard, area_ratio, mission, propeller, lo, hi
        )
    if eval_lo["clearance"] >= 0.0:
        return eval_lo                      # shortest arm already fits

    # clearance(lo) < 0 <= clearance(hi): bisect for the smallest feasible arm.
    a, best = lo, eval_hi
    for _ in range(40):
        mid = 0.5 * (a + best["wing_le_m"])
        eval_mid = evaluate(mid)
        if eval_mid["clearance"] >= 0.0:
            best = eval_mid
        else:
            a = mid
        if (best["wing_le_m"] - a) < 1e-3:
            break
    return best


def canard_and_wing_iteration(wing, mission, propeller):
    """Pick the canard area ratio + wing position that minimise total mass (MTOW).

    Sc/Sw and the wing station trade against each other: a smaller canard needs a
    longer canard->wing arm to open the scissor band, and the arm sets the fuselage
    length. The mass build-up captures both, so the objective is simply the
    lightest feasible design. Feasibility includes the scissor band and the active
    maximum-stall-speed requirement. For each ratio, solve_wing_station returns the
    lightest (shortest-arm) wing station that fits the CG envelope, so here we only
    sweep the ratio and keep the minimum-mass feasible candidate. If none fit, fall
    back to the largest-clearance candidate so the sweep still reports a near miss.
    """
    half_width = AIRCRAFT["cg_envelope_half_width_over_mac"]
    margin = AIRCRAFT["cg_margin_over_mac"]
    required_width = 2.0 * (half_width + margin)

    candidates = []
    best_candidate = None            # largest clearance (near-miss fallback)
    best_feasible = None             # lightest feasible design (the objective)
    steps = int((AIRCRAFT["canard_area_ratio_max"] - AIRCRAFT["canard_area_ratio_min"]) / AIRCRAFT["canard_area_ratio_step"]) + 1
    for i in range(steps):
        area_ratio = AIRCRAFT["canard_area_ratio_min"] + i * AIRCRAFT["canard_area_ratio_step"]
        canard = canard_geometry(area_ratio, wing)

        # Solve the wing station (canard->wing arm) that fits the CG envelope in the
        # band; the band itself moves with the arm, so the two are solved jointly.
        solution = solve_wing_station(wing, canard, area_ratio, mission, propeller)
        scissor = solution["scissor"]
        mass = solution["mass"]
        if scissor is None:
            continue

        band_is_wide_enough = scissor["cg_range_over_mac"] >= required_width
        x_cg = mass["x_cg_over_mac"]
        operational_fwd = x_cg - half_width
        operational_aft = x_cg + half_width
        # Controllability (forward CG) is always required. The aft (stability) limit
        # and the minimum-band-width check are enforced only when natural static
        # stability is required; otherwise the autopilot recovers it.
        controllable = operational_fwd >= scissor["x_forward_over_mac"] + margin
        if AIRCRAFT["require_static_stability"]:
            stable = (
                band_is_wide_enough
                and operational_aft <= scissor["x_aft_over_mac"] - margin
            )
        else:
            stable = True
        scissor_fits = controllable and stable
        stall_limit = stall_speed_limit(wing, canard)
        stall_margin = stall_limit["stall_EAS_max_m_s"] - wing["stall_EAS_m_s"]
        stall_fits = stall_margin >= 0.0
        fits = scissor_fits and stall_fits

        candidate = {
            "canard": canard,
            "scissor": scissor,
            "mass": mass,
            "wing_le_m": solution["wing_le_m"],
            "arm_m": solution["layout"]["arm_m"],
            "coeffs": solution["coeffs"],
            "target_x_cg_over_mac": solution["band_center"],
            "operational_fwd_over_mac": operational_fwd,
            "operational_aft_over_mac": operational_aft,
            "lower_clearance_over_mac": solution["lower_clearance"],
            "upper_clearance_over_mac": solution["upper_clearance"],
            "clearance_over_mac": solution["clearance"],
            "achieved_static_margin_over_mac": solution["achieved_static_margin_over_mac"],
            "band_is_wide_enough": band_is_wide_enough,
            "statically_stable": operational_aft <= scissor["x_aft_over_mac"] - margin,
            "scissor_feasible": scissor_fits,
            "stall_limit": stall_limit,
            "stall_margin_m_s": stall_margin,
            "stall_feasible": stall_fits,
            "feasible": fits,
        }
        candidates.append(candidate)
        if (
            best_candidate is None
            or candidate["clearance_over_mac"] > best_candidate["clearance_over_mac"]
        ):
            best_candidate = candidate
        if fits and (
            best_feasible is None
            or candidate["mass"]["total_mass_kg"] < best_feasible["mass"]["total_mass_kg"]
        ):
            best_feasible = candidate

    if best_feasible is not None:
        return best_feasible, candidates
    if best_candidate is None:
        raise RuntimeError("No canard/scissor candidates could be evaluated.")
    return best_candidate, candidates
