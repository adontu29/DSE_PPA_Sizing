"""Course-method wing-borne climb from power available.

This file follows the lecture workflow for a constant-EAS climb:

    RC_s = (P_a - P_r) / W
    RC   = RC_s / (1 + V/g * dV/dH)

The result is separate from the main mission model so both approaches can be
compared before choosing which one to use in the sizing workflow.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drag_buildup import build_drag_geometry, parasite_drag_buildup
from xfoil_wrapper import mach_number, reynolds_number


RHO_SEA_LEVEL = 1.225


def setting(values, name, default):
    return values[name] if name in values else default


def linspace(start, stop, count):
    if count == 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [float(start + index * step) for index in range(count)]


def true_airspeed_from_eas(equivalent_airspeed_m_s, density_kg_m3):
    return equivalent_airspeed_m_s * math.sqrt(RHO_SEA_LEVEL / density_kg_m3)


def induced_drag_factor(aircraft):
    return 1.0 / (
        math.pi
        * aircraft["wing_aspect_ratio"]
        * aircraft["oswald_efficiency"]
    )


def trim_lift_split(weight_N, q, wing, trim):
    """Split the total lift between the wing and the canard from trim.

    Moment balance about the CG (nose-up positive), with the wing free pitching
    moment M_ac = Cm_ac * q * S_w * c_w:
        L_c * l_c - L_w * l_w + M_ac = 0,     L_w + L_c = W
    The canard sits ahead of the CG (arm l_c, nose-up when lifting) and the wing
    behind it (arm l_w, nose-down). Returns (L_w, L_c); either may go negative
    (a download) if trim demands it. Total lift is taken as the weight.
    """
    S_w = wing["area_m2"]
    c_w = wing["chord_m"]
    arm_sum = trim["l_w"] + trim["l_c"]
    if arm_sum <= 1e-6:
        return weight_N, 0.0
    M_ac = trim.get("Cm_ac", 0.0) * q * S_w * c_w
    L_c = (weight_N * trim["l_w"] - M_ac) / arm_sum
    L_w = weight_N - L_c
    return L_w, L_c


def _cached_drag_geometry(aircraft, wing):
    """Build the component-drag geometry once per (aircraft, wing, trim) and reuse it.

    The geometry is flow-independent, but parasite_drag_coefficient is called once
    per mission state (tens of thousands of times per sizing pass) with the same
    aircraft/wing/trim objects. Caching it on the aircraft dict turns those
    rebuilds into a single build per mission. The cache key is the identity of the
    wing and trim objects, which are replaced (not mutated) between passes.
    """
    trim = aircraft.get("trim_drag")
    key = (id(wing), id(trim))
    cache = aircraft.get("_drag_geometry_cache")
    if cache is not None and cache[0] == key:
        return cache[1]
    geom = build_drag_geometry(aircraft, wing, trim)
    aircraft["_drag_geometry_cache"] = (key, geom)
    return geom


def parasite_drag_coefficient(aircraft, wing, altitude_m=None, true_speed_m_s=None):
    """Wing-area-referenced zero-lift drag coefficient (AD2 component build-up).

    CD0 is built up component by component once the layout geometry (via the trim
    descriptor) and the flow state (altitude + speed for the Reynolds number) are
    available. On the first mass iteration -- before any canard geometry exists, so
    the build-up cannot run -- it falls back to the fixed aircraft["CD0"].
    """
    if altitude_m is not None and true_speed_m_s is not None:
        geom = _cached_drag_geometry(aircraft, wing)
        if geom is not None:
            reynolds_per_m = reynolds_number(altitude_m, true_speed_m_s, 1.0)
            mach = mach_number(altitude_m, true_speed_m_s)
            return parasite_drag_buildup(geom, mach, reynolds_per_m)["CD0_component"]
    return aircraft["CD0"]


def aero_drag_coefficient(aircraft, q, weight_N, wing, altitude_m=None, true_speed_m_s=None):
    """Total drag coefficient referenced to wing area.

    Parasite drag comes from the AD2 component build-up (parasite_drag_coefficient),
    falling back to the fixed CD0 before the canard geometry exists. With a trim
    descriptor present, induced drag is summed over the wing and canard at their
    trimmed lifts via Di = L^2 / (q*pi*b^2*e) (Munk tandem, plus the mutual
    interference cross term); on the first iteration (no trim) it falls back to
    wing-only induced drag.

    The total CD depends only on the flow state and the lift, not on the available
    power, so the same (altitude, speed, lift) recurs across every power candidate
    of a given EAS. The result is cached per mission (fresh aircraft dict per
    mission, with a fixed wing/geometry) to avoid recomputing the build-up and the
    induced split for each candidate.
    """
    cd_cache = None
    if altitude_m is not None and true_speed_m_s is not None:
        cd_cache = aircraft.get("_cd_cache")
        if cd_cache is None:
            cd_cache = aircraft["_cd_cache"] = {}
        cache_key = (altitude_m, true_speed_m_s, weight_N)
        cached = cd_cache.get(cache_key)
        if cached is not None:
            return cached

    S_w = wing["area_m2"]
    CL_wing_ref = weight_N / (q * S_w)
    trim = aircraft.get("trim_drag")
    parasite = parasite_drag_coefficient(aircraft, wing, altitude_m, true_speed_m_s)
    if trim is None:
        total = parasite + induced_drag_factor(aircraft) * CL_wing_ref**2
        if cd_cache is not None:
            cd_cache[cache_key] = total
        return total

    S_c = trim["S_c"]
    e_w = aircraft["oswald_efficiency"]
    e_c = setting(aircraft, "canard_oswald_efficiency", e_w)
    sigma = setting(aircraft, "canard_wing_induced_interference_factor", 0.0)
    b_w = wing["span_m"]
    b_c = trim["b_c"]
    L_w, L_c = trim_lift_split(weight_N, q, wing, trim)
    # Munk tandem induced drag: each surface's own induced drag plus the mutual
    # interference cross term (positive when both surfaces lift the same way).
    induced_drag_N = (
        L_w**2 / (b_w**2 * e_w)
        + L_c**2 / (b_c**2 * e_c)
        + 2.0 * sigma * L_w * L_c / (b_w * b_c)
    ) / (q * math.pi)
    total = parasite + induced_drag_N / (q * S_w)
    if cd_cache is not None:
        cd_cache[cache_key] = total
    return total


def forward_efficiency(aircraft):
    cached = aircraft.get("_forward_efficiency")
    if cached is not None:
        return cached
    value = setting(
        aircraft,
        "forward_flight_efficiency",
        setting(aircraft, "eta_prop", 0.75)
        * setting(aircraft, "eta_motor", 0.90)
        * setting(aircraft, "eta_ESC", 0.95),
    )
    aircraft["_forward_efficiency"] = value
    return value


def permitted_lift_coefficient(aircraft):
    fraction_limit = (
        setting(aircraft, "mission_CL_limit_fraction", 0.90)
        * aircraft["wing_CL_max"]
    )
    stall_margin_deg = setting(aircraft, "cruise_stall_margin_deg", 3.0)
    margin_limit = (
        aircraft["wing_CL_max"]
        - aircraft["wing_CL_alpha_per_rad"] * math.radians(stall_margin_deg)
    )
    return min(fraction_limit, margin_limit)


def default_eas_grid(weight_N, wing, aircraft):
    CL_allowed = permitted_lift_coefficient(aircraft)
    minimum_eas = math.sqrt(
        2.0 * weight_N
        / (RHO_SEA_LEVEL * wing["area_m2"] * CL_allowed)
    )
    lower = 1.001 * minimum_eas
    upper = max(1.8 * lower, lower + 12.0)
    return linspace(lower, upper, 13)


def default_power_grid(aircraft):
    """Electrical power search grid for the course-method climb."""
    minimum_power_W = setting(aircraft, "course_climb_power_min_W", 500.0)
    maximum_power_W = setting(
        aircraft,
        "course_climb_available_power_W",
        setting(aircraft, "max_affordable_electrical_power_W", 20000.0),
    )
    step_W = setting(aircraft, "course_climb_power_step_W", 500.0)
    count = int((maximum_power_W - minimum_power_W) / step_W) + 1
    return [minimum_power_W + index * step_W for index in range(count)]


_DV_DH_CACHE = {}


def constant_eas_dv_dh(equivalent_airspeed_m_s, altitude_m, isa_density):
    """Numerical dV_TAS/dH for constant EAS.

    Depends only on the EAS, the altitude and the (fixed) atmosphere model, but is
    evaluated once per mission state across every power candidate, so it is
    memoized. The atmosphere function identity is part of the key in case a
    different ISA model is ever passed in.
    """
    key = (equivalent_airspeed_m_s, altitude_m, id(isa_density))
    cached = _DV_DH_CACHE.get(key)
    if cached is not None:
        return cached
    step_m = 1.0
    lower_h = max(0.0, altitude_m - step_m)
    upper_h = altitude_m + step_m
    lower_v = true_airspeed_from_eas(equivalent_airspeed_m_s, isa_density(lower_h))
    upper_v = true_airspeed_from_eas(equivalent_airspeed_m_s, isa_density(upper_h))
    result = (upper_v - lower_v) / (upper_h - lower_h)
    _DV_DH_CACHE[key] = result
    return result


def climb_state_from_power_available(
    weight_N,
    wing,
    aircraft,
    isa_density,
    altitude_m,
    equivalent_airspeed_m_s,
    available_electrical_power_W,
    load_factor=1.0,
):
    density = isa_density(altitude_m)
    true_speed = true_airspeed_from_eas(equivalent_airspeed_m_s, density)
    q = 0.5 * density * true_speed**2
    # In a banked climbing turn the wings carry n*W, so the operating CL and the
    # induced drag are scaled by the load factor; the climb rate still divides the
    # excess power by the actual weight, not n*W.
    lift_N = load_factor * weight_N
    CL = lift_N / (q * wing["area_m2"])
    CD = aero_drag_coefficient(aircraft, q, lift_N, wing, altitude_m, true_speed)
    drag_N = q * wing["area_m2"] * CD
    bank_angle_deg = math.degrees(math.acos(min(1.0, 1.0 / load_factor))) if load_factor > 1.0 else 0.0

    power_required_propulsive_W = drag_N * true_speed
    selected_power_available_propulsive_W = (
        available_electrical_power_W * forward_efficiency(aircraft)
    )
    thrust_from_selected_power_N = selected_power_available_propulsive_W / true_speed
    thrust_limit_N = (
        setting(aircraft, "course_climb_max_thrust_to_weight", 0.50)
        * weight_N
    )
    usable_thrust_N = min(thrust_from_selected_power_N, thrust_limit_N)
    power_available_propulsive_W = usable_thrust_N * true_speed
    electrical_power_used_W = power_available_propulsive_W / forward_efficiency(aircraft)
    steady_rate_of_climb_m_s = (
        power_available_propulsive_W - power_required_propulsive_W
    ) / weight_N

    dV_dH = constant_eas_dv_dh(equivalent_airspeed_m_s, altitude_m, isa_density)
    acceleration_correction = 1.0 + true_speed / aircraft["g_m_s2"] * dV_dH
    rate_of_climb_m_s = steady_rate_of_climb_m_s / acceleration_correction

    return {
        "altitude_m": altitude_m,
        "density_kg_m3": density,
        "EAS_m_s": equivalent_airspeed_m_s,
        "TAS_m_s": true_speed,
        "dV_dH": dV_dH,
        "CL": CL,
        "CD": CD,
        "drag_N": drag_N,
        "power_required_propulsive_W": power_required_propulsive_W,
        "power_available_propulsive_W": power_available_propulsive_W,
        "selected_power_available_propulsive_W": selected_power_available_propulsive_W,
        "power_available_electrical_W": available_electrical_power_W,
        "electrical_power_used_W": electrical_power_used_W,
        "thrust_from_selected_power_N": thrust_from_selected_power_N,
        "usable_thrust_N": usable_thrust_N,
        "thrust_limit_N": thrust_limit_N,
        "thrust_limited": usable_thrust_N < thrust_from_selected_power_N,
        "steady_rate_of_climb_m_s": steady_rate_of_climb_m_s,
        "rate_of_climb_m_s": rate_of_climb_m_s,
        "acceleration_correction": acceleration_correction,
        "load_factor": load_factor,
        "bank_angle_deg": bank_angle_deg,
    }


def _integrate_climb(
    weight_N,
    wing,
    mission,
    aircraft,
    isa_density,
    equivalent_airspeed_m_s,
    available_electrical_power_W,
    spiral_radius_m=None,
    spiral_below_altitude_m=None,
):
    """Integrate the lecture RC equation through altitude.

    Steps below spiral_below_altitude_m (the crossover) are flown as a coordinated
    turn of radius spiral_radius_m, carrying load factor n = sqrt(1 + (V^2/(g*R))^2)
    -- this is the in-place spiral, which makes no progress toward the target but
    pays higher induced drag and a tighter stall (CL) margin. Steps at or above the
    crossover are the straight climb-out toward the target and count as ground-track
    progress. With no radius set the whole climb is straight (load factor 1).
    """
    start_altitude = mission["vertical_takeoff_height_m"]
    target_altitude = mission["altitude_m"]
    altitude_step = setting(mission, "altitude_step_m", 100.0)
    CL_allowed = permitted_lift_coefficient(aircraft)
    g = aircraft["g_m_s2"]

    states = []
    time_s = 0.0
    distance_m = 0.0       # ground-track progress toward the target (straight steps)
    spiral_arc_m = 0.0     # horizontal arc flown while circling in place
    energy_Wh = 0.0
    altitude = start_altitude

    while altitude < target_altitude - 1e-9:
        next_altitude = min(altitude + altitude_step, target_altitude)
        mid_altitude = 0.5 * (altitude + next_altitude)
        spiral_step = (
            bool(spiral_radius_m) and spiral_radius_m > 0.0
            and (spiral_below_altitude_m is None or mid_altitude < spiral_below_altitude_m)
        )
        if spiral_step:
            true_speed = true_airspeed_from_eas(equivalent_airspeed_m_s, isa_density(mid_altitude))
            load_factor = math.sqrt(1.0 + (true_speed**2 / (g * spiral_radius_m))**2)
        else:
            load_factor = 1.0
        state = climb_state_from_power_available(
            weight_N,
            wing,
            aircraft,
            isa_density,
            mid_altitude,
            equivalent_airspeed_m_s,
            available_electrical_power_W,
            load_factor=load_factor,
        )
        if state["CL"] > CL_allowed:
            return {
                "feasible": False,
                "failure_reason": f"CL limit exceeded at {mid_altitude:.0f} m (n={load_factor:.2f}).",
                "states": states,
            }
        if state["rate_of_climb_m_s"] <= 0.0:
            return {
                "feasible": False,
                "failure_reason": f"No positive climb rate at {mid_altitude:.0f} m (n={load_factor:.2f}).",
                "states": states,
            }

        delta_h = next_altitude - altitude
        delta_t = delta_h / state["rate_of_climb_m_s"]
        climb_angle = math.asin(
            min(0.999, state["rate_of_climb_m_s"] / state["TAS_m_s"])
        )
        delta_x = state["TAS_m_s"] * math.cos(climb_angle) * delta_t
        delta_energy = state["electrical_power_used_W"] * delta_t / 3600.0

        time_s += delta_t
        if spiral_step:
            spiral_arc_m += delta_x      # circling, no net progress
        else:
            distance_m += delta_x        # straight climb-out toward the target
        energy_Wh += delta_energy
        state.update({
            "altitude_start_m": altitude,
            "altitude_end_m": next_altitude,
            "delta_h_m": delta_h,
            "delta_t_s": delta_t,
            "delta_x_m": delta_x,
            "delta_energy_Wh": delta_energy,
            "spiral_step": spiral_step,
            "thrust_to_weight": state["usable_thrust_N"] / weight_N,
            "time_s": time_s,
            "distance_m": distance_m,
            "energy_Wh": energy_Wh,
            "climb_angle_deg": math.degrees(climb_angle),
        })
        states.append(state)
        altitude = next_altitude

    return {
        "feasible": True,
        "failure_reason": None,
        "EAS_m_s": equivalent_airspeed_m_s,
        "available_electrical_power_W": available_electrical_power_W,
        "average_electrical_power_used_W": energy_Wh * 3600.0 / time_s,
        "thrust_limited": any(state["thrust_limited"] for state in states),
        "max_thrust_to_weight": max(state["thrust_to_weight"] for state in states),
        "time_s": time_s,
        "distance_m": distance_m,
        "spiral_arc_m": spiral_arc_m,
        "energy_Wh": energy_Wh,
        "states": states,
        "max_load_factor": max(state["load_factor"] for state in states),
        "max_bank_angle_deg": max(state["bank_angle_deg"] for state in states),
    }


def _spiral_crossover_altitude(straight_states, range_m):
    """Altitude above which the straight climb-out covers exactly range_m.

    Walking down from the top of the straight climb, accumulate ground track until
    it reaches the range; the start altitude of that step is the crossover. Below
    it the aircraft spirals up in place; from it, the straight climb-out to the
    target consumes the allowed ground track and arrives at the target altitude.
    """
    cumulative = 0.0
    crossover = straight_states[0]["altitude_start_m"]
    for state in reversed(straight_states):
        cumulative += state["delta_x_m"]
        crossover = state["altitude_start_m"]
        if cumulative >= range_m:
            break
    return crossover


def simulate_course_climb(
    weight_N,
    wing,
    mission,
    aircraft,
    isa_density,
    equivalent_airspeed_m_s,
    available_electrical_power_W=None,
):
    """Course-method climb, straight or spiral-then-straight if it would overshoot.

    A straight climb is integrated first. If spiralling is allowed and that climb's
    ground track would overrun the mission range, the aircraft cannot fly straight
    to the target without overshooting, so it spirals up in place over the launch
    point until only `range` of ground track remains, then climbs straight out to
    the target. The in-place spiral (below the crossover altitude) carries the turn
    load factor -- higher induced drag and a tighter stall margin -- while the
    straight climb-out delivers the ground-track progress, so no level cruise is
    needed.
    """
    if available_electrical_power_W is None:
        available_electrical_power_W = setting(
            aircraft,
            "course_climb_available_power_W",
            setting(aircraft, "max_affordable_electrical_power_W", 20000.0),
        )

    straight = _integrate_climb(
        weight_N, wing, mission, aircraft, isa_density,
        equivalent_airspeed_m_s, available_electrical_power_W,
        spiral_radius_m=None,
    )
    straight["spiral_used"] = False
    straight["spiral_radius_m"] = None
    straight["spiral_crossover_altitude_m"] = None

    spiral_radius_m = setting(mission, "spiral_turn_radius_m", 0.0)
    spiral_allowed = setting(mission, "allow_spiral_climb", False) and spiral_radius_m > 0.0
    if not spiral_allowed or not straight["feasible"]:
        return straight
    if straight["distance_m"] <= mission["range_m"]:
        return straight  # fits within range flying straight; no spiral needed

    crossover_m = _spiral_crossover_altitude(straight["states"], mission["range_m"])
    spiral = _integrate_climb(
        weight_N, wing, mission, aircraft, isa_density,
        equivalent_airspeed_m_s, available_electrical_power_W,
        spiral_radius_m=spiral_radius_m,
        spiral_below_altitude_m=crossover_m,
    )
    if not spiral["feasible"]:
        return spiral
    spiral["spiral_used"] = True
    spiral["spiral_radius_m"] = spiral_radius_m
    spiral["spiral_crossover_altitude_m"] = crossover_m
    return spiral


def estimate_takeoff_transition(weight_N, wing, mission, aircraft, isa_density):
    """Low-altitude takeoff and speed-based transition estimates."""
    transition_altitude = mission["vertical_takeoff_height_m"]
    rho_transition = isa_density(transition_altitude)
    stall_speed = math.sqrt(
        2.0 * weight_N
        / (rho_transition * wing["area_m2"] * aircraft["wing_CL_max"])
    )
    blend_end = setting(aircraft, "transition_blend_end_fraction", 1.20) * stall_speed
    acceleration = setting(aircraft, "transition_accel_m_s2", 1.0)

    takeoff_time = transition_altitude / mission["vertical_takeoff_rate_m_s"]
    takeoff_energy = aircraft["hover_power_W"] * takeoff_time / 3600.0
    transition_time = blend_end / acceleration
    transition_distance = blend_end**2 / (2.0 * acceleration)
    transition_power = setting(aircraft, "transition_power_W", 0.5 * aircraft["hover_power_W"])
    transition_energy = transition_power * transition_time / 3600.0

    return {
        "transition_altitude_m": transition_altitude,
        "takeoff_time_s": takeoff_time,
        "takeoff_energy_Wh": takeoff_energy,
        "stall_speed_transition_m_s": stall_speed,
        "blend_end_m_s": blend_end,
        "transition_time_s": transition_time,
        "transition_distance_m": transition_distance,
        "transition_power_W": transition_power,
        "transition_energy_Wh": transition_energy,
    }


def estimate_level_cruise(weight_N, wing, mission, aircraft, isa_density, distance_m, true_speed_m_s):
    """Simple level-cruise segment after climb, if any range remains."""
    if distance_m <= 0.0:
        return {
            "distance_m": 0.0,
            "time_s": 0.0,
            "energy_Wh": 0.0,
            "electrical_power_W": 0.0,
            "CL": 0.0,
        }

    density = isa_density(mission["altitude_m"])
    q = 0.5 * density * true_speed_m_s**2
    CL = weight_N / (q * wing["area_m2"])
    CD = aero_drag_coefficient(
        aircraft, q, weight_N, wing, mission["altitude_m"], true_speed_m_s
    )
    drag_N = q * wing["area_m2"] * CD
    electrical_power = drag_N * true_speed_m_s / forward_efficiency(aircraft)
    time_s = distance_m / true_speed_m_s
    energy_Wh = electrical_power * time_s / 3600.0
    return {
        "distance_m": distance_m,
        "time_s": time_s,
        "energy_Wh": energy_Wh,
        "electrical_power_W": electrical_power,
        "CL": CL,
    }


def segment_summary(time_s, energy_Wh, distance_m=0.0):
    return {
        "time_s": time_s,
        "energy_Wh": energy_Wh,
        "distance_m": distance_m,
        "average_power_W": energy_Wh * 3600.0 / time_s if time_s > 0.0 else 0.0,
    }


def build_course_mission(weight_N, wing, mission, aircraft, isa_density, selected_climb):
    """Wrap the course-method climb in the full mission energy timeline."""
    takeoff_transition = estimate_takeoff_transition(
        weight_N,
        wing,
        mission,
        aircraft,
        isa_density,
    )
    ground_before_cruise = (
        takeoff_transition["transition_distance_m"]
        + selected_climb["distance_m"]
    )
    remaining_cruise_distance = max(0.0, mission["range_m"] - ground_before_cruise)
    spiral_excess_distance = max(0.0, ground_before_cruise - mission["range_m"])
    final_climb_speed = selected_climb["states"][-1]["TAS_m_s"]
    cruise = estimate_level_cruise(
        weight_N,
        wing,
        mission,
        aircraft,
        isa_density,
        remaining_cruise_distance,
        final_climb_speed,
    )
    hover_energy = aircraft["hover_power_W"] * mission["hover_time_s"] / 3600.0

    segments = {
        "vertical_takeoff": segment_summary(
            takeoff_transition["takeoff_time_s"],
            takeoff_transition["takeoff_energy_Wh"],
        ),
        "transition": segment_summary(
            takeoff_transition["transition_time_s"],
            takeoff_transition["transition_energy_Wh"],
            takeoff_transition["transition_distance_m"],
        ),
        "wing_borne_climb": segment_summary(
            selected_climb["time_s"],
            selected_climb["energy_Wh"],
            selected_climb["distance_m"],
        ),
        "level_cruise": segment_summary(
            cruise["time_s"],
            cruise["energy_Wh"],
            cruise["distance_m"],
        ),
        "mission_hover": segment_summary(
            mission["hover_time_s"],
            hover_energy,
        ),
    }

    total_load_energy_Wh = sum(segment["energy_Wh"] for segment in segments.values())
    installed_battery_energy_Wh = (
        total_load_energy_Wh
        / aircraft["battery_efficiency"]
        / aircraft["battery_usable_fraction"]
    )

    time = 0.0
    distance = 0.0
    altitude = 0.0
    profile = [(time, distance, altitude)]
    time += takeoff_transition["takeoff_time_s"]
    altitude = takeoff_transition["transition_altitude_m"]
    profile.append((time, distance, altitude))
    time += takeoff_transition["transition_time_s"]
    distance += takeoff_transition["transition_distance_m"]
    profile.append((time, distance, altitude))
    for state in selected_climb["states"]:
        profile.append((
            time + state["time_s"],
            distance + state["distance_m"],
            state["altitude_end_m"],
        ))
    time += selected_climb["time_s"]
    distance += selected_climb["distance_m"]
    if cruise["time_s"] > 0.0:
        time += cruise["time_s"]
        distance += cruise["distance_m"]
        profile.append((time, distance, mission["altitude_m"]))
    time += mission["hover_time_s"]
    profile.append((time, distance, mission["altitude_m"]))

    return {
        "takeoff_transition": takeoff_transition,
        "climb": selected_climb,
        "cruise": cruise,
        "segments": segments,
        "profile": profile,
        "total_load_energy_Wh": total_load_energy_Wh,
        "installed_battery_energy_Wh": installed_battery_energy_Wh,
        "battery_mass_kg": installed_battery_energy_Wh / aircraft["battery_specific_energy_Wh_kg"],
        "total_mission_time_s": time,
        "total_ground_track_m": distance,
        "spiral_excess_distance_m": spiral_excess_distance,
    }


def optimize_course_climb(weight_N, wing, mission, aircraft, isa_density):
    """Select the lowest-energy climb that meets the climb-time requirement."""
    candidates = []
    # The 10-minute climb requirement comes from MISSION["time_budget_s"];
    # course_climb_time_limit_s can still override it explicitly when present.
    climb_time_limit_s = setting(
        mission, "course_climb_time_limit_s", setting(mission, "time_budget_s", 600.0)
    )
    for available_power_W in default_power_grid(aircraft):
        for equivalent_airspeed_m_s in default_eas_grid(weight_N, wing, aircraft):
            candidate = simulate_course_climb(
                weight_N,
                wing,
                mission,
                aircraft,
                isa_density,
                equivalent_airspeed_m_s,
                available_electrical_power_W=available_power_W,
            )
            if candidate["feasible"]:
                candidate["meets_time_limit"] = candidate["time_s"] <= climb_time_limit_s
                candidate["climb_time_margin_s"] = climb_time_limit_s - candidate["time_s"]
            else:
                candidate["meets_time_limit"] = False
                candidate["climb_time_margin_s"] = None
            candidates.append(candidate)

    feasible = [
        candidate
        for candidate in candidates
        if candidate["feasible"] and candidate["meets_time_limit"]
    ]
    if not feasible:
        climb_candidates = [candidate for candidate in candidates if candidate["feasible"]]
        if climb_candidates:
            closest = min(climb_candidates, key=lambda candidate: candidate["time_s"])
            closest["failure_reason"] = "No feasible climb candidate meets the 10 min climb time requirement."
            closest["climb_time_limit_s"] = climb_time_limit_s
            closest["complies_with_time_limit"] = False
            return closest, candidates
        return candidates[-1], candidates

    selected = min(feasible, key=lambda candidate: candidate["energy_Wh"])
    selected["climb_time_limit_s"] = climb_time_limit_s
    selected["complies_with_time_limit"] = True
    return selected, candidates


def write_course_climb_csv(path, selected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "altitude_m",
            "TAS_m_s",
            "EAS_m_s",
            "RC_s_m_s",
            "RC_m_s",
            "dV_dH",
            "CL",
            "power_required_propulsive_W",
            "power_available_propulsive_W",
            "selected_power_available_propulsive_W",
            "electrical_power_used_W",
            "usable_thrust_N",
            "thrust_limit_N",
            "thrust_to_weight",
            "delta_t_s",
            "delta_energy_Wh",
        ])
        for state in selected["states"]:
            writer.writerow([
                state["altitude_m"],
                state["TAS_m_s"],
                state["EAS_m_s"],
                state["steady_rate_of_climb_m_s"],
                state["rate_of_climb_m_s"],
                state["dV_dH"],
                state["CL"],
                state["power_required_propulsive_W"],
                state["power_available_propulsive_W"],
                state["selected_power_available_propulsive_W"],
                state["electrical_power_used_W"],
                state["usable_thrust_N"],
                state["thrust_limit_N"],
                state["thrust_to_weight"],
                state["delta_t_s"],
                state["delta_energy_Wh"],
            ])


def write_course_mission_summary(path, mission_result):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "total_load_energy_Wh": mission_result["total_load_energy_Wh"],
        "installed_battery_energy_Wh": mission_result["installed_battery_energy_Wh"],
        "battery_mass_kg": mission_result["battery_mass_kg"],
        "total_mission_time_s": mission_result["total_mission_time_s"],
        "total_ground_track_m": mission_result["total_ground_track_m"],
        "spiral_excess_distance_m": mission_result["spiral_excess_distance_m"],
        "takeoff_energy_Wh": mission_result["segments"]["vertical_takeoff"]["energy_Wh"],
        "transition_energy_Wh": mission_result["segments"]["transition"]["energy_Wh"],
        "climb_energy_Wh": mission_result["segments"]["wing_borne_climb"]["energy_Wh"],
        "cruise_energy_Wh": mission_result["segments"]["level_cruise"]["energy_Wh"],
        "hover_energy_Wh": mission_result["segments"]["mission_hover"]["energy_Wh"],
    }
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["quantity", "value"])
        for name, value in rows.items():
            writer.writerow([name, value])


def plot_course_climb(path, selected):
    states = selected["states"]
    altitude_km = [state["altitude_m"] / 1000.0 for state in states]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    fig.suptitle("Course-method constant-EAS climb", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        (
            f"EAS {selected['EAS_m_s']:.1f} m/s | "
            f"P_avail {selected['available_electrical_power_W'] / 1000.0:.1f} kW | "
            f"T/W cap {selected['max_thrust_to_weight']:.2f} | "
            f"time {selected['time_s'] / 60.0:.1f} min | "
            f"energy {selected['energy_Wh'] / 1000.0:.2f} kWh"
        ),
        ha="center",
        fontsize=10,
        color="#3d4752",
    )
    if not selected.get("complies_with_time_limit", True):
        fig.text(
            0.5,
            0.895,
            "Closest candidate shown: current power/thrust limits do not meet the 10 min climb requirement",
            ha="center",
            fontsize=9,
            color="#9b2226",
        )

    ax = axes[0, 0]
    ax.plot([state["time_s"] / 60.0 for state in states], altitude_km, color="#005f73", linewidth=2.3)
    ax.set_title("Altitude timeline")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot([state["TAS_m_s"] for state in states], altitude_km, color="#001219", linewidth=2.2, label="TAS")
    ax.plot([state["EAS_m_s"] for state in states], altitude_km, color="#ee9b00", linestyle="--", linewidth=2.0, label="EAS")
    ax.set_title("Constant EAS, increasing TAS")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("altitude [km]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot([state["steady_rate_of_climb_m_s"] for state in states], altitude_km, color="#bb3e03", linewidth=2.2, label="RC_s")
    ax.plot([state["rate_of_climb_m_s"] for state in states], altitude_km, color="#0a9396", linewidth=2.0, linestyle="--", label="RC")
    ax.set_title("Lecture rate-of-climb correction")
    ax.set_xlabel("rate of climb [m/s]")
    ax.set_ylabel("altitude [km]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot([state["power_required_propulsive_W"] / 1000.0 for state in states], altitude_km, color="#9b2226", linewidth=2.2, label="P_r")
    ax.plot([state["power_available_propulsive_W"] / 1000.0 for state in states], altitude_km, color="#3d405b", linestyle="--", linewidth=2.0, label="P_a")
    ax.set_title("Power available and required")
    ax.set_xlabel("propulsive power [kW]")
    ax.set_ylabel("altitude [km]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_course_mission_profile(path, mission_result):
    profile = mission_result["profile"]
    segments = mission_result["segments"]
    climb_states = mission_result["climb"]["states"]
    times_min = [point[0] / 60.0 for point in profile]
    distances_km = [point[1] / 1000.0 for point in profile]
    altitudes_km = [point[2] / 1000.0 for point in profile]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    fig.suptitle("Course-method mission energy profile", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        (
            f"Load energy {mission_result['total_load_energy_Wh'] / 1000.0:.2f} kWh | "
            f"Battery {mission_result['battery_mass_kg']:.1f} kg | "
            f"mission time {mission_result['total_mission_time_s'] / 60.0:.1f} min"
        ),
        ha="center",
        fontsize=10,
        color="#3d4752",
    )

    ax = axes[0, 0]
    ax.plot(times_min, altitudes_km, color="#005f73", linewidth=2.3)
    ax.fill_between(times_min, altitudes_km, color="#005f73", alpha=0.10)
    ax.set_title("Altitude timeline")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(distances_km, altitudes_km, color="#0a9396", linewidth=2.3)
    ax.fill_between(distances_km, altitudes_km, color="#0a9396", alpha=0.10)
    ax.set_title("Altitude over ground track")
    ax.set_xlabel("ground track [km]")
    ax.set_ylabel("altitude [km]")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    segment_names = ["vertical_takeoff", "transition", "wing_borne_climb", "level_cruise", "mission_hover"]
    labels = ["takeoff", "transition", "climb", "cruise", "hover"]
    energies = [segments[name]["energy_Wh"] / 1000.0 for name in segment_names]
    colors = ["#005f73", "#0a9396", "#ee9b00", "#ca6702", "#9b2226"]
    ax.bar(labels, energies, color=colors, width=0.65)
    ax.set_title("Segment energy")
    ax.set_ylabel("load energy [kWh]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(energies):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    altitude_km = [state["altitude_m"] / 1000.0 for state in climb_states]
    ax.plot([state["TAS_m_s"] for state in climb_states], altitude_km, color="#001219", linewidth=2.2, label="TAS")
    ax.plot([state["EAS_m_s"] for state in climb_states], altitude_km, color="#ee9b00", linestyle="--", linewidth=2.0, label="EAS")
    ax2 = ax.twiny()
    ax2.plot([state["electrical_power_used_W"] / 1000.0 for state in climb_states], altitude_km, color="#bb3e03", linestyle=":", linewidth=2.0, label="power")
    ax.set_title("Climb speed and power")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("altitude [km]")
    ax2.set_xlabel("electrical power [kW]", color="#bb3e03")
    ax2.tick_params(axis="x", colors="#bb3e03")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_course_climb_demo(output_dir="outputs"):
    """Run the course method with the current sizing inputs."""
    from simple_sizing import AIRCRAFT, MISSION, isa_density, wing_geometry

    output_dir = Path(output_dir)
    weight_N = AIRCRAFT["MTOW_kg"] * AIRCRAFT["g_m_s2"]
    wing = wing_geometry(weight_N, isa_density(MISSION["altitude_m"]))
    selected, candidates = optimize_course_climb(
        weight_N,
        wing,
        MISSION,
        AIRCRAFT,
        isa_density,
    )
    if not selected["feasible"]:
        raise RuntimeError(f"No feasible course-method climb: {selected['failure_reason']}")

    mission_result = build_course_mission(
        weight_N,
        wing,
        MISSION,
        AIRCRAFT,
        isa_density,
        selected,
    )
    write_course_climb_csv(output_dir / "course_climb_profile.csv", selected)
    write_course_mission_summary(output_dir / "course_mission_summary.csv", mission_result)
    plot_course_climb(output_dir / "course_climb_profile.png", selected)
    plot_course_mission_profile(output_dir / "course_mission_profile.png", mission_result)
    return {
        "selected": selected,
        "candidates": candidates,
        "mission": mission_result,
    }


def main():
    result = run_course_climb_demo()
    selected = result["selected"]
    mission = result["mission"]
    print("Course-method wing-borne climb")
    if not selected.get("complies_with_time_limit", True):
        print("  Requirement status: NOT MET")
        print(
            "  Current limits do not allow a 10 min climb; "
            "showing the fastest feasible candidate."
        )
    else:
        print("  Requirement status: MET")
    print(f"  EAS: {selected['EAS_m_s']:.2f} m/s")
    print(f"  Available electrical power: {selected['available_electrical_power_W'] / 1000.0:.2f} kW")
    print(f"  Average electrical power used: {selected['average_electrical_power_used_W'] / 1000.0:.2f} kW")
    print(f"  Max thrust-to-weight: {selected['max_thrust_to_weight']:.2f}")
    print(f"  Time: {selected['time_s'] / 60.0:.2f} min")
    print(f"  Energy: {selected['energy_Wh'] / 1000.0:.2f} kWh")
    print(f"  Distance: {selected['distance_m'] / 1000.0:.2f} km")
    print(f"  Mission load energy incl. hover/transition: {mission['total_load_energy_Wh'] / 1000.0:.2f} kWh")
    print(f"  Battery mass: {mission['battery_mass_kg']:.2f} kg")
    print("  Outputs written to: outputs")


if __name__ == "__main__":
    main()
