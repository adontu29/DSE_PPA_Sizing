

from __future__ import annotations

import math
from collections import Counter
from itertools import product


RHO_SEA_LEVEL = 1.225


def setting(values, name, default):
    return values[name] if name in values else default


def linspace(start, stop, count):
    if count == 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [float(start + i * step) for i in range(count)]


def default_speed_grid(minimum_speed_m_s):
    lower = 1.001 * minimum_speed_m_s
    upper = max(1.8 * lower, lower + 12.0)
    return linspace(lower, upper, 7)


def equivalent_to_true_airspeed(equivalent_speed_m_s, density_kg_m3):
    return equivalent_speed_m_s * math.sqrt(RHO_SEA_LEVEL / density_kg_m3)


def induced_drag_factor(aircraft):
    return 1.0 / (
        math.pi
        * aircraft["wing_aspect_ratio"]
        * aircraft["oswald_efficiency"]
    )


def forward_efficiency(aircraft):
    return setting(
        aircraft,
        "forward_flight_efficiency",
        setting(aircraft, "eta_prop", 0.75)
        * setting(aircraft, "eta_motor", 0.90)
        * setting(aircraft, "eta_ESC", 0.95),
    )


def permitted_lift_coefficient(aircraft):
    fraction_limit = (
        setting(aircraft, "mission_CL_limit_fraction", 0.90)
        * aircraft["wing_CL_max"]
    )
    stall_margin_deg = setting(aircraft, "cruise_stall_margin_deg", 2.0)
    margin_limit = aircraft["wing_CL_max"] - aircraft["wing_CL_alpha_per_rad"] * math.radians(stall_margin_deg)
    return min(fraction_limit, margin_limit) 


def transition_lift_coefficient_limit(aircraft):
    margin_n = setting(aircraft, "transition_stall_margin_n", 1.25)
    return aircraft["wing_CL_max"] / margin_n**2


def transition_completion_speed(weight_N, wing, density, aircraft):
    wing_lift_fraction = setting(aircraft, "transition_wing_lift_fraction_complete", 0.90)
    return math.sqrt(
        2.0 * wing_lift_fraction * weight_N
        / (density * wing["area_m2"] * transition_lift_coefficient_limit(aircraft))
    )


def aerodynamic_speed_limits(weight_N, wing, mission, aircraft, isa_density):
    rho_transition = isa_density(mission["vertical_takeoff_height_m"])
    rho_target = isa_density(mission["altitude_m"])
    CL_allowed = permitted_lift_coefficient(aircraft)

    def stall_speed(density, lift_coefficient):
        return math.sqrt(
            2.0 * weight_N
            / (density * wing["area_m2"] * lift_coefficient)
        )

    stall_EAS = stall_speed(RHO_SEA_LEVEL, aircraft["wing_CL_max"])
    minimum_climb_EAS = stall_speed(RHO_SEA_LEVEL, CL_allowed)
    stall_TAS_transition = stall_speed(rho_transition, aircraft["wing_CL_max"])
    minimum_transition_complete_TAS = transition_completion_speed(weight_N, wing, rho_transition, aircraft)
    minimum_aero_cruise_TAS = stall_speed(rho_target, CL_allowed)
    minimum_cruise_TAS = max(
        minimum_transition_complete_TAS,
        minimum_aero_cruise_TAS,
        setting(aircraft, "minimum_cruise_true_speed_m_s", 0.0),
    )

    return {
        "stall_EAS_m_s": stall_EAS,
        "minimum_climb_EAS_m_s": minimum_climb_EAS,
        "stall_TAS_transition_m_s": stall_TAS_transition,
        "minimum_transition_complete_TAS_m_s": minimum_transition_complete_TAS,
        "minimum_aerodynamic_cruise_TAS_m_s": minimum_aero_cruise_TAS,
        "minimum_cruise_TAS_m_s": minimum_cruise_TAS,
        "rho_target_kg_m3": rho_target,
        "rho_transition_kg_m3": rho_transition,
        "CL_allowed": CL_allowed,
    }


def transition_blending(weight_N, wing, density, mission, aircraft, cruise_true_speed_m_s):
    complete_speed = transition_completion_speed(weight_N, wing, density, aircraft)
    blend_start = (
        setting(aircraft, "transition_blend_start_fraction", 0.50)
        * complete_speed
    )
    acceleration = setting(aircraft, "transition_accel_m_s2", 1.0)
    exit_speed = complete_speed
    time_s = exit_speed / acceleration
    distance_m = exit_speed**2 / (2.0 * acceleration)

    wing_lift_fraction = setting(aircraft, "transition_wing_lift_fraction_complete", 0.90)
    q = 0.5 * density * complete_speed**2
    CL = wing_lift_fraction * weight_N / (q * wing["area_m2"])
    CD = aircraft["CD0"] + induced_drag_factor(aircraft) * CL**2
    drag = q * wing["area_m2"] * CD
    mass_kg = weight_N / aircraft["g_m_s2"]
    forward_force = drag + mass_kg * acceleration
    vertical_force = max(0.0, (1.0 - wing_lift_fraction) * weight_N)
    required_thrust = math.sqrt(forward_force**2 + vertical_force**2)
    available_thrust = setting(aircraft, "thrust_to_weight", 1.0) * weight_N
    thrust_margin = setting(aircraft, "transition_thrust_margin", 1.15)

    forward_power = forward_force * complete_speed / forward_efficiency(aircraft)
    vertical_power = aircraft["hover_power_W"] * (vertical_force / weight_N) ** 1.5
    peak_power = forward_power + vertical_power
    average_power = 0.5 * (aircraft["hover_power_W"] + peak_power)
    maximum_power = setting(aircraft, "max_affordable_electrical_power_W", 18000.0)
    required_power_margin = setting(aircraft, "minimum_power_margin_fraction", 0.05) * maximum_power

    samples = []
    for speed in linspace(0.0, exit_speed, setting(aircraft, "transition_sample_count", 9)):
        xi = (speed - blend_start) / (exit_speed - blend_start)
        xi = min(1.0, max(0.0, xi))
        alpha_fixed_wing = 3.0 * xi**2 - 2.0 * xi**3
        samples.append({
            "speed_m_s": speed,
            "alpha_hover": 1.0 - alpha_fixed_wing,
            "alpha_fixed_wing": alpha_fixed_wing,
        })

    return {
        "V_blend_start_m_s": blend_start,
        "V_blend_end_m_s": exit_speed,
        "V_exit_m_s": exit_speed,
        "t_transition_s": time_s,
        "distance_transition_m": distance_m,
        "transition_complete_speed_m_s": complete_speed,
        "cruise_speed_margin_over_complete_m_s": cruise_true_speed_m_s - complete_speed,
        "wing_lift_fraction_complete": wing_lift_fraction,
        "CL_complete": CL,
        "CL_limit": transition_lift_coefficient_limit(aircraft),
        "drag_N": drag,
        "forward_force_N": forward_force,
        "vertical_force_N": vertical_force,
        "required_thrust_N": required_thrust,
        "available_thrust_N": available_thrust,
        "required_thrust_with_margin_N": thrust_margin * required_thrust,
        "peak_electrical_power_W": peak_power,
        "average_electrical_power_W": average_power,
        "power_margin_W": maximum_power - peak_power,
        "feasible": (
            thrust_margin * required_thrust <= available_thrust
            and peak_power <= maximum_power - required_power_margin
        ),
        "failure_reason": (
            "Transition thrust requirement above installed thrust."
            if thrust_margin * required_thrust > available_thrust
            else "Transition power requirement above available power."
            if peak_power > maximum_power - required_power_margin
            else None
        ),
        "schedule": samples,
    }


def takeoff_and_transition(weight_N, wing, mission, aircraft, isa_density, cruise_true_speed_m_s):
    rho_transition = isa_density(mission["vertical_takeoff_height_m"])
    stall_speed = math.sqrt(
        2.0 * weight_N
        / (rho_transition * wing["area_m2"] * aircraft["wing_CL_max"])
    )
    transition = transition_blending(
        weight_N,
        wing,
        rho_transition,
        mission,
        aircraft,
        cruise_true_speed_m_s,
    )

    takeoff_time = (
        mission["vertical_takeoff_height_m"]
        / mission["vertical_takeoff_rate_m_s"]
    )
    takeoff_energy = aircraft["hover_power_W"] * takeoff_time / 3600.0
    transition_energy = (
        transition["average_electrical_power_W"]
        * transition["t_transition_s"]
        / 3600.0
    )

    return {
        "takeoff_time_s": takeoff_time,
        "takeoff_energy_Wh": takeoff_energy,
        "transition_energy_Wh": transition_energy,
        "transition_altitude_m": mission["vertical_takeoff_height_m"],
        "transition": transition,
        "stall_speed_transition_m_s": stall_speed,
    }


def wing_segment_forces(weight_N, wing, aircraft, density, speed_now, speed_next, climb_angle, delta_h, delta_s):
    speed = 0.5 * (speed_now + speed_next)
    mass_kg = weight_N / aircraft["g_m_s2"]
    lift = weight_N * math.cos(climb_angle)
    q = 0.5 * density * speed**2
    CL = lift / (q * wing["area_m2"])
    CD = aircraft["CD0"] + induced_drag_factor(aircraft) * CL**2
    drag = q * wing["area_m2"] * CD

    potential_energy = weight_N * delta_h
    kinetic_energy = 0.5 * mass_kg * (speed_next**2 - speed_now**2)
    required_thrust = drag + (potential_energy + kinetic_energy) / delta_s

    return {
        "speed_m_s": speed,
        "CL": CL,
        "CD": CD,
        "drag_N": drag,
        "required_thrust_N": required_thrust,
        "potential_energy_J": potential_energy,
        "kinetic_energy_J": kinetic_energy,
        "mechanical_energy_J": potential_energy + kinetic_energy,
    }


def add_power_model(forces, aircraft):
    propulsive_power = max(0.0, forces["required_thrust_N"] * forces["speed_m_s"])
    electrical_power = propulsive_power / forward_efficiency(aircraft)
    state = dict(forces)
    state.update({
        "propulsive_power_W": propulsive_power,
        "electrical_power_W": electrical_power,
        "power_margin_W": setting(
            aircraft,
            "max_affordable_electrical_power_W",
            20000.0,
        ) - electrical_power,
    })
    return state


def finish_segment(states):
    time_s = sum(state["delta_t_s"] for state in states)
    distance_m = sum(state["delta_x_m"] for state in states)
    energy_Wh = sum(state["delta_electrical_energy_Wh"] for state in states)
    return {
        "feasible": True,
        "failure_reason": None,
        "states": states,
        "time_s": time_s,
        "distance_m": distance_m,
        "energy_Wh": energy_Wh,
        "average_electrical_power_W": energy_Wh * 3600.0 / time_s if time_s > 0.0 else 0.0,
    }


def failed_segment(reason, states):
    segment = finish_segment(states)
    segment["feasible"] = False
    segment["failure_reason"] = reason
    return segment


def simulate_wing_borne_climb(weight_N, wing, mission, aircraft, isa_density, equivalent_speed, climb_angle, initial_speed):
    states = []
    altitude = mission["vertical_takeoff_height_m"]
    end_altitude = mission["altitude_m"]
    altitude_step = setting(mission, "altitude_step_m", 100.0)
    required_margin = (
        setting(aircraft, "minimum_power_margin_fraction", 0.05)
        * setting(aircraft, "max_affordable_electrical_power_W", 20000.0)
    )
    CL_allowed = permitted_lift_coefficient(aircraft)

    density_start = isa_density(altitude)
    scheduled_speed = equivalent_to_true_airspeed(equivalent_speed, density_start)
    speed_now = max(initial_speed, scheduled_speed)

    while altitude < end_altitude - 1e-9:
        altitude_next = min(altitude + altitude_step, end_altitude)
        altitude_mid = 0.5 * (altitude + altitude_next)
        density_mid = isa_density(altitude_mid)
        density_next = isa_density(altitude_next)
        speed_next = equivalent_to_true_airspeed(equivalent_speed, density_next)

        delta_h = altitude_next - altitude
        delta_s = delta_h / math.sin(climb_angle)
        delta_x = delta_s * math.cos(climb_angle)
        forces = wing_segment_forces(
            weight_N,
            wing,
            aircraft,
            density_mid,
            speed_now,
            speed_next,
            climb_angle,
            delta_h,
            delta_s,
        )
        state = add_power_model(forces, aircraft)
        delta_t = delta_s / state["speed_m_s"]
        state.update({
            "segment": "wing_borne_climb",
            "altitude_start_m": altitude,
            "altitude_m": altitude_next,
            "altitude_mid_m": altitude_mid,
            "EAS_m_s": equivalent_speed,
            "climb_angle_deg": math.degrees(climb_angle),
            "rate_of_climb_m_s": delta_h / delta_t,
            "delta_h_m": delta_h,
            "delta_s_m": delta_s,
            "delta_x_m": delta_x,
            "delta_t_s": delta_t,
            "delta_electrical_energy_Wh": state["electrical_power_W"] * delta_t / 3600.0,
            "CL_margin": CL_allowed - state["CL"],
        })
        states.append(state)

        if state["CL_margin"] < 0.0:
            return failed_segment(f"CL limit exceeded at {altitude_mid:.0f} m.", states)
        if state["power_margin_W"] < required_margin:
            return failed_segment(f"Power margin too small at {altitude_mid:.0f} m.", states)

        altitude = altitude_next
        speed_now = speed_next

    return finish_segment(states)


def simulate_level_cruise(weight_N, wing, mission, aircraft, isa_density, true_speed, distance):
    if distance <= 0.0:
        return finish_segment([])

    states = []
    density = isa_density(mission["altitude_m"])
    required_margin = (
        setting(aircraft, "minimum_power_margin_fraction", 0.05)
        * setting(aircraft, "max_affordable_electrical_power_W", 20000.0)
    )
    forces = wing_segment_forces(
        weight_N,
        wing,
        aircraft,
        density,
        true_speed,
        true_speed,
        0.0,
        0.0,
        distance,
    )
    state = add_power_model(forces, aircraft)
    delta_t = distance / true_speed
    state.update({
        "segment": "level_cruise",
        "altitude_start_m": mission["altitude_m"],
        "altitude_m": mission["altitude_m"],
        "altitude_mid_m": mission["altitude_m"],
        "EAS_m_s": true_speed * math.sqrt(density / RHO_SEA_LEVEL),
        "climb_angle_deg": 0.0,
        "rate_of_climb_m_s": 0.0,
        "delta_h_m": 0.0,
        "delta_s_m": distance,
        "delta_x_m": distance,
        "delta_t_s": delta_t,
        "delta_electrical_energy_Wh": state["electrical_power_W"] * delta_t / 3600.0,
        "CL_margin": permitted_lift_coefficient(aircraft) - state["CL"],
    })
    states.append(state)

    if state["CL_margin"] < 0.0:
        return failed_segment("CL limit exceeded in level cruise.", states)
    if state["power_margin_W"] < required_margin:
        return failed_segment("Power margin too small in level cruise.", states)
    return finish_segment(states)


def segment_summary(time_s, energy_Wh, distance_m=0.0):
    return {
        "time_s": time_s,
        "energy_Wh": energy_Wh,
        "distance_m": distance_m,
        "average_electrical_power_W": energy_Wh * 3600.0 / time_s if time_s > 0.0 else 0.0,
    }


def complete_candidate(weight_N, wing, mission, aircraft, isa_density, equivalent_speed, climb_angle, cruise_speed):
    if math.degrees(climb_angle) > setting(aircraft, "max_fixed_wing_climb_angle_deg", 30.0):
        return {"feasible": False, "failure_reason": "Climb angle above selected limit."}

    takeoff_transition = takeoff_and_transition(
        weight_N,
        wing,
        mission,
        aircraft,
        isa_density,
        cruise_speed,
    )
    transition = takeoff_transition["transition"]
    if not transition["feasible"]:
        return {"feasible": False, "failure_reason": transition["failure_reason"]}
    if cruise_speed < transition["transition_complete_speed_m_s"]:
        return {"feasible": False, "failure_reason": "Cruise speed below transition completion speed."}

    climb = simulate_wing_borne_climb(
        weight_N,
        wing,
        mission,
        aircraft,
        isa_density,
        equivalent_speed,
        climb_angle,
        transition["V_exit_m_s"],
    )
    if not climb["feasible"]:
        return {"feasible": False, "failure_reason": climb["failure_reason"]}

    ground_before_cruise = transition["distance_transition_m"] + climb["distance_m"]
    remaining_cruise_distance = mission["range_m"] - ground_before_cruise
    if remaining_cruise_distance < 0.0 and not setting(mission, "allow_spiral_climb", True):
        return {"feasible": False, "failure_reason": "Climb ground track exceeds mission range."}

    spiral_excess = max(0.0, -remaining_cruise_distance)
    remaining_cruise_distance = max(0.0, remaining_cruise_distance)
    cruise = simulate_level_cruise(
        weight_N,
        wing,
        mission,
        aircraft,
        isa_density,
        cruise_speed,
        remaining_cruise_distance,
    )
    if not cruise["feasible"]:
        return {"feasible": False, "failure_reason": cruise["failure_reason"]}

    outbound_time = (
        takeoff_transition["takeoff_time_s"]
        + transition["t_transition_s"]
        + climb["time_s"]
        + cruise["time_s"]
    )
    if outbound_time > mission["time_budget_s"]:
        return {"feasible": False, "failure_reason": "Outbound time budget exceeded."}

    hover_energy = aircraft["hover_power_W"] * mission["hover_time_s"] / 3600.0
    segment_summaries = {
        "vertical_takeoff": segment_summary(
            takeoff_transition["takeoff_time_s"],
            takeoff_transition["takeoff_energy_Wh"],
        ),
        "transition": segment_summary(
            transition["t_transition_s"],
            takeoff_transition["transition_energy_Wh"],
            transition["distance_transition_m"],
        ),
        "wing_borne_climb": segment_summary(
            climb["time_s"],
            climb["energy_Wh"],
            climb["distance_m"],
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

    states = climb["states"] + cruise["states"]
    power_values = (
        [state["electrical_power_W"] for state in states]
        + [aircraft["hover_power_W"], transition["peak_electrical_power_W"]]
    )
    total_energy = sum(segment["energy_Wh"] for segment in segment_summaries.values())

    return {
        "feasible": True,
        "failure_reason": None,
        "optimized_climb_EAS_m_s": equivalent_speed,
        "optimized_climb_angle_rad": climb_angle,
        "optimized_climb_angle_deg": math.degrees(climb_angle),
        "cruise_true_speed_m_s": cruise_speed,
        "takeoff_transition": takeoff_transition,
        "climb": climb,
        "cruise": cruise,
        "states": states,
        "segment_summaries": segment_summaries,
        "outbound_time_s": outbound_time,
        "total_mission_time_s": outbound_time + mission["hover_time_s"],
        "total_electrical_energy_Wh": total_energy,
        "peak_electrical_power_W": max(power_values),
        "maximum_wingborne_electrical_power_W": max(
            [state["electrical_power_W"] for state in states] or [0.0]
        ),
        "spiral_excess_ground_track_distance_m": spiral_excess,
        "climb_horizontal_distance_m": climb["distance_m"],
        "level_cruise_distance_m": cruise["distance_m"],
        "total_ground_track_distance_m": ground_before_cruise + remaining_cruise_distance,
    }


def optimize_constant_eas_mission(weight_N, wing, mission, aircraft, isa_density):
    limits = aerodynamic_speed_limits(weight_N, wing, mission, aircraft, isa_density)
    eas_grid = default_speed_grid(limits["minimum_climb_EAS_m_s"])
    cruise_grid = default_speed_grid(limits["minimum_cruise_TAS_m_s"])
    max_angle = setting(aircraft, "max_fixed_wing_climb_angle_deg", 30.0)
    start_angle = min(20.0, max_angle)
    angle_grid_deg = []
    angle = start_angle
    while angle <= max_angle + 1e-9:
        angle_grid_deg.append(angle)
        angle += 2.5

    best = None
    failures = Counter()
    expansion_history = []

    for expansion in range(5):
        best = None
        failures = Counter()
        for eas, angle_deg, cruise_speed in product(eas_grid, angle_grid_deg, cruise_grid):
            candidate = complete_candidate(
                weight_N,
                wing,
                mission,
                aircraft,
                isa_density,
                eas,
                math.radians(angle_deg),
                cruise_speed,
            )
            if not candidate["feasible"]:
                failures[candidate["failure_reason"]] += 1
                continue
            if (
                best is None
                or candidate["total_electrical_energy_Wh"] < best["total_electrical_energy_Wh"]
            ):
                best = candidate

        expand_eas = best is None or abs(best["optimized_climb_EAS_m_s"] - eas_grid[-1]) < 1e-9
        expand_cruise = best is None or abs(best["cruise_true_speed_m_s"] - cruise_grid[-1]) < 1e-9
        if expansion == 4 or not (expand_eas or expand_cruise):
            break
        if expand_eas:
            eas_grid = linspace(eas_grid[0], 1.5 * eas_grid[-1], 7)
        if expand_cruise:
            cruise_grid = linspace(cruise_grid[0], 1.5 * cruise_grid[-1], 7)
        expansion_history.append({
            "expansion": expansion + 1,
            "expanded_climb_EAS": expand_eas,
            "expanded_cruise_TAS": expand_cruise,
            "new_climb_EAS_max_m_s": eas_grid[-1],
            "new_cruise_TAS_max_m_s": cruise_grid[-1],
        })

    grid = {
        "climb_EAS_m_s": eas_grid,
        "climb_angle_deg": angle_grid_deg,
        "cruise_TAS_m_s": cruise_grid,
        "aerodynamic_speed_limits": limits,
        "expansion_history": expansion_history,
        "failure_counts": dict(failures),
    }
    if best is None:
        reason = failures.most_common(1)[0][0] if failures else "No mission candidates were feasible."
        return {
            "feasible": False,
            "failure_reason": reason,
            "grid": grid,
        }

    best["grid"] = grid
    return best


def battery_from_segments(segment_summaries, aircraft):
    load_energy_Wh = sum(segment["energy_Wh"] for segment in segment_summaries.values())
    installed_energy_Wh = (
        load_energy_Wh
        / aircraft["battery_efficiency"]
        / aircraft["battery_usable_fraction"]
    )
    return {
        "load_energy_Wh": load_energy_Wh,
        "installed_energy_Wh": installed_energy_Wh,
        "battery_mass_kg": installed_energy_Wh / aircraft["battery_specific_energy_Wh_kg"],
    }


def level_flight_reference(weight_N, wing, aircraft, isa_density, cruise_speed):
    density = isa_density(aircraft["mission_altitude_m"])
    q = 0.5 * density * cruise_speed**2
    CL = weight_N / (q * wing["area_m2"])
    CD = aircraft["CD0"] + induced_drag_factor(aircraft) * CL**2
    drag = q * wing["area_m2"] * CD
    return {
        "CL": CL,
        "CD": CD,
        "drag_N": drag,
        "electrical_power_W": drag * cruise_speed / forward_efficiency(aircraft),
    }


def build_altitude_profile(mission_result, mission):
    profile = [(0.0, 0.0)]
    time = mission_result["takeoff_transition"]["takeoff_time_s"]
    profile.append((time, mission["vertical_takeoff_height_m"]))
    time += mission_result["takeoff_transition"]["transition"]["t_transition_s"]
    profile.append((time, mission["vertical_takeoff_height_m"]))
    for state in mission_result["climb"]["states"]:
        time += state["delta_t_s"]
        profile.append((time, state["altitude_m"]))
    if mission_result["cruise"]["time_s"] > 0.0:
        time += mission_result["cruise"]["time_s"]
        profile.append((time, mission["altitude_m"]))
    time += mission["hover_time_s"]
    profile.append((time, mission["altitude_m"]))
    return profile


def run_mission_energy(weight_N, wing, mission, aircraft, isa_density):
    aircraft_with_altitude = dict(aircraft)
    aircraft_with_altitude["mission_altitude_m"] = mission["altitude_m"]
    mission_result = optimize_constant_eas_mission(
        weight_N,
        wing,
        mission,
        aircraft_with_altitude,
        isa_density,
    )
    if not mission_result["feasible"]:
        raise RuntimeError(f"No feasible mission profile: {mission_result['failure_reason']}")

    battery = battery_from_segments(
        mission_result["segment_summaries"],
        aircraft_with_altitude,
    )
    cruise_reference = level_flight_reference(
        weight_N,
        wing,
        aircraft_with_altitude,
        isa_density,
        mission_result["cruise_true_speed_m_s"],
    )

    segments = mission_result["segment_summaries"]
    return {
        "drag_N": cruise_reference["drag_N"],
        "CD_trim": cruise_reference["CD"],
        "CL_cruise": cruise_reference["CL"],
        "climb_power_W": mission_result["maximum_wingborne_electrical_power_W"],
        "peak_electrical_power_W": mission_result["peak_electrical_power_W"],
        "total_energy_Wh": battery["load_energy_Wh"],
        "installed_battery_energy_Wh": battery["installed_energy_Wh"],
        "battery_mass_kg": battery["battery_mass_kg"],
        "profile": build_altitude_profile(mission_result, mission),
        "segment_summaries": segments,
        "states": mission_result["states"],
        "cruise_true_speed_m_s": mission_result["cruise_true_speed_m_s"],
        "optimized_climb_EAS_m_s": mission_result["optimized_climb_EAS_m_s"],
        "optimized_climb_angle_deg": mission_result["optimized_climb_angle_deg"],
        "transition_complete_speed_m_s": mission_result["takeoff_transition"]["transition"]["transition_complete_speed_m_s"],
        "transition_peak_electrical_power_W": mission_result["takeoff_transition"]["transition"]["peak_electrical_power_W"],
        "transition_required_thrust_N": mission_result["takeoff_transition"]["transition"]["required_thrust_N"],
        "transition_available_thrust_N": mission_result["takeoff_transition"]["transition"]["available_thrust_N"],
        "outbound_time_s": mission_result["outbound_time_s"],
        "total_mission_time_s": mission_result["total_mission_time_s"],
        "climb_horizontal_distance_m": mission_result["climb_horizontal_distance_m"],
        "level_cruise_distance_m": mission_result["level_cruise_distance_m"],
        "spiral_excess_ground_track_distance_m": mission_result["spiral_excess_ground_track_distance_m"],
        "mission_grid": mission_result["grid"],
        "vertical_takeoff_Wh": segments["vertical_takeoff"]["energy_Wh"],
        "transition_Wh": segments["transition"]["energy_Wh"],
        "wing_borne_climb_Wh": segments["wing_borne_climb"]["energy_Wh"],
        "level_cruise_Wh": segments["level_cruise"]["energy_Wh"],
        "mission_hover_Wh": segments["mission_hover"]["energy_Wh"],
    }
