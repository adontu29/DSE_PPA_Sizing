"""Reduced-order tail-sitter transition simulation -> maximum stall speed.

A 2-DOF point-mass model in the vertical plane (Stone & Clarke inspired). State =
(u, w, x, h): horizontal and vertical velocity, downrange and height. The body
(= thrust = rotor) axis is held at a prescribed pitch attitude theta(t) above the
horizon; thrust T = (T/W)W acts along it and is throttled by an altitude-hold
controller. The angle of attack is alpha = theta - gamma with the flight-path
angle gamma = atan2(w, u); lift and drag follow from a Viterna full-range CL/CD
so the back transition can pitch up through the stall. The model is integrated
forward (explicit Euler) and the stall speed is bisected to find the largest Vs
for which the leg still meets its limits.

Why the cap is size-independent: dividing the equations of motion by the mass,
the thrust term is (T/W)*g and every aerodynamic term carries S_w/m =
2*g/(rho0*Vs^2*CL_max) (from the EAS stall definition). Neither depends on the
absolute weight or wing area, so for fixed T/W, CL_max, drag, altitude, canard
ratio and the speed factors the trajectory -- and hence the cap -- is a pure
function of Vs. That makes the simulation a genuine *generator* of the maximum
allowable stall speed, directly comparable to the candidate's actual stall EAS.

The binding (most demanding) of the forward and back legs sets the wing-area
lower bound, since the smaller EAS cap forces the larger wing.
"""

from __future__ import annotations

import math
from functools import lru_cache

from sizing.inputs import AIRCRAFT, MISSION
from sizing.atmosphere import isa_density


def _viterna_full_range(alpha, cl_alpha, cl_max, alpha0L, aspect_ratio, oswald, cd0_profile):
    """Section-to-surface CL, CD over the full alpha range (attached + post-stall).

    Below the stall angle the surface is linear (CL = CL_alpha*(alpha-alpha0L)) with
    induced drag CL^2/(pi*AR*e) on top of cd0_profile. Beyond the stall angle the
    Viterna-Corrigan flat-plate extrapolation is used (CD_max ~ 1.1 for a finite
    plate). Only the positive-alpha branch is modelled in detail; transition
    manoeuvres stay at alpha >= 0, and the linear law covers small negative ones.
    """
    alpha_stall = alpha0L + cl_max / cl_alpha
    if alpha <= alpha_stall:
        cl = cl_alpha * (alpha - alpha0L)
        cd = cd0_profile + cl * cl / (math.pi * aspect_ratio * oswald)
        return cl, cd

    cd_max = 1.11 + 0.018 * aspect_ratio
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    sin_s, cos_s = math.sin(alpha_stall), math.cos(alpha_stall)
    cos_s = cos_s if abs(cos_s) > 1e-6 else 1e-6
    sin_a = sin_a if abs(sin_a) > 1e-6 else 1e-6
    cl_stall = cl_max
    cd_stall = cd0_profile + cl_stall * cl_stall / (math.pi * aspect_ratio * oswald)
    a1 = cd_max / 2.0
    a2 = (cl_stall - cd_max * sin_s * cos_s) * sin_s / (cos_s * cos_s)
    b1 = cd_max
    b2 = (cd_stall - cd_max * sin_s * sin_s) / cos_s
    cl = a1 * math.sin(2.0 * alpha) + a2 * cos_a * cos_a / sin_a
    cd = b1 * sin_a * sin_a + b2 * cos_a
    return cl, cd


def _transition_sim_context(leg, area_ratio):
    """Hashable bundle of every constant the leg simulation needs (the cache key).

    Everything here is fixed within an XFOIL outer pass except the canard area ratio,
    so caching the cap on this tuple collapses the per-candidate calls to a handful
    of unique evaluations. The canard ratio is rounded coarsely (its effect on the
    transition is second order) to keep the cache small.
    """
    g = AIRCRAFT["g_m_s2"]
    rho0 = isa_density(0.0)
    if leg == "forward":
        altitude = AIRCRAFT["forward_transition_altitude_m"]
        thrust_to_weight = AIRCRAFT["forward_transition_thrust_to_weight"]
    else:
        altitude = AIRCRAFT["back_transition_altitude_m"]
        if altitude is None:
            altitude = MISSION["altitude_m"]
        thrust_to_weight = AIRCRAFT["thrust_to_weight"]

    cl_max_w = AIRCRAFT["wing_CL_max"]
    cl_alpha_w = AIRCRAFT["wing_CL_alpha_per_rad"]
    alpha0L_w = -AIRCRAFT["wing_CL0"] / cl_alpha_w
    cl_max_c = AIRCRAFT["canard_CL_max"]
    cl_alpha_c = AIRCRAFT["canard_CL_alpha_per_rad"]
    alpha0L_c = 0.0          # canard airfoil assumed symmetric (NACA0012)

    return (
        leg,
        round(AIRCRAFT["transition_sim_time_step_s"], 6),
        round(AIRCRAFT["transition_sim_max_time_s"], 3),
        round(g, 6),
        round(rho0, 6),
        round(isa_density(altitude), 6),
        round(math.radians(AIRCRAFT["transition_sim_pitch_rate_deg_s"]), 6),
        round(AIRCRAFT["transition_sim_velocity_epsilon_m_s"], 4),
        round(cl_max_w, 5), round(cl_alpha_w, 5), round(alpha0L_w, 6),
        round(AIRCRAFT["wing_aspect_ratio"], 4), round(AIRCRAFT["oswald_efficiency"], 4),
        round(AIRCRAFT["CD0"], 5),
        round(cl_max_c, 5), round(cl_alpha_c, 5), round(alpha0L_c, 6),
        round(AIRCRAFT["canard_aspect_ratio"], 4),
        round(AIRCRAFT.get("canard_oswald_efficiency", AIRCRAFT["oswald_efficiency"]), 4),
        round(area_ratio, 2),
        round(thrust_to_weight, 4),
        round(AIRCRAFT["transition_sim_altitude_hold_gain"], 4),
        round(AIRCRAFT["forward_transition_climbout_factor"], 4),
        round(AIRCRAFT["forward_transition_final_pitch_min_deg"], 3),
        round(AIRCRAFT["forward_transition_final_pitch_max_deg"], 3),
        round(AIRCRAFT["forward_transition_final_climb_angle_min_deg"], 3),
        round(AIRCRAFT["forward_transition_start_height_m"], 3),
        round(AIRCRAFT["forward_transition_min_height_m"], 3),
        None if AIRCRAFT["forward_transition_max_alpha_deg"] is None
        else round(AIRCRAFT["forward_transition_max_alpha_deg"], 3),
        round(AIRCRAFT["forward_transition_sim_time_limit_s"], 3),
        round(AIRCRAFT["forward_transition_sim_distance_limit_m"], 3),
        round(AIRCRAFT["back_transition_approach_speed_factor"], 4),
        round(AIRCRAFT["back_transition_pitch_max_deg"], 3),
        round(AIRCRAFT["back_transition_capture_speed_m_s"], 3),
        round(AIRCRAFT["back_transition_max_alpha_deg"], 3),
        round(AIRCRAFT["back_transition_height_band_up_m"], 3),
        round(AIRCRAFT["back_transition_height_band_down_m"], 3),
        round(AIRCRAFT["back_transition_sim_time_limit_s"], 3),
        round(AIRCRAFT["back_transition_sim_distance_limit_m"], 3),
    )


# Layout of the aero sub-tuple sliced out of the context for the coefficient call.
_AERO_SLICE = slice(8, 20)   # cl_max_w ... area_ratio (see _transition_sim_context)


def _transition_system_coefficients(alpha, aero):
    """Aircraft-level CL, CD referenced to the wing area (wing + canard).

    `aero` = (cl_max_w, cl_alpha_w, alpha0L_w, ar_w, e_w, cd0, cl_max_c, cl_alpha_c,
    alpha0L_c, ar_c, e_c, area_ratio). All parasite drag is charged once to the wing
    term; the canard term carries induced/post-stall drag only (cd0_profile = 0),
    scaled by the area ratio.
    """
    (
        cl_max_w, cl_alpha_w, alpha0L_w, ar_w, e_w, cd0,
        cl_max_c, cl_alpha_c, alpha0L_c, ar_c, e_c, area_ratio,
    ) = aero
    cl_w, cd_w = _viterna_full_range(alpha, cl_alpha_w, cl_max_w, alpha0L_w, ar_w, e_w, cd0)
    cl_c, cd_c = _viterna_full_range(alpha, cl_alpha_c, cl_max_c, alpha0L_c, ar_c, e_c, 0.0)
    cl = cl_w + area_ratio * cl_c
    cd = cd_w + area_ratio * cd_c
    return cl, cd


@lru_cache(maxsize=8192)
def _simulate_transition_leg(ctx, vs_eas_m_s):
    """Integrate one transition leg at a trial stall speed; return a result dict.

    `ctx` is the constant bundle from _transition_sim_context; `vs_eas_m_s` is the
    trial stall speed (EAS). The leg ("forward"/"back") is ctx[0]. The result holds
    success, the failure reasons, and every requested diagnostic (time, distance,
    max alpha, min height, final speed/pitch/flight-path angle).
    """
    (
        leg, dt, max_time, g, rho0, rho, pitch_rate, v_eps,
        cl_max_w, cl_alpha_w, alpha0L_w, ar_w, e_w, cd0,
        cl_max_c, cl_alpha_c, alpha0L_c, ar_c, e_c, area_ratio,
        thrust_to_weight, altitude_gain,
        *_cache_key_settings,
    ) = ctx
    aero = ctx[_AERO_SLICE]

    # S_w/m from the EAS stall definition (mass/size independent). All aerodynamic
    # accelerations are q * coeff * (S_w/m); thrust acceleration is (T/W)*g.
    s_over_m = 2.0 * g / (rho0 * vs_eas_m_s * vs_eas_m_s * cl_max_w)
    thrust_acc_max = thrust_to_weight * g
    vs_tas = vs_eas_m_s * math.sqrt(rho0 / rho)
    wing_stall_alpha = alpha0L_w + cl_max_w / cl_alpha_w
    half_pi = 0.5 * math.pi
    sin_floor = math.sin(math.radians(3.0))   # guard the 1/sin(theta) thrust solve

    if leg == "forward":
        target_speed = AIRCRAFT["forward_transition_climbout_factor"] * vs_tas
        theta = half_pi                                  # start vertical (hover)
        theta_end = math.radians(AIRCRAFT["forward_transition_final_pitch_min_deg"])
        u, w = 0.0, 0.0                                   # from a stationary hover
        h0 = AIRCRAFT["forward_transition_start_height_m"]
        max_alpha_limit = AIRCRAFT["forward_transition_max_alpha_deg"]
        max_alpha_limit = (
            wing_stall_alpha if max_alpha_limit is None else math.radians(max_alpha_limit)
        )
        time_limit = AIRCRAFT["forward_transition_sim_time_limit_s"]
        distance_limit = AIRCRAFT["forward_transition_sim_distance_limit_m"]
    else:
        approach_speed = AIRCRAFT["back_transition_approach_speed_factor"] * vs_tas
        # Trim alpha at entry (level wing-borne flight). The required system CL is
        # CL_max_w / approach_factor^2, independent of Vs; invert the linear law.
        cl_req = cl_max_w / AIRCRAFT["back_transition_approach_speed_factor"] ** 2
        denom = cl_alpha_w + area_ratio * cl_alpha_c
        alpha_trim = (
            cl_req + cl_alpha_w * alpha0L_w + area_ratio * cl_alpha_c * alpha0L_c
        ) / denom
        theta = alpha_trim                               # start level (gamma = 0)
        theta_end = math.radians(AIRCRAFT["back_transition_pitch_max_deg"])
        u, w = approach_speed, 0.0
        h0 = 0.0
        capture_speed = AIRCRAFT["back_transition_capture_speed_m_s"]
        max_alpha_limit = math.radians(AIRCRAFT["back_transition_max_alpha_deg"])
        time_limit = AIRCRAFT["back_transition_sim_time_limit_s"]
        distance_limit = AIRCRAFT["back_transition_sim_distance_limit_m"]

    x, h, t = 0.0, h0, 0.0
    max_alpha = 0.0
    min_height = h0
    max_height = h0
    alpha = theta
    gamma = 0.0
    thrust_saturated = False
    captured = False
    n_steps = int(max_time / dt) + 1

    for _ in range(n_steps):
        speed = math.hypot(u, w)
        if speed > v_eps:
            gamma = math.atan2(w, u)
            alpha = theta - gamma
            ux, wz = u / speed, w / speed
            cl, cd = _transition_system_coefficients(alpha, aero)
            q_dyn = 0.5 * rho * speed * speed
            lift_acc = q_dyn * cl * s_over_m            # = L/m
            drag_acc = q_dyn * cd * s_over_m            # = D/m
        else:
            gamma = half_pi if leg == "forward" else 0.0
            alpha = 0.0
            ux, wz = (0.0, 1.0) if leg == "forward" else (1.0, 0.0)
            lift_acc = drag_acc = 0.0

        # Thrust controller: throttle (within the available T/W) to hold altitude,
        # i.e. command the vertical acceleration toward -gain*w (drive w -> 0). Solve
        # the vertical EOM az = T*sin(theta) - D*wz + L*ux - g for the thrust needed.
        az_target = -altitude_gain * w
        thrust_acc = (
            az_target + g + drag_acc * wz - lift_acc * ux
        ) / max(math.sin(theta), sin_floor)
        if thrust_acc < 0.0:
            thrust_acc = 0.0
        if thrust_acc > thrust_acc_max:
            thrust_acc = thrust_acc_max
            thrust_saturated = True

        ax = thrust_acc * math.cos(theta) - drag_acc * ux - lift_acc * wz
        az = thrust_acc * math.sin(theta) - drag_acc * wz + lift_acc * ux - g

        # The wing is "aerodynamically engaged" once it carries a fair share of the
        # weight; max alpha is tracked over that regime (forward) or whenever there is
        # appreciable airspeed (back), so the near-hover geometric alpha at V~0 (where
        # the wing sees no dynamic pressure) does not pollute the reported maximum.
        engaged = lift_acc > 0.2 * g
        if speed > v_eps and (leg == "back" or engaged):
            max_alpha = max(max_alpha, alpha)

        u += ax * dt
        w += az * dt
        x += u * dt
        h += w * dt
        t += dt
        min_height = min(min_height, h)
        max_height = max(max_height, h)

        # Pitch schedule (constant rate toward the target attitude). The 4-rotor
        # differential thrust (square layout, ~0.7 m arm) supplies this pitch rate.
        if leg == "forward":
            theta = max(theta_end, theta - pitch_rate * dt)
        else:
            theta = min(theta_end, theta + pitch_rate * dt)

        speed = math.hypot(u, w)
        if leg == "forward":
            if speed >= target_speed:
                captured = True
                break
        else:
            if speed <= capture_speed:
                captured = True
                break
        if t > time_limit or x > distance_limit:
            break

    final_speed = math.hypot(u, w)
    final_gamma = math.degrees(math.atan2(w, u)) if final_speed > v_eps else 0.0
    final_theta = theta
    final_alpha = math.degrees(final_theta) - final_gamma

    reasons = []
    if leg == "forward":
        # During the forward transition the rotors carry and control the aircraft;
        # the wing only unloads them as speed builds, so a transient high (post-stall)
        # alpha is expected and not a failure. What matters is that the climb-out
        # state is reached with the wing *attached* (final alpha below stall) inside
        # the ground-clearance / distance / time corridor.
        if not captured:
            reasons.append(
                f"did not reach climb-out speed {target_speed:.1f} m/s within limits"
            )
        if min_height < AIRCRAFT["forward_transition_min_height_m"]:
            reasons.append(
                f"ground clearance lost (h_min={min_height:.1f} m)"
            )
        if AIRCRAFT["forward_transition_max_alpha_deg"] is not None and (
            max_alpha > max_alpha_limit
        ):
            reasons.append(
                f"transient alpha exceeded limit ({math.degrees(max_alpha):.1f} deg)"
            )
        if captured and final_alpha > math.degrees(wing_stall_alpha):
            reasons.append(
                f"wing not attached at climb-out (alpha={final_alpha:.1f} deg)"
            )
        if captured and final_gamma < AIRCRAFT["forward_transition_final_climb_angle_min_deg"]:
            reasons.append(
                f"final flight-path angle too low ({final_gamma:.1f} deg)"
            )
        if captured and math.degrees(final_theta) > AIRCRAFT["forward_transition_final_pitch_max_deg"]:
            reasons.append(
                f"final pitch attitude too high ({math.degrees(final_theta):.1f} deg)"
            )
        if t > time_limit and not captured:
            reasons.append(f"time limit exceeded ({t:.1f}>{time_limit:.1f} s)")
        if x > distance_limit:
            reasons.append(f"distance limit exceeded ({x:.1f}>{distance_limit:.1f} m)")
    else:
        if not captured:
            reasons.append(
                f"did not decelerate to capture speed {AIRCRAFT['back_transition_capture_speed_m_s']:.1f} m/s within limits"
            )
        if max_alpha > max_alpha_limit:
            reasons.append(
                f"alpha exceeded limit ({math.degrees(max_alpha):.1f} deg)"
            )
        if (max_height - h0) > AIRCRAFT["back_transition_height_band_up_m"]:
            reasons.append(
                f"ballooned above height band (+{max_height - h0:.1f} m)"
            )
        if (h0 - min_height) > AIRCRAFT["back_transition_height_band_down_m"]:
            reasons.append(
                f"sank below height band (-{h0 - min_height:.1f} m)"
            )
        if t > time_limit and not captured:
            reasons.append(f"time limit exceeded ({t:.1f}>{time_limit:.1f} s)")
        if x > distance_limit:
            reasons.append(f"distance limit exceeded ({x:.1f}>{distance_limit:.1f} m)")

    return {
        "leg": leg,
        "vs_eas_m_s": vs_eas_m_s,
        "vs_tas_m_s": vs_tas,
        "success": len(reasons) == 0,
        "failure_reasons": reasons,
        "time_s": t,
        "distance_m": x,
        "max_alpha_deg": math.degrees(max_alpha),
        "min_height_m": min_height,
        "max_height_m": max_height,
        "final_speed_m_s": final_speed,
        "final_pitch_deg": math.degrees(final_theta),
        "final_flight_path_angle_deg": final_gamma,
        "captured": captured,
        "thrust_saturated": thrust_saturated,
    }


def _transition_sim_leg_cap(leg, area_ratio):
    """Largest stall speed (EAS) for which one leg simulation still succeeds.

    The back-transition leg is close to monotone in Vs, but the forward leg can
    fail at *low* Vs by reaching the target speed before the pitch schedule has
    tipped down into the accepted handover attitude. So sample the interval first,
    find the highest feasible region, then bisect that region's upper edge.
    Returns the cap and the simulation result evaluated at the cap for reporting.
    """
    ctx = _transition_sim_context(leg, area_ratio)
    lo = AIRCRAFT["transition_sim_stall_EAS_lo_m_s"]
    hi = AIRCRAFT["transition_sim_stall_EAS_hi_m_s"]
    iterations = int(AIRCRAFT["transition_sim_bisection_iterations"])

    sample_count = max(9, int(AIRCRAFT.get("transition_sim_cap_sample_count", 33)))
    samples = []
    for index in range(sample_count):
        vs = lo + (hi - lo) * index / (sample_count - 1)
        result = _simulate_transition_leg(ctx, round(vs, 2))
        samples.append((vs, result))

    feasible_indices = [
        index for index, (_, result) in enumerate(samples) if result["success"]
    ]
    if not feasible_indices:
        # No sampled point can transition; report the lowest-speed result as the
        # nearest useful failure diagnostic.
        return samples[0][0], samples[0][1], ctx

    best_index = feasible_indices[-1]
    feasible, best_result = samples[best_index]
    if best_index == sample_count - 1:
        return feasible, best_result, ctx

    # The next higher sample failed, so refine the upper edge of the highest
    # feasible region.
    infeasible = samples[best_index + 1][0]
    for _ in range(iterations):
        mid = 0.5 * (feasible + infeasible)
        result = _simulate_transition_leg(ctx, round(mid, 2))
        if result["success"]:
            feasible = mid
            best_result = result
        else:
            infeasible = mid
        if (infeasible - feasible) < 0.05:
            break
    return feasible, best_result, ctx


def _leg_limit_dict(leg, source_label, cap_eas, cap_result, actual_result):
    """Assemble the stall-limit dict for one simulated transition leg."""
    density_ratio = (cap_result["vs_tas_m_s"] / cap_eas) ** 2 if cap_eas > 0 else 1.0
    return {
        "source": source_label,
        "stall_EAS_max_m_s": cap_eas,
        "stall_TAS_m_s": cap_eas * math.sqrt(density_ratio),
        "transition_time_s": cap_result["time_s"],
        "transition_distance_m": cap_result["distance_m"],
        "sim_leg": leg,
        "sim_cap_time_s": cap_result["time_s"],
        "sim_cap_distance_m": cap_result["distance_m"],
        "sim_cap_max_alpha_deg": cap_result["max_alpha_deg"],
        "sim_cap_min_height_m": cap_result["min_height_m"],
        "sim_cap_final_speed_m_s": cap_result["final_speed_m_s"],
        "sim_cap_final_pitch_deg": cap_result["final_pitch_deg"],
        "sim_cap_final_flight_path_angle_deg": cap_result["final_flight_path_angle_deg"],
        "sim_actual_success": actual_result["success"],
        "sim_actual_failure_reasons": actual_result["failure_reasons"],
        "sim_actual_time_s": actual_result["time_s"],
        "sim_actual_distance_m": actual_result["distance_m"],
        "sim_actual_max_alpha_deg": actual_result["max_alpha_deg"],
        "sim_actual_min_height_m": actual_result["min_height_m"],
        "sim_actual_final_speed_m_s": actual_result["final_speed_m_s"],
        "sim_actual_final_pitch_deg": actual_result["final_pitch_deg"],
        "sim_actual_final_flight_path_angle_deg": actual_result["final_flight_path_angle_deg"],
    }


def forward_transition_sim_stall_limit(wing, canard):
    """Forward-transition (hover -> wing-borne) simulated stall-speed cap."""
    area_ratio = canard["area_ratio"]
    cap, cap_result, ctx = _transition_sim_leg_cap("forward", area_ratio)
    actual_result = _simulate_transition_leg(ctx, round(wing["stall_EAS_m_s"], 2))
    return _leg_limit_dict("forward", "transition-sim (forward)", cap, cap_result, actual_result)


def back_transition_sim_stall_limit(wing, canard):
    """Back-transition (wing-borne -> hover) simulated stall-speed cap."""
    area_ratio = canard["area_ratio"]
    cap, cap_result, ctx = _transition_sim_leg_cap("back", area_ratio)
    actual_result = _simulate_transition_leg(ctx, round(wing["stall_EAS_m_s"], 2))
    return _leg_limit_dict("back", "transition-sim (back)", cap, cap_result, actual_result)


def stall_speed_limit(wing, canard):
    """Maximum stall EAS allowed by the binding (most demanding) transition leg.

    The smaller EAS cap forces the larger wing, so it binds; the other leg's cap
    and both legs' actual-Vs success flags are carried along for reporting.
    """
    forward = forward_transition_sim_stall_limit(wing, canard)
    back = back_transition_sim_stall_limit(wing, canard)
    binding = dict(min(forward, back, key=lambda r: r["stall_EAS_max_m_s"]))
    binding_leg = binding["sim_leg"]
    binding["source"] = f"transition-sim ({binding_leg} binding)"
    binding["forward_stall_EAS_max_m_s"] = forward["stall_EAS_max_m_s"]
    binding["back_stall_EAS_max_m_s"] = back["stall_EAS_max_m_s"]
    binding["forward_sim"] = forward
    binding["back_sim"] = back
    binding["binding_leg"] = binding_leg
    return binding
