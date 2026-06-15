"""Sizing inputs: the mission, the aircraft, and the mass model.

These three dictionaries are the only knobs in the workflow. Everything
downstream reads from them. They are mutated in place during a run (the XFOIL
feedback refreshes the airfoil coefficients, and the wing-area sweep restores the
selected design's coefficients), never rebound, so importing the dict object into
another module shares the same live state.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

MISSION = {
    "altitude_m": 6000.0,
    "range_m": 6000.0,
    "time_budget_s": 600.0,
    "hover_time_s": 300.0,
    "vertical_takeoff_height_m": 20.0,
    "vertical_takeoff_rate_m_s": 2.0,
    "altitude_step_m": 100.0,
    "allow_spiral_climb": True,
    # When the straight climb's ground track would overshoot range_m, the aircraft
    # cannot fly straight to the balloon without flying past it. Instead it spirals
    # up in place over the launch point until only `range` of ground track remains,
    # then climbs straight out to the target (arriving at the target altitude, so no
    # level cruise). The in-place spiral is flown at this fixed radius (m); its bank
    # and load factor follow from n = sqrt(1 + (V^2/(g*R))^2), which raises induced
    # drag (~n^2), slows the climb, and lifts the stall speed (Vs*sqrt(n)). Smaller
    # radius = tighter spiral = higher load factor = more demanding.
    "spiral_turn_radius_m": 250.0,
}


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------

AIRCRAFT = {
    "MTOW_kg": 52.78,
    "g_m_s2": 9.80665,
    "wing_area_m2": 6.8,
    "canard_sweep_deg": 0.0,
    "wing_area_sweep_min_m2": 0.15,
    "wing_area_sweep_max_m2": 8.0,
    "wing_area_sweep_step_m2": 0.05,
    "climb_stall_margin_n": 1.25,
    "sizing_iteration_count": 8,
    "sizing_mass_tolerance_kg": 0.02,
    "wing_aspect_ratio": 7.0,
    "wing_taper": 0.40,
    "wing_CL_max": 1.28,
    "wing_CL_alpha_per_rad": 4.74,
    "canard_CL_max": 1,
    "canard_CL_alpha_per_rad": 4.25,
    "canard_aspect_ratio": 5.0,
    "canard_taper": 0.50,
    # Wing longitudinal position is solved as the canard->wing arm (the canard is
    # pinned to the nose, see MASS["nose_to_canard_m"]). These bound the solve;
    # the wing MAC LE station = nose_to_canard + arm.
    "canard_arm_min_m": 0,
    "canard_arm_max_m": 2.5,
    # Minimum clear streamwise gap between the canard root trailing edge and the
    # wing root leading edge (no overlap; keeps the canard clear of the wing
    # upwash, which the scissor's no-interference assumption needs).
    "canard_wing_min_gap_m": 0,
    "canard_area_ratio_min": 0.05,
    "canard_area_ratio_max": 0.80,
    "canard_area_ratio_step": 0.005,
    "static_margin": 0.05,
    "cg_envelope_half_width_over_mac": 0.05,
    "cg_margin_over_mac": 0.02,
    # Canard control authority in the scissor forward-CG limit. Per the course
    # method (AE3211-I Lec 8 slide 17) C_Lh is a *configuration* constant: this
    # tailsitter needs a full-moving canard, so |C_Lh|=1, capped by the canard's
    # real CL_max so XFOIL cannot promise lift the airfoil lacks.
    "canard_control_CLh_full_moving": 1,
    # Enforce the aft-CG/static-stability side of the scissor plot.
    "require_static_stability": True,
    # Scissor controllability lift condition for C_L_A-h. "mission_max" uses the
    # largest wing-borne mission CL (times the margin factor) capped by the
    # permitted CL limit.
    "scissor_control_CL_Ah_source": "mission_max",
    "scissor_control_CL_Ah_margin_factor": 1.15,
    # Fixed-CD0 bootstrap. Used only on the first mass iteration, before any canard
    # geometry exists to run the component build-up; the converged passes use the
    # build-up below. oswald_efficiency is the wing induced-drag factor.
    "CD0": 0.040,
    "oswald_efficiency": 0.78,
    # --- Two-surface (wing + canard) induced-drag split ---
    "canard_oswald_efficiency": 0.70,   # lower than the wing (low-AR canard)
    # Mutual induced-drag interference between the two lifting surfaces (Munk): the
    # wing flies in the canard's downwash, adding a cross term to the induced drag.
    # sigma in [0,1]; ~0.8 for closely-spaced tandem surfaces.
    "canard_wing_induced_interference_factor": 0.80,
    # --- Component drag build-up (AircraftDesign2, AD2 slides 42-60) ---
    #   CD0 = sum(Cf * FF * IF * Swet / Sref) + CD_misc (excrescence/leakage).
    # Reynolds number is evaluated per component at each mission state.
    "excrescence_fraction": 0.10,              # AD2 slide 60 (prop aircraft 5-10%)
    "wing_thickness_ratio": 0.092,             # SD7037 t/c
    "canard_thickness_ratio": 0.12,            # NACA0012 t/c
    "wing_max_thickness_x_c": 0.30,            # (x/c) of max thickness
    "canard_max_thickness_x_c": 0.30,
    # Assumed laminar-flow fractions (AD2 slide 50, smooth molded composite).
    "wing_laminar_fraction": 0.35,
    "canard_laminar_fraction": 0.35,
    "fuselage_laminar_fraction": 0.10,
    "hardware_laminar_fraction": 0.0,
    # Component interference factors (AD2 slide 53).
    "wing_interference_factor": 1.0,
    "canard_interference_factor": 1.0,
    "fuselage_interference_factor": 1.0,
    "hardware_interference_factor": 1.2,
    # Surface roughness for the cut-off Reynolds number (AD2 slide 51).
    "surface_roughness_k_m": 0.052e-5,         # smooth molded composite
    # Exposed rotor/strut/nacelle hardware (placeholder lumped wetted area).
    "exposed_hardware_wetted_area_m2": 0.10,
    "exposed_hardware_form_factor": 1.30,
    # --- Scissor-plot aerodynamics (TU Delft AE3211-I, Lectures 7 & 8) ---
    "datcom_eta": 0.95,                   # DATCOM airfoil efficiency (0.90-1.0)
    "wing_sweep_quarter_chord_deg": 0.0,  # straight wing for this UAV
    "wing_sweep_half_chord_deg": 0.0,
    "canard_sweep_half_chord_deg": 0.0,
    "canard_sweep_quarter_chord_deg": 0.0,
    "wing_airfoil_cm0": -0.05,            # airfoil Cm0 (negative if cambered)
    "wing_CL0": 0.20,                     # aircraft-less-canard CL at alpha=0
    "wing_datcom_eta": 0.95,
    "canard_datcom_eta": 0.95,
    # Canard-wing interference dynamic pressures (see scissor_cg_limits). The
    # canard is forward in clean air, so (Vc/V)^2 ~ 1. The wing sits in the canard
    # wake over its inboard span and loses dynamic pressure there.
    "canard_speed_ratio_sq": 1.0,              # (Vc/V)^2 at the canard (clean air)
    "wing_wake_dynamic_pressure_ratio": 0.85,  # (Vw/V)^2 over the immersed wing
    # --- XFOIL Reynolds-feedback refinement of the section coefficients ---
    "use_xfoil_airfoil_updates": True,
    "xfoil_path": "xfoil/xfoilp4.exe",
    "xfoil_sd7037_file": "xfoil/sd7037.dat",
    "xfoil_transition_x_c": 0.50,
    "xfoil_reynolds_rounding": 0.0,
    "xfoil_reynolds_update_threshold": 5000.0,
    "xfoil_mach_rounding": 0.02,
    "xfoil_mach_command_min": 0.20,
    "xfoil_alpha_start_deg": -6.0,
    "xfoil_alpha_end_deg": 18.0,
    "xfoil_alpha_step_deg": 1.0,
    "xfoil_timeout_s": 30.0,
    # --- Finite-wing CL_max (AircraftDesign2, AD2 slides 16-24) ---
    # The aircraft CL_max is (CL_max/cl_max)(dY, sweep) * cl_max, with the section
    # cl_max read at the low-speed stall Reynolds number (a second XFOIL condition).
    # LE sharpness parameter dY [% chord] per surface (AD2 slide 19); None -> from
    # t/c via dY = le_sharpness_dy_per_tc * (t/c).
    "wing_le_sharpness_dY_pct": None,
    "canard_le_sharpness_dY_pct": None,
    "le_sharpness_dy_per_tc": 26.0,
    # Second XFOIL condition for the section cl_max feeding the finite-wing CL_max
    # (AD2 slide 22 requires cl_max at the stall Reynolds number). The stall speed
    # is taken at sea level from the current CL_max and MTOW; it converges in the
    # XFOIL outer loop.
    "clmax_stall_xfoil_altitude_m": 0.0,
    "clmax_stall_speed_margin": 1.0,            # V used = margin * V_stall
    "xfoil_update_relaxation": 1.0,
    "xfoil_update_tolerance_fraction": 0.005,
    # XFOIL runs in an OUTER Reynolds-feedback loop around the whole wing-area
    # sweep: each outer pass = one full sweep + one XFOIL airfoil-pair run,
    # repeated until the section coefficients stop changing.
    "xfoil_outer_iteration_count": 4,
    "cruise_true_speed_m_s": 15.0,
    # Cruise/scissor permitted-CL cap. The wing-borne CLIMB derives its own
    # (tighter) CL limit from climb_stall_margin_n.
    "mission_CL_limit_fraction": 0.90,
    "cruise_stall_margin_deg": 3.0,
    "max_affordable_electrical_power_W": 18000.0,
    "transition_accel_m_s2": 1.0,
    "forward_flight_efficiency": 0.75 * 0.90 * 0.95,
    "hover_power_W": 14000.0,
    "battery_specific_energy_Wh_kg": 310.0,
    "battery_usable_fraction": 0.85,
    "battery_efficiency": 0.95,
    "n_rotors": 4,
    "thrust_to_weight": 1.30,
    "disc_loading_N_m2": 170.0,

    # --- Maximum-stall-speed requirement (the wing-area lower bound) ---
    # The cap is generated by a reduced-order point-mass transition simulation
    # (Stone & Clarke inspired): the binding (most demanding) of the forward and
    # back transition legs. A 2-DOF vertical-plane model with a prescribed pitch
    # schedule, body-axis thrust, and a Viterna full-range CL/CD bisects the stall
    # speed (EAS) to find the largest Vs whose trajectory still meets the leg's
    # time / distance / height / angle-of-attack limits. The cap is mass- and
    # size-independent, so it is a clean generator of the maximum allowable stall
    # speed comparable to the candidate's actual wing stall EAS.
    "back_transition_approach_speed_factor": 1.30,
    "forward_transition_thrust_to_weight": 1.30,
    "forward_transition_climbout_factor": 1.25,

    # --- Reduced-order transition simulation knobs ---
    "transition_sim_time_step_s": 0.05,
    "transition_sim_max_time_s": 40.0,
    "transition_sim_pitch_rate_deg_s": 12.0,
    "transition_sim_bisection_iterations": 24,
    "transition_sim_stall_EAS_lo_m_s": 3.0,
    "transition_sim_stall_EAS_hi_m_s": 45.0,
    "transition_sim_velocity_epsilon_m_s": 0.5,
    # Altitude-hold gain (1/s) for the thrust controller: the commanded thrust is
    # throttled (within the available T/W) to drive the vertical speed back to zero
    # -- how a tail-sitter brakes in the back transition and holds height while
    # accelerating in the forward transition.
    "transition_sim_altitude_hold_gain": 1.5,
    # Separate transition altitudes (set the air density per leg). Forward near the
    # take-off site; back near the mission altitude (None resolves to MISSION alt).
    "forward_transition_altitude_m": 0.0,
    "back_transition_altitude_m": None,
    # Forward leg (hover -> wing-borne climb-out): target V_co = climbout factor *
    # Vs, plus the dynamic-corridor limits.
    "forward_transition_start_height_m": 20.0,        # height AGL at hover entry
    "forward_transition_min_height_m": 0.0,           # ground-clearance floor (AGL)
    "forward_transition_final_pitch_min_deg": 0.0,    # attitude the pitch-down stops at
    "forward_transition_final_pitch_max_deg": 30.0,   # max acceptable attitude at completion
    "forward_transition_final_climb_angle_min_deg": 0,
    "forward_transition_sim_time_limit_s": 25.0,
    "forward_transition_sim_distance_limit_m": 250.0,
    "forward_transition_max_alpha_deg": None,         # None -> wing stall alpha
    # Back leg (wing-borne -> near hover): entry at approach factor * Vs, pitch up
    # to vertical, capture when |V| <= the capture speed.
    "back_transition_capture_speed_m_s": 2.0,
    "back_transition_pitch_max_deg": 110.0,            # attitude the pitch-up stops at
    "back_transition_height_band_up_m": 100.0,         # max altitude gain (balloon-up)
    "back_transition_height_band_down_m": 100.0,       # max altitude loss (sink)
    "back_transition_sim_time_limit_s": 20.0,
    "back_transition_sim_distance_limit_m": 3000.0,
    "back_transition_max_alpha_deg": 120.0,            # pitch-up through stall is allowed

    "battery_cg_offset_over_mac": -0.25,   # battery CG as fraction of MAC from wing MAC LE
}


# ---------------------------------------------------------------------------
# Mass model
# ---------------------------------------------------------------------------

MASS = {
    # Areal/linear densities calibrated to colleague estimates:
    "wing_areal_density_kg_m2": 3.36,    # 12 kg @ 5 m span, AR 7 (3.57 m^2)
    "canard_areal_density_kg_m2": 2.50,  # 2 kg @ 2 m span, AR 5 (0.80 m^2)
    "fuselage_linear_density_kg_m": 2.61,  # 6 kg @ 2.30 m
    # Fuselage length is DERIVED from the layout, not fixed: the canard sits
    # nose_to_canard behind the nose, the wing station is solved, and the tail
    # sits a fixed margin behind the wing root trailing edge:
    #   L_fus = wing_LE + wing_root_chord + wing_te_to_tail.
    "nose_to_canard_m": 0.35,
    "wing_te_to_tail_m": 0.4,         # aft body length behind wing root TE
    "fuselage_width_m": 0.30,
    "nose_bay_x_m": 0.10,             # station of nose electronics/payload
    "motor_mass_each_kg": 2.8,        # per motor; on struts from the wing
    "prop_mass_coeff_kg_m2": 0.10,
    "avionics_mass_kg": 0.60,         # in the nose bay
    "sensor_mass_kg": 1.00,           # sensors in the nose
    # Balloon-capture subsystem
    "net_gun_mass_kg": 1.50,          # net + netgun, in the nose
    "reel_mass_kg": 0.50,             # reel, ~at the CG (mid-body)
    "parachute_mass_kg": 0.60,        # parachute, just ahead of the wing LE
    "parachute_ahead_of_wing_le_m": 0.10,
    "wiring_fraction": 0.06,
    "contingency_fraction": 0.08,
}


# Section-derived aerodynamic inputs that the XFOIL feedback overwrites. The
# baseline values double as the fallback used when XFOIL is disabled.
AIRFOIL_AERO_KEYS = (
    "wing_CL_max",
    "wing_CL_alpha_per_rad",
    "wing_airfoil_cm0",
    "wing_CL0",
    "wing_datcom_eta",
    "canard_CL_max",
    "canard_CL_alpha_per_rad",
    "canard_datcom_eta",
)
BASE_AIRFOIL_AERO = {key: AIRCRAFT[key] for key in AIRFOIL_AERO_KEYS}


def reset_airfoil_aero_defaults():
    """Restore the fallback airfoil assumptions before a new sizing run."""
    AIRCRAFT.update(BASE_AIRFOIL_AERO)


def snapshot_aircraft():
    """Copy the current aircraft input state for later reporting."""
    return dict(AIRCRAFT)


def restore_aircraft_snapshot(snapshot):
    """Restore a stored aircraft state without replacing the global object."""
    AIRCRAFT.update(snapshot)
