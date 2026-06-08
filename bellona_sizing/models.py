"""Mission inputs and first-cut assumptions used throughout the Bellona sizing workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Mission:
    """Mission inputs that should be easy to override from the CLI later."""

    altitude_m: float = 6000.0
    range_m: float = 6000.0
    time_budget_s: float = 600.0
    hover_time_s: float = 300.0
    mission_equipment_mass_kg: float = 7.3
    external_tow_load_N: float = 0.0


@dataclass
class Assumptions:
    """First-cut sizing assumptions shared across the phase functions."""

    g: float = 9.80665
    n_rotors: int = 4
    thrust_to_weight: float = 1.30
    vtol_climb_rate_m_s: float = 0.0
    vertical_takeoff_height_m: float = 20.0
    vertical_takeoff_rate_m_s: float = 2.0
    disc_loading_target_N_m2: float = 170.0
    prop_diameter_max_m: Optional[float] = None
    CD0: float = 0.040
    AR: float = 7.0
    oswald_e: float = 0.78
    CL_max_guess: float = 1.30
    preliminary_wing_area_m2: float = 5.0
    wing_area_m2: Optional[float] = None
    wing_taper: float = 0.40
    wing_sweep_c4_rad: float = 0.0
    canard_volume_coeff: float = 0.375
    canard_arm_chord_ratio: float = 2.50
    canard_AR: float = 5.0
    canard_taper: float = 0.50
    canard_sweep_c4_rad: float = 0.0
    canard_CL_limit_fraction: float = 0.90
    canard_eps_alpha_c: float = 0.0
    wing_eps_alpha_w: float = 0.0
    wing_mac_le_x_m: Optional[float] = None
    solve_wing_position_for_cg: bool = True
    use_xfoil: bool = True
    xfoil_path: Optional[str] = None
    figure_of_merit: float = 0.70
    eta_motor: float = 0.90
    eta_ESC: float = 0.95
    eta_prop: float = 0.75
    max_affordable_electrical_power_W: float = 20000.0
    preliminary_hover_power_W: float = 14000.0
    preliminary_transition_power_W: float = 8000.0
    eta_batt: float = 0.95
    usable_battery_fraction: float = 0.85
    battery_specific_energy_Wh_kg: float = 310.0
    mission_altitude_step_m: float = 100.0
    mission_CL_limit_fraction: float = 0.90
    max_transition_complete_speed_m_s: Optional[float] = 20.0
    max_stall_EAS_m_s: Optional[float] = None
    minimum_power_margin_fraction: float = 0.05
    max_fixed_wing_climb_angle_deg: float = 30.0
    cruise_stall_margin_deg: float = 3.0
    canard_stall_before_wing_margin_deg: float = 2.0
    allow_spiral_climb: bool = True
    run_multistage_mission_extension: bool = False
    run_minimum_time_reference: bool = False
    fuselage_length_m: float = 2.2
    static_margin_min: float = 0.10
    elevon_q_slipstream_ratio: float = 1.50
    elevon_max_deflection_deg: float = 25.0
    elevon_tau: float = 0.50
    elevon_eta: float = 0.90
    elevon_l_e_over_c: float = 0.75
    elevon_pitch_trim_margin: float = 1.20
    elevon_chord_fraction_min: float = 0.12
    elevon_chord_fraction_max: float = 0.35
    elevon_span_fraction_min: float = 0.20
    elevon_span_fraction_max: float = 0.80
    elevon_grid_points: int = 49
    roll_rate_required_deg_s: float = 60.0
    hover_pitch_arm_m: Optional[float] = None
    hover_roll_arm_m: Optional[float] = None
    hover_yaw_arm_m: Optional[float] = None
    hover_pitch_arm_fraction_fuselage: float = 0.37
    hover_roll_arm_fraction_span: float = 0.375
    hover_yaw_arm_fraction_chord: float = 0.75
    hover_Ixx_kg_m2: Optional[float] = None
    hover_Iyy_kg_m2: Optional[float] = None
    hover_Izz_kg_m2: Optional[float] = None
    hover_Ixx_radius_fraction_span: float = 0.20
    hover_Iyy_radius_fraction_fuselage: float = 0.35
    hover_Izz_radius_fraction_span: float = 0.22
    hover_angular_accel_required_rad_s2: float = 2.0
    hover_yaw_rate_required_deg_s: float = 30.0
    hover_yaw_response_time_s: float = 1.0
    hover_control_CL_de: Optional[float] = None
    transition_blend_start_frac: float = 0.50
    transition_blend_end_frac: float = 1.20
    transition_cruise_margin_frac: float = 0.05
    transition_accel_m_s2: float = 1.0
    transition_sample_count: int = 9
    dynamic_Cm_q: Optional[float] = None
    dynamic_Cm_alpha_dot: float = 0.0
    dynamic_Cl_beta: float = -0.08
    dynamic_Cl_p: Optional[float] = None
    dynamic_Cl_r: float = 0.02
    dynamic_Cn_beta: float = 0.06
    dynamic_Cn_p: float = -0.02
    dynamic_Cn_r: float = -0.20
    short_period_zeta_min: float = 0.30
    short_period_zeta_max: float = 2.00
    short_period_omega_min_rad_s: float = 1.00
    phugoid_zeta_min: float = 0.04
    dutch_roll_zeta_min: float = 0.08
    dutch_roll_omega_min_rad_s: float = 0.40
    spiral_time_to_double_min_s: float = 12.0
    wing_areal_density_kg_m2: float = 2.00
    canard_areal_density_kg_m2: float = 1.70
    fuselage_linear_density_kg_m: float = 2.20
    boom_landing_gear_mass_kg: float = 2.50
    motor_specific_mass_kg_W: float = 0.00027
    esc_specific_mass_kg_W: float = 0.00008
    prop_mass_coeff_kg_m2: float = 0.10
    propeller_mass_total_kg: Optional[float] = None
    avionics_mass_kg: float = 1.50
    wiring_fraction: float = 0.06
    mass_contingency_fraction: float = 0.08
    cg_envelope_half_width_over_mac: float = 0.05
    cg_required_margin_over_mac: float = 0.02
    canard_volume_grid_min: float = 0.05
    canard_volume_grid_max: float = 1.60
    canard_volume_grid_step: float = 0.01
    hover_control_margin_min: float = 1.05
    hover_pitch_arm_fraction_fuselage_max: float = 0.45
    hover_roll_arm_fraction_span_max: float = 0.45
    thrust_to_weight_min: float = 1.20
    thrust_to_weight_max: float = 1.60
    control_closure_max_iter: int = 4
