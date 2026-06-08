import numpy as np
import pytest

from bellona_sizing.common import isa
from bellona_sizing.mission_profile import (
    MissionPowerAssumptions,
    MissionProfileConfig,
    aerodynamic_speed_limits,
    equivalent_to_true_airspeed,
    minimum_time_reference_climb,
    optimize_constant_eas_mission,
    optimize_multistage_mission,
    simulate_level_cruise,
    simulate_wing_borne_climb,
    sweep_target_distances,
    wing_segment_forces,
)


def test_constant_speed_climb_reduces_to_drag_plus_weight_sine(aircraft):
    gamma = np.deg2rad(12.0)
    forces = wing_segment_forces(aircraft, 1.0, 25.0, 25.0, gamma, 100.0)
    assert forces["required_thrust_N"] == pytest.approx(
        forces["drag_N"] + aircraft.weight_N * np.sin(gamma)
    )


def test_level_flight_reduces_to_drag(aircraft):
    forces = wing_segment_forces(
        aircraft, 1.0, 25.0, 25.0, 0.0, 0.0, delta_s_m=200.0
    )
    assert forces["required_thrust_N"] == pytest.approx(forces["drag_N"])


def test_energy_height_includes_acceleration_energy(aircraft):
    delta_s = 500.0
    accelerated = wing_segment_forces(
        aircraft, 1.0, 20.0, 25.0, 0.0, 0.0, delta_s_m=delta_s
    )
    expected_kinetic = 0.5 * aircraft.mass_kg * (25.0**2 - 20.0**2)
    assert accelerated["kinetic_energy_change_J"] == pytest.approx(expected_kinetic)
    assert accelerated["required_thrust_N"] - accelerated["drag_N"] == pytest.approx(
        expected_kinetic / delta_s
    )


def test_electrical_power_uses_declared_efficiencies(
        aircraft, power_assumptions):
    result = simulate_level_cruise(
        aircraft, power_assumptions, 1000.0, 25.0, 500.0
    )
    state = result["states"][0]
    expected = (
        state["required_thrust_N"]
        * state["speed_m_s"]
        / power_assumptions.total_forward_efficiency
    )
    assert state["electrical_power_W"] == pytest.approx(expected)
    assert state["power_margin_W"] == pytest.approx(
        power_assumptions.max_affordable_electrical_power_W - expected
    )


def test_constant_eas_increases_tas_as_density_falls():
    sea_level = equivalent_to_true_airspeed(20.0, isa(0.0)[0])
    altitude = equivalent_to_true_airspeed(20.0, isa(5000.0)[0])
    assert altitude > sea_level


def test_integrated_climb_is_internally_consistent(
        aircraft, power_assumptions):
    h0 = 20.0
    h1 = 320.0
    result = simulate_wing_borne_climb(
        aircraft,
        power_assumptions,
        h0,
        h1,
        20.0,
        np.deg2rad(35.0),
        altitude_step_m=50.0,
    )
    assert result["feasible"]
    states = result["states"]
    assert sum(state["delta_h_m"] for state in states) == pytest.approx(h1 - h0)
    assert sum(state["delta_x_m"] for state in states) == pytest.approx(result["distance_m"])
    assert sum(state["delta_t_s"] for state in states) == pytest.approx(result["time_s"])
    assert sum(state["delta_electrical_energy_Wh"] for state in states) == pytest.approx(
        result["energy_Wh"]
    )
    expected_mechanical = (
        aircraft.weight_N * (h1 - h0)
        + 0.5 * aircraft.mass_kg
        * (states[-1]["speed_next_m_s"]**2 - states[0]["speed_current_m_s"]**2)
    )
    assert result["mechanical_energy_change_J"] == pytest.approx(expected_mechanical)


def test_affordable_power_limit_rejects_infeasible_climb(aircraft):
    limited = MissionPowerAssumptions(
        propulsive_efficiency=0.70,
        motor_efficiency=0.90,
        esc_efficiency=0.95,
        max_affordable_electrical_power_W=1500.0,
        preliminary_hover_power_W=1000.0,
        preliminary_transition_power_W=1000.0,
    )
    result = simulate_wing_borne_climb(
        aircraft, limited, 20.0, 320.0, 22.0, np.deg2rad(50.0)
    )
    assert not result["feasible"]
    assert "Affordable electrical power" in result["failure_reason"]


def test_optimizer_rejects_hover_assumption_above_total_power_cap(aircraft):
    power = MissionPowerAssumptions(
        propulsive_efficiency=0.75,
        motor_efficiency=0.90,
        esc_efficiency=0.95,
        max_affordable_electrical_power_W=10000.0,
        preliminary_hover_power_W=14000.0,
        preliminary_transition_power_W=8000.0,
    )
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        vertical_takeoff_height_m=10.0,
    )
    result = optimize_constant_eas_mission(aircraft, power, config)
    assert not result["feasible"]
    assert result["failure_reason"] == (
        "Preliminary hover power exceeds affordable electrical power."
    )


def test_constant_eas_optimizer_returns_feasible_consistent_profile(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        altitude_step_m=50.0,
        vertical_takeoff_height_m=10.0,
        vertical_takeoff_rate_m_s=2.0,
        transition_accel_m_s2=2.0,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[18.0, 22.0],
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
        cruise_tas_grid_m_s=[20.0, 25.0],
    )
    assert result["feasible"]
    assert result["outbound_time_s"] <= config.outbound_time_budget_s
    assert result["final_altitude_m"] == pytest.approx(config.target_altitude_m)
    assert result["total_horizontal_distance_m"] == pytest.approx(config.horizontal_range_m)
    assert result["total_electrical_energy_Wh"] == pytest.approx(
        sum(segment["energy_Wh"] for segment in result["segment_summaries"].values())
    )
    assert min(result["constraint_margins"].values()) >= -1e-8


def test_default_speed_grids_follow_aerodynamic_limits(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        altitude_step_m=50.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
    )
    limits = aerodynamic_speed_limits(aircraft, config)
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
    )

    assert result["feasible"]
    grid = result["grid"]
    assert grid["adaptive_climb_EAS"]
    assert grid["adaptive_cruise_TAS"]
    assert grid["climb_EAS_m_s"][0] == pytest.approx(
        1.001 * limits["minimum_climb_EAS_m_s"]
    )
    assert grid["cruise_TAS_m_s"][0] == pytest.approx(
        1.001 * limits["minimum_cruise_TAS_m_s"]
    )
    assert grid["climb_EAS_m_s"][0] != pytest.approx(14.0)
    assert grid["cruise_TAS_m_s"][0] != pytest.approx(16.0)
    assert max(grid["climb_angle_deg"]) <= config.max_fixed_wing_climb_angle_deg


def test_transition_complete_speed_limit_rejects_small_wing(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        altitude_step_m=50.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
        max_transition_complete_speed_m_s=9.0,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
    )

    assert not result["feasible"]
    assert result["failure_reason"] == (
        "Transition-complete speed exceeds the selected limit."
    )
    limits = result["grid"]["aerodynamic_speed_limits"]
    assert limits["transition_complete_speed_margin_m_s"] < 0.0


def test_optional_stall_eas_limit_rejects_candidate(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        altitude_step_m=50.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
        max_transition_complete_speed_m_s=None,
        max_stall_EAS_m_s=8.0,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
    )

    assert not result["feasible"]
    assert result["failure_reason"] == "Stall EAS exceeds the selected limit."
    limits = result["grid"]["aerodynamic_speed_limits"]
    assert limits["stall_EAS_margin_m_s"] < 0.0


def test_preliminary_power_must_keep_selected_reserve(aircraft):
    power = MissionPowerAssumptions(
        propulsive_efficiency=0.75,
        motor_efficiency=0.90,
        esc_efficiency=0.95,
        max_affordable_electrical_power_W=10000.0,
        preliminary_hover_power_W=9600.0,
        preliminary_transition_power_W=4000.0,
    )
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=20.0,
        vertical_takeoff_height_m=10.0,
        minimum_power_margin_fraction=0.05,
    )
    result = optimize_constant_eas_mission(aircraft, power, config)

    assert not result["feasible"]
    assert "Required electrical power margin" in result["failure_reason"]


def test_better_efficiency_reduces_mission_energy(aircraft, power_assumptions):
    better = MissionPowerAssumptions(
        propulsive_efficiency=0.85,
        motor_efficiency=0.95,
        esc_efficiency=0.98,
        max_affordable_electrical_power_W=100000.0,
        preliminary_hover_power_W=power_assumptions.preliminary_hover_power_W,
        preliminary_transition_power_W=power_assumptions.preliminary_transition_power_W,
    )
    baseline = simulate_level_cruise(
        aircraft, power_assumptions, 3000.0, 25.0, 1000.0
    )
    improved = simulate_level_cruise(aircraft, better, 3000.0, 25.0, 1000.0)
    assert improved["energy_Wh"] < baseline["energy_Wh"]


def test_mission_states_do_not_claim_component_level_propulsion_data(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=200.0,
        horizontal_range_m=800.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=0.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[20.0],
        climb_angle_grid_rad=np.deg2rad([30.0]),
        cruise_tas_grid_m_s=[20.0],
    )
    assert result["feasible"]
    forbidden = {
        "rpm",
        "advance_ratio",
        "tip_mach",
        "available_thrust_N",
        "torque_Nm_per_motor",
        "battery_current_A",
    }
    assert all(forbidden.isdisjoint(state) for state in result["states"])


def test_spiral_climb_can_absorb_excess_ground_track(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=100.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=0.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
        allow_spiral_climb=True,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[20.0],
        climb_angle_grid_rad=np.deg2rad([20.0]),
        cruise_tas_grid_m_s=[20.0],
    )
    assert result["feasible"]
    assert result["cruise"]["distance_m"] == pytest.approx(0.0)
    assert result["spiral_excess_ground_track_distance_m"] > 0.0
    assert result["total_ground_track_distance_m"] > config.horizontal_range_m


def test_zero_distance_target_is_supported_by_spiral_climb(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=0.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=0.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
        allow_spiral_climb=True,
    )
    result = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[20.0],
        climb_angle_grid_rad=np.deg2rad([30.0]),
        cruise_tas_grid_m_s=[20.0],
    )
    assert result["feasible"]
    assert result["required_horizontal_displacement_m"] == pytest.approx(0.0)
    assert result["spiral_excess_ground_track_distance_m"] > 0.0


def test_target_distance_sweep_keeps_aircraft_fixed_and_reports_results(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=0.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
    )
    sweep = sweep_target_distances(
        aircraft,
        power_assumptions,
        config,
        [0.0, 500.0],
        eas_grid_m_s=[20.0],
        climb_angle_grid_rad=np.deg2rad([30.0]),
        cruise_tas_grid_m_s=[20.0],
    )
    assert [record["target_distance_m"] for record in sweep] == [0.0, 500.0]
    assert all(record["feasible"] for record in sweep)
    assert (
        sweep[0]["spiral_excess_ground_track_distance_m"]
        > sweep[1]["spiral_excess_ground_track_distance_m"]
    )


def test_multistage_and_minimum_time_extensions_return_feasible_profiles(
        aircraft, power_assumptions):
    config = MissionProfileConfig(
        target_altitude_m=300.0,
        horizontal_range_m=1000.0,
        outbound_time_budget_s=180.0,
        hover_duration_s=0.0,
        altitude_step_m=50.0,
        vertical_takeoff_height_m=10.0,
        transition_accel_m_s2=2.0,
    )
    initial = optimize_constant_eas_mission(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[18.0, 22.0],
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
        cruise_tas_grid_m_s=[20.0],
    )
    multi_stage = optimize_multistage_mission(
        aircraft,
        power_assumptions,
        config,
        initial_result=initial,
        eas_candidates_m_s=[18.0, 22.0],
        climb_angle_candidates_rad=np.deg2rad([25.0, 30.0]),
    )
    minimum_time = minimum_time_reference_climb(
        aircraft,
        power_assumptions,
        config,
        eas_grid_m_s=[18.0, 22.0],
        climb_angle_grid_rad=np.deg2rad([25.0, 30.0]),
    )
    assert multi_stage["feasible"]
    assert len(multi_stage["optimized_climb_EAS_schedule_m_s"]) == 3
    assert minimum_time["feasible"]
