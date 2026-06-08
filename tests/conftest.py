import pytest

from bellona_sizing.mission_profile import (
    AircraftPerformance,
    MissionPowerAssumptions,
)


@pytest.fixture
def aircraft():
    return AircraftPerformance(
        mass_kg=20.0,
        wing_area_m2=3.0,
        CD0=0.035,
        aspect_ratio=7.0,
        oswald_efficiency=0.80,
        CL_max=1.5,
        CL_limit_fraction=0.90,
    )


@pytest.fixture
def power_assumptions():
    return MissionPowerAssumptions(
        propulsive_efficiency=0.75,
        motor_efficiency=0.90,
        esc_efficiency=0.95,
        max_affordable_electrical_power_W=100000.0,
        preliminary_hover_power_W=5000.0,
        preliminary_transition_power_W=4000.0,
    )
