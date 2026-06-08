import pytest

from bellona_sizing.phases.phase08_wing import phase8_wing_planform


def _wing(area_m2=6.0, cl_max_2d=1.4):
    return phase8_wing_planform(
        MTOW_N=500.0,
        S_wing=area_m2,
        cl_max_2D=cl_max_2d,
        cl_a_2D=6.0,
        rho_mission=0.66,
        rho_transition=1.22,
        rho_reference=1.225,
        AR_guess=7.0,
    )


def test_phase8_preserves_selected_area_and_calculates_stall_speeds():
    result = _wing(area_m2=6.4)

    assert result["S"] == pytest.approx(6.4)
    assert result["W_S_design"] == pytest.approx(500.0 / 6.4)
    assert result["V_stall"] == pytest.approx(result["stall_TAS_mission_m_s"])
    assert result["stall_TAS_mission_m_s"] > result["stall_EAS_m_s"]
    assert result["stall_TAS_transition_m_s"] == pytest.approx(
        result["stall_EAS_m_s"] * (1.225 / 1.22) ** 0.5
    )


def test_larger_area_and_higher_cl_max_lower_calculated_stall_speed():
    baseline = _wing(area_m2=6.0, cl_max_2d=1.4)
    larger = _wing(area_m2=8.0, cl_max_2d=1.4)
    higher_cl = _wing(area_m2=6.0, cl_max_2d=1.6)

    assert larger["stall_EAS_m_s"] < baseline["stall_EAS_m_s"]
    assert higher_cl["stall_EAS_m_s"] < baseline["stall_EAS_m_s"]
