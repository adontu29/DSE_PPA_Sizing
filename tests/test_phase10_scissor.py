import pytest

from bellona_sizing.phases.phase10_scissor import (
    CanardScissorInputs,
    fixed_cg_area_bounds,
    minimum_ShS_for_fixed_cg,
    phase10_scissor_canard,
    plot_phase10_scissor,
)


def test_phase10_uses_lecture_scissor_line_equations():
    result = phase10_scissor_canard(
        S_w=5.0,
        c_bar_w=0.95,
        x_ac_w=0.4 * 0.95,
        CL_a_w=4.751,
        S_c=1.0,
        l_c=2.38,
        CL_a_c=4.262,
        eps_alpha_c=0.0,
        eps_alpha_w=0.0,
        CL_c_max=0.815,
        CL_trim=0.703,
        SM_min=0.05,
        Cm_ac=-0.0845,
        Cm_ac_source="test",
        CL_Ah_control=0.703,
    )

    area_ratio = 1.0 / 5.0
    l_h_over_c = -2.38 / 0.95
    expected_stability = (
        0.4
        + (4.262 / 4.751) * l_h_over_c * area_ratio
        - 0.05
    )
    expected_controllability = (
        0.4
        - (-0.0845) / 0.703
        + (0.815 / 0.703) * l_h_over_c * area_ratio
    )

    assert result["x_cg_aft_over_c"] == pytest.approx(expected_stability)
    assert result["x_cg_fwd_over_c"] == pytest.approx(expected_controllability)
    assert result["x_np_over_c"] == pytest.approx(expected_stability + 0.05)
    assert result["scissor_inputs"]["l_h"] == pytest.approx(-2.38)
    assert result["Cm_ac"] == pytest.approx(-0.0845)


def test_fixed_cg_area_bounds_report_canard_lower_and_upper_bounds():
    inputs = CanardScissorInputs(
        x_ac=0.25,
        Cm_ac=0.0,
        CL_Ah=0.7,
        CLa_Ah=5.0,
        CL_h=0.9,
        CLa_h=4.0,
        l_h=-2.0,
        c_bar=1.0,
        SM=0.10,
        x_cg_fixed=-0.15,
    )

    bounds = fixed_cg_area_bounds(inputs)

    assert bounds["ShS_min"] == pytest.approx((-0.15 - 0.25) / ((0.9 / 0.7) * -2.0))
    assert bounds["ShS_max"] == pytest.approx((-0.15 - 0.15) / ((4.0 / 5.0) * -2.0))
    assert bounds["lower_bound_governed_by"] == "controllability"
    assert bounds["upper_bound_governed_by"] == "stability"
    assert bounds["feasible_area_window"]

    compatibility = minimum_ShS_for_fixed_cg(inputs)
    assert compatibility["ShS_min"] == pytest.approx(bounds["ShS_min"])
    assert compatibility["ShS_max"] == pytest.approx(bounds["ShS_max"])
    assert compatibility["governed_by"] == "controllability"


def test_phase10_scissor_plot_writes_png(tmp_path):
    result = phase10_scissor_canard(
        S_w=5.0,
        c_bar_w=1.0,
        x_ac_w=0.25,
        CL_a_w=5.0,
        S_c=0.82,
        l_c=2.0,
        CL_a_c=4.0,
        eps_alpha_c=0.0,
        eps_alpha_w=0.0,
        CL_c_max=0.9,
        CL_trim=0.7,
        SM_min=0.10,
        x_cg_fixed=-0.15,
    )

    out_path = tmp_path / "canard_scissor.png"
    returned = plot_phase10_scissor(result, out_path)

    assert returned == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
