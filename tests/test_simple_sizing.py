import csv
import math

import simple_sizing
from scissor_plot import scissor_cg_limits


def test_scissor_interference_hits_wing_not_canard_control():
    inputs = {
        "x_ac_over_mac": 0.30,
        "static_margin": 0.0,
        "cl_alpha_canard": 4.5,
        "cl_alpha_A_h": 5.0,
        "cl_h_control": 1.0,
        "cl_A_h_control": 1.2,
        "cmac": 0.0,
        "lh_over_mac": -0.8,
        "de_da": 0.45,
        "wing_wake_dynamic_pressure_ratio": 0.85,
        "canard_speed_ratio_sq": 1.0,
    }

    clean = scissor_cg_limits(0.25, wing_immersed_fraction=0.0, **inputs)
    partial_wake = scissor_cg_limits(0.25, wing_immersed_fraction=0.5, **inputs)
    full_wake = scissor_cg_limits(0.25, wing_immersed_fraction=1.0, **inputs)

    assert math.isclose(partial_wake[0], clean[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(full_wake[0], clean[0], rel_tol=0.0, abs_tol=1e-12)
    assert full_wake[1] < partial_wake[1] < clean[1]


def test_simple_sizing_runs_and_creates_outputs(tmp_path):
    result = simple_sizing.run_sizing(
        output_dir=tmp_path,
        use_xfoil=False,
        show_progress=False,
    )
    summary = result["summary"]
    selected = result["selected"]

    assert summary["canard_area_ratio"] > 0.0
    assert selected["feasible"] is True
    assert selected["coeffs"]["canard_speed_ratio_sq"] == 1.0
    assert 0.0 < selected["coeffs"]["wing_immersed_fraction"] < 1.0
    assert selected["coeffs"]["wing_wake_dynamic_pressure_ratio"] == 0.85
    assert selected["scissor"]["wing_lift_slope_factor"] < 1.0
    # Controllability (forward-CG limit) is always enforced: the canard must be
    # able to trim at the forward-most operational CG.
    assert selected["operational_fwd_over_mac"] >= selected["scissor"]["x_forward_over_mac"]
    # Static stability (aft-CG limit) is only enforced when required; otherwise
    # the design may be statically unstable (autopilot-stabilised) and the
    # reported achieved static margin is negative.
    if simple_sizing.AIRCRAFT["require_static_stability"]:
        assert selected["operational_aft_over_mac"] <= selected["scissor"]["x_aft_over_mac"]
        assert selected["statically_stable"] is True
        assert selected["achieved_static_margin_over_mac"] >= 0.0
    else:
        assert selected["achieved_static_margin_over_mac"] == (
            selected["scissor"]["x_aft_over_mac"]
            + simple_sizing.AIRCRAFT["static_margin"]
            - selected["mass"]["x_cg_over_mac"]
        )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "mass_breakdown.csv").exists()
    assert (tmp_path / "iteration_history.csv").exists()
    assert (tmp_path / "wing_area_sweep.csv").exists()
    assert (tmp_path / "scissor_plot.png").exists()
    assert (tmp_path / "mission_profile.png").exists()
    assert (tmp_path / "wing_area_sweep.png").exists()
    assert result["mission"]["segment_summaries"]["wing_borne_climb"]["energy_Wh"] > 0.0
    assert result["mission"]["mission_grid"]["climb_EAS_m_s"]
    assert abs(result["iteration_history"][-1]["mass_change_kg"]) < abs(result["iteration_history"][0]["mass_change_kg"])


def test_summary_table_contains_main_report_values(tmp_path):
    simple_sizing.run_sizing(
        output_dir=tmp_path,
        make_plots=False,
        use_xfoil=False,
        show_progress=False,
    )

    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as csv_file:
        rows = {row["quantity"]: row["value"] for row in csv.DictReader(csv_file)}

    assert "MTOW_mass_estimate_kg" in rows
    assert "mass_closure_error_kg" in rows
    assert "wing_area_m2" in rows
    assert "wing_stall_EAS_m_s" in rows
    assert "climb_CL_limit" in rows
    assert "max_climb_CL" in rows
    assert "wing_CL_max" in rows
    assert "xfoil_enabled" in rows
    assert "canard_area_ratio" in rows
    assert "x_CG_over_MAC" in rows
    assert "optimized_climb_EAS_m_s" in rows
    assert "installed_battery_energy_Wh" in rows
