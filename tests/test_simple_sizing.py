import csv

import simple_sizing


def test_simple_sizing_runs_and_creates_outputs(tmp_path):
    result = simple_sizing.run_sizing(output_dir=tmp_path)
    summary = result["summary"]
    selected = result["selected"]

    assert summary["canard_area_ratio"] > 0.0
    assert selected["feasible"] is True
    assert selected["operational_fwd_over_mac"] >= selected["scissor"]["x_forward_over_mac"]
    assert selected["operational_aft_over_mac"] <= selected["scissor"]["x_aft_over_mac"]

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "mass_breakdown.csv").exists()
    assert (tmp_path / "scissor_plot.png").exists()
    assert (tmp_path / "mission_profile.png").exists()


def test_summary_table_contains_main_report_values(tmp_path):
    simple_sizing.run_sizing(output_dir=tmp_path, make_plots=False)

    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as csv_file:
        rows = {row["quantity"]: row["value"] for row in csv.DictReader(csv_file)}

    assert "MTOW_mass_estimate_kg" in rows
    assert "wing_area_m2" in rows
    assert "canard_area_ratio" in rows
    assert "x_CG_over_MAC" in rows
