import math

from xfoil_wrapper import (
    datcom_efficiency_from_section_slope,
    summarize_polar_rows,
)


def test_summarize_polar_rows_extracts_airfoil_coefficients():
    rows = [
        {"alpha_deg": -2.0, "cl": -0.10, "cd": 0.012, "cdp": 0.011, "cm": -0.050},
        {"alpha_deg": 0.0, "cl": 0.12, "cd": 0.010, "cdp": 0.009, "cm": -0.055},
        {"alpha_deg": 2.0, "cl": 0.34, "cd": 0.011, "cdp": 0.010, "cm": -0.060},
        {"alpha_deg": 4.0, "cl": 0.56, "cd": 0.014, "cdp": 0.013, "cm": -0.065},
        {"alpha_deg": 8.0, "cl": 0.98, "cd": 0.030, "cdp": 0.028, "cm": -0.070},
        {"alpha_deg": 12.0, "cl": 1.20, "cd": 0.060, "cdp": 0.055, "cm": -0.080},
    ]

    summary = summarize_polar_rows(
        rows,
        airfoil="Example",
        reynolds=500000.0,
        mach=0.10,
        x_transition=0.50,
        source="test",
    )

    assert summary["cl_max"] == 1.20
    assert summary["alpha_cl_max_deg"] == 12.0
    assert math.isclose(summary["cl_at_alpha0"], 0.12, rel_tol=1e-9)
    assert summary["cl_alpha_per_rad"] > 5.0
    assert summary["cm_at_zero_lift"] < 0.0


def test_datcom_efficiency_from_section_slope_is_bounded():
    assert 0.70 <= datcom_efficiency_from_section_slope(0.1) <= 1.05
    assert math.isclose(
        datcom_efficiency_from_section_slope(2.0 * math.pi),
        1.0,
        rel_tol=1e-12,
    )
