from pathlib import Path

from bellona_sizing.models import Assumptions
from bellona_sizing.phases import phase07_airfoil


def test_bundled_xfoil_assets_are_project_defaults(monkeypatch):
    assert Assumptions().use_xfoil

    calls = []

    def fake_run(xfoil_exe, airfoil, reynolds, x_transition, design_cl,
                 coordinate_file=None, **kwargs):
        calls.append({
            "xfoil_exe": xfoil_exe,
            "airfoil": airfoil,
            "coordinate_file": coordinate_file,
        })
        return {
            "airfoil": airfoil,
            "Re": reynolds,
            "x_transition": x_transition,
            "cl_a": 6.0,
            "cl_max": 1.2,
            "source": "xfoil",
        }, None

    monkeypatch.setattr(phase07_airfoil, "_run_xfoil_airfoil", fake_run)
    result = phase07_airfoil.phase7_airfoil_xfoil(
        700_000.0,
        300_000.0,
    )

    assert result["xfoil"]["requested"]
    assert result["xfoil"]["used"]
    assert Path(result["xfoil"]["executable"]).name == "xfoilp4.exe"
    assert calls[0]["airfoil"] == "SD7037"
    assert Path(calls[0]["coordinate_file"]).name == "sd7037.dat"
    assert calls[1]["airfoil"] == "NACA 0012"
    assert calls[1]["coordinate_file"] is None
