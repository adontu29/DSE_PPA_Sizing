import pytest

from bellona_sizing.models import Assumptions
from bellona_sizing.phases.phase05_energy import phase5_energy_battery


def test_project_battery_specific_energy_default_is_used_by_phase5():
    result = phase5_energy_battery(
        {"mission_reserve": (1000.0, 3600.0)},
        eta_batt=1.0,
        f_usable=1.0,
    )

    assert Assumptions().battery_specific_energy_Wh_kg == pytest.approx(310.0)
    assert result["e_batt_Wh_kg"] == pytest.approx(310.0)
    assert result["m_batt_kg"] == pytest.approx(1000.0 / 310.0)
