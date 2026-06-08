import numpy as np
import pytest

from bellona_sizing.phases.phase06_constraints import phase6_constraint_diagram


def test_phase2_hover_power_loading_is_a_sizing_constraint():
    hover_power_loading = 30.0
    result = phase6_constraint_diagram(
        np.linspace(40.0, 100.0, 5),
        np.array([]),
        V_cruise=20.0,
        ROC=5.0,
        gamma=np.deg2rad(15.0),
        rho_6km=0.66,
        CD0=0.04,
        AR=7.0,
        e=0.78,
        eta_prop=0.75 * 0.90 * 0.95,
        T_W_floor=1.3,
        selected_W_S=70.0,
        hover_power_loading_W_N=hover_power_loading,
    )

    assert result["P_W_hover"] == pytest.approx([hover_power_loading] * 5)
    assert result["hover_power_loading_W_N"] == pytest.approx(hover_power_loading)
    assert np.all(
        np.asarray(result["P_W_envelope"]) >= np.asarray(result["P_W_hover"])
    )
    assert not any("hover sizing constraint is absent" in w for w in result["warnings"])
    assert result["selected_W_S"] == pytest.approx(70.0)
    assert result["selected_P_W"] is not None
    assert "W_S_stall_max" not in result
