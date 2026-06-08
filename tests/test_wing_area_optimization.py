import pytest

from bellona_sizing.workflows.wing_area import optimize_wing_area


def _fake_result(area_m2, objective, mission_feasible=True,
                 yaw_margin=1.2, transition_margin=5.0,
                 power_reserve_margin=1000.0):
    return {
        "converged": True,
        "phase3": {
            "E_total_Wh": 1000.0 + area_m2,
            "reference_level_flight_TAS_m_s": 20.0,
            "carry_forward": {
                "all_required_cases_feasible": mission_feasible,
                "transition_reference": {
                    "minimum_transition_complete_TAS_m_s": 15.0,
                    "max_transition_complete_speed_m_s": 20.0,
                    "transition_complete_speed_margin_m_s": transition_margin,
                    "stall_EAS_margin_m_s": None,
                },
                "minimum_constraint_margins": {
                    "power_margin_over_required_W": power_reserve_margin,
                },
            },
        },
        "phase4": {"feasible": mission_feasible},
        "phase8": {
            "stall_EAS_m_s": 12.0,
            "stall_TAS_mission_m_s": 16.0,
            "stall_TAS_transition_m_s": 12.1,
        },
        "phase10": {"operational_CG_feasible": True},
        "phase11": {"feasible_preliminary_elevon": True},
        "phase12": {
            "feasible_preliminary_hover_control": True,
            "pitch_margin": 1.2,
            "roll_margin": 1.2,
            "yaw_margin": yaw_margin,
        },
        "phase13": {
            "V_cruise": 20.0,
            "V_cruise_required_for_margin": 15.0,
            "V_blend_end": 14.0,
        },
        "phase15": {"MTOW_estimate_kg": objective},
    }


def test_optimizer_expands_and_refines_to_interior_minimum():
    def evaluator(area_m2):
        return _fake_result(area_m2, 50.0 + (area_m2 - 12.0) ** 2)

    result, report = optimize_wing_area(
        evaluator,
        seed_area_m2=5.0,
        area_tol=0.02,
    )

    assert report["mode"] == "optimized"
    assert any(item["direction"] == "upper" for item in report["expansion_history"])
    assert report["selected_area_m2"] == pytest.approx(12.0, rel=0.03)
    assert report["local_minimum_verified"]
    assert result["phase15"]["MTOW_estimate_kg"] == pytest.approx(
        report["selected_objective_MTOW_estimate_kg"]
    )


def test_optimizer_rejects_mission_infeasible_candidates():
    def evaluator(area_m2):
        mission_feasible = area_m2 >= 4.0
        return _fake_result(
            area_m2,
            40.0 + (area_m2 - 3.0) ** 2,
            mission_feasible=mission_feasible,
        )

    _, report = optimize_wing_area(
        evaluator,
        seed_area_m2=5.0,
        area_tol=0.02,
    )

    assert report["selected_area_m2"] >= 4.0
    rejected = [item for item in report["candidates"] if not item["feasible"]]
    assert rejected
    assert any(
        "mission" in reason
        for item in rejected
        for reason in item["failure_reasons"]
    )


def test_optimizer_rejects_hover_margin_below_target():
    def evaluator(area_m2):
        return _fake_result(
            area_m2,
            20.0 + area_m2,
            yaw_margin=1.02 if area_m2 < 6.0 else 1.10,
        )

    _, report = optimize_wing_area(
        evaluator,
        seed_area_m2=5.0,
        area_tol=0.05,
        hover_control_margin_min=1.05,
    )

    assert report["selected_area_m2"] >= 6.0
    rejected = [item for item in report["candidates"] if not item["feasible"]]
    assert any(
        "hover-control margin" in reason
        for item in rejected
        for reason in item["failure_reasons"]
    )


def test_optimizer_rejects_transition_speed_limit_failures():
    def evaluator(area_m2):
        return _fake_result(
            area_m2,
            20.0 + area_m2,
            transition_margin=-0.5 if area_m2 < 6.0 else 1.0,
        )

    _, report = optimize_wing_area(
        evaluator,
        seed_area_m2=5.0,
        area_tol=0.05,
    )

    assert report["selected_area_m2"] >= 6.0
    rejected = [item for item in report["candidates"] if not item["feasible"]]
    assert any(
        "transition-complete speed" in reason
        for item in rejected
        for reason in item["failure_reasons"]
    )


def test_optimizer_rejects_power_reserve_margin_failures():
    def evaluator(area_m2):
        return _fake_result(
            area_m2,
            20.0 + area_m2,
            power_reserve_margin=-10.0 if area_m2 < 6.0 else 100.0,
        )

    _, report = optimize_wing_area(
        evaluator,
        seed_area_m2=5.0,
        area_tol=0.05,
    )

    assert report["selected_area_m2"] >= 6.0
    rejected = [item for item in report["candidates"] if not item["feasible"]]
    assert any(
        "power reserve margin" in reason
        for item in rejected
        for reason in item["failure_reasons"]
    )
