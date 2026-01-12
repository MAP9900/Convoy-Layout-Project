"""Tests for objective scoring helpers."""

from convoy_sim.entities import ShipClass
from convoy_sim.objectives import ObjectiveSpec, score_trial_result


def test_objective_loss_increases_with_value() -> None:
    obj = ObjectiveSpec(w_total_value=1.0, w_total_hits=0.0, mode="defender_minimize")
    low = {
        "total_value_destroyed": 1.0,
        "n_hits": 1,
        "value_destroyed_by_class": {ShipClass.FREIGHTER: 1.0},
    }
    high = {
        "total_value_destroyed": 3.0,
        "n_hits": 1,
        "value_destroyed_by_class": {ShipClass.FREIGHTER: 3.0},
    }
    assert score_trial_result(high, obj) > score_trial_result(low, obj)


def test_class_weighting_changes_score() -> None:
    obj = ObjectiveSpec(
        w_total_value=1.0,
        class_value_weights={ShipClass.TANKER: 2.0},
        escort_loss_discount=1.0,
        mode="defender_minimize",
    )
    base = {
        "total_value_destroyed": 2.0,
        "n_hits": 1,
        "value_destroyed_by_class": {ShipClass.TANKER: 2.0},
    }
    unweighted = ObjectiveSpec(w_total_value=1.0, mode="defender_minimize")
    assert score_trial_result(base, obj) > score_trial_result(base, unweighted)
