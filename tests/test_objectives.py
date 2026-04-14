"""Tests for objective scoring helpers."""

from convoy_sim.entities import ShipClass
from convoy_sim.objectives import ObjectiveSpec, objective_from_config, score_trial_result


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


def test_unique_ship_penalty_exceeds_repeat_hit_penalty() -> None:
    obj = ObjectiveSpec(
        w_total_value=0.0,
        w_unique_ships_hit=1.0,
        w_repeat_hits=0.2,
        mode="defender_minimize",
    )
    concentrated = {
        "n_hits": 4,
        "unique_ships_hit": 1,
        "repeat_hits": 3,
        "total_value_destroyed": 0.0,
        "value_destroyed_by_class": {},
    }
    distributed = {
        "n_hits": 4,
        "unique_ships_hit": 4,
        "repeat_hits": 0,
        "total_value_destroyed": 0.0,
        "value_destroyed_by_class": {},
    }
    assert score_trial_result(distributed, obj) > score_trial_result(concentrated, obj)


def test_objective_from_config_parses_class_weights() -> None:
    obj = objective_from_config(
        {
            "w_total_value": 1.0,
            "w_unique_ships_hit": 1.0,
            "w_repeat_hits": 0.2,
            "escort_loss_discount": 0.5,
            "class_value_weights": {
                "freighter": 1.0,
                "tanker": 1.8,
                "escort": 0.5,
            },
        }
    )
    assert obj is not None
    assert obj.class_value_weights is not None
    assert obj.class_value_weights[ShipClass.TANKER] == 1.8
