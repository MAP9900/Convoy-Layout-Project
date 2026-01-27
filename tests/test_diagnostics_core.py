"""Tests for core diagnostics helpers."""

import numpy as np

from convoy_sim.diagnostics import compare_attack_outcomes, compare_layout_metrics, lane_vulnerability_proxy
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec


def _ship_at(pos: np.ndarray, ship_id: str) -> Ship:
    return Ship(
        id=ship_id,
        position=pos,
        speed=0.0,
        heading_rad=0.0,
        length=40.0,
        beam=10.0,
        ship_class=ShipClass.FREIGHTER,
    )


def test_compare_layout_metrics_keys() -> None:
    ships_a = [_ship_at(as_vec(0.0, 0.0), "S1"), _ship_at(as_vec(10.0, 0.0), "S2")]
    ships_b = [_ship_at(as_vec(0.0, 0.0), "S1"), _ship_at(as_vec(20.0, 0.0), "S2")]
    result = compare_layout_metrics(ships_a, ships_b)
    assert "before" in result
    assert "after" in result
    assert "delta" in result
    assert result["delta"]["nn_min"] >= 0.0


def test_compare_attack_outcomes_deltas() -> None:
    a = {"expected_hits": 2.0, "expected_value_destroyed": 5.0, "hit_prob_at_least_one": 0.7}
    b = {"expected_hits": 1.5, "expected_value_destroyed": 4.0, "hit_prob_at_least_one": 0.5}
    delta = compare_attack_outcomes(a, b)
    assert np.isclose(delta["delta_expected_hits"], -0.5)
    assert np.isclose(delta["delta_expected_value"], -1.0)


def test_lane_vulnerability_proxy_shapes() -> None:
    ships = [_ship_at(as_vec(0.0, 0.0), "S1"), _ship_at(as_vec(10.0, 0.0), "S2")]
    headings = np.linspace(-np.pi, np.pi, 5)
    lane = lane_vulnerability_proxy(ships, headings=headings, n_rays=10)
    assert len(lane["max_hits"]) == len(headings)
    assert all(value >= 0.0 for value in lane["lane_score"])
