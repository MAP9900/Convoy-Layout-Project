"""Tests for value-based hit scoring."""

import numpy as np

from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec
from convoy_sim.simulation import simulate_attack_once_scored


def test_value_destroyed_single_hit() -> None:
    ships = [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=120.0,
            beam=20.0,
            ship_class=ShipClass.FREIGHTER,
            value_weight=1.0,
        ),
        Ship(
            id="S2",
            position=as_vec(0.0, 400.0),
            speed=0.0,
            heading_rad=0.0,
            length=120.0,
            beam=20.0,
            ship_class=ShipClass.TANKER,
            value_weight=3.0,
        ),
    ]
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-500.0, 0.0),
            speed=20.0,
            heading_rad=0.0,
            max_run_time=200.0,
        )
    ]
    scored = simulate_attack_once_scored(ships, torpedoes, t_max=200.0)
    assert scored["total_value_destroyed"] == 1.0
    assert scored["hits_by_class"][ShipClass.FREIGHTER] == 1


def test_value_counted_once_per_ship() -> None:
    ships = [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=120.0,
            beam=20.0,
            ship_class=ShipClass.FREIGHTER,
            value_weight=2.0,
        )
    ]
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-500.0, 0.0),
            speed=20.0,
            heading_rad=0.0,
            max_run_time=200.0,
        ),
        Torpedo(
            id="T2",
            launch_position=as_vec(-500.0, 0.0),
            speed=20.0,
            heading_rad=0.0,
            max_run_time=200.0,
        ),
    ]
    scored = simulate_attack_once_scored(ships, torpedoes, t_max=200.0)
    assert scored["n_hits"] >= 1
    assert scored["total_value_destroyed"] == 2.0


def test_scored_attack_reports_unique_and_repeat_hits() -> None:
    ships = [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=120.0,
            beam=20.0,
            ship_class=ShipClass.FREIGHTER,
            value_weight=2.0,
        )
    ]
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-500.0, 0.0),
            speed=20.0,
            heading_rad=0.0,
            max_run_time=200.0,
        ),
        Torpedo(
            id="T2",
            launch_position=as_vec(-500.0, 0.0),
            speed=20.0,
            heading_rad=0.0,
            max_run_time=200.0,
        ),
    ]
    scored = simulate_attack_once_scored(ships, torpedoes, t_max=200.0)
    assert scored["unique_ships_hit"] == 1
    assert scored["repeat_hits"] == scored["n_hits"] - 1
