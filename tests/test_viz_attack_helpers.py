"""Tests for attack visualization helpers without matplotlib."""

import numpy as np

from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec
from convoy_sim.viz_attack import attack_debug_metrics, min_miss_distance_ship_torpedo


def _ship_at(pos: np.ndarray) -> Ship:
    return Ship(
        id="S1",
        position=pos,
        speed=0.0,
        heading_rad=0.0,
        length=40.0,
        beam=10.0,
        ship_class=ShipClass.FREIGHTER,
    )


def test_min_miss_distance_direct_hit() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-100.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min <= ship.effective_hit_radius()


def test_min_miss_distance_clean_miss() -> None:
    ship = _ship_at(as_vec(0.0, 100.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-100.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min > ship.effective_hit_radius()


def test_attack_debug_metrics_keys() -> None:
    ships = [
        _ship_at(as_vec(0.0, 0.0)),
        Ship(
            id="S2",
            position=as_vec(50.0, 50.0),
            speed=0.0,
            heading_rad=0.0,
            length=40.0,
            beam=10.0,
            ship_class=ShipClass.ESCORT,
        ),
    ]
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-100.0, 0.0),
            speed=10.0,
            heading_rad=0.0,
            max_run_time=20.0,
        )
    ]
    metrics = attack_debug_metrics(ships, torpedoes, t_max=20.0)
    assert "torpedoes" in metrics
    assert "ships" in metrics
    assert metrics["torpedoes"][0]["closest_ship_id"] in {"S1", "S2"}


def test_min_miss_distance_handles_gyro_turn_path() -> None:
    ship = _ship_at(as_vec(20.0, 80.0))
    torpedo = Torpedo(
        id="Tgyro",
        launch_position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=np.pi / 2.0,
        max_run_time=20.0,
        launch_heading_rad=0.0,
        gyro_turn_distance_m=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min <= ship.effective_hit_radius()
