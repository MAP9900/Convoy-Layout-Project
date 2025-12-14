"""Unit tests for basic ship and torpedo motion utilities."""

import math

import numpy as np

from convoy_sim import as_vec
from convoy_sim.geometry import min_distance_over_interval
from convoy_sim.entities import Ship, Torpedo, torpedo_hits_ship


def test_ship_position_at_moves_linearly() -> None:
    ship = Ship(
        id="cargo-1",
        position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        length=150.0,
        beam=20.0,
    )
    future = ship.position_at(5.0)
    assert np.allclose(future, np.array([50.0, 0.0]))


def test_torpedo_position_at_moves_linearly() -> None:
    torpedo = Torpedo(
        id="type-a",
        launch_position=as_vec(-100.0, 0.0),
        speed=20.0,
        heading_rad=0.0,
        max_run_time=200.0,
    )
    assert np.allclose(torpedo.position_at(3.0), np.array([-40.0, 0.0]))


def _stationary_ship() -> Ship:
    return Ship(
        id="cargo",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=120.0,
        beam=40.0,
    )


def test_torpedo_hits_stationary_ship_centerline() -> None:
    ship = _stationary_ship()
    torpedo = Torpedo(
        id="torp-hit",
        launch_position=as_vec(-500.0, 0.0),
        speed=50.0,
        heading_rad=0.0,
        max_run_time=400.0,
    )
    assert torpedo_hits_ship(ship, torpedo, t_max=60.0)


def test_torpedo_misses_parallel_offset_gt_radius() -> None:
    ship = _stationary_ship()
    radius = max(ship.length, ship.beam) * 0.5
    torpedo = Torpedo(
        id="torp-miss",
        launch_position=as_vec(-500.0, radius + 20.0),
        speed=50.0,
        heading_rad=0.0,
        max_run_time=400.0,
    )
    assert not torpedo_hits_ship(ship, torpedo, t_max=60.0)


def test_torpedo_hits_moving_crossing_ship() -> None:
    ship = Ship(
        id="moving",
        position=as_vec(0.0, -200.0),
        speed=10.0,
        heading_rad=math.pi / 2,
        length=120.0,
        beam=40.0,
    )
    torpedo = Torpedo(
        id="torp-cross",
        launch_position=as_vec(-300.0, 0.0),
        speed=20.0,
        heading_rad=0.0,
        max_run_time=400.0,
    )
    assert torpedo_hits_ship(ship, torpedo, t_max=40.0)
    d_min = min_distance_over_interval(
        ship.position,
        ship.velocity_vec(),
        torpedo.launch_position,
        torpedo.velocity_vec(),
        0.0,
        40.0,
    )
    assert math.isclose(d_min, math.sqrt(2000.0), rel_tol=1e-6)
