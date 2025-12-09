"""Unit tests for basic ship and torpedo motion utilities."""

import math

import numpy as np

from convoy_sim import as_vec
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


def test_torpedo_hits_stationary_ship() -> None:
    ship = Ship(
        id="cargo-1",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=math.pi / 2,
        length=120.0,
        beam=18.0,
    )
    torpedo = Torpedo(
        id="torp-hit",
        launch_position=as_vec(-500.0, 0.0),
        speed=50.0,
        heading_rad=0.0,
        max_run_time=400.0,
    )
    assert torpedo_hits_ship(ship, torpedo, t_max=60.0)


def test_torpedo_misses_offset_ship() -> None:
    ship = Ship(
        id="cargo-2",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
    )
    torpedo = Torpedo(
        id="torp-miss",
        launch_position=as_vec(-500.0, 300.0),
        speed=50.0,
        heading_rad=0.0,
        max_run_time=400.0,
    )
    assert not torpedo_hits_ship(ship, torpedo, t_max=60.0)
