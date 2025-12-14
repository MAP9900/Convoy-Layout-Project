"""Smoke tests for geometry scaffolding."""

import math

import numpy as np

from convoy_sim.geometry import (
    Point2D,
    Vec2,
    as_vec,
    closest_approach_time,
    distance,
    min_distance_over_interval,
    step_position,
)


def test_distance_three_four_five() -> None:
    origin = as_vec(0.0, 0.0)
    target = as_vec(3.0, 4.0)
    assert distance(origin, target) == 5.0
    assert Point2D(0.0, 0.0).distance_to(Point2D(3.0, 4.0)) == 5.0


def test_step_position_linear_motion() -> None:
    p0: Vec2 = as_vec(0.0, 0.0)
    v: Vec2 = as_vec(2.0, -1.0)
    result = step_position(p0, v, 3.0)
    assert np.allclose(result, np.array([6.0, -3.0]))


def test_closest_approach_time_for_crossing_tracks() -> None:
    ship_a_pos = as_vec(0.0, 0.0)
    ship_a_vel = as_vec(10.0, 0.0)
    ship_b_pos = as_vec(100.0, 100.0)
    ship_b_vel = as_vec(0.0, -10.0)
    t_ca = closest_approach_time(ship_a_pos, ship_a_vel, ship_b_pos, ship_b_vel)
    assert math.isclose(t_ca, 10.0)


def test_min_distance_over_interval_crossing_tracks() -> None:
    ship_a_pos = as_vec(-50.0, 0.0)
    ship_a_vel = as_vec(10.0, 0.0)
    ship_b_pos = as_vec(0.0, -50.0)
    ship_b_vel = as_vec(0.0, 10.0)
    d_min = min_distance_over_interval(
        ship_a_pos,
        ship_a_vel,
        ship_b_pos,
        ship_b_vel,
        t_min=0.0,
        t_max=10.0,
    )
    assert math.isclose(d_min, 0.0, abs_tol=1e-6)
