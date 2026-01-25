"""Tests for convoy formation motion helpers."""

import math

import numpy as np

from convoy_sim.dynamics import (
    ConvoyFormation,
    ConvoyKinematics,
    RouteLeg,
    RoutePlan,
    convoy_pose_at,
    ship_positions_at,
)
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec


def _rotate_inv(angle_rad: float, vec: np.ndarray) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=float)
    return rotation @ vec


def test_convoy_pose_at_straight_motion() -> None:
    kin = ConvoyKinematics()
    origin0 = as_vec(0.0, 0.0)
    origin_t, heading_t, speed_t = convoy_pose_at(
        5.0,
        origin0,
        heading0=0.0,
        speed0=2.0,
        kin=kin,
        dt=1.0,
    )
    assert np.allclose(origin_t, np.array([10.0, 0.0]))
    assert math.isclose(heading_t, 0.0)
    assert math.isclose(speed_t, 2.0)


def test_convoy_pose_at_route_turn() -> None:
    route = RoutePlan(
        legs=[
            RouteLeg(duration_s=5.0, heading_rad=0.0),
            RouteLeg(duration_s=5.0, heading_rad=math.pi / 2.0),
        ]
    )
    kin = ConvoyKinematics(route=route)
    origin_t, heading_t, _speed_t = convoy_pose_at(
        7.0,
        as_vec(0.0, 0.0),
        heading0=0.0,
        speed0=1.0,
        kin=kin,
        dt=1.0,
    )
    assert np.allclose(origin_t, np.array([5.0, 2.0]))
    assert math.isclose(heading_t, math.pi / 2.0)


def test_ship_offsets_remain_constant_in_convoy_frame() -> None:
    ships = [
        Ship(
            id="lead",
            position=as_vec(0.0, 0.0),
            speed=2.0,
            heading_rad=0.0,
            length=10.0,
            beam=3.0,
            ship_class=ShipClass.FREIGHTER,
        ),
        Ship(
            id="wing",
            position=as_vec(0.0, 10.0),
            speed=2.0,
            heading_rad=0.0,
            length=10.0,
            beam=3.0,
            ship_class=ShipClass.FREIGHTER,
        ),
    ]
    formation = ConvoyFormation(
        ships0=ships,
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    route = RoutePlan(
        legs=[
            RouteLeg(duration_s=5.0, heading_rad=0.0),
            RouteLeg(duration_s=5.0, heading_rad=math.pi / 2.0),
        ]
    )
    kin = ConvoyKinematics(route=route)
    positions = ship_positions_at(6.0, formation, kin, dt=1.0)
    origin_t, heading_t, _speed_t = convoy_pose_at(
        6.0,
        formation.convoy_origin0,
        formation.convoy_heading0,
        speed0=2.0,
        kin=kin,
        dt=1.0,
    )
    for pos, offset0 in zip(positions, formation.offsets_at_t0()):
        offset_t = _rotate_inv(heading_t, pos - origin_t)
        assert np.allclose(offset_t, offset0)
