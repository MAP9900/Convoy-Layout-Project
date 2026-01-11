"""Deterministic torpedo spread samplers for attacker configurations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from convoy_sim.entities import Torpedo
from convoy_sim.geometry import Vec2, as_vec


def _vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector")
    return arr


def fan_spread(
    u_pos: Vec2 | Sequence[float],
    base_bearing_rad: float,
    n: int,
    spread_rad: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from ``u_pos`` spread evenly across ``spread_rad``."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    if n == 1 or spread_rad == 0.0:
        headings = [base_bearing_rad]
    else:
        half_spread = spread_rad / 2.0
        headings = [
            base_bearing_rad - half_spread + (spread_rad * i / (n - 1))
            for i in range(n)
        ]
    return [
        Torpedo(
            id=f"F{i+1:02d}",
            launch_position=origin_vec,
            speed=speed,
            heading_rad=heading,
            max_run_time=max_run_time,
        )
        for i, heading in enumerate(headings)
    ]


def parallel_spread(
    u_pos: Vec2 | Sequence[float],
    bearing_rad: float,
    n: int,
    lateral_spacing: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from parallel launchers offset laterally."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    perp = as_vec(-math.sin(bearing_rad), math.cos(bearing_rad))
    center = (n - 1) / 2.0
    launches = [
        origin_vec + perp * ((i - center) * lateral_spacing)
        for i in range(n)
    ]
    return [
        Torpedo(
            id=f"P{i+1:02d}",
            launch_position=launch_pos,
            speed=speed,
            heading_rad=bearing_rad,
            max_run_time=max_run_time,
        )
        for i, launch_pos in enumerate(launches)
    ]
