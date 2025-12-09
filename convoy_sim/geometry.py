"""Basic 2D geometry utilities for convoy and torpedo dynamics.

All coordinates are expressed in meters on a flat Euclidean plane.  The helpers
here intentionally contain only typed interfaces for now so downstream modules
can depend on a stable API as the math models evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

Vec2 = np.ndarray


def as_vec(x: float, y: float) -> Vec2:
    """Return a 2D vector expressed in meters."""

    return np.array([float(x), float(y)], dtype=float)


def _ensure_vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector")
    return arr


def norm(v: Vec2) -> float:
    """Return the Euclidean norm of ``v`` in meters."""

    return float(np.linalg.norm(_ensure_vec(v)))


def unit(v: Vec2) -> Vec2:
    """Return the unit vector pointing along ``v``."""

    vector = _ensure_vec(v)
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise ValueError("Cannot normalise a zero-length vector")
    return vector / magnitude


def distance(a: Vec2, b: Vec2) -> float:
    """Return the Euclidean distance between two positions."""

    return norm(_ensure_vec(a) - _ensure_vec(b))


def step_position(p0: Vec2, v: Vec2, dt: float) -> Vec2:
    """Advance ``p0`` by velocity ``v`` over ``dt`` seconds."""

    return _ensure_vec(p0) + _ensure_vec(v) * float(dt)


def closest_approach_time(p0_a: Vec2, v_a: Vec2, p0_b: Vec2, v_b: Vec2) -> float:
    """Return the time of closest approach between two tracks (seconds, ≥ 0)."""

    r = _ensure_vec(p0_a) - _ensure_vec(p0_b)
    v_rel = _ensure_vec(v_a) - _ensure_vec(v_b)
    speed_sq = float(np.dot(v_rel, v_rel))
    if speed_sq == 0.0:
        return 0.0
    t_star = -float(np.dot(v_rel, r)) / speed_sq
    return max(t_star, 0.0)


def min_distance_over_interval(
    p0_a: Vec2,
    v_a: Vec2,
    p0_b: Vec2,
    v_b: Vec2,
    t_min: float,
    t_max: float,
) -> float:
    """Compute the minimum distance between two paths over ``[t_min, t_max]`` seconds."""

    if t_min > t_max:
        raise ValueError("t_min must be <= t_max")
    r = _ensure_vec(p0_a) - _ensure_vec(p0_b)
    v_rel = _ensure_vec(v_a) - _ensure_vec(v_b)
    speed_sq = float(np.dot(v_rel, v_rel))
    candidate_times = [t_min, t_max]
    if speed_sq > 0.0:
        t_star = -float(np.dot(v_rel, r)) / speed_sq
        candidate_times.append(float(np.clip(t_star, t_min, t_max)))
    distances = []
    for t in candidate_times:
        pos_a = step_position(p0_a, v_a, t)
        pos_b = step_position(p0_b, v_b, t)
        distances.append(distance(pos_a, pos_b))
    return min(distances)


@dataclass(frozen=True)
class Point2D:
    """Cartesian point measured in meters."""

    x: float
    y: float

    def as_vec(self) -> Vec2:
        """Return the point as a Vec2."""

        return as_vec(self.x, self.y)

    def distance_to(self, other: "Point2D") -> float:
        """Return the Euclidean distance to ``other`` in meters."""

        return distance(self.as_vec(), other.as_vec())


@dataclass(frozen=True)
class Vector2D:
    """2D vector expressed in meters along x and y axes."""

    dx: float
    dy: float

    def as_vec(self) -> Vec2:
        """Return the vector as a Vec2."""

        return as_vec(self.dx, self.dy)

    def magnitude(self) -> float:
        """Return the vector magnitude (meters)."""

        return norm(self.as_vec())


def translate_point(point: Point2D, vector: Vector2D) -> Point2D:
    """Return a new point translated by ``vector``."""

    vec = point.as_vec() + vector.as_vec()
    return Point2D(float(vec[0]), float(vec[1]))


def rotate_point(point: Point2D, angle_deg: float, origin: Point2D | None = None) -> Point2D:
    """Rotate ``point`` around ``origin`` (default: global origin) by ``angle_deg``."""

    origin_vec = as_vec(0.0, 0.0) if origin is None else origin.as_vec()
    translated = point.as_vec() - origin_vec
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    rotated = rotation_matrix @ translated + origin_vec
    return Point2D(float(rotated[0]), float(rotated[1]))


def bearing_between(start: Point2D, end: Point2D) -> float:
    """Return the compass bearing in degrees from ``start`` to ``end``."""

    delta = end.as_vec() - start.as_vec()
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    return angle % 360.0
