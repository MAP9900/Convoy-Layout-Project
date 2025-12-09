"""Basic 2D geometry utilities for convoy and torpedo dynamics.

All coordinates are expressed in meters on a flat Euclidean plane.  The helpers
here intentionally contain only typed interfaces for now so downstream modules
can depend on a stable API as the math models evolve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """Cartesian point measured in meters."""

    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        """Return the Euclidean distance to ``other`` in meters."""

        raise NotImplementedError


@dataclass(frozen=True)
class Vector2D:
    """2D vector expressed in meters along x and y axes."""

    dx: float
    dy: float

    def magnitude(self) -> float:
        """Return the vector magnitude (meters)."""

        raise NotImplementedError


def translate_point(point: Point2D, vector: Vector2D) -> Point2D:
    """Return a new point translated by ``vector``."""

    raise NotImplementedError


def rotate_point(point: Point2D, angle_deg: float, origin: Point2D | None = None) -> Point2D:
    """Rotate ``point`` around ``origin`` (default: global origin) by ``angle_deg``."""

    raise NotImplementedError


def bearing_between(start: Point2D, end: Point2D) -> float:
    """Return the compass bearing in degrees from ``start`` to ``end``."""

    raise NotImplementedError
