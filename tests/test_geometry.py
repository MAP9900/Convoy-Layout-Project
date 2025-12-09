"""Smoke tests for geometry scaffolding."""

import pytest

from convoy_sim import (
    Point2D,
    Vector2D,
    bearing_between,
    rotate_point,
    translate_point,
)


def test_geometry_imports() -> None:
    point = Point2D(0.0, 0.0)
    vector = Vector2D(1.0, 1.0)
    assert point.x == 0.0
    assert vector.dy == 1.0


def test_geometry_methods_raise_not_implemented() -> None:
    start = Point2D(0.0, 0.0)
    end = Point2D(10.0, 0.0)
    with pytest.raises(NotImplementedError):
        start.distance_to(end)
    with pytest.raises(NotImplementedError):
        translate_point(start, Vector2D(1.0, 2.0))
    with pytest.raises(NotImplementedError):
        rotate_point(start, angle_deg=45.0)
    with pytest.raises(NotImplementedError):
        bearing_between(start, end)
