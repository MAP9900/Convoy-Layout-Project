"""Tests for visualization helpers without requiring matplotlib."""

import sys

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.viz import compute_footprint_polygon


def test_viz_import_does_not_require_matplotlib() -> None:
    assert "matplotlib" not in sys.modules


def test_compute_footprint_polygon_bbox() -> None:
    ships = [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=50.0,
            beam=10.0,
            ship_class=ShipClass.FREIGHTER,
        ),
        Ship(
            id="S2",
            position=as_vec(10.0, 20.0),
            speed=0.0,
            heading_rad=0.0,
            length=50.0,
            beam=10.0,
            ship_class=ShipClass.ESCORT,
        ),
    ]
    poly = compute_footprint_polygon(ships, padding=5.0)
    assert poly.shape == (5, 2)
    assert poly[0, 0] <= 0.0
    assert poly[2, 1] >= 20.0
