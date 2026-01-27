"""Tests for pure-numpy visualization helpers."""

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.viz import layout_summary


def test_layout_summary_counts_and_bbox() -> None:
    ships = [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=40.0,
            beam=10.0,
            ship_class=ShipClass.FREIGHTER,
            value_weight=2.0,
        ),
        Ship(
            id="S2",
            position=as_vec(10.0, 20.0),
            speed=0.0,
            heading_rad=0.0,
            length=40.0,
            beam=10.0,
            ship_class=ShipClass.ESCORT,
            value_weight=3.0,
        ),
    ]
    summary = layout_summary(ships)
    counts = summary["counts_by_class"]
    assert counts["freighter"] == 1
    assert counts["escort"] == 1
    assert summary["total_value"] == 5.0
    assert summary["bbox_along"] == 10.0
    assert summary["bbox_across"] == 20.0
