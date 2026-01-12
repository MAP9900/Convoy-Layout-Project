"""Ship class catalog for heterogeneous convoy modeling."""

from __future__ import annotations

from typing import Any

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import Vec2

SHIP_CATALOG: dict[ShipClass, dict[str, float]] = {
    ShipClass.FREIGHTER: {"length": 140.0, "beam": 20.0, "value_weight": 1.0},
    ShipClass.TANKER: {"length": 180.0, "beam": 28.0, "value_weight": 1.5},
    ShipClass.ESCORT: {"length": 90.0, "beam": 12.0, "value_weight": 0.5},
    ShipClass.DECOY: {"length": 120.0, "beam": 18.0, "value_weight": 0.2},
}


def make_ship(
    ship_id: str,
    ship_class: ShipClass,
    position: Vec2,
    speed: float,
    heading_rad: float,
    overrides: dict[str, Any] | None = None,
) -> Ship:
    """Create a Ship using catalog defaults with optional overrides."""

    base = dict(SHIP_CATALOG[ship_class])
    if overrides:
        base.update(overrides)
    return Ship(
        id=ship_id,
        position=position,
        speed=speed,
        heading_rad=heading_rad,
        length=float(base["length"]),
        beam=float(base["beam"]),
        ship_class=ship_class,
        value_weight=float(base.get("value_weight", 1.0)),
        hit_radius=base.get("hit_radius"),
    )
