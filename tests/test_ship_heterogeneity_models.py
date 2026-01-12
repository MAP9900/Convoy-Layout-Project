"""Tests for heterogeneous ship models and catalog defaults."""

import numpy as np

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.ship_catalog import SHIP_CATALOG, make_ship


def test_ship_defaults_backwards_compatible() -> None:
    ship = Ship(
        id="S1",
        position=as_vec(0.0, 0.0),
        speed=5.0,
        heading_rad=0.0,
        length=120.0,
        beam=20.0,
    )
    assert ship.ship_class == ShipClass.FREIGHTER
    assert ship.value_weight == 1.0
    assert ship.effective_hit_radius() > 0.0


def test_catalog_distinguishes_ship_classes() -> None:
    freighter = make_ship("F1", ShipClass.FREIGHTER, as_vec(0.0, 0.0), 5.0, 0.0)
    tanker = make_ship("T1", ShipClass.TANKER, as_vec(0.0, 0.0), 5.0, 0.0)
    assert freighter.length != tanker.length or freighter.beam != tanker.beam
    assert freighter.value_weight != tanker.value_weight


def test_effective_hit_radius_override() -> None:
    ship = Ship(
        id="S2",
        position=as_vec(0.0, 0.0),
        speed=5.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
        hit_radius=30.0,
    )
    assert ship.effective_hit_radius() == 30.0
    derived = Ship(
        id="S3",
        position=as_vec(0.0, 0.0),
        speed=5.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
    )
    assert derived.effective_hit_radius() == np.hypot(50.0, 10.0)
