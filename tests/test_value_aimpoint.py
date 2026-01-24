"""Tests for value-based aimpoint selection."""

import numpy as np

from convoy_sim.attack_proposals import choose_aimpoint
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec


def _ships_for_aimpoint() -> list[Ship]:
    return [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=5.0,
            heading_rad=0.0,
            length=120.0,
            beam=20.0,
            ship_class=ShipClass.FREIGHTER,
            value_weight=1.0,
        ),
        Ship(
            id="S2",
            position=as_vec(1000.0, 0.0),
            speed=5.0,
            heading_rad=0.0,
            length=150.0,
            beam=25.0,
            ship_class=ShipClass.TANKER,
            value_weight=10.0,
        ),
    ]


def test_max_value_aimpoint() -> None:
    ships = _ships_for_aimpoint()
    rng = np.random.default_rng(0)
    aimpoint = choose_aimpoint(ships, "max_value", rng)
    assert np.allclose(aimpoint, ships[1].position)


def test_value_weighted_centroid_shifts() -> None:
    ships = _ships_for_aimpoint()
    rng = np.random.default_rng(0)
    aimpoint = choose_aimpoint(ships, "value_weighted_centroid", rng)
    assert aimpoint[0] > 500.0
