"""Tests for heterogeneous layout role maps."""

from collections import Counter

import numpy as np

from convoy_sim.entities import ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.layout_roles import center_high_value, perimeter_escorts
from convoy_sim.layouts import make_rectangular_convoy


def test_role_maps_assign_classes() -> None:
    role_map = center_high_value(n_cols=3)
    ships = make_rectangular_convoy(
        n_rows=2,
        n_cols=3,
        spacing_along=400.0,
        spacing_across=200.0,
        speed=5.0,
        heading_rad=0.0,
        length=120.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
        ship_class_map=role_map,
    )
    counts = Counter(ship.ship_class for ship in ships)
    assert counts[ShipClass.TANKER] == 2
    assert counts[ShipClass.FREIGHTER] == 4


def test_perimeter_escorts_counts() -> None:
    role_map = perimeter_escorts(n_rows=3, n_cols=3)
    ships = make_rectangular_convoy(
        n_rows=3,
        n_cols=3,
        spacing_along=400.0,
        spacing_across=200.0,
        speed=5.0,
        heading_rad=0.0,
        length=120.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
        ship_class_map=role_map,
    )
    counts = Counter(ship.ship_class for ship in ships)
    assert counts[ShipClass.ESCORT] == 8
    assert counts[ShipClass.FREIGHTER] == 1


def test_default_layout_is_freighter() -> None:
    ships = make_rectangular_convoy(
        n_rows=1,
        n_cols=2,
        spacing_along=300.0,
        spacing_across=200.0,
        speed=5.0,
        heading_rad=0.0,
        length=120.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
    )
    assert all(ship.ship_class == ShipClass.FREIGHTER for ship in ships)


def test_jitter_preserves_heading_speed() -> None:
    role_map = perimeter_escorts(n_rows=2, n_cols=2)
    ships = make_rectangular_convoy(
        n_rows=2,
        n_cols=2,
        spacing_along=300.0,
        spacing_across=200.0,
        speed=6.0,
        heading_rad=0.5,
        length=120.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
        ship_class_map=role_map,
        jitter_std=5.0,
        rng=np.random.default_rng(1),
    )
    assert all(ship.speed == 6.0 for ship in ships)
    assert all(ship.heading_rad == 0.5 for ship in ships)
