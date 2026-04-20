"""Tests for heterogeneous ship models and catalog defaults."""

import numpy as np

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.ship_catalog import HULL_VARIANT_CATALOG, SHIP_CATALOG, build_fleet_assignment_maps, make_ship


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


def test_hull_variant_catalog_has_multiple_templates_for_major_classes() -> None:
    assert len(HULL_VARIANT_CATALOG[ShipClass.FREIGHTER]) >= 2
    assert len(HULL_VARIANT_CATALOG[ShipClass.TANKER]) >= 2
    assert len(HULL_VARIANT_CATALOG[ShipClass.ESCORT]) >= 2


def test_build_fleet_assignment_maps_is_seeded_and_reproducible() -> None:
    class_map_a, overrides_map_a = build_fleet_assignment_maps(
        n_rows=3,
        n_cols=4,
        fleet_profile="mixed_convoy_v1",
        fleet_seed=1947,
    )
    class_map_b, overrides_map_b = build_fleet_assignment_maps(
        n_rows=3,
        n_cols=4,
        fleet_profile="mixed_convoy_v1",
        fleet_seed=1947,
    )
    assert class_map_a is not None and overrides_map_a is not None
    assert class_map_b is not None and overrides_map_b is not None
    assert class_map_a(0, 0) == class_map_b(0, 0)
    assert overrides_map_a(1, 1) == overrides_map_b(1, 1)


def test_build_fleet_assignment_maps_respects_placement_policy() -> None:
    balanced_map, _ = build_fleet_assignment_maps(
        n_rows=6,
        n_cols=7,
        row_counts=[6, 7, 8, 8, 7, 6],
        fleet_profile="mixed_convoy_v1",
        fleet_seed=1947,
        class_placement_policy="mixed_balanced",
    )
    rear_map, _ = build_fleet_assignment_maps(
        n_rows=6,
        n_cols=7,
        row_counts=[6, 7, 8, 8, 7, 6],
        fleet_profile="mixed_convoy_v1",
        fleet_seed=1947,
        class_placement_policy="high_value_rear_center",
    )
    assert balanced_map is not None and rear_map is not None
    balanced_tankers = {
        (row_idx, col_idx)
        for row_idx, count in enumerate([6, 7, 8, 8, 7, 6])
        for col_idx in range(count)
        if balanced_map(row_idx, col_idx) == ShipClass.TANKER
    }
    rear_tankers = {
        (row_idx, col_idx)
        for row_idx, count in enumerate([6, 7, 8, 8, 7, 6])
        for col_idx in range(count)
        if rear_map(row_idx, col_idx) == ShipClass.TANKER
    }
    assert balanced_tankers != rear_tankers


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
