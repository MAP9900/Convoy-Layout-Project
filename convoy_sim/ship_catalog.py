"""Ship class catalog for heterogeneous convoy modeling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import Vec2

SHIP_CATALOG: dict[ShipClass, dict[str, float]] = {
    ShipClass.FREIGHTER: {"length": 140.0, "beam": 20.0, "value_weight": 1.0},
    ShipClass.TANKER: {"length": 180.0, "beam": 28.0, "value_weight": 1.5},
    ShipClass.ESCORT: {"length": 90.0, "beam": 12.0, "value_weight": 0.5},
    ShipClass.DECOY: {"length": 120.0, "beam": 18.0, "value_weight": 0.2},
}

HULL_VARIANT_CATALOG: dict[ShipClass, list[dict[str, float]]] = {
    ShipClass.FREIGHTER: [
        {"length": 126.0, "beam": 18.0, "value_weight": 0.9, "weight": 1.0},
        {"length": 138.0, "beam": 19.5, "value_weight": 1.0, "weight": 1.4},
        {"length": 152.0, "beam": 20.5, "value_weight": 1.05, "weight": 1.4},
        {"length": 166.0, "beam": 22.0, "value_weight": 1.1, "weight": 1.0},
    ],
    ShipClass.TANKER: [
        {"length": 168.0, "beam": 25.0, "value_weight": 1.4, "weight": 1.0},
        {"length": 182.0, "beam": 27.0, "value_weight": 1.55, "weight": 1.3},
        {"length": 196.0, "beam": 29.0, "value_weight": 1.7, "weight": 1.1},
    ],
    ShipClass.ESCORT: [
        {"length": 82.0, "beam": 11.0, "value_weight": 0.45, "weight": 1.0},
        {"length": 92.0, "beam": 12.0, "value_weight": 0.5, "weight": 1.3},
        {"length": 104.0, "beam": 13.0, "value_weight": 0.6, "weight": 0.8},
    ],
    ShipClass.DECOY: [
        {"length": 110.0, "beam": 17.0, "value_weight": 0.18, "weight": 1.0},
        {"length": 124.0, "beam": 18.5, "value_weight": 0.22, "weight": 1.0},
    ],
}

FLEET_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "freighter_heterogeneous_v1": {
        "class_counts": {},
        "default_class": ShipClass.FREIGHTER,
    },
    "mixed_convoy_v1": {
        "class_counts": {
            ShipClass.ESCORT: 4,
            ShipClass.TANKER: 6,
        },
        "default_class": ShipClass.FREIGHTER,
    },
}


def sample_hull_variant(ship_class: ShipClass, rng: np.random.Generator) -> dict[str, float]:
    """Return one weighted hull variant for the requested ship class."""

    variants = HULL_VARIANT_CATALOG.get(ship_class)
    if not variants:
        return dict(SHIP_CATALOG[ship_class])
    weights = np.array([float(item.get("weight", 1.0)) for item in variants], dtype=float)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(variants), p=weights))
    chosen = dict(variants[idx])
    chosen.pop("weight", None)
    return chosen


def build_fleet_assignment_maps(
    *,
    n_rows: int,
    n_cols: int,
    fleet_profile: str | None,
    fleet_seed: int | None,
) -> tuple[Callable[[int, int], ShipClass] | None, Callable[[int, int], dict[str, float]] | None]:
    """Return deterministic class/override maps for a seeded fleet realization."""

    if not fleet_profile:
        return None, None
    if fleet_profile not in FLEET_PROFILE_SPECS:
        raise ValueError(f"Unknown fleet_profile: {fleet_profile}")

    spec = FLEET_PROFILE_SPECS[fleet_profile]
    rng = np.random.default_rng(0 if fleet_seed is None else int(fleet_seed))
    slots = [(row_idx, col_idx) for row_idx in range(n_rows) for col_idx in range(n_cols)]
    assigned: dict[tuple[int, int], ShipClass] = {}

    class_counts: dict[ShipClass, int] = {
        ShipClass(ship_class): int(count)
        for ship_class, count in dict(spec.get("class_counts", {})).items()
    }
    default_class = ShipClass(spec.get("default_class", ShipClass.FREIGHTER))

    escort_count = int(class_counts.get(ShipClass.ESCORT, 0))
    if escort_count > 0:
        perimeter_slots = [slot for slot in slots if _is_perimeter_cell(*slot, n_rows=n_rows, n_cols=n_cols)]
        chosen_escorts = _sample_cells(perimeter_slots, escort_count, rng)
        for slot in chosen_escorts:
            assigned[slot] = ShipClass.ESCORT

    tanker_count = int(class_counts.get(ShipClass.TANKER, 0))
    if tanker_count > 0:
        tanker_pool = [slot for slot in _core_priority_cells(n_rows=n_rows, n_cols=n_cols) if slot not in assigned]
        chosen_tankers = tanker_pool[:tanker_count]
        if len(chosen_tankers) < tanker_count:
            raise ValueError("Not enough open cells to assign tanker slots for fleet profile")
        for slot in chosen_tankers:
            assigned[slot] = ShipClass.TANKER

    for slot in slots:
        assigned.setdefault(slot, default_class)

    overrides_map: dict[tuple[int, int], dict[str, float]] = {}
    for slot in slots:
        ship_class = assigned[slot]
        overrides_map[slot] = sample_hull_variant(ship_class, rng)

    def ship_class_map(row_idx: int, col_idx: int) -> ShipClass:
        return assigned[(row_idx, col_idx)]

    def ship_overrides_map(row_idx: int, col_idx: int) -> dict[str, float]:
        return dict(overrides_map[(row_idx, col_idx)])

    return ship_class_map, ship_overrides_map


def _is_perimeter_cell(row_idx: int, col_idx: int, *, n_rows: int, n_cols: int) -> bool:
    return row_idx in {0, n_rows - 1} or col_idx in {0, n_cols - 1}


def _sample_cells(
    cells: list[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    if count > len(cells):
        raise ValueError("Requested more cells than available")
    indices = list(rng.choice(len(cells), size=count, replace=False))
    return [cells[idx] for idx in indices]


def _core_priority_cells(*, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    row_center = (n_rows - 1) / 2.0
    col_center = (n_cols - 1) / 2.0
    slots = [(row_idx, col_idx) for row_idx in range(n_rows) for col_idx in range(n_cols)]
    slots.sort(
        key=lambda slot: (
            abs(slot[0] - row_center) + abs(slot[1] - col_center),
            abs(slot[0] - row_center),
            abs(slot[1] - col_center),
        )
    )
    return slots


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
