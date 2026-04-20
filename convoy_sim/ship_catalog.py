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

VALID_CLASS_PLACEMENT_POLICIES = (
    "mixed_balanced",
    "high_value_center",
    "high_value_rear_center",
    "escort_perimeter",
)


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
    row_counts: list[int] | tuple[int, ...] | None = None,
    fleet_profile: str | None,
    fleet_seed: int | None,
    class_placement_policy: str | None = None,
) -> tuple[Callable[[int, int], ShipClass] | None, Callable[[int, int], dict[str, float]] | None]:
    """Return deterministic class/override maps for a seeded fleet realization."""

    if not fleet_profile:
        return None, None
    if fleet_profile not in FLEET_PROFILE_SPECS:
        raise ValueError(f"Unknown fleet_profile: {fleet_profile}")

    spec = FLEET_PROFILE_SPECS[fleet_profile]
    rng = np.random.default_rng(0 if fleet_seed is None else int(fleet_seed))
    actual_row_counts = _resolve_row_counts(n_rows=n_rows, n_cols=n_cols, row_counts=row_counts)
    slots = [(row_idx, col_idx) for row_idx, count in enumerate(actual_row_counts) for col_idx in range(count)]
    assigned: dict[tuple[int, int], ShipClass] = {}
    placement_policy = str(class_placement_policy or spec.get("class_placement_policy", "mixed_balanced"))
    if placement_policy not in VALID_CLASS_PLACEMENT_POLICIES:
        raise ValueError(
            f"Unknown class_placement_policy: {placement_policy}. "
            f"Valid policies: {sorted(VALID_CLASS_PLACEMENT_POLICIES)}"
        )

    class_counts: dict[ShipClass, int] = {
        ShipClass(ship_class): int(count)
        for ship_class, count in dict(spec.get("class_counts", {})).items()
    }
    default_class = ShipClass(spec.get("default_class", ShipClass.FREIGHTER))

    escort_count = int(class_counts.get(ShipClass.ESCORT, 0))
    if escort_count > 0:
        perimeter_slots = [
            slot for slot in slots if _is_perimeter_cell(slot[0], slot[1], row_counts=actual_row_counts)
        ]
        chosen_escorts = _sample_cells(perimeter_slots, escort_count, rng)
        for slot in chosen_escorts:
            assigned[slot] = ShipClass.ESCORT

    tanker_count = int(class_counts.get(ShipClass.TANKER, 0))
    if tanker_count > 0:
        tanker_pool = [
            slot
            for slot in _priority_cells_for_policy(actual_row_counts, placement_policy=placement_policy)
            if slot not in assigned
        ]
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


def _resolve_row_counts(
    *,
    n_rows: int,
    n_cols: int,
    row_counts: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if row_counts is None:
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive")
        return [int(n_cols)] * int(n_rows)
    resolved = [int(value) for value in row_counts]
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError("row_counts must be a non-empty sequence of positive integers")
    return resolved


def _is_perimeter_cell(row_idx: int, col_idx: int, *, row_counts: list[int]) -> bool:
    return row_idx in {0, len(row_counts) - 1} or col_idx in {0, row_counts[row_idx] - 1}


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


def _priority_cells_for_policy(
    row_counts: list[int],
    *,
    placement_policy: str,
) -> list[tuple[int, int]]:
    if placement_policy == "high_value_center":
        return _center_priority_cells(row_counts)
    if placement_policy == "high_value_rear_center":
        return _rear_center_priority_cells(row_counts)
    if placement_policy == "escort_perimeter":
        return _center_priority_cells(row_counts)
    return _distributed_core_priority_cells(row_counts)


def _slot_center_distance(slot: tuple[int, int], row_counts: list[int]) -> tuple[float, float, float]:
    row_idx, col_idx = slot
    row_center = (len(row_counts) - 1) / 2.0
    col_center = (row_counts[row_idx] - 1) / 2.0
    row_delta = abs(row_idx - row_center)
    col_delta = abs(col_idx - col_center)
    return row_delta + col_delta, row_delta, col_delta


def _center_priority_cells(row_counts: list[int]) -> list[tuple[int, int]]:
    slots = [(row_idx, col_idx) for row_idx, count in enumerate(row_counts) for col_idx in range(count)]
    slots.sort(key=lambda slot: _slot_center_distance(slot, row_counts))
    return slots


def _rear_center_priority_cells(row_counts: list[int]) -> list[tuple[int, int]]:
    slots = [(row_idx, col_idx) for row_idx, count in enumerate(row_counts) for col_idx in range(count)]
    slots.sort(
        key=lambda slot: (
            slot[0],
            abs(slot[1] - (row_counts[slot[0]] - 1) / 2.0),
            *_slot_center_distance(slot, row_counts),
        )
    )
    return slots


def _distributed_core_priority_cells(row_counts: list[int]) -> list[tuple[int, int]]:
    row_center = (len(row_counts) - 1) / 2.0
    ordered_rows = sorted(range(len(row_counts)), key=lambda row_idx: (abs(row_idx - row_center), row_idx))
    row_queues: dict[int, list[int]] = {
        row_idx: sorted(
            range(row_counts[row_idx]),
            key=lambda col_idx: abs(col_idx - (row_counts[row_idx] - 1) / 2.0),
        )
        for row_idx in ordered_rows
    }
    slots: list[tuple[int, int]] = []
    while any(row_queues.values()):
        for row_idx in ordered_rows:
            queue = row_queues[row_idx]
            if not queue:
                continue
            slots.append((row_idx, queue.pop(0)))
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
