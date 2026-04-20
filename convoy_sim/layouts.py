"""Convoy layout generators for research experiments."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import Vec2, as_vec
from convoy_sim.ship_catalog import build_fleet_assignment_maps, make_ship

VALID_ROW_OFFSET_POLICIES = ("none", "centered_alt", "forward_taper", "rear_taper")


def _rotation_matrix(heading_rad: float) -> np.ndarray:
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    return np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)


def _grid_offsets(count: int, spacing: float) -> list[float]:
    center = (count - 1) / 2.0
    return [(idx - center) * spacing for idx in range(count)]


def _origin_vec(origin: Vec2 | Sequence[float] | None) -> Vec2:
    if origin is None:
        return as_vec(0.0, 0.0)
    arr = np.asarray(origin, dtype=float)
    if arr.shape != (2,):
        raise ValueError("origin must be a 2D vector")
    return arr


def _resolve_row_counts(n_rows: int, n_cols: int, row_counts: Sequence[int] | None) -> list[int]:
    if row_counts is None:
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive")
        return [int(n_cols)] * int(n_rows)
    resolved = [int(value) for value in row_counts]
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError("row_counts must be a non-empty sequence of positive integers")
    return resolved


def _row_offset_shift(
    row_idx: int,
    *,
    n_rows: int,
    spacing_across: float,
    row_offset_policy: str,
) -> float:
    if row_offset_policy not in VALID_ROW_OFFSET_POLICIES:
        raise ValueError(
            f"Unknown row_offset_policy: {row_offset_policy}. "
            f"Valid policies: {sorted(VALID_ROW_OFFSET_POLICIES)}"
        )
    if row_offset_policy == "none":
        return 0.0
    if row_offset_policy == "centered_alt":
        return (0.25 if row_idx % 2 else -0.25) * spacing_across
    row_center = (n_rows - 1) / 2.0
    normalized = 0.0 if row_center == 0.0 else (row_idx - row_center) / row_center
    if row_offset_policy == "forward_taper":
        return 0.35 * spacing_across * normalized
    return -0.35 * spacing_across * normalized


def _make_ships_from_local_positions(
    local_positions: Sequence[tuple[Vec2, int, int]],
    origin: Vec2,
    heading_rad: float,
    speed: float,
    length: float,
    beam: float,
    ship_class_map: Callable[[int, int], ShipClass] | None,
    ship_overrides_map: Callable[[int, int], dict] | None,
) -> list[Ship]:
    rotation = _rotation_matrix(heading_rad)
    ships = []
    for idx, (local, row_idx, col_idx) in enumerate(local_positions, start=1):
        local_vec = np.asarray(local, dtype=float)
        world_pos = origin + rotation @ local_vec
        ship_class = ShipClass.FREIGHTER
        if ship_class_map is not None:
            ship_class = ShipClass(ship_class_map(row_idx, col_idx))
        overrides = {"length": length, "beam": beam}
        if ship_overrides_map is not None:
            overrides.update(ship_overrides_map(row_idx, col_idx))
        ship = make_ship(
            ship_id=f"S{idx:02d}",
            ship_class=ship_class,
            position=world_pos,
            speed=speed,
            heading_rad=heading_rad,
            overrides=overrides,
        )
        ships.append(ship)
    return ships


def apply_jitter(
    ships: list[Ship],
    jitter_std: float,
    rng: np.random.Generator | None = None,
) -> list[Ship]:
    """Return ships with independent Gaussian offsets applied to their centers."""

    if jitter_std <= 0.0:
        return ships
    generator = rng or np.random.default_rng()
    for ship in ships:
        offset = generator.normal(loc=0.0, scale=jitter_std, size=2)
        ship.position = ship.position + offset
    return ships


def make_rectangular_convoy(
    n_rows: int,
    n_cols: int,
    spacing_along: float,
    spacing_across: float,
    speed: float,
    heading_rad: float,
    length: float,
    beam: float,
    origin: Vec2 | None = None,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
    ship_class_map: Callable[[int, int], ShipClass] | None = None,
    ship_overrides_map: Callable[[int, int], dict] | None = None,
    fleet_profile: str | None = None,
    fleet_seed: int | None = None,
    row_counts: Sequence[int] | None = None,
    row_offset_policy: str = "none",
    class_placement_policy: str | None = None,
) -> list[Ship]:
    """Return a rectangular grid of ships centered around ``origin``."""

    actual_row_counts = _resolve_row_counts(n_rows, n_cols, row_counts)
    if fleet_profile is not None:
        fleet_class_map, fleet_overrides_map = build_fleet_assignment_maps(
            n_rows=len(actual_row_counts),
            n_cols=max(actual_row_counts),
            row_counts=actual_row_counts,
            fleet_profile=fleet_profile,
            fleet_seed=fleet_seed,
            class_placement_policy=class_placement_policy,
        )
        if ship_class_map is None:
            ship_class_map = fleet_class_map
        if ship_overrides_map is None:
            ship_overrides_map = fleet_overrides_map
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(len(actual_row_counts), spacing_along)
    local_positions: list[tuple[Vec2, int, int]] = []
    for row_idx, x in enumerate(row_offsets):
        row_shift = _row_offset_shift(
            row_idx,
            n_rows=len(actual_row_counts),
            spacing_across=spacing_across,
            row_offset_policy=row_offset_policy,
        )
        col_offsets = _grid_offsets(actual_row_counts[row_idx], spacing_across)
        local_positions.extend(
            (as_vec(x, y + row_shift), row_idx, col_idx) for col_idx, y in enumerate(col_offsets)
        )
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
        ship_class_map,
        ship_overrides_map,
    )
    return apply_jitter(ships, jitter_std=jitter_std, rng=rng)


def make_staggered_convoy(
    n_rows: int,
    n_cols: int,
    spacing_along: float,
    spacing_across: float,
    speed: float,
    heading_rad: float,
    length: float,
    beam: float,
    origin: Vec2 | None = None,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
    ship_class_map: Callable[[int, int], ShipClass] | None = None,
    ship_overrides_map: Callable[[int, int], dict] | None = None,
    fleet_profile: str | None = None,
    fleet_seed: int | None = None,
    row_counts: Sequence[int] | None = None,
    row_offset_policy: str = "none",
    class_placement_policy: str | None = None,
) -> list[Ship]:
    """Return a grid with alternating row offsets along the across direction."""

    actual_row_counts = _resolve_row_counts(n_rows, n_cols, row_counts)
    if fleet_profile is not None:
        fleet_class_map, fleet_overrides_map = build_fleet_assignment_maps(
            n_rows=len(actual_row_counts),
            n_cols=max(actual_row_counts),
            row_counts=actual_row_counts,
            fleet_profile=fleet_profile,
            fleet_seed=fleet_seed,
            class_placement_policy=class_placement_policy,
        )
        if ship_class_map is None:
            ship_class_map = fleet_class_map
        if ship_overrides_map is None:
            ship_overrides_map = fleet_overrides_map
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(len(actual_row_counts), spacing_along)
    local_positions: list[tuple[Vec2, int, int]] = []
    for row_idx, x in enumerate(row_offsets):
        base_shift = 0.5 * spacing_across if row_idx % 2 == 1 else 0.0
        row_shift = _row_offset_shift(
            row_idx,
            n_rows=len(actual_row_counts),
            spacing_across=spacing_across,
            row_offset_policy=row_offset_policy,
        )
        cols = [offset + base_shift + row_shift for offset in _grid_offsets(actual_row_counts[row_idx], spacing_across)]
        local_positions.extend(
            (as_vec(x, y), row_idx, col_idx) for col_idx, y in enumerate(cols)
        )
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
        ship_class_map,
        ship_overrides_map,
    )
    return apply_jitter(ships, jitter_std=jitter_std, rng=rng)


def make_hexagonal_convoy(
    n_rows: int,
    n_cols: int,
    spacing_along: float,
    spacing_across: float,
    speed: float,
    heading_rad: float,
    length: float,
    beam: float,
    origin: Vec2 | None = None,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
    ship_class_map: Callable[[int, int], ShipClass] | None = None,
    ship_overrides_map: Callable[[int, int], dict] | None = None,
    fleet_profile: str | None = None,
    fleet_seed: int | None = None,
) -> list[Ship]:
    """Return ships arranged using a hexagonal (triangular) packing approximation.

    Each row is offset along the across-axis by half ``spacing_across`` and along
    the convoy axis by half ``spacing_along`` for alternating columns, producing
    a close-packed layout suitable for hex formations.
    """

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    if fleet_profile is not None:
        fleet_class_map, fleet_overrides_map = build_fleet_assignment_maps(
            n_rows=n_rows,
            n_cols=n_cols,
            fleet_profile=fleet_profile,
            fleet_seed=fleet_seed,
        )
        if ship_class_map is None:
            ship_class_map = fleet_class_map
        if ship_overrides_map is None:
            ship_overrides_map = fleet_overrides_map
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(n_rows, spacing_along)
    center = (n_cols - 1) / 2.0
    local_positions: list[tuple[Vec2, int, int]] = []
    for row_idx, x in enumerate(row_offsets):
        for col_idx in range(n_cols):
            y = (col_idx - center) * spacing_across
            x_shift = (col_idx % 2) * (spacing_along * 0.5)
            y_shift = (row_idx % 2) * (spacing_across * 0.5)
            local_positions.append((as_vec(x + x_shift, y + y_shift), row_idx, col_idx))
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
        ship_class_map,
        ship_overrides_map,
    )
    return apply_jitter(ships, jitter_std=jitter_std, rng=rng)
