"""Convoy layout generators for research experiments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .entities import Ship
from .geometry import Vec2, as_vec


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


def _make_ships_from_local_positions(
    local_positions: Sequence[Vec2],
    origin: Vec2,
    heading_rad: float,
    speed: float,
    length: float,
    beam: float,) -> list[Ship]:
    rotation = _rotation_matrix(heading_rad)
    ships = []
    for idx, local in enumerate(local_positions, start=1):
        local_vec = np.asarray(local, dtype=float)
        world_pos = origin + rotation @ local_vec
        ship = Ship(
            id=f"S{idx:02d}",
            position=world_pos,
            speed=speed,
            heading_rad=heading_rad,
            length=length,
            beam=beam,
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
) -> list[Ship]:
    """Return a rectangular grid of ships centered around ``origin``."""

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(n_rows, spacing_along)
    col_offsets = _grid_offsets(n_cols, spacing_across)
    local_positions = [as_vec(x, y) for x in row_offsets for y in col_offsets]
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
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
) -> list[Ship]:
    """Return a grid with alternating row offsets along the across direction."""

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(n_rows, spacing_along)
    center = (n_cols - 1) / 2.0
    local_positions: list[Vec2] = []
    for row_idx, x in enumerate(row_offsets):
        col_shift = 0.5 * spacing_across if row_idx % 2 == 1 else 0.0
        cols = [(col - center) * spacing_across + col_shift for col in range(n_cols)]
        local_positions.extend(as_vec(x, y) for y in cols)
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
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
) -> list[Ship]:
    """Return ships arranged using a hexagonal (triangular) packing approximation.

    Each row is offset along the across-axis by half ``spacing_across`` and along
    the convoy axis by half ``spacing_along`` for alternating columns, producing
    a close-packed layout suitable for hex formations.
    """

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    origin_vec = _origin_vec(origin)
    row_offsets = _grid_offsets(n_rows, spacing_along)
    center = (n_cols - 1) / 2.0
    local_positions: list[Vec2] = []
    for row_idx, x in enumerate(row_offsets):
        for col_idx in range(n_cols):
            y = (col_idx - center) * spacing_across
            x_shift = (col_idx % 2) * (spacing_along * 0.5)
            y_shift = (row_idx % 2) * (spacing_across * 0.5)
            local_positions.append(as_vec(x + x_shift, y + y_shift))
    ships = _make_ships_from_local_positions(
        local_positions,
        origin_vec,
        heading_rad,
        speed,
        length,
        beam,
    )
    return apply_jitter(ships, jitter_std=jitter_std, rng=rng)
