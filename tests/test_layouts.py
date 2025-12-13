"""Ensure layout generators produce sane ship formations."""

import numpy as np

from convoy_sim import as_vec
from convoy_sim.layouts import (
    apply_jitter,
    make_hexagonal_convoy,
    make_rectangular_convoy,
    make_staggered_convoy,
)


def _basic_params() -> dict:
    return {
        "speed": 5.0,
        "heading_rad": 0.0,
        "length": 100.0,
        "beam": 20.0,
    }


def test_rectangular_convoy_spacing() -> None:
    ships = make_rectangular_convoy(
        n_rows=2,
        n_cols=3,
        spacing_along=500.0,
        spacing_across=300.0,
        origin=as_vec(0.0, 0.0),
        **_basic_params(),
    )
    assert len(ships) == 6
    positions = np.array([ship.position for ship in ships])
    x_coords = np.sort(np.unique(positions[:, 0]))
    y_coords = np.sort(np.unique(positions[:, 1]))
    assert np.isclose(x_coords[1] - x_coords[0], 500.0)
    assert np.isclose(y_coords[1] - y_coords[0], 300.0)


def test_staggered_convoy_offsets_every_other_row() -> None:
    ships = make_staggered_convoy(
        n_rows=2,
        n_cols=2,
        spacing_along=400.0,
        spacing_across=200.0,
        origin=as_vec(0.0, 0.0),
        **_basic_params(),
    )
    assert len(ships) == 4
    positions = np.array([ship.position for ship in ships])
    # Extract rows via approximate x-coordinate
    row0 = positions[positions[:, 0] < 0]
    row1 = positions[positions[:, 0] > 0]
    assert np.allclose(np.diff(np.sort(row0[:, 1])), 200.0)
    assert np.allclose(np.diff(np.sort(row1[:, 1])), 200.0)
    # Row1 should be offset by ~100 meters relative to row0
    assert np.allclose(row1[:, 1].mean() - row0[:, 1].mean(), 100.0)


def test_hexagonal_convoy_smoke() -> None:
    ships = make_hexagonal_convoy(
        n_rows=2,
        n_cols=3,
        spacing_along=400.0,
        spacing_across=200.0,
        origin=as_vec(0.0, 0.0),
        **_basic_params(),
    )
    assert len(ships) == 6
    positions = np.array([ship.position for ship in ships])
    assert np.ptp(positions[:, 0]) > 0.0
    assert np.ptp(positions[:, 1]) > 0.0


def test_apply_jitter_changes_positions() -> None:
    ships = make_rectangular_convoy(
        n_rows=1,
        n_cols=1,
        spacing_along=100.0,
        spacing_across=100.0,
        origin=as_vec(0.0, 0.0),
        **_basic_params(),
    )
    base_position = ships[0].position.copy()
    rng = np.random.default_rng(42)
    apply_jitter(ships, jitter_std=5.0, rng=rng)
    assert not np.allclose(base_position, ships[0].position)
