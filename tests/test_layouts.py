"""Ensure layout generators are discoverable."""

import pytest

from convoy_sim import (
    Point2D,
    hexagonal_layout,
    jittered_layout,
    rectangular_layout,
    staggered_layout,
)


def test_rectangular_layout_stub() -> None:
    with pytest.raises(NotImplementedError):
        rectangular_layout(rows=2, cols=3, spacing_m=400.0)


def test_staggered_layout_stub() -> None:
    with pytest.raises(NotImplementedError):
        staggered_layout(rows=2, cols=3, spacing_m=400.0, row_offset_m=200.0)


def test_hexagonal_layout_stub() -> None:
    with pytest.raises(NotImplementedError):
        hexagonal_layout(count=6, spacing_m=350.0)


def test_jittered_layout_stub() -> None:
    anchors = [Point2D(0.0, 0.0), Point2D(100.0, 0.0)]
    with pytest.raises(NotImplementedError):
        jittered_layout(anchors, jitter_std_m=5.0, seed=1)
