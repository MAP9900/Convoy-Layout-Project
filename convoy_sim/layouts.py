"""Convoy layout generators for research experiments."""

from __future__ import annotations

from typing import Sequence

from .geometry import Point2D


def rectangular_layout(
    rows: int,
    cols: int,
    spacing_m: float,
    origin: Point2D | None = None,
) -> list[Point2D]:
    """Return a row/column aligned grid of ship anchor points."""

    raise NotImplementedError


def staggered_layout(
    rows: int,
    cols: int,
    spacing_m: float,
    row_offset_m: float | None = None,
) -> list[Point2D]:
    """Return a rectangular grid with alternating row offsets."""

    raise NotImplementedError


def hexagonal_layout(
    count: int,
    spacing_m: float,
    origin: Point2D | None = None,
) -> list[Point2D]:
    """Return anchor points arranged in a hexagonal/triangular packing."""

    raise NotImplementedError


def jittered_layout(
    base_layout: Sequence[Point2D],
    jitter_std_m: float,
    seed: int | None = None,
) -> list[Point2D]:
    """Return jittered variants of ``base_layout`` using small random offsets."""

    raise NotImplementedError
