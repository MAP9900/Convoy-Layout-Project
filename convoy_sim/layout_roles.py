"""Reusable ship class maps for convoy layouts."""

from __future__ import annotations

from typing import Callable

from convoy_sim.entities import ShipClass


def center_high_value(n_cols: int) -> Callable[[int, int], ShipClass]:
    """Return a role map with tankers in center columns, freighters elsewhere."""

    center_left = (n_cols - 1) / 2.0
    center_right = center_left

    def role_map(_: int, col_idx: int) -> ShipClass:
        if abs(col_idx - center_left) <= 0.5 or abs(col_idx - center_right) <= 0.5:
            return ShipClass.TANKER
        return ShipClass.FREIGHTER

    return role_map


def perimeter_escorts(n_rows: int, n_cols: int) -> Callable[[int, int], ShipClass]:
    """Return a role map with escorts on the perimeter and freighters inside."""

    def role_map(row_idx: int, col_idx: int) -> ShipClass:
        if row_idx in {0, n_rows - 1} or col_idx in {0, n_cols - 1}:
            return ShipClass.ESCORT
        return ShipClass.FREIGHTER

    return role_map
