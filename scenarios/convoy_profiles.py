"""Convoy layout profile registry for scenarios and visuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.layouts import make_rectangular_convoy


LayoutFn = Callable[..., list[Ship]]


@dataclass(frozen=True)
class ConvoyLayoutProfile:
    """Named convoy layout profile."""

    name: str
    layout_fn: LayoutFn
    layout_kwargs: dict[str, Any]
    description: str = ""

    def build_ships(self) -> list[Ship]:
        return self.layout_fn(**self.layout_kwargs)


def _ship_class_map_rl_large(row_idx: int, col_idx: int) -> ShipClass:
    """Per-cell ship class map for RL scaffold profile.

    Coordinates are (row_idx, col_idx) in the generated grid.
    Update `ship_class_overrides` to place specific classes by cell.
    """

    # TODO(RL_PROFILE): replace with your exact class plan.
    # Example intent: place high-value tankers near convoy center.
    ship_class_overrides: dict[tuple[int, int], ShipClass] = {
        # Center row examples (for n_rows=5, n_cols=10):
        (2, 4): ShipClass.TANKER,
        (2, 5): ShipClass.TANKER,
        # Escort examples near perimeter:
        (0, 0): ShipClass.ESCORT,
        (0, 9): ShipClass.ESCORT,
        (4, 0): ShipClass.ESCORT,
        (4, 9): ShipClass.ESCORT,
        # TODO(RL_PROFILE): add/remove overrides for your exact fleet composition.
    }
    return ship_class_overrides.get((row_idx, col_idx), ShipClass.FREIGHTER)


def build_small_demo_profile() -> ConvoyLayoutProfile:
    """Current small visual default profile (kept intentionally unchanged)."""

    return ConvoyLayoutProfile(
        name="small_demo",
        layout_fn=make_rectangular_convoy,
        layout_kwargs={
            "n_rows": 3,
            "n_cols": 4,
            "spacing_along": 600.0,
            "spacing_across": 350.0,
            "speed": 5.0,
            "heading_rad": 0.0,
            "length": 150.0,
            "beam": 20.0,
            "origin": as_vec(0.0, 0.0),
        },
        description="Small baseline layout used by current visual demos.",
    )


def build_rl_large_profile_scaffold() -> ConvoyLayoutProfile:
    """RL-focused large convoy scaffold profile.

    TODO(RL_PROFILE): update all numeric values and class mapping as needed.
    """

    return ConvoyLayoutProfile(
        name="rl_large",
        layout_fn=make_rectangular_convoy,
        layout_kwargs={
            # TODO(RL_PROFILE): tune grid size and spacing.
            "n_rows": 5,
            "n_cols": 10,
            "spacing_along": 700.0,
            "spacing_across": 450.0,
            # TODO(RL_PROFILE): tune convoy kinematics and hull defaults.
            "speed": 5.0,
            "heading_rad": 0.0,
            "length": 150.0,
            "beam": 20.0,
            "origin": as_vec(0.0, 0.0),
            # Key extension: per-cell class assignment hook.
            "ship_class_map": _ship_class_map_rl_large,
            # TODO(RL_PROFILE): optionally add ship_overrides_map for per-cell hull changes.
        },
        description="Large RL scaffold with per-cell ship class mapping.",
    )


def get_convoy_layout_profile_registry() -> dict[str, ConvoyLayoutProfile]:
    return {
        "small_demo": build_small_demo_profile(),
        "rl_large": build_rl_large_profile_scaffold(),
    }


def get_convoy_layout_profile(profile_name: str) -> ConvoyLayoutProfile:
    registry = get_convoy_layout_profile_registry()
    if profile_name not in registry:
        raise ValueError(f"Unknown convoy layout profile: {profile_name}")
    return registry[profile_name]


def list_convoy_layout_profiles() -> list[str]:
    return list(get_convoy_layout_profile_registry().keys())

