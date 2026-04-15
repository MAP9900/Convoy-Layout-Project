"""Bounded multi-step convoy layout builder for RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from convoy_sim.defender_policy import LayoutAction
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy


_LAYOUTS = {
    "rectangular": make_rectangular_convoy,
    "staggered": make_staggered_convoy,
}

_STEP_PREFIX = {
    0: "family",
    1: "along",
    2: "across",
}


@dataclass(frozen=True)
class RLLayoutBuilderState:
    """Partial builder state across the bounded layout-construction episode."""

    family: str | None = None
    spacing_along_bucket: str | None = None
    spacing_across_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "spacing_along_bucket": self.spacing_along_bucket,
            "spacing_across_bucket": self.spacing_across_bucket,
        }


@dataclass(frozen=True)
class RLLayoutBuilderConfig:
    """Config for the minimal Phase 2 layout builder."""

    enabled: bool
    base_n_rows: int
    base_n_cols: int
    speed: float
    heading_rad: float
    length: float
    beam: float
    origin: np.ndarray
    fleet_profile: str | None
    fleet_seed: int | None
    layout_families: tuple[str, ...]
    spacing_along_options: dict[str, float]
    spacing_across_options: dict[str, float]
    family_complexity: dict[str, float]
    spacing_along_complexity: dict[str, float]
    spacing_across_complexity: dict[str, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RLLayoutBuilderConfig":
        cfg = dict(payload or {})
        origin = np.asarray(cfg.get("origin", [0.0, 0.0]), dtype=float)
        families = tuple(str(item) for item in cfg.get("layout_families", ("rectangular", "staggered")))
        for family in families:
            if family not in _LAYOUTS:
                raise ValueError(f"Unknown builder layout family: {family}")
        spacing_along_options = {str(k): float(v) for k, v in dict(cfg.get("spacing_along_options", {})).items()}
        spacing_across_options = {str(k): float(v) for k, v in dict(cfg.get("spacing_across_options", {})).items()}
        if cfg.get("enabled", False):
            if not spacing_along_options:
                raise ValueError("Builder mode requires spacing_along_options")
            if not spacing_across_options:
                raise ValueError("Builder mode requires spacing_across_options")
        family_complexity = {str(k): float(v) for k, v in dict(cfg.get("family_complexity", {})).items()}
        spacing_along_complexity = {
            str(k): float(v) for k, v in dict(cfg.get("spacing_along_complexity", {})).items()
        }
        spacing_across_complexity = {
            str(k): float(v) for k, v in dict(cfg.get("spacing_across_complexity", {})).items()
        }
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            base_n_rows=int(cfg.get("base_n_rows", 6)),
            base_n_cols=int(cfg.get("base_n_cols", 7)),
            speed=float(cfg.get("speed", 5.0)),
            heading_rad=float(cfg.get("heading_rad", 0.0)),
            length=float(cfg.get("length", 150.0)),
            beam=float(cfg.get("beam", 20.0)),
            origin=origin,
            fleet_profile=(None if cfg.get("fleet_profile") in {None, ""} else str(cfg.get("fleet_profile"))),
            fleet_seed=(None if cfg.get("fleet_seed") is None else int(cfg.get("fleet_seed"))),
            layout_families=families,
            spacing_along_options=spacing_along_options,
            spacing_across_options=spacing_across_options,
            family_complexity=family_complexity,
            spacing_along_complexity=spacing_along_complexity,
            spacing_across_complexity=spacing_across_complexity,
        )

    @property
    def step_count(self) -> int:
        return 3

    def action_space_names(self) -> list[str]:
        names = [f"family:{family}" for family in self.layout_families]
        names.extend(f"along:{bucket}" for bucket in self.spacing_along_options)
        names.extend(f"across:{bucket}" for bucket in self.spacing_across_options)
        return names

    def valid_action_names(self, state: RLLayoutBuilderState) -> list[str]:
        if state.family is None:
            return [f"family:{family}" for family in self.layout_families]
        if state.spacing_along_bucket is None:
            return [f"along:{bucket}" for bucket in self.spacing_along_options]
        if state.spacing_across_bucket is None:
            return [f"across:{bucket}" for bucket in self.spacing_across_options]
        return []

    def apply_action(self, state: RLLayoutBuilderState, action_name: str) -> RLLayoutBuilderState:
        valid = set(self.valid_action_names(state))
        if action_name not in valid:
            raise ValueError(f"Invalid builder action {action_name!r}; valid actions: {sorted(valid)}")
        prefix, value = action_name.split(":", 1)
        if prefix == "family":
            return RLLayoutBuilderState(
                family=value,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "along":
            return RLLayoutBuilderState(
                family=state.family,
                spacing_along_bucket=value,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "across":
            return RLLayoutBuilderState(
                family=state.family,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=value,
            )
        raise ValueError(f"Unknown builder action prefix: {prefix}")

    def is_complete(self, state: RLLayoutBuilderState) -> bool:
        return (
            state.family is not None
            and state.spacing_along_bucket is not None
            and state.spacing_across_bucket is not None
        )

    def materialize_layout_action(
        self,
        state: RLLayoutBuilderState,
        *,
        ship_movement_realism: dict[str, Any] | None = None,
    ) -> LayoutAction:
        if not self.is_complete(state):
            raise ValueError("Cannot materialize incomplete builder state")
        assert state.family is not None
        assert state.spacing_along_bucket is not None
        assert state.spacing_across_bucket is not None
        layout_fn = _LAYOUTS[state.family]
        layout_kwargs: dict[str, Any] = {
            "n_rows": self.base_n_rows,
            "n_cols": self.base_n_cols,
            "spacing_along": float(self.spacing_along_options[state.spacing_along_bucket]),
            "spacing_across": float(self.spacing_across_options[state.spacing_across_bucket]),
            "speed": self.speed,
            "heading_rad": self.heading_rad,
            "length": self.length,
            "beam": self.beam,
            "origin": np.asarray(self.origin, dtype=float),
        }
        if self.fleet_profile is not None:
            layout_kwargs["fleet_profile"] = self.fleet_profile
        if self.fleet_seed is not None:
            layout_kwargs["fleet_seed"] = int(self.fleet_seed)
        if ship_movement_realism:
            layout_kwargs["ship_movement_realism"] = dict(ship_movement_realism)
        complexity_cost = (
            float(self.family_complexity.get(state.family, 0.0))
            + float(self.spacing_along_complexity.get(state.spacing_along_bucket, 0.0))
            + float(self.spacing_across_complexity.get(state.spacing_across_bucket, 0.0))
        )
        family_label = "rect" if state.family == "rectangular" else state.family
        return LayoutAction(
            name=f"{family_label}_{state.spacing_along_bucket}_{state.spacing_across_bucket}",
            layout_fn=layout_fn,
            layout_kwargs=layout_kwargs,
            complexity_cost=complexity_cost,
        )

    def enumerate_layout_actions(
        self,
        *,
        ship_movement_realism: dict[str, Any] | None = None,
    ) -> list[LayoutAction]:
        actions: list[LayoutAction] = []
        for family in self.layout_families:
            for along_bucket in self.spacing_along_options:
                for across_bucket in self.spacing_across_options:
                    state = RLLayoutBuilderState(
                        family=family,
                        spacing_along_bucket=along_bucket,
                        spacing_across_bucket=across_bucket,
                    )
                    actions.append(
                        self.materialize_layout_action(
                            state,
                            ship_movement_realism=ship_movement_realism,
                        )
                    )
        return actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_n_rows": self.base_n_rows,
            "base_n_cols": self.base_n_cols,
            "speed": self.speed,
            "heading_rad": self.heading_rad,
            "length": self.length,
            "beam": self.beam,
            "origin": self.origin.tolist(),
            "fleet_profile": self.fleet_profile,
            "fleet_seed": self.fleet_seed,
            "layout_families": list(self.layout_families),
            "spacing_along_options": dict(self.spacing_along_options),
            "spacing_across_options": dict(self.spacing_across_options),
            "family_complexity": dict(self.family_complexity),
            "spacing_along_complexity": dict(self.spacing_along_complexity),
            "spacing_across_complexity": dict(self.spacing_across_complexity),
        }
