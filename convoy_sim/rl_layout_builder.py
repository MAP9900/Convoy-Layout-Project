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

@dataclass(frozen=True)
class RLLayoutBuilderState:
    """Partial builder state across the bounded layout-construction episode."""

    family: str | None = None
    row_pattern: str | None = None
    row_offset_policy: str | None = None
    class_placement_policy: str | None = None
    spacing_along_bucket: str | None = None
    spacing_across_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "row_pattern": self.row_pattern,
            "row_offset_policy": self.row_offset_policy,
            "class_placement_policy": self.class_placement_policy,
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
    row_patterns: dict[str, tuple[int, ...]]
    row_offset_policies: tuple[str, ...]
    class_placement_policies: tuple[str, ...]
    spacing_along_options: dict[str, float]
    spacing_across_options: dict[str, float]
    family_complexity: dict[str, float]
    row_pattern_complexity: dict[str, float]
    row_offset_complexity: dict[str, float]
    class_placement_complexity: dict[str, float]
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
        row_patterns = _parse_row_patterns(
            cfg.get("row_patterns"),
            default_rows=int(cfg.get("base_n_rows", 6)),
            default_cols=int(cfg.get("base_n_cols", 7)),
        )
        row_offset_policies = tuple(str(item) for item in cfg.get("row_offset_policies", ("none",)))
        class_placement_policies = tuple(
            str(item) for item in cfg.get("class_placement_policies", ("mixed_balanced",))
        )
        spacing_along_options = {str(k): float(v) for k, v in dict(cfg.get("spacing_along_options", {})).items()}
        spacing_across_options = {str(k): float(v) for k, v in dict(cfg.get("spacing_across_options", {})).items()}
        if cfg.get("enabled", False):
            if not row_patterns:
                raise ValueError("Builder mode requires at least one row pattern")
            if not spacing_along_options:
                raise ValueError("Builder mode requires spacing_along_options")
            if not spacing_across_options:
                raise ValueError("Builder mode requires spacing_across_options")
        family_complexity = {str(k): float(v) for k, v in dict(cfg.get("family_complexity", {})).items()}
        row_pattern_complexity = {
            str(k): float(v) for k, v in dict(cfg.get("row_pattern_complexity", {})).items()
        }
        row_offset_complexity = {
            str(k): float(v) for k, v in dict(cfg.get("row_offset_complexity", {})).items()
        }
        class_placement_complexity = {
            str(k): float(v) for k, v in dict(cfg.get("class_placement_complexity", {})).items()
        }
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
            row_patterns=row_patterns,
            row_offset_policies=row_offset_policies,
            class_placement_policies=class_placement_policies,
            spacing_along_options=spacing_along_options,
            spacing_across_options=spacing_across_options,
            family_complexity=family_complexity,
            row_pattern_complexity=row_pattern_complexity,
            row_offset_complexity=row_offset_complexity,
            class_placement_complexity=class_placement_complexity,
            spacing_along_complexity=spacing_along_complexity,
            spacing_across_complexity=spacing_across_complexity,
        )

    @property
    def step_count(self) -> int:
        return 6

    def action_space_names(self) -> list[str]:
        names = [f"family:{family}" for family in self.layout_families]
        names.extend(f"pattern:{name}" for name in self.row_patterns)
        names.extend(f"offset:{name}" for name in self.row_offset_policies)
        names.extend(f"placement:{name}" for name in self.class_placement_policies)
        names.extend(f"along:{bucket}" for bucket in self.spacing_along_options)
        names.extend(f"across:{bucket}" for bucket in self.spacing_across_options)
        return names

    def valid_action_names(self, state: RLLayoutBuilderState) -> list[str]:
        if state.family is None:
            return [f"family:{family}" for family in self.layout_families]
        if state.row_pattern is None:
            return [f"pattern:{name}" for name in self.row_patterns]
        if state.row_offset_policy is None:
            return [f"offset:{name}" for name in self.row_offset_policies]
        if state.class_placement_policy is None:
            return [f"placement:{name}" for name in self.class_placement_policies]
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
                row_pattern=state.row_pattern,
                row_offset_policy=state.row_offset_policy,
                class_placement_policy=state.class_placement_policy,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "pattern":
            return RLLayoutBuilderState(
                family=state.family,
                row_pattern=value,
                row_offset_policy=state.row_offset_policy,
                class_placement_policy=state.class_placement_policy,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "offset":
            return RLLayoutBuilderState(
                family=state.family,
                row_pattern=state.row_pattern,
                row_offset_policy=value,
                class_placement_policy=state.class_placement_policy,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "placement":
            return RLLayoutBuilderState(
                family=state.family,
                row_pattern=state.row_pattern,
                row_offset_policy=state.row_offset_policy,
                class_placement_policy=value,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "along":
            return RLLayoutBuilderState(
                family=state.family,
                row_pattern=state.row_pattern,
                row_offset_policy=state.row_offset_policy,
                class_placement_policy=state.class_placement_policy,
                spacing_along_bucket=value,
                spacing_across_bucket=state.spacing_across_bucket,
            )
        if prefix == "across":
            return RLLayoutBuilderState(
                family=state.family,
                row_pattern=state.row_pattern,
                row_offset_policy=state.row_offset_policy,
                class_placement_policy=state.class_placement_policy,
                spacing_along_bucket=state.spacing_along_bucket,
                spacing_across_bucket=value,
            )
        raise ValueError(f"Unknown builder action prefix: {prefix}")

    def is_complete(self, state: RLLayoutBuilderState) -> bool:
        return (
            state.family is not None
            and state.row_pattern is not None
            and state.row_offset_policy is not None
            and state.class_placement_policy is not None
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
        assert state.row_pattern is not None
        assert state.row_offset_policy is not None
        assert state.class_placement_policy is not None
        assert state.spacing_along_bucket is not None
        assert state.spacing_across_bucket is not None
        layout_fn = _LAYOUTS[state.family]
        row_counts = list(self.row_patterns[state.row_pattern])
        layout_kwargs: dict[str, Any] = {
            "n_rows": len(row_counts),
            "n_cols": max(row_counts),
            "spacing_along": float(self.spacing_along_options[state.spacing_along_bucket]),
            "spacing_across": float(self.spacing_across_options[state.spacing_across_bucket]),
            "speed": self.speed,
            "heading_rad": self.heading_rad,
            "length": self.length,
            "beam": self.beam,
            "origin": np.asarray(self.origin, dtype=float),
            "row_counts": row_counts,
            "row_offset_policy": state.row_offset_policy,
            "class_placement_policy": state.class_placement_policy,
        }
        if self.fleet_profile is not None:
            layout_kwargs["fleet_profile"] = self.fleet_profile
        if self.fleet_seed is not None:
            layout_kwargs["fleet_seed"] = int(self.fleet_seed)
        if ship_movement_realism:
            layout_kwargs["ship_movement_realism"] = dict(ship_movement_realism)
        complexity_cost = (
            float(self.family_complexity.get(state.family, 0.0))
            + float(self.row_pattern_complexity.get(state.row_pattern, 0.0))
            + float(self.row_offset_complexity.get(state.row_offset_policy, 0.0))
            + float(self.class_placement_complexity.get(state.class_placement_policy, 0.0))
            + float(self.spacing_along_complexity.get(state.spacing_along_bucket, 0.0))
            + float(self.spacing_across_complexity.get(state.spacing_across_bucket, 0.0))
        )
        family_label = "rect" if state.family == "rectangular" else state.family
        return LayoutAction(
            name=(
                f"{family_label}_{state.row_pattern}_{state.row_offset_policy}_"
                f"{state.class_placement_policy}_{state.spacing_along_bucket}_{state.spacing_across_bucket}"
            ),
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
            for row_pattern in self.row_patterns:
                for row_offset_policy in self.row_offset_policies:
                    for class_placement_policy in self.class_placement_policies:
                        for along_bucket in self.spacing_along_options:
                            for across_bucket in self.spacing_across_options:
                                state = RLLayoutBuilderState(
                                    family=family,
                                    row_pattern=row_pattern,
                                    row_offset_policy=row_offset_policy,
                                    class_placement_policy=class_placement_policy,
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
            "row_patterns": {name: list(values) for name, values in self.row_patterns.items()},
            "row_offset_policies": list(self.row_offset_policies),
            "class_placement_policies": list(self.class_placement_policies),
            "spacing_along_options": dict(self.spacing_along_options),
            "spacing_across_options": dict(self.spacing_across_options),
            "family_complexity": dict(self.family_complexity),
            "row_pattern_complexity": dict(self.row_pattern_complexity),
            "row_offset_complexity": dict(self.row_offset_complexity),
            "class_placement_complexity": dict(self.class_placement_complexity),
            "spacing_along_complexity": dict(self.spacing_along_complexity),
            "spacing_across_complexity": dict(self.spacing_across_complexity),
        }


def _parse_row_patterns(
    payload: Any,
    *,
    default_rows: int,
    default_cols: int,
) -> dict[str, tuple[int, ...]]:
    if not payload:
        return {"uniform": tuple([default_cols] * default_rows)}
    parsed: dict[str, tuple[int, ...]] = {}
    for name, raw in dict(payload).items():
        if isinstance(raw, str):
            values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
        else:
            values = tuple(int(part) for part in raw)
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"Invalid row pattern {name!r}: {raw!r}")
        parsed[str(name)] = values
    return parsed
