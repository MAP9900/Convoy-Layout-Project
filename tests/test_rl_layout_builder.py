"""Tests for the bounded RL layout builder."""

from __future__ import annotations

from convoy_sim.rl_layout_builder import RLLayoutBuilderConfig, RLLayoutBuilderState


def test_builder_enumerates_all_family_and_spacing_combinations() -> None:
    builder = RLLayoutBuilderConfig.from_dict(
        {
            "enabled": True,
            "base_n_rows": 2,
            "base_n_cols": 3,
            "speed": 5.0,
            "heading_rad": 0.0,
            "length": 120.0,
            "beam": 18.0,
            "origin": [0.0, 0.0],
            "layout_families": ["rectangular", "staggered"],
            "row_patterns": {"uniform": [3, 3], "center_heavy": [2, 4]},
            "row_offset_policies": ["none", "centered_alt"],
            "class_placement_policies": ["mixed_balanced", "high_value_center"],
            "spacing_along_options": {"compact": 350.0, "standard": 450.0},
            "spacing_across_options": {"compact": 300.0, "loose": 500.0},
            "family_complexity": {"rectangular": 1.0, "staggered": 1.2},
            "row_pattern_complexity": {"uniform": 0.0, "center_heavy": 0.2},
            "row_offset_complexity": {"none": 0.0, "centered_alt": 0.1},
            "class_placement_complexity": {"mixed_balanced": 0.0, "high_value_center": 0.1},
            "spacing_along_complexity": {"compact": 0.0, "standard": 0.1},
            "spacing_across_complexity": {"compact": 0.0, "loose": 0.2},
        }
    )

    actions = builder.enumerate_layout_actions()
    assert len(actions) == 64
    assert "rect_uniform_none_mixed_balanced_compact_compact" in {action.name for action in actions}
    assert (
        "staggered_center_heavy_centered_alt_high_value_center_standard_loose"
        in {action.name for action in actions}
    )


def test_builder_valid_actions_progress_by_step() -> None:
    builder = RLLayoutBuilderConfig.from_dict(
        {
            "enabled": True,
            "row_patterns": {"uniform": [2, 2]},
            "row_offset_policies": ["none", "centered_alt"],
            "class_placement_policies": ["mixed_balanced", "high_value_center"],
            "spacing_along_options": {"compact": 1.0, "loose": 2.0},
            "spacing_across_options": {"compact": 1.0, "loose": 2.0},
        }
    )
    state0 = RLLayoutBuilderState()
    assert builder.valid_action_names(state0) == ["family:rectangular", "family:staggered"]

    state1 = builder.apply_action(state0, "family:rectangular")
    assert builder.valid_action_names(state1) == ["pattern:uniform"]

    state2 = builder.apply_action(state1, "pattern:uniform")
    assert builder.valid_action_names(state2) == ["offset:none", "offset:centered_alt"]

    state3 = builder.apply_action(state2, "offset:none")
    assert builder.valid_action_names(state3) == ["placement:mixed_balanced", "placement:high_value_center"]

    state4 = builder.apply_action(state3, "placement:mixed_balanced")
    assert builder.valid_action_names(state4) == ["along:compact", "along:loose"]

    state5 = builder.apply_action(state4, "along:compact")
    assert builder.valid_action_names(state5) == ["across:compact", "across:loose"]

    state6 = builder.apply_action(state5, "across:loose")
    assert builder.is_complete(state6) is True
    assert builder.valid_action_names(state6) == []


def test_builder_materialized_action_preserves_fleet_profile_metadata() -> None:
    builder = RLLayoutBuilderConfig.from_dict(
        {
            "enabled": True,
            "base_n_rows": 2,
            "base_n_cols": 3,
            "speed": 5.0,
            "heading_rad": 0.0,
            "length": 120.0,
            "beam": 18.0,
            "origin": [0.0, 0.0],
            "fleet_profile": "mixed_convoy_v1",
            "fleet_seed": 1947,
            "layout_families": ["rectangular"],
            "row_patterns": {"center_heavy": [2, 4]},
            "row_offset_policies": ["centered_alt"],
            "class_placement_policies": ["high_value_center"],
            "spacing_along_options": {"compact": 350.0},
            "spacing_across_options": {"standard": 300.0},
        }
    )
    state = RLLayoutBuilderState(
        family="rectangular",
        row_pattern="center_heavy",
        row_offset_policy="centered_alt",
        class_placement_policy="high_value_center",
        spacing_along_bucket="compact",
        spacing_across_bucket="standard",
    )
    action = builder.materialize_layout_action(state)
    assert action.layout_kwargs["fleet_profile"] == "mixed_convoy_v1"
    assert action.layout_kwargs["fleet_seed"] == 1947
    assert action.layout_kwargs["row_counts"] == [2, 4]
    assert action.layout_kwargs["row_offset_policy"] == "centered_alt"
    assert action.layout_kwargs["class_placement_policy"] == "high_value_center"
