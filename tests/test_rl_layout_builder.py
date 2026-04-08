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
            "spacing_along_options": {"compact": 350.0, "standard": 450.0},
            "spacing_across_options": {"compact": 300.0, "loose": 500.0},
            "family_complexity": {"rectangular": 1.0, "staggered": 1.2},
            "spacing_along_complexity": {"compact": 0.0, "standard": 0.1},
            "spacing_across_complexity": {"compact": 0.0, "loose": 0.2},
        }
    )

    actions = builder.enumerate_layout_actions()
    assert len(actions) == 8
    assert {action.name for action in actions} == {
        "rect_compact_compact",
        "rect_compact_loose",
        "rect_standard_compact",
        "rect_standard_loose",
        "staggered_compact_compact",
        "staggered_compact_loose",
        "staggered_standard_compact",
        "staggered_standard_loose",
    }


def test_builder_valid_actions_progress_by_step() -> None:
    builder = RLLayoutBuilderConfig.from_dict(
        {
            "enabled": True,
            "spacing_along_options": {"compact": 1.0, "loose": 2.0},
            "spacing_across_options": {"compact": 1.0, "loose": 2.0},
        }
    )
    state0 = RLLayoutBuilderState()
    assert builder.valid_action_names(state0) == ["family:rectangular", "family:staggered"]

    state1 = builder.apply_action(state0, "family:rectangular")
    assert builder.valid_action_names(state1) == ["along:compact", "along:loose"]

    state2 = builder.apply_action(state1, "along:compact")
    assert builder.valid_action_names(state2) == ["across:compact", "across:loose"]

    state3 = builder.apply_action(state2, "across:loose")
    assert builder.is_complete(state3) is True
    assert builder.valid_action_names(state3) == []
