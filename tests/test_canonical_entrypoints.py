"""Smoke tests for canonical baseline and RL entrypoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.run_baseline_suite import run_from_config as run_baseline_from_config
from experiments.run_rl_train import (
    _select_best_action_from_train_summaries,
    run_from_config as run_rl_from_config,
)
from convoy_sim.defender_policy import LayoutAction
from convoy_sim.layouts import make_rectangular_convoy


REQUIRED_BASE_FILES = {
    "config_resolved.yaml",
    "metrics_summary.json",
    "per_profile_metrics.csv",
    "run_manifest.json",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


REQUIRED_PROFILE_CSV_COLUMNS = {
    "model_name",
    "profile_id",
    "samples",
    "expected_hits",
    "CVaR_90",
    "p_hit_ge_1",
    "value_lost",
    "expected_unique_ships_hit",
    "expected_repeat_hits",
    "expected_loss",
    "CVaR_90_loss",
}


REQUIRED_SUMMARY_KEYS = {
    "profiles",
    "samples",
    "expected_hits",
    "CVaR_90",
    "VaR_90",
    "p_hit_ge_1",
    "value_lost",
    "expected_unique_ships_hit",
    "expected_repeat_hits",
    "expected_loss",
    "CVaR_90_loss",
}


def _assert_manifest_common(manifest: dict) -> None:
    assert {"workflow", "git_sha", "profile_splits", "seed_sets", "n_trials_per_seed", "t_max"} <= set(manifest)
    assert {"train", "eval"} <= set(manifest["profile_splits"])
    assert {"train", "eval"} <= set(manifest["seed_sets"])


def test_run_baseline_suite_writes_canonical_artifacts(tmp_path: Path) -> None:
    cfg = {
        "run": {"name": "smoke", "output_root": "runs"},
        "simulation": {"t_max": 120.0, "n_trials_per_seed": 2, "max_hits_per_torpedo": 1},
        "splits": {
            "train_profiles": ["P01", "P02"],
            "eval_profiles": ["P03", "P04"],
            "train_seeds": [11],
            "eval_seeds": [21],
        },
        "baseline": {
            "static_layout": {
                "type": "rectangular",
                "n_rows": 1,
                "n_cols": 2,
                "spacing_along": 400.0,
                "spacing_across": 300.0,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
            },
            "heuristic_search": {
                "max_candidates": 2,
                "grid": {
                    "spacing_along": [350.0, 450.0],
                },
            },
        },
    }

    run_dir = run_baseline_from_config(cfg, project_root=tmp_path)

    assert REQUIRED_BASE_FILES.issubset({item.name for item in run_dir.iterdir()})
    rows_path = run_dir / "per_profile_metrics.csv"
    rows = _read_csv_rows(rows_path)
    assert rows
    assert REQUIRED_PROFILE_CSV_COLUMNS <= set(rows[0].keys())
    assert {row["model_name"] for row in rows} == {"static_baseline", "heuristic_baseline"}

    metrics = _read_json(run_dir / "metrics_summary.json")
    assert {"static_baseline", "heuristic_baseline", "winner"} <= set(metrics)
    assert metrics["winner"] in {"static_baseline", "heuristic_baseline"}
    assert REQUIRED_SUMMARY_KEYS <= set(metrics["static_baseline"])
    assert REQUIRED_SUMMARY_KEYS <= set(metrics["heuristic_baseline"])

    manifest = _read_json(run_dir / "run_manifest.json")
    _assert_manifest_common(manifest)
    assert manifest["workflow"] == "baseline"
    assert "realism" in manifest
    assert manifest["realism"]["u_boat_mode_default"] == "moving"
    assert "objective" in manifest


def test_run_rl_train_writes_canonical_artifacts_and_checkpoint(tmp_path: Path) -> None:
    cfg = {
        "run": {"name": "smoke", "output_root": "runs"},
        "simulation": {"t_max": 120.0, "n_trials_per_seed": 2, "max_hits_per_torpedo": 1},
        "splits": {
            "train_profiles": ["P01", "P02"],
            "eval_profiles": ["P03", "P04"],
            "train_seeds": [31],
            "eval_seeds": [41],
        },
        "training": {
            "episodes": 8,
            "epsilon": 0.2,
            "epsilon_decay": 0.9,
            "epsilon_min": 0.05,
            "alpha": 0.3,
            "seed": 5,
        },
        "rl": {
            "selection": {
                "risk_cvar_weight": 0.05,
                "complexity_tiebreak_tolerance": 0.1,
            },
            "actions": [
                {
                    "name": "rect_a",
                    "type": "rectangular",
                    "complexity_cost": 1.0,
                    "n_rows": 1,
                    "n_cols": 2,
                    "spacing_along": 400.0,
                    "spacing_across": 300.0,
                    "speed": 5.0,
                    "heading_rad": 0.0,
                    "length": 120.0,
                    "beam": 18.0,
                    "origin": [0.0, 0.0],
                },
                {
                    "name": "stagger_b",
                    "type": "staggered",
                    "complexity_cost": 1.2,
                    "n_rows": 1,
                    "n_cols": 2,
                    "spacing_along": 420.0,
                    "spacing_across": 280.0,
                    "speed": 5.0,
                    "heading_rad": 0.0,
                    "length": 120.0,
                    "beam": 18.0,
                    "origin": [0.0, 0.0],
                },
            ]
        },
    }

    run_dir = run_rl_from_config(cfg, project_root=tmp_path)

    assert REQUIRED_BASE_FILES.issubset({item.name for item in run_dir.iterdir()})
    assert (run_dir / "checkpoints" / "policy_latest.json").exists()
    rows = _read_csv_rows(run_dir / "per_profile_metrics.csv")
    assert rows
    assert REQUIRED_PROFILE_CSV_COLUMNS <= set(rows[0].keys())
    assert {row["model_name"] for row in rows} == {"rl_policy"}

    metrics = _read_json(run_dir / "metrics_summary.json")
    assert {"training", "selection", "evaluation"} <= set(metrics)
    assert {
        "episodes",
        "epsilon_final",
        "avg_reward_last_50",
        "selected_action",
        "selected_action_by_q_value",
        "selection_method",
    } <= set(metrics["training"])
    assert {"risk_cvar_weight", "complexity_tiebreak_tolerance", "ranked_train_actions"} <= set(metrics["selection"])
    assert REQUIRED_SUMMARY_KEYS <= set(metrics["evaluation"])

    manifest = _read_json(run_dir / "run_manifest.json")
    _assert_manifest_common(manifest)
    assert manifest["workflow"] == "rl"
    assert "episodes" in manifest
    assert "realism" in manifest
    assert "selection" in manifest
    assert manifest["realism"]["u_boat_mode_default"] == "moving"
    assert "objective" in manifest


def test_run_rl_train_builder_mode_writes_selection_metadata(tmp_path: Path) -> None:
    cfg = {
        "run": {"name": "builder_smoke", "output_root": "runs"},
        "simulation": {"t_max": 120.0, "n_trials_per_seed": 2, "max_hits_per_torpedo": 1},
        "splits": {
            "train_profiles": ["P01", "P02"],
            "eval_profiles": ["P03", "P04"],
            "train_seeds": [31],
            "eval_seeds": [41],
        },
        "training": {
            "episodes": 6,
            "epsilon": 0.2,
            "epsilon_decay": 0.9,
            "epsilon_min": 0.05,
            "alpha": 0.3,
            "seed": 5,
        },
        "rl": {
            "builder": {
                "enabled": True,
                "base_n_rows": 1,
                "base_n_cols": 2,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
                "layout_families": ["rectangular", "staggered"],
                "spacing_along_options": {"compact": 400.0, "loose": 500.0},
                "spacing_across_options": {"compact": 300.0, "loose": 450.0},
                "family_complexity": {"rectangular": 1.0, "staggered": 1.2},
                "spacing_along_complexity": {"compact": 0.0, "loose": 0.1},
                "spacing_across_complexity": {"compact": 0.0, "loose": 0.1},
            },
            "selection": {
                "risk_cvar_weight": 0.05,
                "complexity_tiebreak_tolerance": 0.1,
            },
        },
    }

    run_dir = run_rl_from_config(cfg, project_root=tmp_path)
    metrics = _read_json(run_dir / "metrics_summary.json")
    assert metrics["training"]["mode"] == "builder"
    assert "builder_greedy_trace" in metrics["selection"]

    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["selection"]["mode"] == "builder"
    assert "builder" in manifest["selection"]


def test_rl_train_selection_prefers_simpler_action_when_scores_are_nearly_tied() -> None:
    actions = [
        LayoutAction(
            name="simple",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 1,
                "n_cols": 1,
                "spacing_along": 100.0,
                "spacing_across": 100.0,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
            },
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="complex",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 1,
                "n_cols": 1,
                "spacing_along": 100.0,
                "spacing_across": 100.0,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
            },
            complexity_cost=1.4,
        ),
    ]
    train_summaries = [
        {"expected_hits": 2.05, "CVaR_90": 3.0},
        {"expected_hits": 2.0, "CVaR_90": 3.0},
    ]
    best_idx, ranked = _select_best_action_from_train_summaries(
        actions,
        train_summaries,
        risk_cvar_weight=0.0,
        complexity_tiebreak_tolerance=0.1,
    )
    assert best_idx == 0
    assert ranked[0]["name"] == "complex"


def test_rl_train_builder_selection_does_not_override_better_primary_score() -> None:
    actions = [
        LayoutAction(
            name="rect_compact_standard",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 1,
                "n_cols": 1,
                "spacing_along": 100.0,
                "spacing_across": 100.0,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
            },
            complexity_cost=1.05,
        ),
        LayoutAction(
            name="rect_compact_loose",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 1,
                "n_cols": 1,
                "spacing_along": 100.0,
                "spacing_across": 120.0,
                "speed": 5.0,
                "heading_rad": 0.0,
                "length": 120.0,
                "beam": 18.0,
                "origin": [0.0, 0.0],
            },
            complexity_cost=1.15,
        ),
    ]
    train_summaries = [
        {"expected_hits": 2.5725, "CVaR_90": 3.1384668883586913},
        {"expected_hits": 2.5704166666666666, "CVaR_90": 3.083746117133406},
    ]
    best_idx, ranked = _select_best_action_from_train_summaries(
        actions,
        train_summaries,
        risk_cvar_weight=0.05,
        complexity_tiebreak_tolerance=0.1,
        selection_mode="builder",
    )
    assert best_idx == 1
    assert ranked[0]["name"] == "rect_compact_loose"
    assert ranked[0]["effective_tolerance"] < 0.004
