"""Canonical config-first baseline suite runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.attack_profiles import DEFAULT_ATTACK_PROFILE_LIBRARY
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.workflows import (
    evaluate_layout_over_profiles,
    git_sha,
    iter_param_overrides,
    load_config,
    resolve_run_dir,
    summarize_profile_rows,
    write_json,
    write_profile_rows_csv,
    write_yaml,
)


_LAYOUTS = {
    "rectangular": make_rectangular_convoy,
    "staggered": make_staggered_convoy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical baseline suite")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline/default.toml"),
        help="Path to baseline config file (.toml/.json/.yaml)",
    )
    return parser.parse_args()


def _layout_from_config(layout_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    layout_type = str(layout_cfg.get("type", "rectangular"))
    if layout_type not in _LAYOUTS:
        raise ValueError(f"Unknown layout type: {layout_type}")
    layout_fn = _LAYOUTS[layout_type]

    kwargs = {k: v for k, v in layout_cfg.items() if k != "type"}
    if "origin" in kwargs:
        kwargs["origin"] = np.asarray(kwargs["origin"], dtype=float)
    return layout_fn, kwargs


def run_from_config(config: dict[str, Any], *, project_root: Path) -> Path:
    run_cfg = dict(config.get("run", {}))
    sim_cfg = dict(config.get("simulation", {}))
    split_cfg = dict(config.get("splits", {}))
    baseline_cfg = dict(config.get("baseline", {}))

    output_root = project_root / str(run_cfg.get("output_root", "results/runs"))
    run_dir = resolve_run_dir(output_root, "baseline", run_cfg.get("name"))

    static_layout_cfg = dict(baseline_cfg.get("static_layout", {}))
    static_layout_fn, static_layout_kwargs = _layout_from_config(static_layout_cfg)

    train_profile_ids = [str(x) for x in split_cfg.get("train_profiles", [])]
    eval_profile_ids = [str(x) for x in split_cfg.get("eval_profiles", [])]
    train_seeds = [int(x) for x in split_cfg.get("train_seeds", [0])]
    eval_seeds = [int(x) for x in split_cfg.get("eval_seeds", [1])]

    n_trials_per_seed = int(sim_cfg.get("n_trials_per_seed", 50))
    t_max = float(sim_cfg.get("t_max", 400.0))
    max_hits_per_torpedo = sim_cfg.get("max_hits_per_torpedo")
    max_hits_per_torpedo = None if max_hits_per_torpedo is None else int(max_hits_per_torpedo)

    static_rows = evaluate_layout_over_profiles(
        model_name="static_baseline",
        layout_fn=static_layout_fn,
        layout_kwargs=static_layout_kwargs,
        library=DEFAULT_ATTACK_PROFILE_LIBRARY,
        profile_ids=eval_profile_ids,
        seeds=eval_seeds,
        n_trials_per_seed=n_trials_per_seed,
        t_max=t_max,
        max_hits_per_torpedo=max_hits_per_torpedo,
    )

    search_cfg = dict(baseline_cfg.get("heuristic_search", {}))
    grid = dict(search_cfg.get("grid", {}))
    max_candidates = search_cfg.get("max_candidates")
    max_candidates = None if max_candidates is None else int(max_candidates)

    best_kwargs = dict(static_layout_kwargs)
    best_train_score = float("inf")

    for override in iter_param_overrides(grid, max_candidates=max_candidates):
        candidate_kwargs = dict(static_layout_kwargs)
        candidate_kwargs.update(override)
        candidate_rows = evaluate_layout_over_profiles(
            model_name="heuristic_baseline_train",
            layout_fn=static_layout_fn,
            layout_kwargs=candidate_kwargs,
            library=DEFAULT_ATTACK_PROFILE_LIBRARY,
            profile_ids=train_profile_ids,
            seeds=train_seeds,
            n_trials_per_seed=n_trials_per_seed,
            t_max=t_max,
            max_hits_per_torpedo=max_hits_per_torpedo,
        )
        candidate_summary = summarize_profile_rows(candidate_rows)
        score = float(candidate_summary["expected_hits"])
        if score < best_train_score:
            best_train_score = score
            best_kwargs = candidate_kwargs

    heuristic_rows = evaluate_layout_over_profiles(
        model_name="heuristic_baseline",
        layout_fn=static_layout_fn,
        layout_kwargs=best_kwargs,
        library=DEFAULT_ATTACK_PROFILE_LIBRARY,
        profile_ids=eval_profile_ids,
        seeds=eval_seeds,
        n_trials_per_seed=n_trials_per_seed,
        t_max=t_max,
        max_hits_per_torpedo=max_hits_per_torpedo,
    )

    all_rows = static_rows + heuristic_rows
    static_summary = summarize_profile_rows(static_rows)
    heuristic_summary = summarize_profile_rows(heuristic_rows)
    winner = "heuristic_baseline" if heuristic_summary["expected_hits"] < static_summary["expected_hits"] else "static_baseline"

    resolved = {
        "config": config,
        "resolved": {
            "static_layout": {
                "layout_fn": getattr(static_layout_fn, "__name__", str(static_layout_fn)),
                "layout_kwargs": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in static_layout_kwargs.items()},
            },
            "heuristic_best_layout": {
                "layout_fn": getattr(static_layout_fn, "__name__", str(static_layout_fn)),
                "layout_kwargs": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_kwargs.items()},
                "train_objective_expected_hits": best_train_score,
            },
        },
    }

    metrics_summary = {
        "static_baseline": static_summary,
        "heuristic_baseline": heuristic_summary,
        "winner": winner,
    }

    manifest = {
        "workflow": "baseline",
        "git_sha": git_sha(project_root),
        "profile_splits": {
            "train": train_profile_ids,
            "eval": eval_profile_ids,
        },
        "seed_sets": {
            "train": train_seeds,
            "eval": eval_seeds,
        },
        "n_trials_per_seed": n_trials_per_seed,
        "t_max": t_max,
    }

    write_yaml(run_dir / "config_resolved.yaml", resolved)
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_profile_rows_csv(run_dir / "per_profile_metrics.csv", all_rows)
    write_json(run_dir / "run_manifest.json", manifest)

    return run_dir


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    run_dir = run_from_config(config, project_root=project_root)
    print(f"Baseline suite completed: {run_dir}")


if __name__ == "__main__":
    main()
