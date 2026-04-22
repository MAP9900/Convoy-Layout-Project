"""Canonical config-first baseline suite runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.attack_profiles import DEFAULT_ATTACK_PROFILE_LIBRARY
from convoy_sim.feasibility import Environment
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.noise import NoiseModel
from convoy_sim.objectives import objective_from_config, objective_to_dict
from convoy_sim.workflows import (
    evaluate_layout_over_profiles,
    git_sha,
    iter_param_overrides,
    load_config,
    resolve_run_dir,
    summarize_profile_rows,
    write_layout_plot,
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


def _render_layout_kwargs(layout_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in layout_kwargs.items() if k != "ship_movement_realism"}


def run_from_config(config: dict[str, Any], *, project_root: Path) -> Path:
    started_at = time.perf_counter()
    run_cfg = dict(config.get("run", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    sim_cfg = dict(config.get("simulation", {}))
    split_cfg = dict(config.get("splits", {}))
    baseline_cfg = dict(config.get("baseline", {}))
    objective_cfg = dict(config.get("objective", {}))
    plot_cfg = dict(config.get("plot", {}))

    output_root = project_root / str(run_cfg.get("output_root", "results/runs"))
    run_dir = resolve_run_dir(output_root, "baseline", run_cfg.get("name"))

    static_layout_cfg = dict(baseline_cfg.get("static_layout", {}))
    static_layout_fn, static_layout_kwargs = _layout_from_config(static_layout_cfg)

    train_profile_ids = [str(x) for x in split_cfg.get("train_profiles", [])]
    eval_profile_ids = [str(x) for x in split_cfg.get("eval_profiles", [])]
    train_seeds = [int(x) for x in split_cfg.get("train_seeds", [0])]
    eval_seeds = [int(x) for x in split_cfg.get("eval_seeds", [1])]

    n_trials_per_seed = int(sim_cfg.get("n_trials_per_seed", 50))
    baseline_search_n_trials_per_seed = int(
        runtime_cfg.get("baseline_search_n_trials_per_seed", n_trials_per_seed)
    )
    t_max = float(sim_cfg.get("t_max", 400.0))
    noise_model = NoiseModel.from_dict(dict(sim_cfg.get("noise", {})))
    env_cfg = dict(sim_cfg.get("environment", {}))
    env = Environment(
        time_of_day=str(env_cfg.get("time_of_day", "night")),
        visibility_m=float(env_cfg.get("visibility_m", 3500.0)),
        sea_state=int(env_cfg.get("sea_state", 4)),
        detection_risk_scale=float(env_cfg.get("detection_risk_scale", 1.0)),
    )
    ship_movement_realism = dict(sim_cfg.get("ship_movement_realism", {}))
    if ship_movement_realism:
        static_layout_kwargs["ship_movement_realism"] = ship_movement_realism
    max_hits_per_torpedo = sim_cfg.get("max_hits_per_torpedo")
    max_hits_per_torpedo = None if max_hits_per_torpedo is None else int(max_hits_per_torpedo)
    objective = objective_from_config(objective_cfg)

    static_eval_started = time.perf_counter()
    static_rows = evaluate_layout_over_profiles(
        model_name="static_baseline",
        layout_fn=static_layout_fn,
        layout_kwargs=static_layout_kwargs,
        library=DEFAULT_ATTACK_PROFILE_LIBRARY,
        profile_ids=eval_profile_ids,
        seeds=eval_seeds,
        n_trials_per_seed=n_trials_per_seed,
        t_max=t_max,
        noise_model=noise_model,
        env=env,
        max_hits_per_torpedo=max_hits_per_torpedo,
        objective=objective,
    )
    static_eval_seconds = time.perf_counter() - static_eval_started

    search_cfg = dict(baseline_cfg.get("heuristic_search", {}))
    grid = dict(search_cfg.get("grid", {}))
    max_candidates = search_cfg.get("max_candidates")
    max_candidates = None if max_candidates is None else int(max_candidates)

    best_kwargs = dict(static_layout_kwargs)
    best_train_score = float("inf")
    heuristic_search_started = time.perf_counter()
    heuristic_candidate_count = 0

    for override in iter_param_overrides(grid, max_candidates=max_candidates):
        heuristic_candidate_count += 1
        candidate_kwargs = dict(static_layout_kwargs)
        candidate_kwargs.update(override)
        candidate_rows = evaluate_layout_over_profiles(
            model_name="heuristic_baseline_train",
            layout_fn=static_layout_fn,
            layout_kwargs=candidate_kwargs,
            library=DEFAULT_ATTACK_PROFILE_LIBRARY,
            profile_ids=train_profile_ids,
            seeds=train_seeds,
            n_trials_per_seed=baseline_search_n_trials_per_seed,
            t_max=t_max,
            noise_model=noise_model,
            env=env,
            max_hits_per_torpedo=max_hits_per_torpedo,
            objective=objective,
        )
        candidate_summary = summarize_profile_rows(candidate_rows)
        score = float(candidate_summary["expected_loss"] if objective is not None else candidate_summary["expected_hits"])
        if score < best_train_score:
            best_train_score = score
            best_kwargs = candidate_kwargs
    heuristic_search_seconds = time.perf_counter() - heuristic_search_started

    heuristic_eval_started = time.perf_counter()
    heuristic_rows = evaluate_layout_over_profiles(
        model_name="heuristic_baseline",
        layout_fn=static_layout_fn,
        layout_kwargs=best_kwargs,
        library=DEFAULT_ATTACK_PROFILE_LIBRARY,
        profile_ids=eval_profile_ids,
        seeds=eval_seeds,
        n_trials_per_seed=n_trials_per_seed,
        t_max=t_max,
        noise_model=noise_model,
        env=env,
        max_hits_per_torpedo=max_hits_per_torpedo,
        objective=objective,
    )
    heuristic_eval_seconds = time.perf_counter() - heuristic_eval_started

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
                "train_objective_score": best_train_score,
            },
            "objective": objective_to_dict(objective),
        },
    }

    metrics_summary = {
        "static_baseline": static_summary,
        "heuristic_baseline": heuristic_summary,
        "winner": winner,
        "runtime_budgets": {
            "baseline_search_n_trials_per_seed": baseline_search_n_trials_per_seed,
            "final_eval_n_trials_per_seed": n_trials_per_seed,
        },
        "timing": {
            "static_eval_seconds": static_eval_seconds,
            "heuristic_search_seconds": heuristic_search_seconds,
            "heuristic_eval_seconds": heuristic_eval_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "heuristic_candidate_count": heuristic_candidate_count,
        },
    }

    plot_xlim = tuple(float(x) for x in plot_cfg["xlim"]) if "xlim" in plot_cfg else (-5000.0, 5000.0)
    plot_ylim = tuple(float(y) for y in plot_cfg["ylim"]) if "ylim" in plot_cfg else (-5000.0, 5000.0)
    plot_dpi = int(plot_cfg.get("dpi", 150))
    show_plot = bool(plot_cfg.get("show", True))
    figures_dir = run_dir / "figures"
    static_ships = static_layout_fn(**_render_layout_kwargs(static_layout_kwargs))
    heuristic_ships = static_layout_fn(**_render_layout_kwargs(best_kwargs))
    static_plot_path = figures_dir / "layout_static.png"
    heuristic_plot_path = figures_dir / "layout_heuristic_best.png"
    static_plot_written = write_layout_plot(
        ships=static_ships,
        output_path=static_plot_path,
        title="Baseline Static Layout",
        xlim=plot_xlim,
        ylim=plot_ylim,
        dpi=plot_dpi,
        show_plot=show_plot,
    )
    heuristic_plot_written = write_layout_plot(
        ships=heuristic_ships,
        output_path=heuristic_plot_path,
        title="Baseline Heuristic Best Layout",
        xlim=plot_xlim,
        ylim=plot_ylim,
        dpi=plot_dpi,
        show_plot=show_plot,
    )

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
        "realism": {
            "u_boat_mode_default": "moving",
            "noise_model": noise_model.to_dict(),
            "environment": {
                "time_of_day": env.time_of_day,
                "visibility_m": env.visibility_m,
                "sea_state": env.sea_state,
                "detection_risk_scale": env.detection_risk_scale,
            },
            "ship_movement_realism_enabled": bool(ship_movement_realism),
        },
        "objective": objective_to_dict(objective),
        "runtime_budgets": {
            "baseline_search_n_trials_per_seed": baseline_search_n_trials_per_seed,
            "final_eval_n_trials_per_seed": n_trials_per_seed,
        },
        "timing": {
            "static_eval_seconds": static_eval_seconds,
            "heuristic_search_seconds": heuristic_search_seconds,
            "heuristic_eval_seconds": heuristic_eval_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "heuristic_candidate_count": heuristic_candidate_count,
        },
        "layout_plots": {
            "static": str(static_plot_path.relative_to(project_root)) if static_plot_written else None,
            "heuristic_best": str(heuristic_plot_path.relative_to(project_root)) if heuristic_plot_written else None,
        },
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
