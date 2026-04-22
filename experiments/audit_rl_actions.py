"""Audit the current RL action menu directly on train/eval splits.

This answers a simple question:
- do the configured RL actions contain any good layouts at all?
- or is the current learner failing to discover them?
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.attack_profiles import DEFAULT_ATTACK_PROFILE_LIBRARY
from convoy_sim.defender_policy import LayoutAction
from convoy_sim.feasibility import Environment
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.noise import NoiseModel
from convoy_sim.objectives import objective_from_config, objective_to_dict
from convoy_sim.rl_layout_builder import RLLayoutBuilderConfig
from convoy_sim.workflows import (
    evaluate_layout_over_profiles,
    git_sha,
    load_config,
    resolve_run_dir,
    summarize_profile_rows,
    write_json,
    write_layout_plot,
    write_profile_rows_csv,
    write_yaml,
)


_LAYOUTS = {
    "rectangular": make_rectangular_convoy,
    "staggered": make_staggered_convoy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit configured RL actions directly on train/eval splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/default.toml"),
        help="Path to RL config file (.toml/.json/.yaml)",
    )
    return parser.parse_args()


def _layout_action_from_cfg(cfg: dict[str, Any]) -> LayoutAction:
    layout_type = str(cfg.get("type", "rectangular"))
    if layout_type not in _LAYOUTS:
        raise ValueError(f"Unknown layout type: {layout_type}")
    layout_fn = _LAYOUTS[layout_type]
    kwargs = {k: v for k, v in cfg.items() if k not in {"name", "type", "complexity_cost"}}
    if "origin" in kwargs:
        kwargs["origin"] = np.asarray(kwargs["origin"], dtype=float)
    return LayoutAction(
        name=str(cfg["name"]),
        layout_fn=layout_fn,
        layout_kwargs=kwargs,
        complexity_cost=float(cfg.get("complexity_cost", 0.0)),
    )


def _render_layout_kwargs(layout_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in layout_kwargs.items() if k != "ship_movement_realism"}


def _resolve_candidate_actions(
    rl_cfg: dict[str, Any],
    *,
    ship_movement_realism: dict[str, Any] | None,
) -> tuple[list[LayoutAction], RLLayoutBuilderConfig | None]:
    builder_cfg = RLLayoutBuilderConfig.from_dict(dict(rl_cfg.get("builder", {})))
    if builder_cfg.enabled:
        return (
            builder_cfg.enumerate_layout_actions(ship_movement_realism=ship_movement_realism),
            builder_cfg,
        )

    action_cfgs = list(rl_cfg.get("actions", []))
    actions = [_layout_action_from_cfg(item) for item in action_cfgs]
    if not actions:
        raise ValueError("Config must define at least one RL action")
    if ship_movement_realism:
        for action in actions:
            action.layout_kwargs["ship_movement_realism"] = ship_movement_realism
    return actions, None


def _write_action_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action_name",
        "split",
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
        "complexity_cost",
        "layout_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_from_config(config: dict[str, Any], *, project_root: Path) -> Path:
    started_at = time.perf_counter()
    run_cfg = dict(config.get("run", {}))
    sim_cfg = dict(config.get("simulation", {}))
    split_cfg = dict(config.get("splits", {}))
    rl_cfg = dict(config.get("rl", {}))
    objective_cfg = dict(config.get("objective", {}))
    plot_cfg = dict(config.get("plot", {}))

    output_root = project_root / str(run_cfg.get("output_root", "results/runs"))
    run_name = f"{run_cfg.get('name', 'rl')}_action_audit"
    run_dir = resolve_run_dir(output_root, "rl_action_audit", run_name)

    train_profile_ids = [str(x) for x in split_cfg.get("train_profiles", [])]
    eval_profile_ids = [str(x) for x in split_cfg.get("eval_profiles", [])]
    train_seeds = [int(x) for x in split_cfg.get("train_seeds", [0])]
    eval_seeds = [int(x) for x in split_cfg.get("eval_seeds", [1])]

    n_trials_per_seed = int(sim_cfg.get("n_trials_per_seed", 50))
    t_max = float(sim_cfg.get("t_max", 400.0))
    noise_model = NoiseModel.from_dict(dict(sim_cfg.get("noise", {})))
    env_cfg = dict(sim_cfg.get("environment", {}))
    env_profile = Environment(
        time_of_day=str(env_cfg.get("time_of_day", "night")),
        visibility_m=float(env_cfg.get("visibility_m", 3500.0)),
        sea_state=int(env_cfg.get("sea_state", 4)),
        detection_risk_scale=float(env_cfg.get("detection_risk_scale", 1.0)),
    )
    max_hits_per_torpedo = sim_cfg.get("max_hits_per_torpedo")
    max_hits_per_torpedo = None if max_hits_per_torpedo is None else int(max_hits_per_torpedo)
    objective = objective_from_config(objective_cfg)

    ship_movement_realism = dict(sim_cfg.get("ship_movement_realism", {}))
    actions, builder_cfg = _resolve_candidate_actions(
        rl_cfg,
        ship_movement_realism=ship_movement_realism or None,
    )

    plot_xlim = tuple(float(x) for x in plot_cfg["xlim"]) if "xlim" in plot_cfg else (-5000.0, 5000.0)
    plot_ylim = tuple(float(y) for y in plot_cfg["ylim"]) if "ylim" in plot_cfg else (-5000.0, 5000.0)
    plot_dpi = int(plot_cfg.get("dpi", 150))
    show_plot = bool(plot_cfg.get("show", True))
    figures_dir = run_dir / "figures"

    summary_rows: list[dict[str, Any]] = []
    all_profile_rows = []
    action_reports: list[dict[str, Any]] = []
    action_timing_rows: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()

    for action in actions:
        action_started = time.perf_counter()
        train_rows = evaluate_layout_over_profiles(
            model_name=f"{action.name}_train",
            layout_fn=action.layout_fn,
            layout_kwargs=action.layout_kwargs,
            library=DEFAULT_ATTACK_PROFILE_LIBRARY,
            profile_ids=train_profile_ids,
            seeds=train_seeds,
            n_trials_per_seed=n_trials_per_seed,
            t_max=t_max,
            noise_model=noise_model,
            env=env_profile,
            max_hits_per_torpedo=max_hits_per_torpedo,
            objective=objective,
        )
        eval_rows = evaluate_layout_over_profiles(
            model_name=f"{action.name}_eval",
            layout_fn=action.layout_fn,
            layout_kwargs=action.layout_kwargs,
            library=DEFAULT_ATTACK_PROFILE_LIBRARY,
            profile_ids=eval_profile_ids,
            seeds=eval_seeds,
            n_trials_per_seed=n_trials_per_seed,
            t_max=t_max,
            noise_model=noise_model,
            env=env_profile,
            max_hits_per_torpedo=max_hits_per_torpedo,
            objective=objective,
        )
        all_profile_rows.extend(train_rows)
        all_profile_rows.extend(eval_rows)

        train_summary = summarize_profile_rows(train_rows)
        eval_summary = summarize_profile_rows(eval_rows)
        layout_type = getattr(action.layout_fn, "__name__", str(action.layout_fn))

        summary_rows.append(
            {
                "action_name": action.name,
                "split": "train",
                "profiles": train_summary["profiles"],
                "samples": train_summary["samples"],
                "expected_hits": train_summary["expected_hits"],
                "CVaR_90": train_summary["CVaR_90"],
                "VaR_90": train_summary["VaR_90"],
                "p_hit_ge_1": train_summary["p_hit_ge_1"],
                "value_lost": train_summary["value_lost"],
                "expected_unique_ships_hit": train_summary["expected_unique_ships_hit"],
                "expected_repeat_hits": train_summary["expected_repeat_hits"],
                "expected_loss": train_summary["expected_loss"],
                "CVaR_90_loss": train_summary["CVaR_90_loss"],
                "complexity_cost": action.complexity_cost,
                "layout_type": layout_type,
            }
        )
        summary_rows.append(
            {
                "action_name": action.name,
                "split": "eval",
                "profiles": eval_summary["profiles"],
                "samples": eval_summary["samples"],
                "expected_hits": eval_summary["expected_hits"],
                "CVaR_90": eval_summary["CVaR_90"],
                "VaR_90": eval_summary["VaR_90"],
                "p_hit_ge_1": eval_summary["p_hit_ge_1"],
                "value_lost": eval_summary["value_lost"],
                "expected_unique_ships_hit": eval_summary["expected_unique_ships_hit"],
                "expected_repeat_hits": eval_summary["expected_repeat_hits"],
                "expected_loss": eval_summary["expected_loss"],
                "CVaR_90_loss": eval_summary["CVaR_90_loss"],
                "complexity_cost": action.complexity_cost,
                "layout_type": layout_type,
            }
        )

        ships = action.layout_fn(**_render_layout_kwargs(action.layout_kwargs))
        plot_path = figures_dir / f"{action.name}.png"
        plot_written = write_layout_plot(
            ships=ships,
            output_path=plot_path,
            title=f"RL Action Audit: {action.name}",
            xlim=plot_xlim,
            ylim=plot_ylim,
            dpi=plot_dpi,
            show_plot=show_plot,
        )
        action_elapsed = time.perf_counter() - action_started

        action_reports.append(
            {
                "name": action.name,
                "complexity_cost": float(action.complexity_cost),
                "layout_type": layout_type,
                "layout_kwargs": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in action.layout_kwargs.items()
                },
                "train_summary": train_summary,
                "eval_summary": eval_summary,
                "figure": str(plot_path.relative_to(project_root)) if plot_written else None,
            }
        )
        action_timing_rows.append(
            {
                "name": action.name,
                "seconds": action_elapsed,
                "complexity_cost": float(action.complexity_cost),
            }
        )
    evaluation_seconds = time.perf_counter() - evaluation_started

    best_train = min(action_reports, key=lambda item: float(item["train_summary"].get("expected_loss", item["train_summary"]["expected_hits"])))
    best_eval = min(action_reports, key=lambda item: float(item["eval_summary"].get("expected_loss", item["eval_summary"]["expected_hits"])))

    resolved = {
        "config": config,
        "resolved": {
            "actions": action_reports,
            "best_train_action": best_train["name"],
            "best_eval_action": best_eval["name"],
            "builder": builder_cfg.to_dict() if builder_cfg is not None and builder_cfg.enabled else None,
            "objective": objective_to_dict(objective),
        },
    }

    metrics_summary = {
        "best_train_action": {
            "name": best_train["name"],
            "expected_hits": best_train["train_summary"]["expected_hits"],
            "expected_loss": best_train["train_summary"].get("expected_loss"),
            "CVaR_90": best_train["train_summary"]["CVaR_90"],
            "CVaR_90_loss": best_train["train_summary"].get("CVaR_90_loss"),
            "eval_expected_hits": best_train["eval_summary"]["expected_hits"],
        },
        "best_eval_action": {
            "name": best_eval["name"],
            "expected_hits": best_eval["eval_summary"]["expected_hits"],
            "expected_loss": best_eval["eval_summary"].get("expected_loss"),
            "CVaR_90": best_eval["eval_summary"]["CVaR_90"],
            "CVaR_90_loss": best_eval["eval_summary"].get("CVaR_90_loss"),
            "train_expected_hits": best_eval["train_summary"]["expected_hits"],
        },
        "actions": action_reports,
        "timing": {
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "candidate_action_count": len(actions),
            "per_action_seconds": action_timing_rows,
        },
    }

    manifest = {
        "workflow": "rl_action_audit",
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
                "time_of_day": env_profile.time_of_day,
                "visibility_m": env_profile.visibility_m,
                "sea_state": env_profile.sea_state,
                "detection_risk_scale": env_profile.detection_risk_scale,
            },
            "ship_movement_realism_enabled": bool(ship_movement_realism),
        },
        "best_train_action": best_train["name"],
        "best_eval_action": best_eval["name"],
        "selection": {
            "mode": "builder" if builder_cfg is not None and builder_cfg.enabled else "flat_action_menu",
        },
        "objective": objective_to_dict(objective),
        "timing": {
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "candidate_action_count": len(actions),
        },
    }

    write_yaml(run_dir / "config_resolved.yaml", resolved)
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_json(run_dir / "run_manifest.json", manifest)
    write_profile_rows_csv(run_dir / "per_profile_metrics.csv", all_profile_rows)
    _write_action_summary_csv(run_dir / "per_action_metrics.csv", summary_rows)
    return run_dir


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    run_dir = run_from_config(config, project_root=project_root)
    print(f"RL action audit completed: {run_dir}")


if __name__ == "__main__":
    main()
