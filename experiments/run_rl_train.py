"""Canonical config-first RL training/evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.attack_profiles import DEFAULT_ATTACK_PROFILE_LIBRARY, AttackProfileLibrary
from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import LayoutAction, ThreatPrior, ThreatType
from convoy_sim.game import AttackerStrategy, DefenderStrategy
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.rl_wrapper import RLEpisode
from convoy_sim.workflows import (
    evaluate_layout_over_profiles,
    git_sha,
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
    parser = argparse.ArgumentParser(description="Run canonical RL train/eval workflow")
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


def _train_library(default_library: AttackProfileLibrary, train_ids: list[str]) -> AttackProfileLibrary:
    keep = [profile for profile in default_library.profiles if profile.profile_id in set(train_ids)]
    if not keep:
        raise ValueError("No train profiles matched the default library")
    return AttackProfileLibrary(profiles=keep)


def run_from_config(config: dict[str, Any], *, project_root: Path) -> Path:
    run_cfg = dict(config.get("run", {}))
    sim_cfg = dict(config.get("simulation", {}))
    split_cfg = dict(config.get("splits", {}))
    train_cfg = dict(config.get("training", {}))
    rl_cfg = dict(config.get("rl", {}))
    plot_cfg = dict(config.get("plot", {}))

    output_root = project_root / str(run_cfg.get("output_root", "results/runs"))
    run_dir = resolve_run_dir(output_root, "rl", run_cfg.get("name"))

    train_profile_ids = [str(x) for x in split_cfg.get("train_profiles", [])]
    eval_profile_ids = [str(x) for x in split_cfg.get("eval_profiles", [])]
    train_seeds = [int(x) for x in split_cfg.get("train_seeds", [0])]
    eval_seeds = [int(x) for x in split_cfg.get("eval_seeds", [1])]

    n_trials_per_seed = int(sim_cfg.get("n_trials_per_seed", 50))
    t_max = float(sim_cfg.get("t_max", 400.0))
    max_hits_per_torpedo = sim_cfg.get("max_hits_per_torpedo")
    max_hits_per_torpedo = None if max_hits_per_torpedo is None else int(max_hits_per_torpedo)

    action_cfgs = list(rl_cfg.get("actions", []))
    actions = [_layout_action_from_cfg(item) for item in action_cfgs]
    if not actions:
        raise ValueError("Config must define at least one RL action")

    defenders = [
        DefenderStrategy(name=action.name, kind="layout_action", payload=action)
        for action in actions
    ]

    attacker = AttackerStrategy(
        name="profile_sampler",
        kind="torpedo_sampler",
        payload=lambda rng: fan_spread(
            u_pos=np.array([-2000.0, 0.0]),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=0.0,
            speed=25.0,
            max_run_time=800.0,
        ),
    )

    episodes = int(train_cfg.get("episodes", 500))
    epsilon = float(train_cfg.get("epsilon", 0.25))
    epsilon_decay = float(train_cfg.get("epsilon_decay", 0.995))
    epsilon_min = float(train_cfg.get("epsilon_min", 0.02))
    alpha = float(train_cfg.get("alpha", 0.1))
    train_seed = int(train_cfg.get("seed", 0))

    profile_lib = _train_library(DEFAULT_ATTACK_PROFILE_LIBRARY, train_profile_ids)

    env = RLEpisode(
        defenders=defenders,
        attackers=[attacker],
        prior=ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0}),
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={
            "t_max": t_max,
            "max_hits_per_torpedo": max_hits_per_torpedo,
        },
        max_steps=1,
        reward_perspective="defender",
        attack_profile_library=profile_lib,
        use_sampled_attack_profile_for_torpedo_sampler=True,
        rng=np.random.default_rng(train_seed),
    )

    q_values = np.zeros(len(actions), dtype=float)
    action_counts = np.zeros(len(actions), dtype=int)
    reward_history: list[float] = []

    choose_rng = np.random.default_rng(train_seed)

    for episode in range(episodes):
        reset_seed = int(train_seeds[episode % len(train_seeds)] + episode)
        env.reset(seed=reset_seed)
        if choose_rng.random() < epsilon:
            action_idx = int(choose_rng.integers(len(actions)))
        else:
            action_idx = int(np.argmax(q_values))

        _obs, reward, _done, _info = env.step(action_idx, 0)
        q_values[action_idx] = q_values[action_idx] + alpha * (float(reward) - q_values[action_idx])
        action_counts[action_idx] += 1
        reward_history.append(float(reward))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    best_action_idx = int(np.argmax(q_values))
    best_action = actions[best_action_idx]

    eval_rows = evaluate_layout_over_profiles(
        model_name="rl_policy",
        layout_fn=best_action.layout_fn,
        layout_kwargs=best_action.layout_kwargs,
        library=DEFAULT_ATTACK_PROFILE_LIBRARY,
        profile_ids=eval_profile_ids,
        seeds=eval_seeds,
        n_trials_per_seed=n_trials_per_seed,
        t_max=t_max,
        max_hits_per_torpedo=max_hits_per_torpedo,
    )

    eval_summary = summarize_profile_rows(eval_rows)
    training_summary = {
        "episodes": episodes,
        "epsilon_final": epsilon,
        "avg_reward_last_50": float(np.mean(reward_history[-50:])) if reward_history else 0.0,
        "selected_action": best_action.name,
    }

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "selected_action": best_action.name,
        "q_values": {actions[idx].name: float(q_values[idx]) for idx in range(len(actions))},
        "action_counts": {actions[idx].name: int(action_counts[idx]) for idx in range(len(actions))},
        "training_summary": training_summary,
    }
    (checkpoint_dir / "policy_latest.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    resolved = {
        "config": config,
        "resolved": {
            "selected_action": best_action.name,
            "selected_layout_fn": getattr(best_action.layout_fn, "__name__", str(best_action.layout_fn)),
            "selected_layout_kwargs": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in best_action.layout_kwargs.items()
            },
        },
    }

    metrics_summary = {
        "training": training_summary,
        "evaluation": eval_summary,
    }

    plot_xlim = tuple(float(x) for x in plot_cfg["xlim"]) if "xlim" in plot_cfg else (-5000.0, 5000.0)
    plot_ylim = tuple(float(y) for y in plot_cfg["ylim"]) if "ylim" in plot_cfg else (-5000.0, 5000.0)
    plot_dpi = int(plot_cfg.get("dpi", 150))
    show_plot = bool(plot_cfg.get("show", True))
    figures_dir = run_dir / "figures"
    selected_ships = best_action.layout_fn(**best_action.layout_kwargs)
    selected_plot_path = figures_dir / "layout_selected_policy.png"
    selected_plot_written = write_layout_plot(
        ships=selected_ships,
        output_path=selected_plot_path,
        title=f"RL Selected Layout: {best_action.name}",
        xlim=plot_xlim,
        ylim=plot_ylim,
        dpi=plot_dpi,
        show_plot=show_plot,
    )

    manifest = {
        "workflow": "rl",
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
        "episodes": episodes,
        "layout_plots": {
            "selected_policy": str(selected_plot_path.relative_to(project_root)) if selected_plot_written else None,
        },
    }

    write_yaml(run_dir / "config_resolved.yaml", resolved)
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_profile_rows_csv(run_dir / "per_profile_metrics.csv", eval_rows)
    write_json(run_dir / "run_manifest.json", manifest)

    return run_dir


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    run_dir = run_from_config(config, project_root=project_root)
    print(f"RL run completed: {run_dir}")


if __name__ == "__main__":
    main()
