"""CLI runner for alternating defender/attacker best-response loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convoy_sim.minmax_loop import run_minmax_loop
from scenarios.scenario_a import build_scenario_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minmax loop")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/minmax_history.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)

    initial_defense = {
        "layout_fn": scenario.layout_fn,
        "layout_kwargs": dict(scenario.layout_kwargs),
        "t_max": scenario.t_max,
        "n_trials": scenario.n_trials,
        "noise_model": scenario.noise_model,
    }
    initial_attack = {
        "mode": "fan",
        "torpedo_origin": (-2000.0, 0.0),
        "torpedo_speed": 25.0,
        "torpedo_max_run_time": 800.0,
        "base_bearing_rad": 0.0,
        "spread_rad": 0.2,
        "n": 4,
        "launch_delay_mean": 0.0,
    }
    budgets = {
        "defense_grid": {
            "spacing_along": [400.0, 600.0],
            "spacing_across": [250.0, 350.0],
            "jitter_std": [0.0, 20.0],
        },
        "attack_grid": {
            "base_bearing_rad": [-0.2, 0.0, 0.2],
            "spread_rad": [0.0, 0.2, 0.4],
            "n": [2, 4],
            "launch_delay_mean": [0.0],
        },
        "n_trials": args.trials,
        "epsilon": 0.01,
        "patience": 2,
    }

    result = run_minmax_loop(
        initial_defense=initial_defense,
        initial_attack=initial_attack,
        n_rounds=args.rounds,
        rng_seed=args.seed,
        budgets=budgets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote minmax history to {args.output}")


if __name__ == "__main__":
    main()
