"""CLI to generate datasets for surrogate modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convoy_sim.datasets import generate_dataset
from scenarios.scenario_a import build_scenario_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate surrogate modeling dataset")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/datasets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)
    parameter_space = {
        "defense": {
            "spacing_along": (400.0, 800.0),
            "spacing_across": (250.0, 450.0),
            "jitter_std": (0.0, 40.0),
        },
        "attack": {
            "mode": ["fan"],
            "base_bearing_rad": (-0.2, 0.2),
            "spread_rad": (0.0, 0.4),
            "n": [2, 4, 6],
            "launch_delay_mean": (0.0, 2.0),
            "torpedo_origin": [(-2000.0, 0.0)],
            "torpedo_speed": (20.0, 30.0),
            "torpedo_max_run_time": (600.0, 900.0),
        },
        "noise": {
            "sigma_heading_rad": (0.0, 0.05),
            "sigma_launch_delay": (0.0, 2.0),
            "p_dud": (0.0, 0.2),
        },
    }
    rng = np.random.default_rng(args.seed)
    result = generate_dataset(
        n_samples=args.samples,
        parameter_space=parameter_space,
        scenario_template=scenario,
        rng=rng,
        output_dir=args.output,
    )
    print(f"Wrote dataset to {result['csv_path']} and {result['npz_path']}")


if __name__ == "__main__":
    main()
