"""Generate robustness report across noise settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from convoy_sim import NoiseModel, run_monte_carlo_attack
from scenarios.scenario_a import build_scenario_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness report")
    parser.add_argument("--defense-json", type=Path, default=Path("results/defender_best.json"))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/robustness_report.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)

    baseline_kwargs = dict(scenario.layout_kwargs)
    optimized_kwargs = dict(scenario.layout_kwargs)
    if args.defense_json.exists():
        optimized = json.loads(args.defense_json.read_text())
        optimized_kwargs.update(optimized.get("params", {}))

    noise_settings = [
        NoiseModel(),
        NoiseModel(sigma_heading_rad=0.02, p_dud=0.0),
        NoiseModel(sigma_heading_rad=0.05, p_dud=0.1),
        NoiseModel(sigma_heading_rad=0.05, p_dud=0.2),
    ]

    rows = []
    for label, layout_kwargs in (
        ("baseline", baseline_kwargs),
        ("optimized", optimized_kwargs),
    ):
        for noise in noise_settings:
            rng = np.random.default_rng(args.seed)
            result = run_monte_carlo_attack(
                layout_fn=scenario.layout_fn,
                layout_kwargs=layout_kwargs,
                torpedo_sampler=scenario.torpedo_sampler,
                n_trials=args.trials,
                t_max=scenario.t_max,
                rng=rng,
                noise_model=noise,
                risk_alpha=0.9,
            )
            rows.append(
                {
                    "defense": label,
                    "sigma_heading_rad": noise.sigma_heading_rad,
                    "sigma_launch_delay": noise.sigma_launch_delay,
                    "p_dud": noise.p_dud,
                    "expected_hits": result["expected_hits"],
                    "var_hits": result["var_hits"],
                    "VaR_90": result["VaR_90"],
                    "CVaR_90": result["CVaR_90"],
                    "seed": args.seed,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "defense",
                "sigma_heading_rad",
                "sigma_launch_delay",
                "p_dud",
                "expected_hits",
                "var_hits",
                "VaR_90",
                "CVaR_90",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote robustness report to {args.output}")


if __name__ == "__main__":
    main()
