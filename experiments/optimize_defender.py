"""CLI runner for defender layout parameter search."""

from __future__ import annotations

import argparse
from pathlib import Path

from convoy_sim.defender_opt import search_layout_params
from scenarios.scenario_a import build_scenario_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defender layout parameter search")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)
    grid = {
        "spacing_along": [400.0, 600.0, 800.0],
        "spacing_across": [250.0, 350.0, 450.0],
        "jitter_std": [0.0, 20.0, 40.0],
    }
    csv_path = args.output / "defender_opt.csv"
    json_path = args.output / "defender_best.json"
    results = search_layout_params(
        scenario=scenario,
        param_grid=grid,
        n_trials=args.trials,
        rng_seed=args.seed,
        constraints=None,
        output_csv=csv_path,
        output_json=json_path,
    )
    print("Top 10 candidates:")
    for rank, candidate in enumerate(results[:10], start=1):
        params = ", ".join(f"{k}={v}" for k, v in candidate.params.items())
        print(
            f"{rank:2d}. E[hits]={candidate.expected_hits:.2f}, "
            f"P(hit>=1)={candidate.p_hit_ge_1:.2f}, "
            f"Area={candidate.footprint_area:.1f}, "
            f"{params}"
        )
    print(f"Wrote CSV to {csv_path} and best candidate to {json_path}")


if __name__ == "__main__":
    main()
