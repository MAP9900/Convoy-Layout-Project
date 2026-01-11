"""CLI runner for attacker parameter search against a fixed defense."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convoy_sim.attacker_opt import search_attack_params
from scenarios.scenario_a import build_scenario_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attacker parameter search")
    parser.add_argument("--defense-json", type=Path, default=Path("results/defender_best.json"))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=["fan", "parallel"], default="fan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)

    defense_params = {}
    if args.defense_json.exists():
        defense_params = json.loads(args.defense_json.read_text()).get("params", {})
    scenario.layout_kwargs.update(defense_params)

    grid = {
        "base_bearing_rad": [-0.2, 0.0, 0.2],
        "spread_rad": [0.0, 0.2, 0.4],
        "n": [2, 4, 6],
        "launch_delay_mean": [0.0, 2.0],
    }
    if args.mode == "parallel":
        grid.pop("spread_rad")
        grid["lateral_spacing"] = [50.0, 100.0, 150.0]

    csv_path = args.output / "attacker_opt.csv"
    json_path = args.output / "attacker_best.json"
    results = search_attack_params(
        layout_fn=scenario.layout_fn,
        layout_kwargs=scenario.layout_kwargs,
        param_grid=grid,
        torpedo_origin=(-2000.0, 0.0),
        torpedo_speed=25.0,
        torpedo_max_run_time=800.0,
        t_max=scenario.t_max,
        n_trials=args.trials,
        rng_seed=args.seed,
        mode=args.mode,
        output_csv=csv_path,
        output_json=json_path,
    )

    print("Top 10 attack candidates:")
    for rank, candidate in enumerate(results[:10], start=1):
        params = ", ".join(f"{k}={v}" for k, v in candidate.params.items())
        print(
            f"{rank:2d}. E[hits]={candidate.expected_hits:.2f}, "
            f"P(hit>=1)={candidate.p_hit_ge_1:.2f}, {params}"
        )
    print(f"Wrote CSV to {csv_path} and best candidate to {json_path}")


if __name__ == "__main__":
    main()
