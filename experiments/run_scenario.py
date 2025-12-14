"""CLI runner for convoy attack scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenarios.scenario_a import build_scenario_a

SCENARIOS = {
    "scenario_a": build_scenario_a,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run convoy attack scenario experiments")
    parser.add_argument("scenario", choices=SCENARIOS.keys())
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory to store JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = SCENARIOS[args.scenario]
    scenario = builder(n_trials=args.trials or 1000, rng_seed=args.seed)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    result = scenario.run()
    out_path = output_dir / f"{args.scenario}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
