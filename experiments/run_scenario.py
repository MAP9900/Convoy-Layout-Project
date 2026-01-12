"""CLI runner for convoy attack scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_a1_constraints import build_scenario_a1

SCENARIOS = {
    "scenario_a": build_scenario_a,
    "scenario_a1": build_scenario_a,
}

import numpy as np

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o

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
    out_path.write_text(json.dumps(result, indent=2, default=_json_default))
    summary = result["result"]
    print(f"Wrote results to {out_path}")
    print(
        "Summary: E[hits]={:.2f}, Var={:.3f}, P(hit>=1)={:.2f}".format(
            summary["expected_hits"],
            summary["var_hits"],
            summary["hit_prob_at_least_one"],
        )
    )


if __name__ == "__main__":
    main()
