"""CLI runner for defender policy scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.defender_policy import evaluate_defender_policy
from scenarios.scenario_b1_policy_demo import build_scenario_b1_policy_demo


SCENARIOS = {
    "scenario_b1": build_scenario_b1_policy_demo,
}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defender policy evaluation scenarios")
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
    scenario = builder(n_trials=args.trials or 200, rng_seed=args.seed)
    rng = np.random.default_rng(scenario["rng_seed"])

    result = evaluate_defender_policy(
        prior=scenario["prior"],
        policy=scenario["policy"],
        attacker_factory=scenario["attacker_factory"],
        n_trials=scenario["n_trials"],
        sim_kwargs=scenario["sim_kwargs"],
        objective=scenario.get("objective"),
        rng=rng,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.scenario}.json"
    payload = {
        "scenario": scenario["name"],
        "summary": result["summary"],
        "trials": result["trials"],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    summary = result["summary"]
    print(f"Wrote results to {out_path}")
    print(
        "Summary: E[hits]={:.2f}, E[value]={:.2f}, E[loss]={}".format(
            summary["expected_hits"],
            summary["expected_value_destroyed"],
            "n/a" if summary["expected_loss"] is None else f"{summary['expected_loss']:.2f}",
        )
    )


if __name__ == "__main__":
    main()
