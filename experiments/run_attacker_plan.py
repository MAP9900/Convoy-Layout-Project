"""CLI runner for multi-pass attacker plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.attacker_tactics import execute_attacker_plan
from scenarios.scenario_b2_multisalvo_demo import build_scenario_b2_multisalvo_demo


SCENARIOS = {
    "scenario_b2": build_scenario_b2_multisalvo_demo,
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
    parser = argparse.ArgumentParser(description="Run attacker plan simulations")
    parser.add_argument("scenario", choices=SCENARIOS.keys())
    parser.add_argument("--trials", type=int, default=50, help="Number of plan executions")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
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
    scenario = builder(rng_seed=args.seed)
    rng = np.random.default_rng(scenario["rng_seed"])

    totals = []
    per_trial = []
    for _ in range(args.trials):
        result = execute_attacker_plan(
            ships_t0=scenario["ships"],
            plan=scenario["plan"],
            constraints=scenario["constraints"],
            env=scenario["env"],
            dynamics=None,
            torpedo_params=scenario["torpedo_params"],
            t_max_global=scenario["t_max_global"],
            rng=rng,
        )
        per_trial.append(result)
        totals.append(result["totals"])

    expected_hits = float(np.mean([t["total_hits"] for t in totals])) if totals else 0.0
    expected_value = float(np.mean([t["total_value_destroyed"] for t in totals])) if totals else 0.0
    summary = {
        "expected_hits": expected_hits,
        "expected_value_destroyed": expected_value,
        "n_trials": args.trials,
    }

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.scenario}.json"
    payload = {
        "scenario": scenario["name"],
        "summary": summary,
        "trials": per_trial,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print(f"Wrote results to {out_path}")
    print(
        "Summary: E[hits]={:.2f}, E[value]={:.2f}".format(
            expected_hits,
            expected_value,
        )
    )
    if per_trial:
        per_pass = per_trial[0]["per_pass"]
        for idx, entry in enumerate(per_pass, start=1):
            print(
                "Pass {}: status={}, hits={}, torps={}".format(
                    idx,
                    entry["status"],
                    entry["n_hits"],
                    entry["n_torpedoes_fired"],
                )
            )


if __name__ == "__main__":
    main()
