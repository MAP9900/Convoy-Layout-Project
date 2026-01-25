"""CLI runner for defender policy optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.defender_policy_opt import (
    PolicyObjective,
    optimize_policy_deterministic,
    optimize_policy_mixture_pairwise,
)
from convoy_sim.objectives import ObjectiveSpec
from scenarios.scenario_b1_policy_demo import build_scenario_b1_policy_demo


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize defender policy tables")
    parser.add_argument("--trials", type=int, default=200, help="Trials per evaluation")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory to store JSON results",
    )
    parser.add_argument("--w-loss", type=float, default=1.0, help="Weight on expected loss")
    parser.add_argument("--w-footprint", type=float, default=0.0, help="Weight on mean footprint")
    parser.add_argument("--w-complexity", type=float, default=0.0, help="Weight on mean complexity")
    parser.add_argument("--footprint-budget", type=float, default=None, help="Hard footprint budget")
    parser.add_argument("--complexity-budget", type=float, default=None, help="Hard complexity budget")
    parser.add_argument("--w-total-hits", type=float, default=0.0, help="Objective: weight hits")
    parser.add_argument("--w-total-value", type=float, default=1.0, help="Objective: weight value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_b1_policy_demo(n_trials=args.trials, rng_seed=args.seed)

    policy_obj = PolicyObjective(
        w_loss=args.w_loss,
        w_footprint=args.w_footprint,
        w_complexity=args.w_complexity,
        footprint_budget=args.footprint_budget,
        complexity_budget=args.complexity_budget,
    )
    objective = ObjectiveSpec(w_total_value=args.w_total_value, w_total_hits=args.w_total_hits)

    deterministic_policy, deterministic_result = optimize_policy_deterministic(
        prior=scenario["prior"],
        actions=scenario["policy"].actions,
        threats=list(scenario["prior"].probs.keys()),
        attacker_factory=scenario["attacker_factory"],
        n_trials=args.trials,
        sim_kwargs=scenario["sim_kwargs"],
        objective_spec=objective,
        policy_obj=policy_obj,
        rng_seed=args.seed,
    )
    mixture_policy, mixture_result = optimize_policy_mixture_pairwise(
        prior=scenario["prior"],
        actions=scenario["policy"].actions,
        threats=list(scenario["prior"].probs.keys()),
        attacker_factory=scenario["attacker_factory"],
        n_trials=args.trials,
        sim_kwargs=scenario["sim_kwargs"],
        objective_spec=objective,
        policy_obj=policy_obj,
        rng_seed=args.seed,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "policy_opt.json"
    payload = {
        "scenario": scenario["name"],
        "deterministic_policy": deterministic_policy.policy_table,
        "deterministic_eval": deterministic_result,
        "mixture_policy": mixture_policy.policy_table,
        "mixture_eval": mixture_result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    det_summary = deterministic_result["final_eval"]["summary"]
    mix_summary = mixture_result["final_eval"]["summary"]
    print(f"Wrote results to {out_path}")
    print(
        "Deterministic: E[hits]={:.2f}, E[value]={:.2f}".format(
            det_summary["expected_hits"], det_summary["expected_value_destroyed"]
        )
    )
    print(
        "Mixture:       E[hits]={:.2f}, E[value]={:.2f}".format(
            mix_summary["expected_hits"], mix_summary["expected_value_destroyed"]
        )
    )


if __name__ == "__main__":
    main()
