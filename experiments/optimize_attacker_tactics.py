"""CLI runner for coarse attacker tactics search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.attacker_tactics_opt import PlanTemplate, search_attacker_plans
from convoy_sim.feasibility import ApproachMode
from scenarios.scenario_b2_multisalvo_demo import build_scenario_b2_multisalvo_demo


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize attacker tactics plans")
    parser.add_argument("--trials", type=int, default=50, help="Trials per plan")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--top-k", type=int, default=10, help="Number of plans to keep")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory to store JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = build_scenario_b2_multisalvo_demo(rng_seed=args.seed)
    template = PlanTemplate(
        n_passes_options=[1, 2],
        launch_time_1=0.0,
        u_boat_pos_1=np.array([-1800.0, 0.0]),
        bearing_rad_1=0.0,
        approach_mode_1=ApproachMode.STERN_CHASE,
        pattern_1="fan",
        salvo_sizes_1=[2, 4],
        spread_options_1=[np.radians(6.0), np.radians(12.0)],
        asymmetry_options_1=[-0.2, 0.0, 0.2],
        edge_bias_options_1=[0.0, 0.5],
        launch_delay_options_2=[20.0, 40.0],
        u_boat_pos_2=np.array([-1800.0, 200.0]),
        bearing_rad_2=0.0,
        approach_mode_2=ApproachMode.STERN_CHASE,
        pattern_2="fan",
        salvo_sizes_2=[2, 4],
        spread_options_2=[np.radians(6.0), np.radians(12.0)],
        asymmetry_options_2=[0.0],
        edge_bias_options_2=[0.0, 0.5],
        abort_if_risk_above_options=[None, 2.0],
    )

    results = search_attacker_plans(
        ships_t0=scenario["ships"],
        template=template,
        constraints=scenario["constraints"],
        env=scenario["env"],
        dynamics=None,
        torpedo_params=scenario["torpedo_params"],
        n_trials=args.trials,
        t_max_global=scenario["t_max_global"],
        objective=None,
        rng_seed=args.seed,
        top_k=args.top_k,
    )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "attacker_tactics_opt.json"
    out_path.write_text(json.dumps({"results": results}, indent=2, default=_json_default))

    print(f"Wrote results to {out_path}")
    for rank, entry in enumerate(results, start=1):
        plan = entry["plan"]
        metrics = entry["metrics"]
        print(
            "#{:02d} utility={:.3f} E[hits]={:.2f} E[value]={:.2f}".format(
                rank,
                entry["utility"],
                metrics["expected_hits"],
                metrics["expected_value_destroyed"],
            )
        )
        print("  passes={}".format(len(plan["passes"])))


if __name__ == "__main__":
    main()
