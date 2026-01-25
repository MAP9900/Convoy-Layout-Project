"""CLI runner to estimate payoff matrices and exploitability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.defender_policy import LayoutAction, ThreatPrior, ThreatType
from convoy_sim.game import (
    AttackerStrategy,
    DefenderStrategy,
    estimate_payoff_matrix,
    exploitability,
)
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate game payoff matrix")
    parser.add_argument("--trials", type=int, default=50, help="Trials per entry")
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
    rng = np.random.default_rng(args.seed)

    defenders = [
        DefenderStrategy(
            name="rect_tight",
            kind="layout_action",
            payload=LayoutAction(
                name="rect_tight",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 3,
                    "n_cols": 4,
                    "spacing_along": 500.0,
                    "spacing_across": 300.0,
                    "speed": 5.0,
                    "heading_rad": 0.0,
                    "length": 120.0,
                    "beam": 18.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        ),
        DefenderStrategy(
            name="rect_wide",
            kind="layout_action",
            payload=LayoutAction(
                name="rect_wide",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 3,
                    "n_cols": 4,
                    "spacing_along": 700.0,
                    "spacing_across": 500.0,
                    "speed": 5.0,
                    "heading_rad": 0.0,
                    "length": 120.0,
                    "beam": 18.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.5,
            ),
        ),
        DefenderStrategy(
            name="staggered",
            kind="layout_action",
            payload=LayoutAction(
                name="staggered",
                layout_fn=make_staggered_convoy,
                layout_kwargs={
                    "n_rows": 3,
                    "n_cols": 4,
                    "spacing_along": 600.0,
                    "spacing_across": 350.0,
                    "speed": 5.0,
                    "heading_rad": 0.0,
                    "length": 120.0,
                    "beam": 18.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=2.0,
            ),
        ),
    ]

    attackers = [
        AttackerStrategy(
            name="fan_small",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-2000.0, 0.0]),
                base_bearing_rad=0.0,
                n=3,
                spread_rad=np.radians(8.0),
                speed=25.0,
                max_run_time=800.0,
            ),
        ),
        AttackerStrategy(
            name="fan_wide",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-2000.0, 0.0]),
                base_bearing_rad=0.0,
                n=4,
                spread_rad=np.radians(15.0),
                speed=25.0,
                max_run_time=800.0,
            ),
        ),
        AttackerStrategy(
            name="parallel",
            kind="torpedo_sampler",
            payload=lambda rng: parallel_spread(
                u_pos=np.array([-2000.0, 0.0]),
                bearing_rad=0.0,
                n=3,
                lateral_spacing=120.0,
                speed=25.0,
                max_run_time=800.0,
            ),
        ),
    ]

    prior = ThreatPrior(
        probs={
            ThreatType.ABEAM_FAN: 0.7,
            ThreatType.BOW_ON_FAN: 0.3,
        }
    )

    result = estimate_payoff_matrix(
        defenders=defenders,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 400.0},
        objective=None,
        n_trials=args.trials,
        rng=rng,
    )

    m = result["matrix_mean_loss"]
    p_uniform = np.ones(m.shape[0]) / m.shape[0]
    q_uniform = np.ones(m.shape[1]) / m.shape[1]
    exp_uniform = exploitability(p_uniform, q_uniform, m)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "game_matrix.json"
    payload = {
        "matrix": result,
        "exploitability_uniform": exp_uniform,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    print(f"Wrote results to {out_path}")
    print("Mean loss matrix:")
    print(m)
    print("Exploitability (uniform):", exp_uniform)


if __name__ == "__main__":
    main()
