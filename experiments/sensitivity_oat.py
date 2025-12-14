"""One-at-a-time sensitivity study for convoy scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from scenarios.scenario_a import build_scenario_a

PARAMETERS = {
    "spacing_along": lambda scenario, value: scenario.layout_kwargs.__setitem__("spacing_along", value),
    "spacing_across": lambda scenario, value: scenario.layout_kwargs.__setitem__("spacing_across", value),
    "spread_rad": None,
    "sigma_heading_rad": None,
    "p_dud": None,
}


def _update_spread(scenario, value):
    # Scenario uses fan spread; spread_rad stored via metadata
    scenario.metadata["spread_rad"] = value


def _update_noise(scenario, attr: str, value: float):
    if scenario.noise_model is None:
        from convoy_sim.noise import NoiseModel

        scenario.noise_model = NoiseModel()
    scenario.noise_model = scenario.noise_model.__class__(
        **{field.name: (value if field.name == attr else getattr(scenario.noise_model, field.name))
           for field in scenario.noise_model.__dataclass_fields__.values()}
    )


PARAMETERS["spread_rad"] = _update_spread
PARAMETERS["sigma_heading_rad"] = lambda s, v: _update_noise(s, "sigma_heading_rad", v)
PARAMETERS["p_dud"] = lambda s, v: _update_noise(s, "p_dud", v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 sensitivity sweep")
    parser.add_argument("--scenario", default="scenario_a")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/sensitivity.csv"))
    return parser.parse_args()


def run_sweep(parameter: str, values: Iterable[float], base_scenario_builder: Callable[..., any], n_trials: int, seed: int) -> list[dict]:
    rows = []
    for value in values:
        scenario = base_scenario_builder(n_trials=n_trials, rng_seed=seed)
        updater = PARAMETERS[parameter]
        updater(scenario, value)
        result = scenario.run()
        rows.append(
            {
                "param_name": parameter,
                "param_value": value,
                "expected_hits": result["result"]["expected_hits"],
                "var_hits": result["result"]["var_hits"],
                "p_hit_ge_1": result["result"]["hit_prob_at_least_one"],
                "seed": seed,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    builder = build_scenario_a
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sweeps = {
        "spacing_along": np.linspace(400.0, 800.0, num=3),
        "spacing_across": np.linspace(250.0, 450.0, num=3),
        "spread_rad": np.linspace(np.radians(5.0), np.radians(25.0), num=3),
        "sigma_heading_rad": np.linspace(0.0, 0.05, num=3),
        "p_dud": np.linspace(0.0, 0.3, num=3),
    }
    all_rows = []
    for param, values in sweeps.items():
        rows = run_sweep(param, values, builder, args.n_trials, args.seed)
        all_rows.extend(rows)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["param_name", "param_value", "expected_hits", "var_hits", "p_hit_ge_1", "seed"])
        writer.writeheader()
        writer.writerows(all_rows)
    importance = {}
    for param in sweeps:
        vals = [row["expected_hits"] for row in all_rows if row["param_name"] == param]
        importance[param] = max(vals) - min(vals)
    ranking = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    print("Importance ranking (ΔE[hits]):")
    for param, delta in ranking:
        print(f"  {param}: {delta:.3f}")
    print(f"Wrote sensitivity results to {output_path}")


if __name__ == "__main__":
    main()
