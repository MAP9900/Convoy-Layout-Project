"""Dataset generation utilities for surrogate modeling experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from convoy_sim.noise import NoiseModel
from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.simulation import run_monte_carlo_attack
from scenarios.scenario_base import Scenario


def _sample_value(rng: np.random.Generator, spec: Any) -> Any:
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            low, high = float(spec[0]), float(spec[1])
            return rng.uniform(low, high)
        return spec[int(rng.integers(0, len(spec)))]
    return spec


def _sample_params(rng: np.random.Generator, space: dict[str, Any]) -> dict[str, Any]:
    return {key: _sample_value(rng, spec) for key, spec in space.items()}


def generate_dataset(
    n_samples: int,
    parameter_space: dict[str, dict[str, Any]],
    scenario_template: Scenario,
    rng: np.random.Generator,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a dataset by sampling defense/attack/noise parameters.

    parameter_space should contain keys: "defense", "attack", "noise".
    Each sub-dict maps param names to either a (min, max) range or a list of options.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if output_dir is None:
        output_dir = Path("results/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    defense_space = parameter_space.get("defense", {})
    attack_space = parameter_space.get("attack", {})
    noise_space = parameter_space.get("noise", {})

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    hits_matrix: list[np.ndarray] = []

    for sample_idx in range(n_samples):
        defense_params = _sample_params(rng, defense_space)
        attack_params = _sample_params(rng, attack_space)
        noise_params = _sample_params(rng, noise_space)

        layout_kwargs = dict(scenario_template.layout_kwargs)
        layout_kwargs.update(defense_params)

        mode = attack_params.get("mode", "fan")
        base_bearing = float(attack_params.get("base_bearing_rad", layout_kwargs.get("heading_rad", 0.0)))
        n = int(round(attack_params.get("n", 1)))
        spread_rad = float(attack_params.get("spread_rad", 0.0))
        lateral_spacing = float(attack_params.get("lateral_spacing", 0.0))
        launch_delay_mean = float(attack_params.get("launch_delay_mean", 0.0))
        torpedo_origin = attack_params.get("torpedo_origin", (-2000.0, 0.0))
        torpedo_speed = float(attack_params.get("torpedo_speed", 25.0))
        torpedo_max_run_time = float(attack_params.get("torpedo_max_run_time", 800.0))

        def sampler(_: np.random.Generator):
            if mode == "parallel":
                torps = parallel_spread(
                    u_pos=torpedo_origin,
                    bearing_rad=base_bearing,
                    n=n,
                    lateral_spacing=lateral_spacing,
                    speed=torpedo_speed,
                    max_run_time=torpedo_max_run_time,
                )
            else:
                torps = fan_spread(
                    u_pos=torpedo_origin,
                    base_bearing_rad=base_bearing,
                    n=n,
                    spread_rad=spread_rad,
                    speed=torpedo_speed,
                    max_run_time=torpedo_max_run_time,
                )
            if launch_delay_mean > 0.0:
                for torpedo in torps:
                    torpedo.launch_delay = launch_delay_mean
            return torps

        noise_model = NoiseModel(
            sigma_heading_rad=float(noise_params.get("sigma_heading_rad", 0.0)),
            sigma_launch_delay=float(noise_params.get("sigma_launch_delay", 0.0)),
            p_dud=float(noise_params.get("p_dud", 0.0)),
        )

        result = run_monte_carlo_attack(
            layout_fn=scenario_template.layout_fn,
            layout_kwargs=layout_kwargs,
            torpedo_sampler=sampler,
            n_trials=scenario_template.n_trials,
            t_max=scenario_template.t_max,
            rng=rng,
            noise_model=noise_model,
            risk_alpha=0.9,
        )

        feature_rows.append(
            {
                **{f"def_{k}": v for k, v in defense_params.items()},
                **{f"atk_{k}": v for k, v in attack_params.items()},
                **{f"noise_{k}": v for k, v in noise_params.items()},
            }
        )
        target_rows.append(
            {
                "expected_hits": result["expected_hits"],
                "p_hit_ge_1": result["hit_prob_at_least_one"],
                "CVaR_90": result["CVaR_90"],
            }
        )
        hits_matrix.append(np.asarray(result["hits_per_trial"], dtype=float))

    feature_columns = sorted(feature_rows[0].keys()) if feature_rows else []
    target_columns = ["expected_hits", "p_hit_ge_1", "CVaR_90"]

    csv_path = output_dir / "dataset.csv"
    with csv_path.open("w", newline="") as handle:
        header = feature_columns + target_columns
        handle.write(",".join(header) + "\n")
        for features, targets in zip(feature_rows, target_rows):
            row = [features.get(col, "") for col in feature_columns] + [targets[col] for col in target_columns]
            handle.write(",".join(str(val) for val in row) + "\n")

    npz_path = output_dir / "dataset.npz"
    if hits_matrix:
        hits_array = np.vstack(hits_matrix)
        np.savez(
            npz_path,
            hits_per_trial=hits_array,
            feature_columns=np.array(feature_columns, dtype=object),
            target_columns=np.array(target_columns, dtype=object),
        )

    return {
        "csv_path": csv_path,
        "npz_path": npz_path,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "n_samples": n_samples,
    }
