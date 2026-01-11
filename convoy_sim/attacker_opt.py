"""Brute-force attack parameter search against a fixed defensive layout."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from convoy_sim.entities import Torpedo
from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.simulation import run_monte_carlo_attack


@dataclass(frozen=True)
class AttackCandidateResult:
    """Summary metrics for a single attack candidate."""

    params: dict[str, Any]
    expected_hits: float
    p_hit_ge_1: float
    var_hits: float


def _resolve_bearing(params: dict[str, Any], convoy_heading_rad: float) -> float:
    if "base_bearing_rad" in params:
        return float(params["base_bearing_rad"])
    if "bearing_offset_rad" in params:
        return convoy_heading_rad + float(params["bearing_offset_rad"])
    return convoy_heading_rad


def _apply_launch_delay_mean(torpedoes: Sequence[Torpedo], delay_mean: float) -> list[Torpedo]:
    if delay_mean <= 0.0:
        return list(torpedoes)
    adjusted: list[Torpedo] = []
    for torpedo in torpedoes:
        adjusted.append(
            Torpedo(
                id=torpedo.id,
                launch_position=torpedo.launch_position,
                speed=torpedo.speed,
                heading_rad=torpedo.heading_rad,
                max_run_time=torpedo.max_run_time,
                launch_delay=delay_mean,
                is_dud=torpedo.is_dud,
            )
        )
    return adjusted


def search_attack_params(
    layout_fn,
    layout_kwargs: dict[str, Any],
    param_grid: dict[str, Sequence[Any]],
    torpedo_origin,
    torpedo_speed: float,
    torpedo_max_run_time: float,
    t_max: float,
    n_trials: int,
    rng_seed: int | None = None,
    mode: str = "fan",
    convoy_heading_rad: float | None = None,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> list[AttackCandidateResult]:
    """Search attack parameters and return candidates ranked by expected hits."""

    if mode not in {"fan", "parallel"}:
        raise ValueError("mode must be 'fan' or 'parallel'")
    convoy_heading_rad = float(convoy_heading_rad or layout_kwargs.get("heading_rad", 0.0))

    grid_keys = list(param_grid.keys())
    results: list[AttackCandidateResult] = []

    for idx, values in enumerate(product(*[param_grid[k] for k in grid_keys])):
        params = dict(zip(grid_keys, values))
        base_bearing = _resolve_bearing(params, convoy_heading_rad)
        n = int(params.get("n", 1))
        spread_rad = float(params.get("spread_rad", 0.0))
        lateral_spacing = float(params.get("lateral_spacing", 0.0))
        delay_mean = float(params.get("launch_delay_mean", 0.0))

        def sampler(_: np.random.Generator):
            if mode == "fan":
                torps = fan_spread(
                    u_pos=torpedo_origin,
                    base_bearing_rad=base_bearing,
                    n=n,
                    spread_rad=spread_rad,
                    speed=torpedo_speed,
                    max_run_time=torpedo_max_run_time,
                )
            else:
                torps = parallel_spread(
                    u_pos=torpedo_origin,
                    bearing_rad=base_bearing,
                    n=n,
                    lateral_spacing=lateral_spacing,
                    speed=torpedo_speed,
                    max_run_time=torpedo_max_run_time,
                )
            return _apply_launch_delay_mean(torps, delay_mean)

        rng = np.random.default_rng(None if rng_seed is None else rng_seed + idx)
        result = run_monte_carlo_attack(
            layout_fn=layout_fn,
            layout_kwargs=layout_kwargs,
            torpedo_sampler=sampler,
            n_trials=n_trials,
            t_max=t_max,
            rng=rng,
        )
        results.append(
            AttackCandidateResult(
                params=params,
                expected_hits=result["expected_hits"],
                p_hit_ge_1=result["hit_prob_at_least_one"],
                var_hits=result["var_hits"],
            )
        )

    results.sort(key=lambda r: (-r.expected_hits, -r.p_hit_ge_1))

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(grid_keys) + ["expected_hits", "p_hit_ge_1", "var_hits", "seed"]
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in results:
                row = dict(candidate.params)
                row.update(
                    {
                        "expected_hits": candidate.expected_hits,
                        "p_hit_ge_1": candidate.p_hit_ge_1,
                        "var_hits": candidate.var_hits,
                        "seed": rng_seed,
                    }
                )
                writer.writerow(row)

    if output_json and results:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        best = results[0]
        payload = {
            "params": best.params,
            "expected_hits": best.expected_hits,
            "p_hit_ge_1": best.p_hit_ge_1,
            "var_hits": best.var_hits,
            "seed": rng_seed,
        }
        output_json.write_text(json.dumps(payload, indent=2))

    return results
