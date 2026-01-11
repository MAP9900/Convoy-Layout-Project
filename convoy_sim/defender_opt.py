"""Brute-force layout parameter search for defender optimization."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from convoy_sim.geometry import Vec2, as_vec
from convoy_sim.simulation import run_monte_carlo_attack
from scenarios.scenario_base import Scenario


@dataclass(frozen=True)
class LayoutCandidateResult:
    """Summary metrics for a single layout candidate."""

    params: dict[str, Any]
    expected_hits: float
    p_hit_ge_1: float
    var_hits: float
    footprint_area: float
    max_extent_along: float
    max_extent_across: float


def _convoy_frame_extents(positions: Sequence[Vec2], heading_rad: float) -> tuple[float, float]:
    if not positions:
        return 0.0, 0.0
    cos_h = math.cos(-heading_rad)
    sin_h = math.sin(-heading_rad)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
    rotated = np.array([rotation @ np.asarray(pos, dtype=float) for pos in positions])
    min_x = float(np.min(rotated[:, 0]))
    max_x = float(np.max(rotated[:, 0]))
    min_y = float(np.min(rotated[:, 1]))
    max_y = float(np.max(rotated[:, 1]))
    return max_x - min_x, max_y - min_y


def _candidate_params(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    params = dict(base)
    params.update(override)
    return params


def search_layout_params(
    scenario: Scenario,
    param_grid: dict[str, Sequence[Any]],
    n_trials: int | None = None,
    rng_seed: int | None = None,
    constraints: dict[str, float] | None = None,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> list[LayoutCandidateResult]:
    """Run a brute-force layout search and return ranked candidate metrics."""

    constraints = constraints or {}
    base_kwargs = dict(scenario.layout_kwargs)
    layout_fn = scenario.layout_fn
    n_trials = n_trials or scenario.n_trials
    seed = rng_seed if rng_seed is not None else scenario.rng_seed

    grid_keys = list(param_grid.keys())
    results: list[LayoutCandidateResult] = []

    for idx, values in enumerate(product(*[param_grid[k] for k in grid_keys])):
        overrides = dict(zip(grid_keys, values))
        candidate_kwargs = _candidate_params(base_kwargs, overrides)
        ships = layout_fn(**candidate_kwargs)
        heading_rad = float(candidate_kwargs.get("heading_rad", 0.0))
        ext_along, ext_across = _convoy_frame_extents([ship.position for ship in ships], heading_rad)
        area = ext_along * ext_across

        if "max_footprint_area" in constraints and area > constraints["max_footprint_area"]:
            continue
        if "max_extent_across" in constraints and ext_across > constraints["max_extent_across"]:
            continue
        if "max_extent_along" in constraints and ext_along > constraints["max_extent_along"]:
            continue

        rng = np.random.default_rng(None if seed is None else seed + idx)
        result = run_monte_carlo_attack(
            layout_fn=layout_fn,
            layout_kwargs=candidate_kwargs,
            torpedo_sampler=scenario.torpedo_sampler,
            n_trials=n_trials,
            t_max=scenario.t_max,
            rng=rng,
            noise_model=scenario.noise_model,
        )
        results.append(
            LayoutCandidateResult(
                params=overrides,
                expected_hits=result["expected_hits"],
                p_hit_ge_1=result["hit_prob_at_least_one"],
                var_hits=result["var_hits"],
                footprint_area=area,
                max_extent_along=ext_along,
                max_extent_across=ext_across,
            )
        )

    results.sort(key=lambda r: (r.expected_hits, r.p_hit_ge_1))

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "param_name",
                    "param_value",
                    "expected_hits",
                    "p_hit_ge_1",
                    "var_hits",
                    "footprint_area",
                    "max_extent_along",
                    "max_extent_across",
                    "seed",
                ],
            )
            writer.writeheader()
            for candidate in results:
                for name, value in candidate.params.items():
                    writer.writerow(
                        {
                            "param_name": name,
                            "param_value": value,
                            "expected_hits": candidate.expected_hits,
                            "p_hit_ge_1": candidate.p_hit_ge_1,
                            "var_hits": candidate.var_hits,
                            "footprint_area": candidate.footprint_area,
                            "max_extent_along": candidate.max_extent_along,
                            "max_extent_across": candidate.max_extent_across,
                            "seed": seed,
                        }
                    )

    if output_json and results:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        best = results[0]
        payload = {
            "params": best.params,
            "expected_hits": best.expected_hits,
            "p_hit_ge_1": best.p_hit_ge_1,
            "var_hits": best.var_hits,
            "footprint_area": best.footprint_area,
            "max_extent_along": best.max_extent_along,
            "max_extent_across": best.max_extent_across,
            "seed": seed,
        }
        output_json.write_text(json.dumps(payload, indent=2))

    return results
