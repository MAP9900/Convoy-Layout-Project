"""Alternating defender/attacker best-response loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from convoy_sim.attacker_opt import search_attack_params
from convoy_sim.defender_opt import LayoutCandidateResult, search_layout_params
from convoy_sim.entities import Torpedo
from convoy_sim.noise import NoiseModel
from convoy_sim.attackers import fan_spread, parallel_spread
from scenarios.scenario_base import Scenario


@dataclass
class MinMaxRoundResult:
    """Container for per-round minmax results."""

    round_index: int
    defense_params: dict[str, Any]
    attack_params: dict[str, Any]
    expected_hits: float
    p_hit_ge_1: float
    var_hits: float
    footprint_area: float
    max_extent_along: float
    max_extent_across: float


def _apply_launch_delay_mean(torpedoes: Sequence[Torpedo], delay_mean: float) -> list[Torpedo]:
    if delay_mean <= 0.0:
        return list(torpedoes)
    adjusted = []
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


def _attack_sampler_factory(attack_state: dict[str, Any]):
    mode = attack_state.get("mode", "fan")
    origin = attack_state["torpedo_origin"]
    speed = attack_state["torpedo_speed"]
    max_run_time = attack_state["torpedo_max_run_time"]
    base_bearing = attack_state.get("base_bearing_rad", 0.0)
    n = int(attack_state.get("n", 1))
    spread_rad = float(attack_state.get("spread_rad", 0.0))
    lateral_spacing = float(attack_state.get("lateral_spacing", 0.0))
    delay_mean = float(attack_state.get("launch_delay_mean", 0.0))

    def sampler(_: np.random.Generator):
        if mode == "parallel":
            torps = parallel_spread(
                u_pos=origin,
                bearing_rad=base_bearing,
                n=n,
                lateral_spacing=lateral_spacing,
                speed=speed,
                max_run_time=max_run_time,
            )
        else:
            torps = fan_spread(
                u_pos=origin,
                base_bearing_rad=base_bearing,
                n=n,
                spread_rad=spread_rad,
                speed=speed,
                max_run_time=max_run_time,
            )
        return _apply_launch_delay_mean(torps, delay_mean)

    return sampler


def run_minmax_loop(
    initial_defense: dict[str, Any],
    initial_attack: dict[str, Any],
    n_rounds: int,
    rng_seed: int | None = None,
    budgets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run alternating defender/attacker best-response rounds.

    budgets keys:
      - defense_grid: dict[str, Sequence[Any]]
      - attack_grid: dict[str, Sequence[Any]]
      - constraints: dict[str, float]
      - n_trials: int
      - epsilon: float
      - patience: int
    """

    budgets = budgets or {}
    defense_grid = budgets.get("defense_grid", {})
    attack_grid = budgets.get("attack_grid", {})
    constraints = budgets.get("constraints")
    n_trials = budgets.get("n_trials")
    epsilon = float(budgets.get("epsilon", 0.0))
    patience = int(budgets.get("patience", 1))

    layout_fn = initial_defense["layout_fn"]
    layout_kwargs = dict(initial_defense["layout_kwargs"])
    t_max = float(initial_defense["t_max"])
    noise_model: NoiseModel | None = initial_defense.get("noise_model")

    attack_state = dict(initial_attack)

    history: list[MinMaxRoundResult] = []
    best_expected = -math.inf
    no_improve = 0

    for round_idx in range(n_rounds):
        sampler = _attack_sampler_factory(attack_state)
        scenario = Scenario(
            name=f"minmax_round_{round_idx}",
            layout_fn=layout_fn,
            layout_kwargs=layout_kwargs,
            torpedo_sampler=sampler,
            n_trials=n_trials or initial_defense.get("n_trials", 100),
            t_max=t_max,
            rng_seed=rng_seed,
            noise_model=noise_model,
        )

        defense_results = search_layout_params(
            scenario=scenario,
            param_grid=defense_grid,
            n_trials=n_trials,
            rng_seed=rng_seed,
            constraints=constraints,
        )
        if not defense_results:
            break
        best_defense: LayoutCandidateResult = defense_results[0]
        layout_kwargs.update(best_defense.params)

        attack_results = search_attack_params(
            layout_fn=layout_fn,
            layout_kwargs=layout_kwargs,
            param_grid=attack_grid,
            torpedo_origin=attack_state["torpedo_origin"],
            torpedo_speed=attack_state["torpedo_speed"],
            torpedo_max_run_time=attack_state["torpedo_max_run_time"],
            t_max=t_max,
            n_trials=n_trials or initial_defense.get("n_trials", 100),
            rng_seed=rng_seed,
            mode=attack_state.get("mode", "fan"),
            convoy_heading_rad=layout_kwargs.get("heading_rad", 0.0),
        )
        if not attack_results:
            break
        best_attack = attack_results[0]
        attack_state.update(best_attack.params)

        history.append(
            MinMaxRoundResult(
                round_index=round_idx,
                defense_params=dict(best_defense.params),
                attack_params=dict(best_attack.params),
                expected_hits=best_attack.expected_hits,
                p_hit_ge_1=best_attack.p_hit_ge_1,
                var_hits=best_attack.var_hits,
                footprint_area=best_defense.footprint_area,
                max_extent_along=best_defense.max_extent_along,
                max_extent_across=best_defense.max_extent_across,
            )
        )

        if best_attack.expected_hits > best_expected + epsilon:
            best_expected = best_attack.expected_hits
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return {
        "history": [result.__dict__ for result in history],
        "rounds_completed": len(history),
    }
