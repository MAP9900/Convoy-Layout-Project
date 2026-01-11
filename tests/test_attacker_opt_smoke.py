"""Smoke test for attacker optimization search."""

import numpy as np

from convoy_sim.attacker_opt import search_attack_params
from scenarios.scenario_a import build_scenario_a


def test_attacker_opt_deterministic() -> None:
    scenario = build_scenario_a(n_trials=50, rng_seed=1)
    grid = {
        "base_bearing_rad": [0.0, 0.1],
        "spread_rad": [0.0, 0.2],
        "n": [2],
    }
    results_a = search_attack_params(
        layout_fn=scenario.layout_fn,
        layout_kwargs=scenario.layout_kwargs,
        param_grid=grid,
        torpedo_origin=(-1000.0, 0.0),
        torpedo_speed=20.0,
        torpedo_max_run_time=500.0,
        t_max=scenario.t_max,
        n_trials=50,
        rng_seed=10,
        mode="fan",
    )
    results_b = search_attack_params(
        layout_fn=scenario.layout_fn,
        layout_kwargs=scenario.layout_kwargs,
        param_grid=grid,
        torpedo_origin=(-1000.0, 0.0),
        torpedo_speed=20.0,
        torpedo_max_run_time=500.0,
        t_max=scenario.t_max,
        n_trials=50,
        rng_seed=10,
        mode="fan",
    )
    assert len(results_a) == 4
    assert [r.expected_hits for r in results_a] == [r.expected_hits for r in results_b]
    assert [r.p_hit_ge_1 for r in results_a] == [r.p_hit_ge_1 for r in results_b]
    assert results_a[0].expected_hits >= results_a[-1].expected_hits
