"""Smoke test for minmax loop."""

from convoy_sim.minmax_loop import run_minmax_loop
from scenarios.scenario_a import build_scenario_a


def test_minmax_loop_deterministic() -> None:
    scenario = build_scenario_a(n_trials=50, rng_seed=2)
    initial_defense = {
        "layout_fn": scenario.layout_fn,
        "layout_kwargs": dict(scenario.layout_kwargs),
        "t_max": scenario.t_max,
        "n_trials": scenario.n_trials,
        "noise_model": scenario.noise_model,
    }
    initial_attack = {
        "mode": "fan",
        "torpedo_origin": (-1500.0, 0.0),
        "torpedo_speed": 20.0,
        "torpedo_max_run_time": 600.0,
        "base_bearing_rad": 0.0,
        "spread_rad": 0.2,
        "n": 2,
        "launch_delay_mean": 0.0,
    }
    budgets = {
        "defense_grid": {
            "spacing_along": [500.0],
            "spacing_across": [300.0],
            "jitter_std": [0.0],
        },
        "attack_grid": {
            "base_bearing_rad": [0.0, 0.1],
            "spread_rad": [0.0, 0.2],
            "n": [2],
            "launch_delay_mean": [0.0],
        },
        "n_trials": 50,
        "epsilon": 0.0,
        "patience": 1,
    }
    result_a = run_minmax_loop(
        initial_defense=initial_defense,
        initial_attack=initial_attack,
        n_rounds=2,
        rng_seed=5,
        budgets=budgets,
    )
    result_b = run_minmax_loop(
        initial_defense=initial_defense,
        initial_attack=initial_attack,
        n_rounds=2,
        rng_seed=5,
        budgets=budgets,
    )
    assert result_a["rounds_completed"] >= 1
    assert result_a == result_b
