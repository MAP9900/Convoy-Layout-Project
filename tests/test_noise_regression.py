"""Noise regression tests across multiple scenarios."""

import numpy as np

from convoy_sim import NoiseModel, run_monte_carlo_attack
from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_a1_constraints import build_scenario_a1


def _assert_zero_noise_matches_baseline(scenario) -> None:
    baseline = run_monte_carlo_attack(
        layout_fn=scenario.layout_fn,
        layout_kwargs=scenario.layout_kwargs,
        torpedo_sampler=scenario.torpedo_sampler,
        n_trials=scenario.n_trials,
        t_max=scenario.t_max,
        rng=np.random.default_rng(scenario.rng_seed),
        noise_model=None,
    )
    zero_noise = run_monte_carlo_attack(
        layout_fn=scenario.layout_fn,
        layout_kwargs=scenario.layout_kwargs,
        torpedo_sampler=scenario.torpedo_sampler,
        n_trials=scenario.n_trials,
        t_max=scenario.t_max,
        rng=np.random.default_rng(scenario.rng_seed),
        noise_model=NoiseModel(),
    )
    assert np.array_equal(baseline["hits_per_trial"], zero_noise["hits_per_trial"])


def test_zero_noise_matches_baseline_scenario_a() -> None:
    scenario = build_scenario_a(n_trials=25, rng_seed=7)
    _assert_zero_noise_matches_baseline(scenario)


def test_zero_noise_matches_baseline_scenario_a1() -> None:
    scenario = build_scenario_a1(n_trials=25, rng_seed=7)
    _assert_zero_noise_matches_baseline(scenario)
