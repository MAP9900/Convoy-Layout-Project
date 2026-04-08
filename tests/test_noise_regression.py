"""Noise regression tests across current scenario/profile configurations."""

from __future__ import annotations

import math

import numpy as np

from convoy_sim import NoiseModel, run_monte_carlo_attack
from convoy_sim.attackers import fan_spread
from convoy_sim.geometry import as_vec
from scenarios.convoy_profiles import get_convoy_layout_profile
from scenarios.scenario_base import Scenario


def _build_noise_scenario(*, profile_name: str) -> Scenario:
    profile = get_convoy_layout_profile(profile_name)

    def sampler(rng):
        return fan_spread(
            u_pos=as_vec(-2000.0, 0.0),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=math.radians(5.0),
            speed=15.0,
            max_run_time=500.0,
        )

    return Scenario(
        name=f"Noise Regression: {profile_name}",
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        torpedo_sampler=sampler,
        n_trials=25,
        t_max=500.0,
        rng_seed=7,
        metadata={"profile_name": profile_name},
    )


def _assert_zero_noise_matches_baseline(scenario: Scenario) -> None:
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


def test_zero_noise_matches_baseline_small_demo() -> None:
    scenario = _build_noise_scenario(profile_name="small_demo")
    _assert_zero_noise_matches_baseline(scenario)


def test_zero_noise_matches_baseline_convoy_layout_1() -> None:
    scenario = _build_noise_scenario(profile_name="convoy_layout_1")
    _assert_zero_noise_matches_baseline(scenario)
