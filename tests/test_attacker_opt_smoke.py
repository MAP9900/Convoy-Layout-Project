"""Smoke test for attacker optimization search."""

from __future__ import annotations

from convoy_sim.attacker_opt import search_attack_params
from scenarios.convoy_profiles import get_convoy_layout_profile


def test_attacker_opt_deterministic() -> None:
    profile = get_convoy_layout_profile("small_demo")
    grid = {
        "base_bearing_rad": [0.0, 0.1],
        "spread_rad": [0.0, 0.2],
        "n": [2],
    }
    results_a = search_attack_params(
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        param_grid=grid,
        torpedo_origin=(-1000.0, 0.0),
        torpedo_speed=20.0,
        torpedo_max_run_time=500.0,
        t_max=500.0,
        n_trials=50,
        rng_seed=10,
        mode="fan",
    )
    results_b = search_attack_params(
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        param_grid=grid,
        torpedo_origin=(-1000.0, 0.0),
        torpedo_speed=20.0,
        torpedo_max_run_time=500.0,
        t_max=500.0,
        n_trials=50,
        rng_seed=10,
        mode="fan",
    )
    assert len(results_a) == 4
    assert [r.expected_hits for r in results_a] == [r.expected_hits for r in results_b]
    assert [r.p_hit_ge_1 for r in results_a] == [r.p_hit_ge_1 for r in results_b]
    assert results_a[0].expected_hits >= results_a[-1].expected_hits
