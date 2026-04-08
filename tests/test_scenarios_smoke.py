"""Scenario scaffold smoke tests for the current profile registry."""

from __future__ import annotations

import math

from convoy_sim.attackers import fan_spread
from convoy_sim.geometry import as_vec
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles
from scenarios.scenario_base import Scenario


def _build_smoke_scenario(*, profile_name: str, n_trials: int, rng_seed: int) -> Scenario:
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
        name=f"Scenario Smoke: {profile_name}",
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        torpedo_sampler=sampler,
        n_trials=n_trials,
        t_max=500.0,
        rng_seed=rng_seed,
        metadata={"profile_name": profile_name},
    )


def test_profile_registry_contains_current_named_profiles() -> None:
    profile_names = set(list_convoy_layout_profiles())
    assert {"small_demo", "convoy_layout_1", "convoy_layout_2"} <= profile_names


def test_scenario_runs_end_to_end_for_current_profile() -> None:
    scenario = _build_smoke_scenario(profile_name="small_demo", n_trials=10, rng_seed=42)
    result = scenario.run()
    assert result["scenario"] == "Scenario Smoke: small_demo"
    payload = result["result"]
    assert "hits_per_trial" in payload
    assert len(payload["hits_per_trial"]) == 10
    assert payload["expected_hits"] >= 0.0


def test_scenario_serialization_roundtrip() -> None:
    scenario = _build_smoke_scenario(profile_name="convoy_layout_1", n_trials=5, rng_seed=11)
    payload = scenario.to_dict()
    restored = Scenario.from_dict(
        payload,
        layout_fn=scenario.layout_fn,
        torpedo_sampler=scenario.torpedo_sampler,
    )
    assert restored.name == scenario.name
    assert restored.layout_kwargs == scenario.layout_kwargs
    assert restored.t_max == scenario.t_max
