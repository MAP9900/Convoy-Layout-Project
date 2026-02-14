"""Scenario scaffold smoke tests."""

import numpy as np

from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_rl import build_scenario_rl
from scenarios.convoy_profiles import get_convoy_layout_profile
from scenarios.scenario_base import Scenario
from convoy_sim.entities import ShipClass


def test_scenario_a_runs_end_to_end() -> None:
    scenario = build_scenario_a(n_trials=10, rng_seed=42)
    result = scenario.run()
    assert result["scenario"] == "Scenario A"
    payload = result["result"]
    assert "hits_per_trial" in payload
    assert len(payload["hits_per_trial"]) == 10
    assert payload["expected_hits"] >= 0.0


def test_scenario_serialization_roundtrip() -> None:
    scenario = build_scenario_a(n_trials=5, rng_seed=11)
    payload = scenario.to_dict()
    restored = Scenario.from_dict(
        payload,
        layout_fn=scenario.layout_fn,
        torpedo_sampler=scenario.torpedo_sampler,
    )
    assert restored.name == scenario.name
    assert restored.layout_kwargs == scenario.layout_kwargs
    assert restored.t_max == scenario.t_max


def test_rl_profile_scaffold_builds_heterogeneous_ships() -> None:
    profile = get_convoy_layout_profile("rl_large")
    ships = profile.build_ships()
    classes = {ship.ship_class for ship in ships}
    assert ShipClass.FREIGHTER in classes
    assert ShipClass.TANKER in classes
    assert ShipClass.ESCORT in classes


def test_scenario_rl_runs_end_to_end() -> None:
    scenario = build_scenario_rl(n_trials=5, rng_seed=7)
    result = scenario.run()
    assert result["scenario"] == "Scenario RL"
    payload = result["result"]
    assert len(payload["hits_per_trial"]) == 5
