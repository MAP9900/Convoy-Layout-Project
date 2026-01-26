"""Scenario scaffold smoke tests."""

import numpy as np

from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_base import Scenario


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
