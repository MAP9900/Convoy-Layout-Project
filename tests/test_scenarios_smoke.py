"""Scenario scaffold smoke tests."""

import numpy as np

from scenarios.scenario_a import build_scenario_a


def test_scenario_a_runs_end_to_end() -> None:
    scenario = build_scenario_a(n_trials=10, rng_seed=42)
    result = scenario.run()
    assert result["scenario"] == "Scenario A"
    payload = result["result"]
    assert "hits_per_trial" in payload
    assert len(payload["hits_per_trial"]) == 10
    assert payload["expected_hits"] >= 0.0
