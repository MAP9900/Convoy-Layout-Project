"""Smoke test for defender optimization search."""

from pathlib import Path

from convoy_sim.defender_opt import search_layout_params
from scenarios.scenario_a import build_scenario_a


def test_defender_opt_smoke(tmp_path: Path) -> None:
    scenario = build_scenario_a(n_trials=50, rng_seed=1)
    results = search_layout_params(
        scenario=scenario,
        param_grid={
            "spacing_along": [500.0, 600.0],
            "spacing_across": [300.0],
            "jitter_std": [0.0],
        },
        n_trials=50,
        rng_seed=1,
        output_csv=tmp_path / "defender_opt.csv",
        output_json=tmp_path / "defender_best.json",
    )
    assert len(results) == 2
    assert (tmp_path / "defender_opt.csv").exists()
    assert (tmp_path / "defender_best.json").exists()
