"""Smoke test for sensitivity analysis."""

import csv
from pathlib import Path

from experiments.sensitivity_oat import run_sweep
from scenarios.scenario_a import build_scenario_a


def test_sensitivity_oat_smoke(tmp_path: Path) -> None:
    rows = run_sweep(
        parameter="spacing_along",
        values=[500.0, 550.0],
        base_scenario_builder=lambda n_trials, rng_seed: build_scenario_a(n_trials=n_trials, rng_seed=rng_seed),
        n_trials=50,
        seed=10,
    )
    assert len(rows) == 2
    output = tmp_path / "sensitivity.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert output.exists()
