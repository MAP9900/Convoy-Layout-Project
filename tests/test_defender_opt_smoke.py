"""Smoke test for defender optimization search."""

from __future__ import annotations

import math
from pathlib import Path

from convoy_sim.attackers import fan_spread
from convoy_sim.defender_opt import search_layout_params
from convoy_sim.geometry import as_vec
from scenarios.convoy_profiles import get_convoy_layout_profile
from scenarios.scenario_base import Scenario


def _build_scenario() -> Scenario:
    profile = get_convoy_layout_profile("small_demo")

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
        name="Defender Opt Smoke",
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        torpedo_sampler=sampler,
        n_trials=50,
        t_max=500.0,
        rng_seed=1,
    )


def test_defender_opt_smoke(tmp_path: Path) -> None:
    scenario = _build_scenario()
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
