"""Baseline scenario configuration (Scenario A)."""

from __future__ import annotations

import math
import numpy as np

from convoy_sim import as_vec, make_rectangular_convoy
from convoy_sim.attackers import fan_spread
from scenarios.scenario_base import Scenario


def build_scenario_a(n_trials: int = 200, rng_seed: int | None = 1234) -> Scenario:
    layout_kwargs = dict(
        n_rows=3,
        n_cols=4,
        spacing_along=600.0,
        spacing_across=350.0,
        speed=5.0,
        heading_rad=0.0,
        length=150.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
    )

    def sampler(_: np.random.Generator):
        return fan_spread(
            u_pos=as_vec(-2000.0, 0.0),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=math.radians(15.0),
            speed=25.0,
            max_run_time=800.0,
        )

    return Scenario(
        name="Scenario A",
        layout_fn=make_rectangular_convoy,
        layout_kwargs=layout_kwargs,
        torpedo_sampler=sampler,
        n_trials=n_trials,
        t_max=400.0,
        rng_seed=rng_seed,
        metadata={
            "description": "Baseline rectangular convoy vs straight torpedo fan spread",
            "enable_value_scoring": False,
        },
    )
