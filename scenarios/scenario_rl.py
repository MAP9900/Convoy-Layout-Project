"""RL convoy layout scaffold scenario."""

from __future__ import annotations

import math
import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.geometry import as_vec
from .convoy_profiles import get_convoy_layout_profile
from .scenario_base import Scenario


def build_scenario_rl(n_trials: int = 200, rng_seed: int | None = 1234) -> Scenario:
    """Return a scaffold scenario that uses the RL convoy layout profile."""

    profile = get_convoy_layout_profile("rl_large")

    # TODO(RL_SCENARIO): tune attacker settings for RL evaluation runs.
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
        name="Scenario RL",
        layout_fn=profile.layout_fn,
        layout_kwargs=profile.layout_kwargs,
        torpedo_sampler=sampler,
        n_trials=n_trials,
        t_max=400.0,
        rng_seed=rng_seed,
        metadata={
            "description": "RL convoy layout scaffold using profile 'rl_large'",
            "layout_profile": profile.name,
            "enable_value_scoring": False,
        },
    )

