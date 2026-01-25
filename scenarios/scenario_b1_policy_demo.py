"""Policy demo scenario for defender layout selection under threat priors."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.defender_policy import DefenderPolicy, LayoutAction, ThreatPrior, ThreatType
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy


def build_scenario_b1_policy_demo(
    n_trials: int = 200,
    rng_seed: int | None = 1234,
) -> dict[str, Any]:
    """Return a configuration dict for the B1 policy demo."""

    actions = [
        LayoutAction(
            name="rect_tight",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 3,
                "n_cols": 4,
                "spacing_along": 500.0,
                "spacing_across": 300.0,
                "speed": 6.0,
                "heading_rad": 0.0,
                "length": 140.0,
                "beam": 18.0,
                "origin": np.array([0.0, 0.0]),
            },
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="rect_wide",
            layout_fn=make_rectangular_convoy,
            layout_kwargs={
                "n_rows": 3,
                "n_cols": 4,
                "spacing_along": 700.0,
                "spacing_across": 500.0,
                "speed": 6.0,
                "heading_rad": 0.0,
                "length": 140.0,
                "beam": 18.0,
                "origin": np.array([0.0, 0.0]),
            },
            complexity_cost=1.5,
        ),
        LayoutAction(
            name="staggered",
            layout_fn=make_staggered_convoy,
            layout_kwargs={
                "n_rows": 3,
                "n_cols": 4,
                "spacing_along": 600.0,
                "spacing_across": 350.0,
                "speed": 6.0,
                "heading_rad": 0.0,
                "length": 140.0,
                "beam": 18.0,
                "origin": np.array([0.0, 0.0]),
            },
            complexity_cost=2.0,
        ),
    ]

    prior = ThreatPrior(
        probs={
            ThreatType.ABEAM_FAN: 0.7,
            ThreatType.BOW_ON_FAN: 0.3,
        }
    )

    policy = DefenderPolicy(
        actions=actions,
        policy_table={
            ThreatType.ABEAM_FAN: {
                "rect_tight": 0.7,
                "staggered": 0.3,
            },
            ThreatType.BOW_ON_FAN: {
                "rect_wide": 1.0,
            },
        },
    )

    def attacker_factory(threat: ThreatType) -> Callable[[np.random.Generator], list]:
        def sampler(_: np.random.Generator):
            if threat == ThreatType.PARALLEL_SPREAD:
                return parallel_spread(
                    u_pos=np.array([-2200.0, 0.0]),
                    bearing_rad=0.0,
                    n=3,
                    lateral_spacing=120.0,
                    speed=25.0,
                    max_run_time=900.0,
                )
            bearing = math.pi / 2.0 if threat == ThreatType.ABEAM_FAN else 0.0
            return fan_spread(
                u_pos=np.array([-2200.0, 0.0]),
                base_bearing_rad=bearing,
                n=3,
                spread_rad=math.radians(12.0),
                speed=25.0,
                max_run_time=900.0,
            )

        return sampler

    return {
        "name": "Scenario B1 Policy Demo",
        "prior": prior,
        "policy": policy,
        "attacker_factory": attacker_factory,
        "n_trials": n_trials,
        "sim_kwargs": {"t_max": 400.0},
        "rng_seed": rng_seed,
    }
