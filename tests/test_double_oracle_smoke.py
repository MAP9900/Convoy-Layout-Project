"""Smoke tests for double-oracle loop."""

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import LayoutAction
from convoy_sim.double_oracle import double_oracle_loop
from convoy_sim.game import AttackerStrategy, DefenderStrategy
from convoy_sim.layouts import make_rectangular_convoy


def test_double_oracle_adds_strategies() -> None:
    defenders = [
        DefenderStrategy(
            name="rect",
            kind="layout_action",
            payload=LayoutAction(
                name="rect",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 1,
                    "n_cols": 1,
                    "spacing_along": 100.0,
                    "spacing_across": 100.0,
                    "speed": 0.0,
                    "heading_rad": 0.0,
                    "length": 40.0,
                    "beam": 10.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        )
    ]
    attackers = [
        AttackerStrategy(
            name="fan1",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=1,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )
    ]

    def br_defender(_: AttackerStrategy) -> DefenderStrategy:
        return DefenderStrategy(
            name="rect_wide",
            kind="layout_action",
            payload=LayoutAction(
                name="rect_wide",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 1,
                    "n_cols": 1,
                    "spacing_along": 300.0,
                    "spacing_across": 300.0,
                    "speed": 0.0,
                    "heading_rad": 0.0,
                    "length": 40.0,
                    "beam": 10.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        )

    def br_attacker(_: DefenderStrategy) -> AttackerStrategy:
        return AttackerStrategy(
            name="fan2",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=2,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )

    result = double_oracle_loop(
        initial_defenders=defenders,
        initial_attackers=attackers,
        br_defender_generator=br_defender,
        br_attacker_generator=br_attacker,
        eval_config={
            "n_trials": 2,
            "sim_params": {"t_max": 200.0},
            "nash_iters": 20,
            "epsilon": -1.0,
        },
        n_outer_iters=2,
    )
    assert "rect_wide" in result["defenders"]
    assert "fan2" in result["attackers"]
