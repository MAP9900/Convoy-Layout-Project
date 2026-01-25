"""Tests for defender policy optimization helpers."""

import math

import numpy as np

from convoy_sim.defender_policy import LayoutAction, ThreatPrior, ThreatType
from convoy_sim.defender_policy_opt import (
    PolicyObjective,
    optimize_policy_deterministic,
    optimize_policy_mixture_pairwise,
)
from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec


def _make_ships(rect_x: float, rect_y: float, y_offset: float = 0.0) -> list[Ship]:
    positions = [
        as_vec(0.0, 0.0 + y_offset),
        as_vec(rect_x, 0.0 + y_offset),
        as_vec(0.0, rect_y + y_offset),
        as_vec(rect_x, rect_y + y_offset),
    ]
    ships = []
    for idx, pos in enumerate(positions, start=1):
        ships.append(
            Ship(
                id=f"S{idx}",
                position=pos,
                speed=0.0,
                heading_rad=0.0,
                length=40.0,
                beam=10.0,
                ship_class=ShipClass.FREIGHTER,
            )
        )
    return ships


def _layout_wide() -> list[Ship]:
    return _make_ships(1000.0, 1000.0, y_offset=500.0)


def _layout_tight() -> list[Ship]:
    return _make_ships(100.0, 100.0)


def _attacker_factory(_: ThreatType):
    def sampler(_: np.random.Generator):
        return [
            Torpedo(
                id="T1",
                launch_position=as_vec(-200.0, 0.0),
                speed=20.0,
                heading_rad=0.0,
                max_run_time=200.0,
            )
        ]

    return sampler


def test_deterministic_optimizer_selects_best_action() -> None:
    actions = [
        LayoutAction(
            name="wide",
            layout_fn=_layout_wide,
            layout_kwargs={},
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="tight",
            layout_fn=_layout_tight,
            layout_kwargs={},
            complexity_cost=1.0,
        ),
    ]
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    policy, result = optimize_policy_deterministic(
        prior=prior,
        actions=actions,
        threats=[ThreatType.ABEAM_FAN],
        attacker_factory=_attacker_factory,
        n_trials=5,
        sim_kwargs={"t_max": 200.0},
        objective_spec=None,
        policy_obj=PolicyObjective(),
        rng_seed=0,
    )
    dist = policy.action_distribution(ThreatType.ABEAM_FAN)
    assert math.isclose(sum(dist.values()), 1.0)
    assert max(dist, key=dist.get) == "wide"


def test_mixture_optimizer_prefers_budget_feasible_mix() -> None:
    actions = [
        LayoutAction(
            name="wide",
            layout_fn=_layout_wide,
            layout_kwargs={},
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="tight",
            layout_fn=_layout_tight,
            layout_kwargs={},
            complexity_cost=1.0,
        ),
    ]
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    policy, _result = optimize_policy_mixture_pairwise(
        prior=prior,
        actions=actions,
        threats=[ThreatType.ABEAM_FAN],
        attacker_factory=_attacker_factory,
        n_trials=200,
        sim_kwargs={"t_max": 200.0},
        objective_spec=None,
        policy_obj=PolicyObjective(footprint_budget=300000.0),
        rng_seed=1,
        mix_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    dist = policy.action_distribution(ThreatType.ABEAM_FAN)
    assert math.isclose(sum(dist.values()), 1.0)
    assert dist["wide"] not in (0.0, 1.0)
