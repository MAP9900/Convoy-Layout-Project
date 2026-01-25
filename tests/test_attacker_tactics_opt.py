"""Tests for attacker tactics plan search."""

import numpy as np

from convoy_sim.attacker_tactics_opt import PlanTemplate, instantiate_plans, search_attacker_plans
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.feasibility import ApproachMode
from convoy_sim.geometry import as_vec


def _make_ships() -> list[Ship]:
    return [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=40.0,
            beam=10.0,
            ship_class=ShipClass.FREIGHTER,
        )
    ]


def _template() -> PlanTemplate:
    return PlanTemplate(
        n_passes_options=[1, 2],
        launch_time_1=0.0,
        u_boat_pos_1=np.array([-100.0, 0.0]),
        bearing_rad_1=0.0,
        approach_mode_1=ApproachMode.STERN_CHASE,
        pattern_1="fan",
        salvo_sizes_1=[1, 2],
        spread_options_1=[0.0],
        asymmetry_options_1=[0.0],
        edge_bias_options_1=[0.0],
        launch_delay_options_2=[10.0],
        u_boat_pos_2=np.array([-100.0, 0.0]),
        bearing_rad_2=0.0,
        approach_mode_2=ApproachMode.STERN_CHASE,
        pattern_2="fan",
        salvo_sizes_2=[1],
        spread_options_2=[0.0],
        asymmetry_options_2=[0.0],
        edge_bias_options_2=[0.0],
        abort_if_risk_above_options=[None],
    )


def test_instantiate_plans_count() -> None:
    plans = instantiate_plans(_template())
    assert len(plans) == 4


def test_search_returns_ranked_results() -> None:
    results = search_attacker_plans(
        ships_t0=_make_ships(),
        template=_template(),
        constraints=None,
        env=None,
        dynamics=None,
        torpedo_params={"speed": 20.0, "max_run_time": 200.0},
        n_trials=3,
        t_max_global=200.0,
        objective=None,
        rng_seed=0,
        top_k=3,
    )
    assert len(results) == 3
    assert "plan" in results[0]
    assert "metrics" in results[0]
    assert "utility" in results[0]


def test_search_is_deterministic_with_seed() -> None:
    results_a = search_attacker_plans(
        ships_t0=_make_ships(),
        template=_template(),
        constraints=None,
        env=None,
        dynamics=None,
        torpedo_params={"speed": 20.0, "max_run_time": 200.0},
        n_trials=5,
        t_max_global=200.0,
        objective=None,
        rng_seed=123,
        top_k=1,
    )
    results_b = search_attacker_plans(
        ships_t0=_make_ships(),
        template=_template(),
        constraints=None,
        env=None,
        dynamics=None,
        torpedo_params={"speed": 20.0, "max_run_time": 200.0},
        n_trials=5,
        t_max_global=200.0,
        objective=None,
        rng_seed=123,
        top_k=1,
    )
    assert results_a[0]["plan"] == results_b[0]["plan"]
