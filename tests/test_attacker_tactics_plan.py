"""Tests for attacker tactics multi-pass execution."""

import math

import numpy as np

from convoy_sim.attacker_tactics import (
    AttackerPlan,
    PassSpec,
    SalvoSpec,
    fan_headings_from_salvo,
    parallel_offsets_from_salvo,
    execute_attacker_plan,
)
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.feasibility import ApproachMode, AttackConstraints, EscortZone
from convoy_sim.geometry import as_vec


def _make_ships() -> list[Ship]:
    return [
        Ship(
            id="S1",
            position=as_vec(0.0, 0.0),
            speed=0.0,
            heading_rad=0.0,
            length=50.0,
            beam=10.0,
            ship_class=ShipClass.FREIGHTER,
        )
    ]


def test_plan_executes_passes_in_time_order() -> None:
    plan = AttackerPlan(
        passes=[
            PassSpec(
                launch_time=10.0,
                u_boat_pos=as_vec(-200.0, 0.0),
                bearing_rad=0.0,
                approach_mode=ApproachMode.STERN_CHASE,
                salvo=SalvoSpec(n_torpedoes=1, pattern="fan", spread_rad=0.0),
            ),
            PassSpec(
                launch_time=0.0,
                u_boat_pos=as_vec(-200.0, 0.0),
                bearing_rad=0.0,
                approach_mode=ApproachMode.STERN_CHASE,
                salvo=SalvoSpec(n_torpedoes=1, pattern="fan", spread_rad=0.0),
            ),
        ],
        name="two-pass",
    )
    result = execute_attacker_plan(
        ships_t0=_make_ships(),
        plan=plan,
        constraints=None,
        env=None,
        dynamics=None,
        torpedo_params={"speed": 20.0, "max_run_time": 100.0},
        t_max_global=200.0,
        rng=np.random.default_rng(0),
    )
    assert [p["launch_time"] for p in result["per_pass"]] == [0.0, 10.0]


def test_pass_aborts_when_infeasible() -> None:
    constraints = AttackConstraints(
        escort_zones=[
            EscortZone(center=np.array([0.0, 0.0]), radius=50.0, hard_exclusion=True, risk_weight=1.0)
        ]
    )
    plan = AttackerPlan(
        passes=[
            PassSpec(
                launch_time=0.0,
                u_boat_pos=as_vec(0.0, 0.0),
                bearing_rad=0.0,
                approach_mode=ApproachMode.STERN_CHASE,
                salvo=SalvoSpec(n_torpedoes=2, pattern="fan", spread_rad=0.1),
            )
        ]
    )
    result = execute_attacker_plan(
        ships_t0=_make_ships(),
        plan=plan,
        constraints=constraints,
        env=None,
        dynamics=None,
        torpedo_params={"speed": 20.0, "max_run_time": 100.0},
        t_max_global=200.0,
    )
    assert result["per_pass"][0]["status"] == "aborted_infeasible"
    assert result["per_pass"][0]["n_torpedoes_fired"] == 0


def test_asymmetry_shifts_headings_and_offsets() -> None:
    base = 0.0
    fan = SalvoSpec(n_torpedoes=5, pattern="fan", spread_rad=math.pi / 2.0, asymmetry=0.5)
    headings = fan_headings_from_salvo(base, fan)
    assert np.mean(headings) > base

    parallel = SalvoSpec(n_torpedoes=5, pattern="parallel", lateral_spacing=10.0, asymmetry=0.5)
    offsets = parallel_offsets_from_salvo(parallel)
    assert np.mean(offsets) > 0.0
