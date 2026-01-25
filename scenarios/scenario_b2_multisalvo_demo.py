"""Multi-pass attacker plan demo (Scenario B2)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from convoy_sim.attacker_tactics import AttackerPlan, PassSpec, SalvoSpec
from convoy_sim.feasibility import ApproachMode, AttackConstraints, EscortZone, Environment
from convoy_sim.layouts import make_rectangular_convoy


def build_scenario_b2_multisalvo_demo(
    rng_seed: int | None = 1234,
) -> dict[str, Any]:
    ships = make_rectangular_convoy(
        n_rows=2,
        n_cols=3,
        spacing_along=500.0,
        spacing_across=300.0,
        speed=0.0,
        heading_rad=0.0,
        length=120.0,
        beam=20.0,
        origin=np.array([0.0, 0.0]),
    )

    constraints = AttackConstraints(
        escort_zones=[
            EscortZone(center=np.array([0.0, 0.0]), radius=300.0, hard_exclusion=True, risk_weight=1.0)
        ]
    )
    env = Environment(time_of_day="day", visibility_m=8000.0, sea_state=4, detection_risk_scale=1.0)

    plan = AttackerPlan(
        passes=[
            PassSpec(
                launch_time=0.0,
                u_boat_pos=np.array([0.0, 0.0]),
                bearing_rad=0.0,
                approach_mode=ApproachMode.STERN_CHASE,
                salvo=SalvoSpec(
                    n_torpedoes=2,
                    pattern="fan",
                    spread_rad=math.radians(6.0),
                    asymmetry=0.2,
                ),
                abort_if_infeasible=True,
            ),
            PassSpec(
                launch_time=40.0,
                u_boat_pos=np.array([-1800.0, 200.0]),
                bearing_rad=0.0,
                approach_mode=ApproachMode.STERN_CHASE,
                salvo=SalvoSpec(
                    n_torpedoes=4,
                    pattern="fan",
                    spread_rad=math.radians(12.0),
                ),
                abort_if_risk_above=2.0,
            ),
        ],
        name="B2 multi-salvo demo",
    )

    return {
        "name": "Scenario B2 Multi-Salvo Demo",
        "ships": ships,
        "plan": plan,
        "constraints": constraints,
        "env": env,
        "torpedo_params": {"speed": 25.0, "max_run_time": 900.0, "dt": 1.0},
        "t_max_global": 400.0,
        "rng_seed": rng_seed,
    }
