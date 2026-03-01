"""Scenario A1 with feasibility constraints enabled."""

from __future__ import annotations

import math
import numpy as np

from convoy_sim import as_vec, make_rectangular_convoy
from convoy_sim.attackers import fan_spread
from convoy_sim.feasibility import (
    ApproachMode,
    AttackConstraints,
    AttackProposal,
    Environment,
    EscortZone,
)
from scenarios.scenario_base import Scenario


def build_scenario_a1(n_trials: int = 200, rng_seed: int | None = 1234) -> Scenario:
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

    constraints = AttackConstraints(
        min_range=800.0,
        max_range=4000.0,
        allowed_modes={ApproachMode.ABEAM, ApproachMode.BOW_ON},
        escort_zones=[
            EscortZone(center=as_vec(0.0, 0.0), radius=1500.0, hard_exclusion=True, risk_weight=1.0),
            EscortZone(center=as_vec(1200.0, 0.0), radius=800.0, hard_exclusion=True, risk_weight=0.5),
        ],
        enable_soft_risk=False,
        max_allowed_risk=float("inf"),
        notes="Baseline constraints for demonstration",
    )
    env = Environment(time_of_day="day", visibility_m=6000.0, sea_state=3)

    proposal_cfg = {
        "u_boat_box": (-4000.0, -1500.0, -1500.0, 1500.0),
        "target_point": "centroid",
        "approach_mode": ApproachMode.ABEAM,
        "salvo_size": 4,
        "bearing_offset_rad": math.pi / 2.0,
    }

    def sampler(rng: np.random.Generator):
        ships = make_rectangular_convoy(**layout_kwargs)
        return fan_spread(
            u_pos=as_vec(-2000.0, 0.0),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=math.radians(15.0),
            speed=25.0,
            max_run_time=800.0,
            ships=ships,
            proposal_cfg=proposal_cfg,
            constraints=constraints,
            env=env,
            rng=rng,
        )

    return Scenario(
        name="Scenario A1",
        layout_fn=make_rectangular_convoy,
        layout_kwargs=layout_kwargs,
        torpedo_sampler=sampler,
        n_trials=n_trials,
        t_max=400.0,
        rng_seed=rng_seed,
        metadata={
            "description": "Scenario A with feasibility constraints",
            "enable_value_scoring": True,
        },
    )
