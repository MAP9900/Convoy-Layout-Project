"""Tests for constrained sampler behavior."""

import math

import numpy as np
import pytest

from convoy_sim.attackers import fan_spread
from convoy_sim.entities import Ship
from convoy_sim.feasibility import ApproachMode, AttackConstraints, Environment, EscortZone
from convoy_sim.geometry import as_vec


def _simple_ships() -> list[Ship]:
    return [
        Ship(id="S1", position=as_vec(0.0, 0.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
        Ship(id="S2", position=as_vec(200.0, 0.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
    ]


def test_constraints_impossible_raises() -> None:
    ships = _simple_ships()
    constraints = AttackConstraints(
        min_range=0.0,
        max_range=2000.0,
        allowed_modes={ApproachMode.ABEAM},
        escort_zones=[EscortZone(center=as_vec(-1000.0, 0.0), radius=500.0, hard_exclusion=True, risk_weight=1.0)],
    )
    cfg = {
        "u_boat_pos": as_vec(-1000.0, 0.0),
        "target_point": "centroid",
        "approach_mode": ApproachMode.ABEAM,
        "salvo_size": 3,
        "bearing_rad": 0.0,
    }
    with pytest.raises(ValueError):
        fan_spread(
            u_pos=as_vec(-1000.0, 0.0),
            base_bearing_rad=0.0,
            n=3,
            spread_rad=math.radians(10.0),
            speed=20.0,
            max_run_time=600.0,
            ships=ships,
            proposal_cfg=cfg,
            constraints=constraints,
            env=Environment(time_of_day="day", visibility_m=5000.0, sea_state=3),
            rng=np.random.default_rng(1),
            max_resample_attempts=5,
        )


def test_constraints_allow_torpedoes() -> None:
    ships = _simple_ships()
    constraints = AttackConstraints(
        min_range=0.0,
        max_range=4000.0,
        allowed_modes={ApproachMode.ABEAM},
        escort_zones=[],
    )
    cfg = {
        "u_boat_box": (-3000.0, -2000.0, -500.0, 500.0),
        "target_point": "centroid",
        "approach_mode": ApproachMode.ABEAM,
        "salvo_size": 2,
        "bearing_offset_rad": math.pi / 2.0,
    }
    torps = fan_spread(
        u_pos=as_vec(-2500.0, 0.0),
        base_bearing_rad=0.0,
        n=2,
        spread_rad=math.radians(5.0),
        speed=20.0,
        max_run_time=600.0,
        ships=ships,
        proposal_cfg=cfg,
        constraints=constraints,
        env=None,
        rng=np.random.default_rng(4),
        max_resample_attempts=10,
    )
    assert len(torps) > 0
