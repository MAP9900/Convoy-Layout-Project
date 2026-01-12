"""Smoke tests for feasibility model scaffolding."""

import numpy as np

from convoy_sim.feasibility import (
    ApproachMode,
    AttackConstraints,
    AttackProposal,
    Environment,
    EscortZone,
)


def test_feasibility_models_construct() -> None:
    env = Environment(time_of_day="day", visibility_m=5000.0, sea_state=3)
    zone = EscortZone(
        center=np.array([0.0, 0.0]),
        radius=2000.0,
        hard_exclusion=True,
        risk_weight=1.0,
    )
    constraints = AttackConstraints(escort_zones=[zone])
    proposal = AttackProposal(
        u_boat_pos=np.array([-1000.0, 0.0]),
        target_point=np.array([0.0, 0.0]),
        bearing_rad=0.0,
        approach_mode=ApproachMode.ABEAM,
        salvo_size=4,
    )
    assert env.visibility_m == 5000.0
    assert constraints.enable_soft_risk is False
    assert proposal.salvo_size == 4
    assert zone.center.shape == (2,)


def test_constraints_optional_for_backwards_compatibility() -> None:
    constraints: AttackConstraints | None = None
    assert constraints is None
