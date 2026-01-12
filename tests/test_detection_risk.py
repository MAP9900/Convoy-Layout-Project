"""Tests for detection risk scoring."""

import numpy as np

from convoy_sim.feasibility import (
    AttackProposal,
    ApproachMode,
    Environment,
    EscortZone,
    detection_risk_score,
)


def _proposal_at(x: float, y: float) -> AttackProposal:
    return AttackProposal(
        u_boat_pos=np.array([x, y], dtype=float),
        target_point=np.array([0.0, 0.0], dtype=float),
        bearing_rad=0.0,
        approach_mode=ApproachMode.ABEAM,
        salvo_size=4,
    )


def test_risk_higher_day_high_visibility() -> None:
    proposal = _proposal_at(5000.0, 0.0)
    zones = []
    day_env = Environment(time_of_day="day", visibility_m=8000.0, sea_state=3)
    night_env = Environment(time_of_day="night", visibility_m=2000.0, sea_state=3)
    assert detection_risk_score(proposal, zones, day_env) > detection_risk_score(proposal, zones, night_env)


def test_risk_increases_near_escort_zone() -> None:
    zone = EscortZone(center=np.array([0.0, 0.0]), radius=1000.0, hard_exclusion=False, risk_weight=1.0)
    env = Environment(time_of_day="day", visibility_m=5000.0, sea_state=3)
    near = _proposal_at(500.0, 0.0)
    far = _proposal_at(5000.0, 0.0)
    assert detection_risk_score(near, [zone], env) > detection_risk_score(far, [zone], env)
