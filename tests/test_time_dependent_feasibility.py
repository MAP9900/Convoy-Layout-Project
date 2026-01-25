"""Tests for time-dependent feasibility checks."""

import math

from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.feasibility import (
    AttackConstraints,
    AttackProposal,
    ApproachMode,
    is_attack_feasible,
)
from convoy_sim.geometry import as_vec


def test_feasibility_changes_with_heading_over_time() -> None:
    ship = Ship(
        id="lead",
        position=as_vec(0.0, 0.0),
        speed=1.0,
        heading_rad=0.0,
        length=10.0,
        beam=4.0,
        ship_class=ShipClass.FREIGHTER,
    )
    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    route = RoutePlan(
        legs=[
            RouteLeg(duration_s=50.0, heading_rad=0.0),
            RouteLeg(duration_s=50.0, heading_rad=math.pi / 2.0),
        ]
    )
    kin = ConvoyKinematics(route=route)
    constraints = AttackConstraints(allowed_modes={ApproachMode.BOW_ON})

    proposal_t1 = AttackProposal(
        u_boat_pos=as_vec(20.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=math.pi,
        approach_mode=ApproachMode.BOW_ON,
        salvo_size=1,
        launch_time=0.0,
    )
    feasible_t1, _ = is_attack_feasible(
        [ship],
        proposal_t1,
        constraints,
        formation=formation,
        kin=kin,
        dt=1.0,
    )

    proposal_t2 = AttackProposal(
        u_boat_pos=as_vec(20.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=math.pi,
        approach_mode=ApproachMode.BOW_ON,
        salvo_size=1,
        launch_time=60.0,
    )
    feasible_t2, _ = is_attack_feasible(
        [ship],
        proposal_t2,
        constraints,
        formation=formation,
        kin=kin,
        dt=1.0,
    )

    assert feasible_t1
    assert not feasible_t2
