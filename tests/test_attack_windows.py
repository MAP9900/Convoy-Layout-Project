"""Tests for time-aware attack windows and feasibility checks."""

from dataclasses import replace
import math

from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.feasibility import (
    AttackConstraints,
    ApproachMode,
    AttackProposal,
    approach_mode_feasible,
)
from convoy_sim.geometry import as_vec
from convoy_sim.simulation import torpedo_hits_ship_dynamic


def test_attack_window_after_turn() -> None:
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

    proposal = AttackProposal(
        u_boat_pos=as_vec(50.0, 10.0),
        target_point=as_vec(50.0, 10.0),
        bearing_rad=math.pi / 2.0,
        approach_mode=ApproachMode.STERN_CHASE,
        salvo_size=1,
    )

    constraints = AttackConstraints()
    heading_t0 = kin.convoy_heading_at(0.0, formation.convoy_heading0)
    heading_t60 = kin.convoy_heading_at(60.0, formation.convoy_heading0)
    assert not approach_mode_feasible(proposal, heading_t0, constraints)
    assert approach_mode_feasible(proposal, heading_t60, constraints)

    hits_now = torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=replace(proposal, launch_time=0.0),
        torpedo_speed=5.0,
        torpedo_max_run_time=5.0,
        spread_rad=0.0,
        t_max_global=120.0,
        dt=1.0,
    )
    hits_wait = torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=replace(proposal, launch_time=60.0),
        torpedo_speed=5.0,
        torpedo_max_run_time=5.0,
        spread_rad=0.0,
        t_max_global=120.0,
        dt=1.0,
    )
    assert hits_now == 0
    assert hits_wait >= 1
