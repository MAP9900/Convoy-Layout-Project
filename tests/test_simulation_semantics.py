"""Tests for static vs dynamic simulation semantics."""

import numpy as np

from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics
from convoy_sim.entities import Ship, ShipClass, Torpedo, torpedo_hits_ship
from convoy_sim.feasibility import ApproachMode, AttackProposal
from convoy_sim.geometry import as_vec
from convoy_sim.simulation import torpedo_hits_ship_dynamic


def _ship_at(pos: np.ndarray) -> Ship:
    return Ship(
        id="S1",
        position=pos,
        speed=0.0,
        heading_rad=0.0,
        length=40.0,
        beam=10.0,
        ship_class=ShipClass.FREIGHTER,
    )


def test_static_dynamic_hit_equivalence_direct_hit() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-1000.0, 0.0),
        speed=20.0,
        heading_rad=0.0,
        max_run_time=200.0,
    )
    static_hit = torpedo_hits_ship(ship, torpedo, t_max=200.0)

    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    kin = ConvoyKinematics()
    proposal = AttackProposal(
        u_boat_pos=as_vec(-1000.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=0.0,
        approach_mode=ApproachMode.STERN_CHASE,
        salvo_size=1,
        launch_time=0.0,
    )
    dynamic_hits = torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=proposal,
        torpedo_speed=20.0,
        torpedo_max_run_time=200.0,
        t_max_global=200.0,
        dt=1.0,
    )
    assert static_hit
    assert dynamic_hits == 1


def test_static_dynamic_miss_equivalence() -> None:
    ship = _ship_at(as_vec(0.0, 500.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-1000.0, 0.0),
        speed=20.0,
        heading_rad=0.0,
        max_run_time=200.0,
    )
    static_hit = torpedo_hits_ship(ship, torpedo, t_max=200.0)

    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 500.0),
        convoy_heading0=0.0,
    )
    kin = ConvoyKinematics()
    proposal = AttackProposal(
        u_boat_pos=as_vec(-1000.0, 0.0),
        target_point=as_vec(0.0, 500.0),
        bearing_rad=0.0,
        approach_mode=ApproachMode.STERN_CHASE,
        salvo_size=1,
        launch_time=0.0,
    )
    dynamic_hits = torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=proposal,
        torpedo_speed=20.0,
        torpedo_max_run_time=200.0,
        t_max_global=200.0,
        dt=1.0,
    )
    assert not static_hit
    assert dynamic_hits == 0


def test_static_dynamic_hit_parity_same_setup() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-500.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=100.0,
    )
    static_hit = torpedo_hits_ship(ship, torpedo, t_max=100.0)
    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    kin = ConvoyKinematics()
    proposal = AttackProposal(
        u_boat_pos=as_vec(-500.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=0.0,
        approach_mode=ApproachMode.STERN_CHASE,
        salvo_size=1,
        launch_time=0.0,
    )
    dynamic_hits = torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=proposal,
        torpedo_speed=10.0,
        torpedo_max_run_time=100.0,
        t_max_global=100.0,
        dt=1.0,
    )
    assert static_hit
    assert dynamic_hits == 1
