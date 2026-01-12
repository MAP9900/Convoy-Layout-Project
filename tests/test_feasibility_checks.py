"""Deterministic tests for feasibility checks."""

import math

import numpy as np

from convoy_sim.entities import Ship
from convoy_sim.feasibility import (
    ApproachMode,
    AttackConstraints,
    AttackProposal,
    EscortZone,
    compute_convoy_reference,
    is_attack_feasible,
)
from convoy_sim.geometry import as_vec


def _simple_convoy() -> list[Ship]:
    return [
        Ship(id="S1", position=as_vec(-50.0, -50.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
        Ship(id="S2", position=as_vec(-50.0, 50.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
        Ship(id="S3", position=as_vec(50.0, -50.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
        Ship(id="S4", position=as_vec(50.0, 50.0), speed=5.0, heading_rad=0.0, length=100.0, beam=20.0),
    ]


def test_exclusion_zone_blocks_attack() -> None:
    ships = _simple_convoy()
    zone = EscortZone(center=as_vec(0.0, 0.0), radius=500.0, hard_exclusion=True, risk_weight=1.0)
    constraints = AttackConstraints(min_range=0.0, max_range=2000.0, escort_zones=[zone])
    proposal = AttackProposal(
        u_boat_pos=as_vec(100.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=0.0,
        approach_mode=ApproachMode.ABEAM,
        salvo_size=4,
    )
    feasible, details = is_attack_feasible(ships, proposal, constraints)
    assert not feasible
    assert "escort_exclusion" in details["failed_checks"]


def test_range_and_mode_pass() -> None:
    ships = _simple_convoy()
    constraints = AttackConstraints(min_range=200.0, max_range=2000.0)
    proposal = AttackProposal(
        u_boat_pos=as_vec(500.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=math.pi,
        approach_mode=ApproachMode.BOW_ON,
        salvo_size=4,
    )
    feasible, details = is_attack_feasible(ships, proposal, constraints)
    assert feasible
    assert details["failed_checks"] == []


def test_approach_mode_feasibility() -> None:
    ships = _simple_convoy()
    constraints = AttackConstraints(min_range=0.0, max_range=2000.0)
    reference = compute_convoy_reference(ships)
    heading = reference["heading_rad"]
    proposal_bow_on = AttackProposal(
        u_boat_pos=as_vec(500.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=heading + math.pi,
        approach_mode=ApproachMode.BOW_ON,
        salvo_size=4,
    )
    proposal_abeam = AttackProposal(
        u_boat_pos=as_vec(500.0, 0.0),
        target_point=as_vec(0.0, 0.0),
        bearing_rad=heading + math.pi / 2.0,
        approach_mode=ApproachMode.BOW_ON,
        salvo_size=4,
    )
    feasible_bow, _ = is_attack_feasible(ships, proposal_bow_on, constraints)
    feasible_abeam, _ = is_attack_feasible(ships, proposal_abeam, constraints)
    assert feasible_bow
    assert not feasible_abeam
