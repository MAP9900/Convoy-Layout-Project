"""Tests for dynamic hit event recording."""

import numpy as np

from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan
from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec
from convoy_sim.simulation import advance_dynamic_hit_state, init_dynamic_hit_state


def _formation_with_one_ship() -> tuple[ConvoyFormation, ConvoyKinematics]:
    ship = Ship(
        id="S1",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=40.0,
        beam=10.0,
        ship_class=ShipClass.FREIGHTER,
    )
    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    kin = ConvoyKinematics(route=RoutePlan(legs=[RouteLeg(duration_s=30.0, heading_rad=0.0)]))
    return formation, kin


def test_dynamic_state_records_single_hit_event() -> None:
    formation, kin = _formation_with_one_ship()
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-40.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
        launch_delay=0.0,
    )
    state = init_dynamic_hit_state(0.0)
    advance_dynamic_hit_state(
        formation,
        kin,
        [torpedo],
        t_target=8.0,
        dt=0.1,
        state=state,
        max_hits_per_torpedo=1,
    )
    assert len(state.hit_events) == 1
    event = state.hit_events[0]
    assert event.torpedo_id == "T1"
    assert event.ship_id == "S1"
    assert event.time_s >= 0.0
    assert np.isclose(state.torpedo_hit_times["T1"], event.time_s)
    assert state.hit_counts["S1"] == 1
    assert len(state.hit_events) == sum(state.hit_counts.values())


def test_dynamic_state_records_multiple_events_same_ship() -> None:
    formation, kin = _formation_with_one_ship()
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-50.0, 0.0),
            speed=10.0,
            heading_rad=0.0,
            max_run_time=20.0,
            launch_delay=0.0,
        ),
        Torpedo(
            id="T2",
            launch_position=as_vec(-50.0, 0.0),
            speed=10.0,
            heading_rad=0.0,
            max_run_time=20.0,
            launch_delay=1.0,
        ),
    ]
    state = init_dynamic_hit_state(0.0)
    advance_dynamic_hit_state(
        formation,
        kin,
        torpedoes,
        t_target=10.0,
        dt=0.1,
        state=state,
        max_hits_per_torpedo=1,
    )
    assert len(state.hit_events) == 2
    assert state.hit_counts["S1"] == 2
    assert {event.torpedo_id for event in state.hit_events} == {"T1", "T2"}
