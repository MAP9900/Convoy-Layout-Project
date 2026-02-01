"""Tests for temporal attack visualization math helpers."""

import numpy as np

from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan
from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec
from convoy_sim.viz_attack import get_ship_positions_dynamic, render_attack_frame, torpedo_position_at_global_time


def _ship_at(pos: np.ndarray) -> Ship:
    return Ship(
        id="S1",
        position=pos,
        speed=5.0,
        heading_rad=0.0,
        length=40.0,
        beam=10.0,
        ship_class=ShipClass.FREIGHTER,
    )


def test_get_ship_positions_dynamic_moves() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    formation = ConvoyFormation(
        ships0=[ship],
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    route = RoutePlan(legs=[RouteLeg(duration_s=10.0, heading_rad=0.0)])
    kin = ConvoyKinematics(route=route)
    pos_t0 = get_ship_positions_dynamic([ship], (formation, kin), t_global=0.0)[0]
    pos_t5 = get_ship_positions_dynamic([ship], (formation, kin), t_global=5.0)[0]
    assert pos_t5[0] > pos_t0[0]


def test_torpedo_position_respects_launch_delay() -> None:
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=100.0,
        launch_delay=5.0,
    )
    pos_before = torpedo_position_at_global_time(torpedo, 2.0)
    pos_after = torpedo_position_at_global_time(torpedo, 7.0)
    assert np.allclose(pos_before, as_vec(0.0, 0.0))
    assert pos_after[0] > 0.0


def test_hit_marker_after_launch_delay() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-10.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
        launch_delay=5.0,
    )
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return
    fig, ax = plt.subplots()
    render_attack_frame([ship], [torpedo], t_global=2.0, t_max=20.0, ax=ax, show_footprint=False)
    before_count = len(ax.collections)
    before_offsets = sum(col.get_offsets().shape[0] for col in ax.collections if hasattr(col, "get_offsets"))
    ax.clear()
    render_attack_frame([ship], [torpedo], t_global=6.0, t_max=20.0, ax=ax, show_footprint=False)
    after_count = len(ax.collections)
    after_offsets = sum(col.get_offsets().shape[0] for col in ax.collections if hasattr(col, "get_offsets"))
    plt.close(fig)
    assert after_count >= before_count
    assert after_offsets > before_offsets
