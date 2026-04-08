"""Tests for attack visualization helpers without matplotlib."""

import numpy as np

from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.geometry import as_vec
from convoy_sim.realism import UBoatMotionPlan
from convoy_sim.viz_attack import (
    attack_debug_metrics,
    format_torpedo_heading_table,
    min_miss_distance_ship_torpedo,
    sample_u_boat_track,
    torpedo_path_points,
    torpedo_heading_table_rows,
)


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


def test_min_miss_distance_direct_hit() -> None:
    ship = _ship_at(as_vec(0.0, 0.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-100.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min <= ship.effective_hit_radius()


def test_min_miss_distance_clean_miss() -> None:
    ship = _ship_at(as_vec(0.0, 100.0))
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-100.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min > ship.effective_hit_radius()


def test_attack_debug_metrics_keys() -> None:
    ships = [
        _ship_at(as_vec(0.0, 0.0)),
        Ship(
            id="S2",
            position=as_vec(50.0, 50.0),
            speed=0.0,
            heading_rad=0.0,
            length=40.0,
            beam=10.0,
            ship_class=ShipClass.ESCORT,
        ),
    ]
    torpedoes = [
        Torpedo(
            id="T1",
            launch_position=as_vec(-100.0, 0.0),
            speed=10.0,
            heading_rad=0.0,
            max_run_time=20.0,
        )
    ]
    metrics = attack_debug_metrics(ships, torpedoes, t_max=20.0)
    assert "torpedoes" in metrics
    assert "ships" in metrics
    assert metrics["torpedoes"][0]["closest_ship_id"] in {"S1", "S2"}


def test_min_miss_distance_handles_gyro_turn_path() -> None:
    ship = _ship_at(as_vec(20.0, 80.0))
    torpedo = Torpedo(
        id="Tgyro",
        launch_position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=np.pi / 2.0,
        max_run_time=20.0,
        launch_heading_rad=0.0,
        gyro_turn_distance_m=20.0,
    )
    d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max=20.0)
    assert d_min <= ship.effective_hit_radius()


def test_torpedo_heading_table_rows_capture_launch_and_final_headings() -> None:
    torpedo = Torpedo(
        id="F01",
        launch_position=as_vec(10.0, 20.0),
        speed=10.0,
        heading_rad=np.deg2rad(100.0),
        max_run_time=30.0,
        launch_delay=12.0,
        launch_heading_rad=np.deg2rad(90.0),
        gyro_turn_distance_m=25.0,
    )
    rows = torpedo_heading_table_rows([torpedo])
    assert len(rows) == 1
    row = rows[0]
    assert row["shot"] == 1
    assert row["torpedo_id"] == "F01"
    assert row["launch_time_s"] == 12.0
    assert np.isclose(row["launch_heading_deg"], 90.0)
    assert np.isclose(row["final_heading_deg"], 100.0)
    assert np.isclose(row["gyro_offset_deg"], 10.0)


def test_format_torpedo_heading_table_has_header_and_values() -> None:
    torpedo = Torpedo(
        id="F02",
        launch_position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=10.0,
    )
    table = format_torpedo_heading_table([torpedo])
    assert "t_launch_s" in table
    assert "F02" in table


def test_sample_u_boat_track_samples_requested_interval() -> None:
    plan = UBoatMotionPlan(
        initial_position=as_vec(0.0, 0.0),
        initial_heading_rad=np.pi / 2.0,
        initial_speed_mps=2.0,
        mode="moving",
        launch_time_s=0.0,
    )
    track = sample_u_boat_track(plan, 0.0, 10.0, n_points=5)
    assert track.shape == (5, 2)
    assert np.allclose(track[0], np.array([0.0, 0.0]))
    assert np.allclose(track[-1], np.array([0.0, 20.0]), atol=1e-6)


def test_torpedo_path_points_respect_absolute_end_time_with_launch_delay() -> None:
    torpedo = Torpedo(
        id="F03",
        launch_position=as_vec(0.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
        launch_delay=5.0,
    )
    path = torpedo_path_points(torpedo, t0=0.0, t1=30.0)
    assert np.allclose(path[-1], np.array([200.0, 0.0]))
