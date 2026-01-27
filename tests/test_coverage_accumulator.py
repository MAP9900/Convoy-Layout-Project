"""Tests for torpedo coverage accumulation."""

import numpy as np

from convoy_sim.coverage import accumulate_torpedo_coverage
from convoy_sim.entities import Torpedo
from convoy_sim.geometry import as_vec


def test_coverage_line_peak_near_track() -> None:
    torpedo = Torpedo(
        id="T1",
        launch_position=as_vec(-100.0, 0.0),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=20.0,
    )
    data = accumulate_torpedo_coverage(
        torpedoes_list=[[torpedo]],
        t_max=20.0,
        bounds=(-150.0, 150.0, -50.0, 50.0),
        grid_n=50,
        dt=1.0,
    )
    counts = data["grid_counts"]
    assert counts.shape == (50, 50)
    max_idx = np.unravel_index(np.argmax(counts), counts.shape)
    y_center = (data["y_edges"][max_idx[0]] + data["y_edges"][max_idx[0] + 1]) / 2.0
    assert abs(y_center) < 10.0


def test_coverage_bounds_match_edges() -> None:
    data = accumulate_torpedo_coverage(
        torpedoes_list=[],
        t_max=10.0,
        bounds=(-10.0, 10.0, -5.0, 5.0),
        grid_n=10,
        dt=1.0,
    )
    assert data["x_edges"][0] == -10.0
    assert data["x_edges"][-1] == 10.0
    assert data["y_edges"][0] == -5.0
    assert data["y_edges"][-1] == 5.0
