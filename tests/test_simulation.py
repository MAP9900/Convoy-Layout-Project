"""Simulation scaffolding smoke tests."""

import math
from typing import Callable

import numpy as np
import pytest

import convoy_sim
from convoy_sim import (
    as_vec,
    make_rectangular_convoy,
    run_monte_carlo_attack,
    sample_fan_spread,
    sample_parallel_spread,
    simulate_attack,
    simulate_attack_once,
)


def test_convoy_sim_imports() -> None:
    assert hasattr(convoy_sim, "simulate_attack")


def _single_ship_layout_kwargs(speed: float = 0.0) -> dict:
    return {
        "n_rows": 1,
        "n_cols": 1,
        "spacing_along": 100.0,
        "spacing_across": 100.0,
        "speed": speed,
        "heading_rad": 0.0,
        "length": 120.0,
        "beam": 20.0,
        "origin": as_vec(0.0, 0.0),
    }


def _fan_sampler(origin: np.ndarray) -> Callable[[np.random.Generator], list]:
    def sampler(_: np.random.Generator):
        return sample_fan_spread(
            u_pos=origin,
            base_bearing_rad=0.0,
            n=1,
            spread_rad=0.0,
            speed=20.0,
            max_run_time=500.0,
        )

    return sampler


def test_simulate_attack_once_direct_hit() -> None:
    ships = make_rectangular_convoy(**_single_ship_layout_kwargs())
    torpedoes = _fan_sampler(as_vec(-1000.0, 0.0))(np.random.default_rng(0))
    hits = simulate_attack_once(ships, torpedoes, t_max=120.0)
    assert hits == 1
    assert simulate_attack(ships, torpedoes, t_max=120.0) == 1


def test_monte_carlo_all_hits() -> None:
    sampler = _fan_sampler(as_vec(-1000.0, 0.0))
    result = run_monte_carlo_attack(
        layout_fn=make_rectangular_convoy,
        layout_kwargs=_single_ship_layout_kwargs(),
        torpedo_sampler=sampler,
        n_trials=100,
        t_max=150.0,
        rng=np.random.default_rng(42),
    )
    assert result["hits_per_trial"].shape == (100,)
    assert np.all(result["hits_per_trial"] == 1)
    assert result["expected_hits"] == pytest.approx(1.0, rel=1e-3)
    assert result["hit_prob_at_least_one"] == pytest.approx(1.0)


def test_monte_carlo_all_misses() -> None:
    sampler = _fan_sampler(as_vec(-1000.0, 500.0))
    result = run_monte_carlo_attack(
        layout_fn=make_rectangular_convoy,
        layout_kwargs=_single_ship_layout_kwargs(),
        torpedo_sampler=sampler,
        n_trials=50,
        t_max=150.0,
        rng=np.random.default_rng(24),
    )
    assert np.all(result["hits_per_trial"] == 0)
    assert result["expected_hits"] == pytest.approx(0.0)
    assert result["hit_prob_at_least_one"] == pytest.approx(0.0)


def test_sample_fan_spread_headings() -> None:
    torpedoes = sample_fan_spread(
        u_pos=as_vec(0.0, 0.0),
        base_bearing_rad=0.0,
        n=3,
        spread_rad=math.radians(30.0),
        speed=20.0,
        max_run_time=200.0,
    )
    headings = [torpedo.heading_rad for torpedo in torpedoes]
    assert math.isclose(headings[0], math.radians(-15.0), abs_tol=1e-9)
    assert math.isclose(headings[1], 0.0, abs_tol=1e-9)
    assert math.isclose(headings[2], math.radians(15.0), abs_tol=1e-9)
    assert np.allclose(torpedoes[0].launch_position, as_vec(0.0, 0.0))


def test_sample_parallel_spread_positions() -> None:
    torpedoes = sample_parallel_spread(
        u_pos=as_vec(0.0, 0.0),
        bearing_rad=0.0,
        n=3,
        lateral_spacing=100.0,
        speed=15.0,
        max_run_time=300.0,
    )
    positions = np.array([t.launch_position for t in torpedoes])
    assert np.allclose(positions[:, 0], 0.0)
    assert np.allclose(positions[:, 1], [-100.0, 0.0, 100.0])
