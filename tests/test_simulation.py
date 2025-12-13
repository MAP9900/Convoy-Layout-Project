"""Simulation scaffolding smoke tests."""

import math
from functools import partial

import numpy as np
import pytest

import convoy_sim
from convoy_sim import (
    as_vec,
    make_rectangular_convoy,
    run_monte_carlo_attack,
    sample_torpedo_spread_fixed_origin,
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


def test_simulate_attack_once_direct_hit() -> None:
    ships = make_rectangular_convoy(**_single_ship_layout_kwargs())
    sampler = partial(
        sample_torpedo_spread_fixed_origin,
        origin=as_vec(-1000.0, 0.0),
        speed=20.0,
        heading_center_rad=0.0,
        spread_deg=0.0,
        count=1,
        max_run_time=500.0,
    )
    torpedoes = sampler(np.random.default_rng(0))
    hits = simulate_attack_once(ships, torpedoes, t_max=120.0)
    assert hits == 1
    # Legacy alias
    assert simulate_attack(ships, torpedoes, t_max=120.0) == 1


def test_monte_carlo_all_hits() -> None:
    sampler = partial(
        sample_torpedo_spread_fixed_origin,
        origin=as_vec(-1000.0, 0.0),
        speed=20.0,
        heading_center_rad=0.0,
        spread_deg=0.0,
        count=1,
        max_run_time=500.0,
    )
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
    sampler = partial(
        sample_torpedo_spread_fixed_origin,
        origin=as_vec(-1000.0, 500.0),
        speed=20.0,
        heading_center_rad=0.0,
        spread_deg=0.0,
        count=1,
        max_run_time=500.0,
    )
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
