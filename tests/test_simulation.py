"""Simulation scaffolding smoke tests."""

import math

import pytest

import convoy_sim
from convoy_sim import Convoy, Ship, Torpedo, as_vec, run_monte_carlo_attack, simulate_attack


def test_convoy_sim_imports() -> None:
    assert hasattr(convoy_sim, "simulate_attack")


def _example_convoy() -> Convoy:
    ship = Ship(
        id="Test Vessel",
        position=as_vec(0.0, 0.0),
        heading_rad=math.radians(90.0),
        speed=5.0,
        length=150.0,
        beam=20.0,
    )
    return Convoy(ships=[ship], layout_name="rectangular", spacing_m=400.0)


def _example_torpedo() -> Torpedo:
    return Torpedo(
        id="Test Torpedo",
        launch_position=as_vec(-1000.0, 0.0),
        heading_rad=0.0,
        speed=12.0,
        max_run_time=1200.0,
    )


def test_simulate_attack_stub() -> None:
    convoy = _example_convoy()
    torpedoes = [_example_torpedo()]
    with pytest.raises(NotImplementedError):
        simulate_attack(convoy=convoy, torpedoes=torpedoes, duration_s=300.0, time_step_s=1.0)


def test_monte_carlo_stub() -> None:
    convoy = _example_convoy()
    torpedo_sequence = [_example_torpedo()]
    with pytest.raises(NotImplementedError):
        run_monte_carlo_attack(convoy=convoy, torpedo_factory=torpedo_sequence, iterations=10)
