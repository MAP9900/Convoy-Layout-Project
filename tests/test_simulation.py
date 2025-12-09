"""Simulation scaffolding smoke tests."""

import pytest

import convoy_sim
from convoy_sim import Convoy, Ship, Torpedo, run_monte_carlo_attack, simulate_attack
from convoy_sim.geometry import Point2D


def test_convoy_sim_imports() -> None:
    assert hasattr(convoy_sim, "simulate_attack")


def _example_convoy() -> Convoy:
    ship = Ship(
        name="Test Vessel",
        position=Point2D(0.0, 0.0),
        heading_deg=90.0,
        speed_mps=5.0,
        length_m=150.0,
        beam_m=20.0,
    )
    return Convoy(ships=[ship], layout_name="rectangular", spacing_m=400.0)


def _example_torpedo() -> Torpedo:
    return Torpedo(
        origin=Point2D(-1000.0, 0.0),
        heading_deg=0.0,
        speed_mps=12.0,
        warhead_radius_m=50.0,
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
