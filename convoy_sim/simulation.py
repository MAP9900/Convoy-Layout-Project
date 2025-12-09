"""Attack simulation scaffolding and Monte Carlo evaluation hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .entities import Convoy, Torpedo
from .geometry import Point2D


@dataclass
class SimulationResult:
    """Outcome from a single simulated torpedo attack."""

    hits: int
    hit_ships: list[str]
    torpedo_tracks: list[list[Point2D]]


@dataclass
class MonteCarloResult:
    """Aggregated statistics from multiple simulated attacks."""

    expected_hits: float
    variance: float
    hit_distribution: list[int]


def simulate_attack(
    convoy: Convoy,
    torpedoes: Sequence[Torpedo],
    duration_s: float,
    time_step_s: float,
) -> SimulationResult:
    """Simulate a single straight-running torpedo attack over a time window."""

    raise NotImplementedError


def run_monte_carlo_attack(
    convoy: Convoy,
    torpedo_factory: Iterable[Torpedo],
    iterations: int,
) -> MonteCarloResult:
    """Run many torpedo attacks and aggregate hit statistics."""

    raise NotImplementedError
