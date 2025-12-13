"""Attack simulation scaffolding and Monte Carlo evaluation hooks."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from .entities import Ship, Torpedo, torpedo_hits_ship
from .geometry import Vec2, as_vec

LayoutFn = Callable[..., list[Ship]]
TorpedoSampler = Callable[[np.random.Generator], Sequence[Torpedo]]


def _vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector")
    return arr


def simulate_attack_once(
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    t_max: float,
) -> int:
    """Simulate one deterministic attack and return the total number of hits."""

    if t_max <= 0.0:
        return 0
    total_hits = 0
    for torpedo in torpedoes:
        for ship in ships:
            if torpedo_hits_ship(ship=ship, torpedo=torpedo, t_max=t_max):
                total_hits += 1
    return total_hits


def run_monte_carlo_attack(
    layout_fn: LayoutFn,
    layout_kwargs: dict[str, Any],
    torpedo_sampler: TorpedoSampler,
    n_trials: int,
    t_max: float,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run a Monte Carlo study of a torpedo attack scenario."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    hits = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        ships = layout_fn(**layout_kwargs)
        torpedoes = list(torpedo_sampler(generator))
        hits[idx] = simulate_attack_once(ships=ships, torpedoes=torpedoes, t_max=t_max)
    expected_hits = float(np.mean(hits))
    variance = float(np.var(hits))
    hit_prob_at_least_one = float(np.mean(hits > 0))
    return {
        "hits_per_trial": hits,
        "expected_hits": expected_hits,
        "var_hits": variance,
        "hit_prob_at_least_one": hit_prob_at_least_one,
        "n_trials": n_trials,
    }


def sample_torpedo_spread_fixed_origin(
    rng: np.random.Generator,
    *,
    origin: Vec2,
    speed: float,
    heading_center_rad: float,
    spread_deg: float,
    count: int,
    max_run_time: float,
) -> list[Torpedo]:
    """Return a deterministic fan of torpedoes from a single launch point."""

    if count <= 0:
        return []
    origin_vec = _vec(origin)
    if count == 1 or spread_deg == 0.0:
        headings = [heading_center_rad]
    else:
        half_spread = math.radians(spread_deg) / 2.0
        headings = []
        for i in range(count):
            fraction = 0.0 if count == 1 else i / (count - 1)
            angle_offset = -half_spread + 2 * half_spread * fraction
            headings.append(heading_center_rad + angle_offset)
    torpedoes = [
        Torpedo(
            id=f"T{i+1:02d}",
            launch_position=origin_vec,
            speed=speed,
            heading_rad=heading,
            max_run_time=max_run_time,
        )
        for i, heading in enumerate(headings)
    ]
    return torpedoes


def sample_parallel_torpedoes(
    rng: np.random.Generator,
    *,
    first_origin: Vec2,
    spacing: float,
    count: int,
    speed: float,
    heading_rad: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Return torpedoes fired from evenly spaced positions along a line."""

    if count <= 0:
        return []
    first_vec = _vec(first_origin)
    offsets = [spacing * i for i in range(count)]
    torpedoes = []
    for idx, offset in enumerate(offsets):
        launch_pos = first_vec + as_vec(0.0, offset)
        torpedoes.append(
            Torpedo(
                id=f"P{idx+1:02d}",
                launch_position=launch_pos,
                speed=speed,
                heading_rad=heading_rad,
                max_run_time=max_run_time,
            )
        )
    return torpedoes

# Backwards-compatible alias (legacy entry point)
simulate_attack = simulate_attack_once
