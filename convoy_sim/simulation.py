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


def sample_fan_spread(
    u_pos: Vec2 | Sequence[float],
    base_bearing_rad: float,
    n: int,
    spread_rad: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from ``u_pos`` spread evenly across ``spread_rad``."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    if n == 1 or spread_rad == 0.0:
        headings = [base_bearing_rad]
    else:
        half_spread = spread_rad / 2.0
        headings = [
            base_bearing_rad - half_spread + (spread_rad * i / (n - 1))
            for i in range(n)
        ]
    return [
        Torpedo(
            id=f"F{i+1:02d}",
            launch_position=origin_vec,
            speed=speed,
            heading_rad=heading,
            max_run_time=max_run_time,
        )
        for i, heading in enumerate(headings)
    ]


def sample_parallel_spread(
    u_pos: Vec2 | Sequence[float],
    bearing_rad: float,
    n: int,
    lateral_spacing: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from parallel launchers offset laterally."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    perp = as_vec(-math.sin(bearing_rad), math.cos(bearing_rad))
    center = (n - 1) / 2.0
    launches = [
        origin_vec + perp * ((i - center) * lateral_spacing)
        for i in range(n)
    ]
    return [
        Torpedo(
            id=f"P{i+1:02d}",
            launch_position=launch_pos,
            speed=speed,
            heading_rad=bearing_rad,
            max_run_time=max_run_time,
        )
        for i, launch_pos in enumerate(launches)
    ]


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
    """Deprecated helper retained for backwards compatibility."""

    return sample_fan_spread(
        origin,
        base_bearing_rad=heading_center_rad,
        n=count,
        spread_rad=math.radians(spread_deg),
        speed=speed,
        max_run_time=max_run_time,
    )


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
    """Deprecated helper retained for backwards compatibility."""

    return sample_parallel_spread(
        u_pos=first_origin,
        bearing_rad=heading_rad,
        n=count,
        lateral_spacing=spacing,
        speed=speed,
        max_run_time=max_run_time,
    )

# Backwards-compatible alias (legacy entry point)
simulate_attack = simulate_attack_once
