"""Attack simulation scaffolding and Monte Carlo evaluation hooks."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Callable, Sequence

import numpy as np

from .attackers import fan_spread, parallel_spread
from .entities import Ship, Torpedo, torpedo_hits_ship
from .geometry import Vec2, as_vec
from .noise import NoiseModel
from .risk import empirical_cvar, empirical_var

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
    noise_model: NoiseModel | None = None,
    risk_alpha: float | None = None,
) -> dict[str, Any]:
    """Run a Monte Carlo study of a torpedo attack scenario."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    hits = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        ships = layout_fn(**layout_kwargs)
        torpedoes = list(torpedo_sampler(generator))
        if noise_model and not noise_model.is_inactive():
            torpedoes = _apply_noise(torpedoes, noise_model, generator)
        hits[idx] = simulate_attack_once(ships=ships, torpedoes=torpedoes, t_max=t_max)
    expected_hits = float(np.mean(hits))
    variance = float(np.var(hits))
    hit_prob_at_least_one = float(np.mean(hits > 0))
    payload = {
        "hits_per_trial": hits,
        "expected_hits": expected_hits,
        "var_hits": variance,
        "hit_prob_at_least_one": hit_prob_at_least_one,
        "n_trials": n_trials,
    }
    if risk_alpha is not None:
        alpha_label = int(round(risk_alpha * 100))
        payload[f"VaR_{alpha_label}"] = empirical_var(hits, risk_alpha)
        payload[f"CVaR_{alpha_label}"] = empirical_cvar(hits, risk_alpha)
    return payload


def sample_fan_spread(
    u_pos: Vec2 | Sequence[float],
    base_bearing_rad: float,
    n: int,
    spread_rad: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from ``u_pos`` spread evenly across ``spread_rad``."""

    return fan_spread(
        u_pos=u_pos,
        base_bearing_rad=base_bearing_rad,
        n=n,
        spread_rad=spread_rad,
        speed=speed,
        max_run_time=max_run_time,
    )


def sample_parallel_spread(
    u_pos: Vec2 | Sequence[float],
    bearing_rad: float,
    n: int,
    lateral_spacing: float,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from parallel launchers offset laterally."""

    return parallel_spread(
        u_pos=u_pos,
        bearing_rad=bearing_rad,
        n=n,
        lateral_spacing=lateral_spacing,
        speed=speed,
        max_run_time=max_run_time,
    )


def _apply_noise(
    torpedoes: Sequence[Torpedo],
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> list[Torpedo]:
    adjusted: list[Torpedo] = []
    for torpedo in torpedoes:
        heading = torpedo.heading_rad
        if noise_model.sigma_heading_rad > 0.0:
            heading += rng.normal(0.0, noise_model.sigma_heading_rad)
        delay = torpedo.launch_delay
        if noise_model.sigma_launch_delay > 0.0:
            delay = max(0.0, delay + rng.normal(0.0, noise_model.sigma_launch_delay))
        is_dud = torpedo.is_dud
        if noise_model.p_dud > 0.0:
            is_dud = is_dud or (rng.uniform(0.0, 1.0) < noise_model.p_dud)
        adjusted.append(
            replace(
                torpedo,
                heading_rad=heading,
                launch_delay=delay,
                is_dud=is_dud,
            )
        )
    return adjusted


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
