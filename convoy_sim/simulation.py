"""Attack simulation scaffolding and Monte Carlo evaluation hooks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from typing import Any, Callable, Sequence

import numpy as np

from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, ship_positions_at, validate_dt
from convoy_sim.entities import Ship, ShipClass, Torpedo, torpedo_hit_time, torpedo_hits_ship
from convoy_sim.feasibility import AttackProposal
from convoy_sim.geometry import Vec2, as_vec, closest_approach_time, distance
from convoy_sim.noise import NoiseModel
from convoy_sim.objectives import ObjectiveSpec, score_trial_result, weighted_value_destroyed_from_trial
from convoy_sim.realism import ShipMovementRealismConfig, apply_ship_movement_realism
from convoy_sim.risk import empirical_cvar, empirical_var

LayoutFn = Callable[..., list[Ship]]
TorpedoSampler = Callable[[np.random.Generator], Sequence[Torpedo]]
ProposalSampler = Callable[[np.random.Generator, list[Ship]], AttackProposal]


def _vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector")
    return arr


def _convoy_reference_from_ships(ships: Sequence[Ship]) -> tuple[Vec2, float, float]:
    if not ships:
        raise ValueError("ships must be non-empty")
    positions = np.array([ship.position for ship in ships], dtype=float)
    centroid = np.mean(positions, axis=0)
    mean_speed = float(np.mean([ship.speed for ship in ships]))
    headings = np.array([ship.heading_rad for ship in ships], dtype=float)
    mean_heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    return centroid, mean_heading, mean_speed


def simulate_attack_once(
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    t_max: float,
    max_hits_per_torpedo: int | None = None,
) -> int:
    """Simulate one deterministic attack and return the total number of hits."""

    if t_max <= 0.0:
        return 0
    total_hits = 0
    for torpedo in torpedoes:
        if max_hits_per_torpedo == 1:
            earliest_hit = None
            for ship in ships:
                hit_time = _earliest_hit_time(ship, torpedo, t_max)
                if hit_time is None:
                    continue
                if earliest_hit is None or hit_time < earliest_hit:
                    earliest_hit = hit_time
            if earliest_hit is not None:
                total_hits += 1
        else:
            for ship in ships:
                if torpedo_hits_ship(ship=ship, torpedo=torpedo, t_max=t_max):
                    total_hits += 1
    return total_hits


def simulate_attack_static(
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    t_max: float,
    max_hits_per_torpedo: int | None = None,
) -> int:
    """Explicit static-path wrapper for straight-line kinematics."""

    return simulate_attack_once(
        ships=ships,
        torpedoes=torpedoes,
        t_max=t_max,
        max_hits_per_torpedo=max_hits_per_torpedo,
    )


def simulate_attack_dynamic_once(
    formation: ConvoyFormation,
    kin: ConvoyKinematics,
    proposal: AttackProposal,
    *,
    torpedo_speed: float,
    torpedo_max_run_time: float,
    t_max_global: float,
    salvo_size: int | None = None,
    spread_rad: float = 0.0,
    dt: float = 1.0,
    safety_margin: float = 0.0,
    max_hits_per_torpedo: int | None = None,
    noise_model: NoiseModel | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Explicit dynamic-path wrapper for convoy-level kinematics."""

    return torpedo_hits_ship_dynamic(
        formation=formation,
        kin=kin,
        proposal=proposal,
        torpedo_speed=torpedo_speed,
        torpedo_max_run_time=torpedo_max_run_time,
        salvo_size=salvo_size,
        spread_rad=spread_rad,
        t_max_global=t_max_global,
        dt=dt,
        safety_margin=safety_margin,
        max_hits_per_torpedo=max_hits_per_torpedo,
        noise_model=noise_model,
        rng=rng,
    )


def torpedo_hits_ship_dynamic(
    formation: ConvoyFormation,
    kin: ConvoyKinematics,
    proposal: AttackProposal,
    *,
    torpedo_speed: float,
    torpedo_max_run_time: float,
    salvo_size: int | None = None,
    spread_rad: float = 0.0,
    t_max_global: float,
    dt: float = 1.0,
    safety_margin: float = 0.0,
    max_hits_per_torpedo: int | None = None,
    noise_model: NoiseModel | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Return total hits for a salvo against a moving convoy using discrete stepping.

    Discrete stepping trades accuracy for simplicity; smaller ``dt`` yields more
    precise hit timing at higher computational cost.
    """

    if t_max_global <= 0.0 or torpedo_max_run_time <= 0.0:
        return 0
    dt = validate_dt(dt)
    salvo = proposal.salvo_size if salvo_size is None else int(salvo_size)
    if salvo <= 0:
        return 0

    torpedoes = fan_spread(
        u_pos=proposal.u_boat_pos,
        base_bearing_rad=proposal.bearing_rad,
        n=salvo,
        spread_rad=spread_rad,
        speed=torpedo_speed,
        max_run_time=torpedo_max_run_time,
    )
    torpedoes = [replace(t, launch_delay=proposal.launch_time) for t in torpedoes]
    if noise_model and not noise_model.is_inactive():
        torpedoes = apply_noise_to_torpedoes(torpedoes, noise_model, rng or np.random.default_rng())

    window_end = min(float(t_max_global), float(proposal.launch_time + torpedo_max_run_time))
    if window_end <= proposal.launch_time:
        return 0

    hits = 0
    hit_pairs: set[tuple[int, int]] = set()
    torpedo_done: set[int] = set()
    time = float(proposal.launch_time)
    while time <= window_end + 1e-9:
        ship_positions = ship_positions_at(time, formation, kin, dt=dt, motion="independent")
        for torp_idx, torpedo in enumerate(torpedoes):
            if torp_idx in torpedo_done or torpedo.is_dud:
                continue
            torp_pos = torpedo.position_at(time)
            for ship_idx, ship in enumerate(formation.ships0):
                key = (torp_idx, ship_idx)
                if key in hit_pairs:
                    continue
                ship_pos = ship_positions[ship_idx]
                radius = ship.effective_hit_radius() + float(safety_margin)
                if distance(ship_pos, torp_pos) <= radius:
                    hit_pairs.add(key)
                    hits += 1
                    if max_hits_per_torpedo == 1:
                        torpedo_done.add(torp_idx)
                        break
        time += dt
    return hits


def _static_hit_events(
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    t_max: float,
    max_hits_per_torpedo: int | None = None,
) -> list[tuple[int, int, float]]:
    """Return static hit events as ``(torpedo_idx, ship_idx, hit_time)`` tuples."""

    if t_max <= 0.0:
        return []
    events: list[tuple[int, int, float]] = []
    for torpedo_idx, torpedo in enumerate(torpedoes):
        if torpedo.is_dud:
            continue
        torpedo_hits: list[tuple[int, float]] = []
        for ship_idx, ship in enumerate(ships):
            hit_time = _earliest_hit_time(ship, torpedo, t_max)
            if hit_time is not None:
                torpedo_hits.append((ship_idx, float(hit_time)))
        if max_hits_per_torpedo == 1:
            if torpedo_hits:
                ship_idx, hit_time = min(torpedo_hits, key=lambda item: (item[1], item[0]))
                events.append((torpedo_idx, ship_idx, hit_time))
        else:
            events.extend(
                (torpedo_idx, ship_idx, hit_time)
                for ship_idx, hit_time in sorted(torpedo_hits, key=lambda item: (item[1], item[0]))
            )
    return sorted(events, key=lambda item: (item[2], item[0], item[1]))


def simulate_attack_once_scored(
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    t_max: float,
    max_hits_per_torpedo: int | None = None,
) -> dict[str, Any]:
    """Simulate one attack and return hit counts plus value-based metrics.

    Ship value is counted once per ship if hit at least once, even if multiple
    torpedoes hit the same ship.
    """

    hit_ship_ids: set[str] = set()
    hits_by_class: dict[ShipClass, int] = {}
    value_by_class: dict[ShipClass, float] = {}

    hit_events = _static_hit_events(
        ships=ships,
        torpedoes=torpedoes,
        t_max=t_max,
        max_hits_per_torpedo=max_hits_per_torpedo,
    )
    n_hits = int(len(hit_events))

    for _, ship_idx, _ in hit_events:
        ship = ships[int(ship_idx)]
        if ship.id in hit_ship_ids:
            continue
        hit_ship_ids.add(ship.id)
        hits_by_class[ship.ship_class] = hits_by_class.get(ship.ship_class, 0) + 1
        value_by_class[ship.ship_class] = value_by_class.get(ship.ship_class, 0.0) + ship.value_weight

    total_value_destroyed = float(sum(value_by_class.values()))
    unique_ships_hit = int(len(hit_ship_ids))
    repeat_hits = max(0, int(n_hits) - unique_ships_hit)
    return {
        "hit_ship_ids": list(hit_ship_ids),
        "n_hits": n_hits,
        "unique_ships_hit": unique_ships_hit,
        "repeat_hits": repeat_hits,
        "total_value_destroyed": total_value_destroyed,
        "value_destroyed_by_class": value_by_class,
        "hits_by_class": hits_by_class,
    }


def run_monte_carlo_attack(
    layout_fn: LayoutFn,
    layout_kwargs: dict[str, Any],
    torpedo_sampler: TorpedoSampler,
    n_trials: int,
    t_max: float,
    rng: np.random.Generator | None = None,
    noise_model: NoiseModel | None = None,
    risk_alpha: float | None = None,
    max_hits_per_torpedo: int | None = None,
) -> dict[str, Any]:
    """Run a Monte Carlo study of a torpedo attack scenario."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    realism_cfg = ShipMovementRealismConfig.from_dict(layout_kwargs.get("ship_movement_realism"))
    layout_kwargs_base = {k: v for k, v in layout_kwargs.items() if k != "ship_movement_realism"}
    sampler_arity = len(inspect.signature(torpedo_sampler).parameters)
    hits = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        ships = layout_fn(**layout_kwargs_base)
        ships = apply_ship_movement_realism(ships, rng=generator, cfg=realism_cfg)
        if sampler_arity >= 2:
            torpedoes = list(torpedo_sampler(generator, ships))
        else:
            torpedoes = list(torpedo_sampler(generator))
        if noise_model and not noise_model.is_inactive():
            torpedoes = _apply_noise(torpedoes, noise_model, generator)
        hits[idx] = simulate_attack_once(
            ships=ships,
            torpedoes=torpedoes,
            t_max=t_max,
            max_hits_per_torpedo=max_hits_per_torpedo,
        )
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


def run_monte_carlo_attack_dynamic(
    layout_fn: LayoutFn,
    layout_kwargs: dict[str, Any],
    proposal_sampler: ProposalSampler,
    n_trials: int,
    t_max_global: float,
    *,
    kin: ConvoyKinematics,
    torpedo_speed: float,
    torpedo_max_run_time: float,
    spread_rad: float = 0.0,
    salvo_size: int | None = None,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
    noise_model: NoiseModel | None = None,
    risk_alpha: float | None = None,
    max_hits_per_torpedo: int | None = None,
) -> dict[str, Any]:
    """Run Monte Carlo with convoy-level motion and time-aware attack windows."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    hits = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        ships = layout_fn(**layout_kwargs)
        centroid, mean_heading, _mean_speed = _convoy_reference_from_ships(ships)
        formation = ConvoyFormation(
            ships0=list(ships),
            convoy_origin0=centroid,
            convoy_heading0=mean_heading,
        )
        proposal = proposal_sampler(generator, list(ships))
        hits[idx] = torpedo_hits_ship_dynamic(
            formation=formation,
            kin=kin,
            proposal=proposal,
            torpedo_speed=torpedo_speed,
            torpedo_max_run_time=torpedo_max_run_time,
            salvo_size=salvo_size,
            spread_rad=spread_rad,
            t_max_global=t_max_global,
            dt=dt,
            noise_model=noise_model,
            rng=generator,
            max_hits_per_torpedo=max_hits_per_torpedo,
        )
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


def run_monte_carlo_attack_scored(
    layout_fn: LayoutFn,
    layout_kwargs: dict[str, Any],
    torpedo_sampler: TorpedoSampler,
    n_trials: int,
    t_max: float,
    rng: np.random.Generator | None = None,
    noise_model: NoiseModel | None = None,
    risk_alpha: float | None = None,
    max_hits_per_torpedo: int | None = None,
    objective: ObjectiveSpec | None = None,
) -> dict[str, Any]:
    """Run Monte Carlo and return hit/value metrics."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    realism_cfg = ShipMovementRealismConfig.from_dict(layout_kwargs.get("ship_movement_realism"))
    layout_kwargs_base = {k: v for k, v in layout_kwargs.items() if k != "ship_movement_realism"}
    sampler_arity = len(inspect.signature(torpedo_sampler).parameters)
    hits = np.zeros(n_trials, dtype=float)
    unique_ships = np.zeros(n_trials, dtype=float)
    repeat_hits = np.zeros(n_trials, dtype=float)
    values = np.zeros(n_trials, dtype=float)
    weighted_values = np.zeros(n_trials, dtype=float)
    losses = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        ships = layout_fn(**layout_kwargs_base)
        ships = apply_ship_movement_realism(ships, rng=generator, cfg=realism_cfg)
        if sampler_arity >= 2:
            torpedoes = list(torpedo_sampler(generator, ships))
        else:
            torpedoes = list(torpedo_sampler(generator))
        if noise_model and not noise_model.is_inactive():
            torpedoes = _apply_noise(torpedoes, noise_model, generator)
        scored = simulate_attack_once_scored(
            ships=ships,
            torpedoes=torpedoes,
            t_max=t_max,
            max_hits_per_torpedo=max_hits_per_torpedo,
        )
        hits[idx] = scored["n_hits"]
        unique_ships[idx] = float(scored.get("unique_ships_hit", 0.0))
        repeat_hits[idx] = float(scored.get("repeat_hits", 0.0))
        values[idx] = scored["total_value_destroyed"]
        weighted_values[idx] = weighted_value_destroyed_from_trial(scored, objective)
        losses[idx] = score_trial_result(scored, objective) if objective is not None else values[idx]
    expected_hits = float(np.mean(hits))
    expected_unique_ships = float(np.mean(unique_ships))
    expected_repeat_hits = float(np.mean(repeat_hits))
    expected_value = float(np.mean(values))
    expected_weighted_value = float(np.mean(weighted_values))
    expected_loss = float(np.mean(losses))
    variance = float(np.var(hits))
    hit_prob_at_least_one = float(np.mean(hits > 0))
    payload = {
        "hits_per_trial": hits,
        "unique_ships_hit_per_trial": unique_ships,
        "repeat_hits_per_trial": repeat_hits,
        "value_destroyed_per_trial": values,
        "weighted_value_destroyed_per_trial": weighted_values,
        "loss_per_trial": losses,
        "expected_hits": expected_hits,
        "expected_unique_ships_hit": expected_unique_ships,
        "expected_repeat_hits": expected_repeat_hits,
        "expected_value_destroyed": expected_value,
        "expected_weighted_value_destroyed": expected_weighted_value,
        "expected_loss": expected_loss,
        "var_hits": variance,
        "hit_prob_at_least_one": hit_prob_at_least_one,
        "n_trials": n_trials,
    }
    alpha = 0.9 if risk_alpha is None else float(risk_alpha)
    alpha_label = int(round(alpha * 100))
    payload[f"VaR_{alpha_label}"] = empirical_var(hits, alpha)
    payload[f"CVaR_{alpha_label}"] = empirical_cvar(hits, alpha)
    payload[f"VaR_{alpha_label}_loss"] = empirical_var(losses, alpha)
    payload[f"CVaR_{alpha_label}_loss"] = empirical_cvar(losses, alpha)
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


def apply_noise_to_torpedoes(
    torpedoes: Sequence[Torpedo],
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> list[Torpedo]:
    """Return torpedoes with optional heading/delay/dud noise applied."""

    adjusted: list[Torpedo] = []
    for torpedo in torpedoes:
        heading = torpedo.heading_rad
        if noise_model.sigma_heading_rad > 0.0:
            heading += rng.normal(0.0, noise_model.sigma_heading_rad)
        speed = float(torpedo.speed)
        if noise_model.sigma_speed_mps > 0.0:
            speed = max(0.01, speed + rng.normal(0.0, noise_model.sigma_speed_mps))
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
                speed=speed,
                launch_delay=delay,
                is_dud=is_dud,
            )
        )
    return adjusted


def _apply_noise(
    torpedoes: Sequence[Torpedo],
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> list[Torpedo]:
    return apply_noise_to_torpedoes(torpedoes, noise_model, rng)


def _earliest_hit_time(ship: Ship, torpedo: Torpedo, t_max: float) -> float | None:
    """Return the earliest hit time within range, or None."""

    return torpedo_hit_time(ship, torpedo, t_max)


@dataclass
class DynamicHitState:
    """Mutable hit state for stepping dynamic torpedo/convoy interactions."""

    time: float
    hit_pairs: set[tuple[int, int]]
    torpedo_done: set[int]
    hit_counts: dict[str, int]
    torpedo_hit_times: dict[str, float]
    hit_time_by_ship: dict[str, float]
    hit_events: list["HitEvent"]


@dataclass(frozen=True)
class HitEvent:
    """Single torpedo-to-ship hit event captured during stepping."""

    torpedo_id: str
    ship_id: str
    time_s: float
    hit_x: float
    hit_y: float


@dataclass(frozen=True)
class HitSlowdownSpec:
    """Optional hit-driven speed reduction model."""

    enabled: bool = False
    decay_rate: float = 0.02
    min_factor: float = 0.3


def init_dynamic_hit_state(start_time: float = 0.0) -> DynamicHitState:
    """Initialize hit state for dynamic stepping."""

    return DynamicHitState(
        time=float(start_time),
        hit_pairs=set(),
        torpedo_done=set(),
        hit_counts={},
        torpedo_hit_times={},
        hit_time_by_ship={},
        hit_events=[],
    )


def advance_dynamic_hit_state(
    formation: ConvoyFormation,
    kin: ConvoyKinematics,
    torpedoes: Sequence[Torpedo],
    t_target: float,
    dt: float,
    state: DynamicHitState,
    *,
    safety_margin: float = 0.0,
    max_hits_per_torpedo: int | None = 1,
    hit_slowdown: HitSlowdownSpec | None = None,
) -> DynamicHitState:
    """Advance hit state up to ``t_target`` using the same stepping logic as simulation."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    time = float(state.time)
    t_target = float(t_target)
    if t_target < time:
        state.time = t_target
        return state
    while time <= t_target + 1e-9:
        ship_positions = ship_positions_at(time, formation, kin, dt=dt, motion="independent")
        for torp_idx, torpedo in enumerate(torpedoes):
            if torpedo.is_dud:
                continue
            if torp_idx in state.torpedo_done:
                continue
            if time < torpedo.launch_delay:
                continue
            torp_pos = torpedo.position_at(time)
            for ship_idx, ship in enumerate(formation.ships0):
                key = (torp_idx, ship_idx)
                if key in state.hit_pairs:
                    continue
                radius = ship.effective_hit_radius() + float(safety_margin)
                if distance(ship_positions[ship_idx], torp_pos) <= radius:
                    state.hit_pairs.add(key)
                    state.hit_counts[ship.id] = int(state.hit_counts.get(ship.id, 0)) + 1
                    state.torpedo_hit_times.setdefault(torpedo.id, time)
                    state.hit_time_by_ship.setdefault(ship.id, time)
                    state.hit_events.append(
                        HitEvent(
                            torpedo_id=str(torpedo.id),
                            ship_id=str(ship.id),
                            time_s=float(time),
                            hit_x=float(torp_pos[0]),
                            hit_y=float(torp_pos[1]),
                        )
                    )
                    if max_hits_per_torpedo == 1:
                        state.torpedo_done.add(torp_idx)
                    break
        time += dt
    state.time = t_target
    return state


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
