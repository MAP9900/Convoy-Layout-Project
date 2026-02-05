"""Deterministic torpedo spread samplers for attacker configurations."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from convoy_sim.attack_proposals import propose_attack_from_config
from convoy_sim.entities import Ship, Torpedo
from convoy_sim.feasibility import ApproachMode, AttackConstraints, Environment, is_attack_feasible
from convoy_sim.geometry import Vec2, as_vec


def _vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector")
    return arr


def fan_spread(
    u_pos: Vec2 | Sequence[float],
    base_bearing_rad: float,
    n: int,
    spread_rad: float,
    speed: float,
    max_run_time: float,
    *,
    ships: list[Ship] | None = None,
    proposal_cfg: dict[str, Any] | None = None,
    constraints: AttackConstraints | None = None,
    env: Environment | None = None,
    rng: np.random.Generator | None = None,
    max_resample_attempts: int = 100,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from ``u_pos`` spread evenly across ``spread_rad``."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    bearing = base_bearing_rad
    salvo_size = n
    if constraints is not None:
        if ships is None:
            raise ValueError("ships must be provided when constraints are enabled")
        rng = rng or np.random.default_rng()
        cfg = dict(proposal_cfg or {})
        cfg.setdefault("u_boat_pos", origin_vec)
        cfg.setdefault("target_point", "centroid")
        if "approach_mode" not in cfg and "approach_modes" not in cfg:
            cfg["approach_modes"] = list(constraints.allowed_modes) if constraints.allowed_modes else [ApproachMode.ABEAM]
        cfg.setdefault("salvo_size", salvo_size)
        cfg.setdefault("bearing_rad", bearing)
        if "bearing_offset_rad" in cfg:
            cfg.pop("bearing_rad", None)
        last_details = {}
        for _ in range(max_resample_attempts):
            proposal = propose_attack_from_config(ships, cfg, rng)
            feasible, details = is_attack_feasible(ships, proposal, constraints, env=env)
            last_details = details
            if feasible:
                origin_vec = proposal.u_boat_pos
                bearing = proposal.bearing_rad
                salvo_size = proposal.salvo_size
                break
        else:
            raise ValueError(f"Unable to generate feasible attack proposal: {last_details}")

    if salvo_size == 1 or spread_rad == 0.0:
        headings = [bearing]
    else:
        half_spread = spread_rad / 2.0
        headings = [
            bearing - half_spread + (spread_rad * i / (salvo_size - 1))
            for i in range(salvo_size)
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


def parallel_spread(
    u_pos: Vec2 | Sequence[float],
    bearing_rad: float,
    n: int,
    lateral_spacing: float,
    speed: float,
    max_run_time: float,
    *,
    ships: list[Ship] | None = None,
    proposal_cfg: dict[str, Any] | None = None,
    constraints: AttackConstraints | None = None,
    env: Environment | None = None,
    rng: np.random.Generator | None = None,
    max_resample_attempts: int = 100,
) -> list[Torpedo]:
    """Fire ``n`` torpedoes from parallel launchers offset laterally."""

    if n <= 0:
        return []
    origin_vec = _vec(u_pos)
    bearing = bearing_rad
    salvo_size = n
    if constraints is not None:
        if ships is None:
            raise ValueError("ships must be provided when constraints are enabled")
        rng = rng or np.random.default_rng()
        cfg = proposal_cfg or {
            "u_boat_pos": origin_vec,
            "target_point": "centroid",
            "approach_mode": (next(iter(constraints.allowed_modes)) if constraints.allowed_modes else ApproachMode.ABEAM),
            "salvo_size": salvo_size,
            "bearing_rad": bearing,
        }
        last_details = {}
        for _ in range(max_resample_attempts):
            proposal = propose_attack_from_config(ships, cfg, rng)
            feasible, details = is_attack_feasible(ships, proposal, constraints, env=env)
            last_details = details
            if feasible:
                origin_vec = proposal.u_boat_pos
                bearing = proposal.bearing_rad
                salvo_size = proposal.salvo_size
                break
        else:
            raise ValueError(f"Unable to generate feasible attack proposal: {last_details}")

    perp = as_vec(-math.sin(bearing), math.cos(bearing))
    center = (salvo_size - 1) / 2.0
    launches = [
        origin_vec + perp * ((i - center) * lateral_spacing)
        for i in range(salvo_size)
    ]
    return [
        Torpedo(
            id=f"P{i+1:02d}",
            launch_position=launch_pos,
            speed=speed,
            heading_rad=bearing,
            max_run_time=max_run_time,
        )
        for i, launch_pos in enumerate(launches)
    ]
