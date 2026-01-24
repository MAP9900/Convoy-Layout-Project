"""Attack proposal generation helpers."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from convoy_sim.entities import Ship
from convoy_sim.feasibility import AttackProposal, ApproachMode, compute_convoy_reference
from convoy_sim.geometry import as_vec

AimpointStrategy = Literal["centroid", "max_value", "value_weighted_centroid"]


def _sample_u_boat_position(cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    if "u_boat_pos" in cfg:
        return np.asarray(cfg["u_boat_pos"], dtype=float)
    if "u_boat_box" in cfg:
        xmin, xmax, ymin, ymax = cfg["u_boat_box"]
        return np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax),
        ], dtype=float)
    raise ValueError("cfg must provide u_boat_pos or u_boat_box")


def choose_aimpoint(
    ships: list[Ship],
    strategy: AimpointStrategy,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select an aimpoint based on ship values."""

    if strategy == "centroid":
        return compute_convoy_reference(ships)["centroid"]
    if strategy == "max_value":
        ship = max(ships, key=lambda s: s.value_weight)
        return ship.position
    if strategy == "value_weighted_centroid":
        weights = np.array([ship.value_weight for ship in ships], dtype=float)
        positions = np.array([ship.position for ship in ships], dtype=float)
        total = float(np.sum(weights))
        if total <= 0.0:
            return compute_convoy_reference(ships)["centroid"]
        return np.average(positions, axis=0, weights=weights)
    raise ValueError(f"Unknown aimpoint strategy: {strategy}")


def _resolve_target_point(
    cfg: dict[str, Any],
    ships: list[Ship],
    centroid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    target = cfg.get("target_point", "centroid")
    if isinstance(target, str) and target == "centroid":
        strategy = cfg.get("aimpoint_strategy", "centroid")
        return choose_aimpoint(ships, strategy, rng)
    return np.asarray(target, dtype=float)


def _sample_approach_mode(cfg: dict[str, Any], rng: np.random.Generator) -> ApproachMode:
    if "approach_mode" in cfg:
        return ApproachMode(cfg["approach_mode"])
    modes = cfg.get("approach_modes", [ApproachMode.ABEAM, ApproachMode.BOW_ON, ApproachMode.STERN_CHASE])
    mode = modes[int(rng.integers(0, len(modes)))]
    return ApproachMode(mode)


def _resolve_bearing(cfg: dict[str, Any], u_boat_pos: np.ndarray, target_point: np.ndarray, convoy_heading: float) -> float:
    if "bearing_rad" in cfg:
        return float(cfg["bearing_rad"])
    if "bearing_offset_rad" in cfg:
        return float(convoy_heading + cfg["bearing_offset_rad"])
    dx, dy = target_point - u_boat_pos
    return float(math.atan2(dy, dx))


def _sample_salvo_size(cfg: dict[str, Any], rng: np.random.Generator) -> int:
    if "salvo_size" in cfg:
        return int(cfg["salvo_size"])
    if "salvo_size_range" in cfg:
        low, high = cfg["salvo_size_range"]
        return int(rng.integers(int(low), int(high) + 1))
    return 1


def propose_attack_from_config(
    ships: list[Ship],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> AttackProposal:
    """Generate an AttackProposal from a minimal config dictionary."""

    reference = compute_convoy_reference(ships)
    centroid = reference["centroid"]
    convoy_heading = reference["heading_rad"]

    u_boat_pos = _sample_u_boat_position(cfg, rng)
    target_point = _resolve_target_point(cfg, ships, centroid, rng)
    approach_mode = _sample_approach_mode(cfg, rng)
    bearing_rad = _resolve_bearing(cfg, u_boat_pos, target_point, convoy_heading)
    salvo_size = _sample_salvo_size(cfg, rng)

    return AttackProposal(
        u_boat_pos=u_boat_pos,
        target_point=target_point,
        bearing_rad=bearing_rad,
        approach_mode=approach_mode,
        salvo_size=salvo_size,
        metadata=cfg.get("metadata", {}),
    )
