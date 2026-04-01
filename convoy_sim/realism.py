"""Historical realism helpers for V2 protocol.

This module provides deterministic U-boat motion, attacker partial-observability
sampling, and bounded ship movement overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from convoy_sim.entities import Ship, ShipClass
from convoy_sim.feasibility import Environment
from convoy_sim.geometry import as_vec


@dataclass(frozen=True)
class UBoatLeg:
    """Single deterministic motion leg for U-boat movement."""

    duration_s: float
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True)
class UBoatMotionPlan:
    """Deterministic U-boat kinematic plan."""

    initial_position: np.ndarray
    initial_heading_rad: float
    initial_speed_mps: float
    mode: str = "moving"
    legs: tuple[UBoatLeg, ...] = ()
    launch_time_s: float = 0.0
    turn_rate_limit_rad_s: float | None = None
    accel_limit_mps2: float | None = None

    def __post_init__(self) -> None:
        pos = np.asarray(self.initial_position, dtype=float)
        if pos.shape != (2,):
            raise ValueError("initial_position must be shape (2,)")
        object.__setattr__(self, "initial_position", pos)
        if self.mode not in {"static", "moving"}:
            raise ValueError("mode must be 'static' or 'moving'")
        if self.launch_time_s < 0.0:
            raise ValueError("launch_time_s must be non-negative")
        if self.turn_rate_limit_rad_s is not None and self.turn_rate_limit_rad_s < 0.0:
            raise ValueError("turn_rate_limit_rad_s must be >= 0")
        if self.accel_limit_mps2 is not None and self.accel_limit_mps2 < 0.0:
            raise ValueError("accel_limit_mps2 must be >= 0")

    def launch_position(self) -> np.ndarray:
        return self.position_at(self.launch_time_s)

    def state_at(self, t: float) -> tuple[np.ndarray, float, float]:
        if t <= 0.0 or self.mode == "static":
            return (
                np.asarray(self.initial_position, dtype=float),
                float(self.initial_heading_rad),
                float(self.initial_speed_mps),
            )

        pos = np.asarray(self.initial_position, dtype=float).copy()
        heading = float(self.initial_heading_rad)
        speed = float(self.initial_speed_mps)
        time_left = float(t)

        if not self.legs:
            direction = as_vec(np.cos(heading), np.sin(heading))
            pos = pos + direction * speed * time_left
            return pos, heading, speed

        for leg in self.legs:
            if time_left <= 0.0:
                break
            step = min(float(leg.duration_s), time_left)
            target_heading = float(leg.heading_rad)
            target_speed = float(leg.speed_mps)
            heading = _bounded_transition(
                current=heading,
                target=target_heading,
                step_s=step,
                rate_limit=self.turn_rate_limit_rad_s,
            )
            speed = _bounded_transition(
                current=speed,
                target=target_speed,
                step_s=step,
                rate_limit=self.accel_limit_mps2,
            )
            direction = as_vec(np.cos(heading), np.sin(heading))
            pos = pos + direction * speed * step
            time_left -= step

        if time_left > 0.0:
            direction = as_vec(np.cos(heading), np.sin(heading))
            pos = pos + direction * speed * time_left
        return pos, heading, speed

    def position_at(self, t: float) -> np.ndarray:
        pos, _heading, _speed = self.state_at(t)
        return pos

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, fallback_u_pos: Sequence[float]) -> "UBoatMotionPlan":
        legs_payload = payload.get("legs", [])
        legs = tuple(
            UBoatLeg(
                duration_s=float(item.get("duration_s", 0.0)),
                heading_rad=float(item.get("heading_rad", 0.0)),
                speed_mps=float(item.get("speed_mps", payload.get("initial_speed_mps", payload.get("speed_mps", 0.0)))),
            )
            for item in legs_payload
        )
        return cls(
            initial_position=np.asarray(payload.get("initial_position", fallback_u_pos), dtype=float),
            initial_heading_rad=float(payload.get("initial_heading_rad", 0.0)),
            initial_speed_mps=float(payload.get("initial_speed_mps", payload.get("speed_mps", 0.0))),
            mode=str(payload.get("mode", "moving")),
            legs=legs,
            launch_time_s=float(payload.get("launch_time_s", 0.0)),
            turn_rate_limit_rad_s=(
                None if payload.get("turn_rate_limit_rad_s") is None else float(payload.get("turn_rate_limit_rad_s"))
            ),
            accel_limit_mps2=(
                None if payload.get("accel_limit_mps2") is None else float(payload.get("accel_limit_mps2"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "initial_position": [float(self.initial_position[0]), float(self.initial_position[1])],
            "initial_heading_rad": float(self.initial_heading_rad),
            "initial_speed_mps": float(self.initial_speed_mps),
            "launch_time_s": float(self.launch_time_s),
            "turn_rate_limit_rad_s": self.turn_rate_limit_rad_s,
            "accel_limit_mps2": self.accel_limit_mps2,
            "legs": [
                {
                    "duration_s": float(leg.duration_s),
                    "heading_rad": float(leg.heading_rad),
                    "speed_mps": float(leg.speed_mps),
                }
                for leg in self.legs
            ],
        }


def _bounded_transition(current: float, target: float, step_s: float, rate_limit: float | None) -> float:
    if rate_limit is None:
        return float(target)
    max_delta = float(rate_limit) * float(step_s)
    delta = float(target) - float(current)
    if abs(delta) <= max_delta:
        return float(target)
    return float(current + np.sign(delta) * max_delta)


@dataclass(frozen=True)
class AttackerObservationConfig:
    """Noise model for partial-observability attacker context."""

    bearing_sigma_rad: float = 0.04
    range_sigma_m: float = 120.0
    heading_sigma_rad: float = 0.06
    speed_sigma_mps: float = 0.5
    contact_count_sigma: float = 0.4

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AttackerObservationConfig":
        if not payload:
            return cls()
        return cls(
            bearing_sigma_rad=float(payload.get("bearing_sigma_rad", 0.04)),
            range_sigma_m=float(payload.get("range_sigma_m", 120.0)),
            heading_sigma_rad=float(payload.get("heading_sigma_rad", 0.06)),
            speed_sigma_mps=float(payload.get("speed_sigma_mps", 0.5)),
            contact_count_sigma=float(payload.get("contact_count_sigma", 0.4)),
        )


def build_attacker_observation(
    *,
    ships: list[Ship],
    u_boat_pos: np.ndarray,
    env: Environment | None,
    rng: np.random.Generator,
    cfg: AttackerObservationConfig,
) -> dict[str, Any]:
    """Return noisy attacker-facing context; does not expose full true state."""

    positions = np.array([ship.position for ship in ships], dtype=float)
    centroid = np.mean(positions, axis=0)
    delta = centroid - np.asarray(u_boat_pos, dtype=float)
    true_range = float(np.linalg.norm(delta))
    true_bearing = float(np.arctan2(delta[1], delta[0]))
    true_heading = float(np.arctan2(np.mean([np.sin(ship.heading_rad) for ship in ships]), np.mean([np.cos(ship.heading_rad) for ship in ships])))
    true_speed = float(np.mean([ship.speed for ship in ships]))
    true_contacts = float(len(ships))

    obs_bearing = float(true_bearing + rng.normal(0.0, cfg.bearing_sigma_rad))
    obs_range = float(max(0.0, true_range + rng.normal(0.0, cfg.range_sigma_m)))
    obs_heading = float(true_heading + rng.normal(0.0, cfg.heading_sigma_rad))
    obs_speed = float(max(0.0, true_speed + rng.normal(0.0, cfg.speed_sigma_mps)))
    obs_contacts = int(max(0, round(true_contacts + rng.normal(0.0, cfg.contact_count_sigma))))

    class_counts = {
        "freighter": int(sum(1 for ship in ships if ship.ship_class == ShipClass.FREIGHTER)),
        "tanker": int(sum(1 for ship in ships if ship.ship_class == ShipClass.TANKER)),
        "escort": int(sum(1 for ship in ships if ship.ship_class == ShipClass.ESCORT)),
        "decoy": int(sum(1 for ship in ships if ship.ship_class == ShipClass.DECOY)),
    }

    return {
        "estimated_bearing_rad": obs_bearing,
        "estimated_range_m": obs_range,
        "estimated_convoy_heading_rad": obs_heading,
        "estimated_convoy_speed_mps": obs_speed,
        "estimated_contact_count": obs_contacts,
        "class_confidence_counts": class_counts,
        "environment": (
            None
            if env is None
            else {
                "time_of_day": env.time_of_day,
                "visibility_m": float(env.visibility_m),
                "sea_state": int(env.sea_state),
            }
        ),
        "observation_quality": {
            "bearing_sigma_rad": float(cfg.bearing_sigma_rad),
            "range_sigma_m": float(cfg.range_sigma_m),
            "heading_sigma_rad": float(cfg.heading_sigma_rad),
            "speed_sigma_mps": float(cfg.speed_sigma_mps),
            "contact_count_sigma": float(cfg.contact_count_sigma),
        },
    }


@dataclass(frozen=True)
class ShipMovementRealismConfig:
    """Bounded overlay around formation-level motion."""

    position_jitter_std_m: float = 0.0
    heading_jitter_std_rad: float = 0.0
    deviation_offset_cap_m: float = 0.0
    enable_slot_swaps: bool = False
    max_swap_fraction: float = 0.0
    freighter_scale: float = 1.0
    tanker_scale: float = 0.9
    escort_scale: float = 0.6
    decoy_scale: float = 1.1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ShipMovementRealismConfig":
        if not payload:
            return cls()
        return cls(
            position_jitter_std_m=float(payload.get("position_jitter_std_m", 0.0)),
            heading_jitter_std_rad=float(payload.get("heading_jitter_std_rad", 0.0)),
            deviation_offset_cap_m=float(payload.get("deviation_offset_cap_m", 0.0)),
            enable_slot_swaps=bool(payload.get("enable_slot_swaps", False)),
            max_swap_fraction=float(payload.get("max_swap_fraction", 0.0)),
            freighter_scale=float(payload.get("freighter_scale", 1.0)),
            tanker_scale=float(payload.get("tanker_scale", 0.9)),
            escort_scale=float(payload.get("escort_scale", 0.6)),
            decoy_scale=float(payload.get("decoy_scale", 1.1)),
        )

    def inactive(self) -> bool:
        return (
            self.position_jitter_std_m <= 0.0
            and self.heading_jitter_std_rad <= 0.0
            and self.deviation_offset_cap_m <= 0.0
            and (not self.enable_slot_swaps or self.max_swap_fraction <= 0.0)
        )


def apply_ship_movement_realism(
    ships: list[Ship],
    *,
    rng: np.random.Generator,
    cfg: ShipMovementRealismConfig,
) -> list[Ship]:
    """Apply bounded per-ship overlays while preserving formation-level baseline."""

    if not ships or cfg.inactive():
        return ships

    for ship in ships:
        scale = _class_scale(ship.ship_class, cfg)
        if cfg.position_jitter_std_m > 0.0:
            offset = rng.normal(0.0, cfg.position_jitter_std_m * scale, size=2)
            if cfg.deviation_offset_cap_m > 0.0:
                mag = float(np.linalg.norm(offset))
                cap = float(cfg.deviation_offset_cap_m)
                if mag > cap and mag > 0.0:
                    offset = offset * (cap / mag)
            ship.position = np.asarray(ship.position, dtype=float) + np.asarray(offset, dtype=float)
        if cfg.heading_jitter_std_rad > 0.0:
            ship.heading_rad = float(ship.heading_rad + rng.normal(0.0, cfg.heading_jitter_std_rad * scale))

    if cfg.enable_slot_swaps and cfg.max_swap_fraction > 0.0 and len(ships) >= 2:
        n_swaps = int(max(0, min(len(ships) // 2, round(len(ships) * cfg.max_swap_fraction))))
        for _ in range(n_swaps):
            i, j = rng.choice(len(ships), size=2, replace=False)
            pos_i = np.asarray(ships[i].position, dtype=float).copy()
            ships[i].position = np.asarray(ships[j].position, dtype=float)
            ships[j].position = pos_i
    return ships


def _class_scale(ship_class: ShipClass, cfg: ShipMovementRealismConfig) -> float:
    if ship_class == ShipClass.ESCORT:
        return float(cfg.escort_scale)
    if ship_class == ShipClass.TANKER:
        return float(cfg.tanker_scale)
    if ship_class == ShipClass.DECOY:
        return float(cfg.decoy_scale)
    return float(cfg.freighter_scale)

