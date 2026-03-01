"""Convoy and weapon entity definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Sequence

import numpy as np

from convoy_sim.geometry import (
    Vec2,
    as_vec,
    closest_approach_time,
    min_distance_over_interval,
    step_position,
)


def _vec(value: Vec2 | Sequence[float]) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector with shape (2,)")
    return arr


class ShipClass(str, Enum):
    """Basic ship taxonomy for heterogeneous convoy modeling."""

    FREIGHTER = "freighter"
    TANKER = "tanker"
    ESCORT = "escort"
    DECOY = "decoy"


@dataclass
class Ship:
    """Simple surface vessel model with straight-line kinematics."""

    id: str
    position: Vec2
    speed: float
    heading_rad: float
    length: float
    beam: float
    ship_class: ShipClass = field(default=ShipClass.FREIGHTER)
    value_weight: float = 1.0
    hit_radius: float | None = None

    def __post_init__(self) -> None:
        self.position = _vec(self.position)

    def velocity_vec(self) -> Vec2:
        """Return the instantaneous velocity vector in meters per second."""

        vx = math.cos(self.heading_rad) * self.speed
        vy = math.sin(self.heading_rad) * self.speed
        return as_vec(vx, vy)

    def position_at(self, t: float) -> Vec2:
        """Return the ship position after ``t`` seconds of straight-line motion."""

        return step_position(self.position, self.velocity_vec(), t)

    def advance(self, delta_t: float) -> Vec2:
        """Project the ship forward by ``delta_t`` seconds at constant speed."""

        self.position = self.position_at(delta_t)
        return self.position

    def outline(self) -> Sequence[Vec2]:
        """Return representative hull vertices for coarse collision checks."""

        half_length = self.length / 2.0
        half_beam = self.beam / 2.0
        corners = np.array(
            [
                (-half_length, -half_beam),
                (-half_length, half_beam),
                (half_length, half_beam),
                (half_length, -half_beam),
            ],
            dtype=float,)
        cos_a = math.cos(self.heading_rad)
        sin_a = math.sin(self.heading_rad)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        return [self.position + rotation @ corner for corner in corners]

    def effective_hit_radius(self) -> float:
        """Return the effective hit radius used for collision checks."""

        if self.hit_radius is not None:
            return float(self.hit_radius)
        half_length = self.length / 2.0
        half_beam = self.beam / 2.0
        return float(np.hypot(half_length, half_beam))


@dataclass
class Torpedo:
    """Straight-running torpedo simplified to a 2D kinematic track."""

    id: str
    launch_position: Vec2
    speed: float
    heading_rad: float
    max_run_time: float
    launch_delay: float = 0.0
    is_dud: bool = False

    def __post_init__(self) -> None:
        self.launch_position = _vec(self.launch_position)
        self.launch_delay = max(0.0, float(self.launch_delay))

    def velocity_vec(self) -> Vec2:
        """Return the constant torpedo velocity vector."""

        vx = math.cos(self.heading_rad) * self.speed
        vy = math.sin(self.heading_rad) * self.speed
        return as_vec(vx, vy)

    def position_at(self, time_s: float) -> Vec2:
        """Return the torpedo position after ``time_s`` seconds."""

        if time_s <= self.launch_delay:
            return self.launch_position
        effective_t = min(time_s - self.launch_delay, max(0.0, self.max_run_time - self.launch_delay))
        return step_position(self.launch_position, self.velocity_vec(), effective_t)

    def time_to_intercept(self, ship: Ship) -> float:
        """Return time-to-intercept with ``ship`` if on a collision course."""

        t = closest_approach_time(
            self.launch_position,
            self.velocity_vec(),
            ship.position,
            ship.velocity_vec(),)
        t += self.launch_delay
        return min(t, self.max_run_time)


@dataclass
class Convoy:
    """Logical convoy composed of individual ships."""

    ships: list[Ship] = field(default_factory=list)
    layout_name: str | None = None
    spacing_m: float | None = None

    def assign_layout(self, anchor_points: Sequence[Vec2]) -> None:
        """Attach ships to layout anchor positions in order."""

        if len(anchor_points) < len(self.ships):
            raise ValueError("Not enough anchor points for convoy ships")
        for ship, anchor in zip(self.ships, anchor_points):
            ship.position = _vec(anchor)

    def advance(self, delta_t: float) -> None:
        """Advance every ship by ``delta_t`` seconds."""

        for ship in self.ships:
            ship.advance(delta_t)

    def as_dict(self) -> dict:
        """Return a JSON-serializable snapshot for downstream analytics."""

        return {
            "layout_name": self.layout_name,
            "spacing_m": self.spacing_m,
            "ships": [
                {
                    "id": ship.id,
                    "position": ship.position.tolist(),
                    "speed": ship.speed,
                    "heading_rad": ship.heading_rad,
                    "length": ship.length,
                    "beam": ship.beam,
                }
                for ship in self.ships
            ],
        }


def torpedo_hits_ship(ship: Ship, torpedo: Torpedo, t_max: float, safety_margin: float = 0.0) -> bool:
    """Return True if the torpedo track intersects the ship footprint within ``t_max`` seconds.

    The ship is approximated as a circle whose radius equals half the greater
    of ``length`` and ``beam`` plus an optional ``safety_margin`` in meters.
    """

    if t_max <= 0 or torpedo.is_dud:
        return False
    window = min(float(t_max), float(torpedo.max_run_time))
    ship_radius = ship.effective_hit_radius() + float(safety_margin)
    segments = []
    delay = min(torpedo.launch_delay, window)
    if delay > 0.0:
        segments.append(
            min_distance_over_interval(
                ship.position,
                ship.velocity_vec(),
                torpedo.launch_position,
                as_vec(0.0, 0.0),
                0.0,
                delay,
            )
        )
    remaining = max(0.0, window - torpedo.launch_delay)
    if remaining > 0.0:
        ship_start = ship.position_at(torpedo.launch_delay)
        torp_start = torpedo.position_at(torpedo.launch_delay)
        segments.append(
            min_distance_over_interval(
                ship_start,
                ship.velocity_vec(),
                torp_start,
                torpedo.velocity_vec(),
                0.0,
                remaining,
            )
        )
    if not segments:
        return False
    min_dist = min(segments)
    return min_dist <= ship_radius
