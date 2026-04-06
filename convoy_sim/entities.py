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
    """Torpedo track with optional gyro deflection after tube exit."""

    id: str
    launch_position: Vec2
    speed: float
    heading_rad: float
    max_run_time: float
    launch_delay: float = 0.0
    is_dud: bool = False
    launch_heading_rad: float | None = None
    gyro_turn_distance_m: float = 0.0

    def __post_init__(self) -> None:
        self.launch_position = _vec(self.launch_position)
        self.launch_delay = max(0.0, float(self.launch_delay))
        if self.launch_heading_rad is not None:
            self.launch_heading_rad = float(self.launch_heading_rad)
        self.gyro_turn_distance_m = max(0.0, float(self.gyro_turn_distance_m))

    def initial_heading_rad(self) -> float:
        """Return the heading used as the torpedo leaves the tube."""

        if self.launch_heading_rad is None:
            return float(self.heading_rad)
        return float(self.launch_heading_rad)

    def uses_gyro_turn(self) -> bool:
        """Return whether this torpedo follows a two-segment gyro path."""

        if self.launch_heading_rad is None or self.gyro_turn_distance_m <= 0.0:
            return False
        delta = math.atan2(
            math.sin(float(self.heading_rad) - float(self.launch_heading_rad)),
            math.cos(float(self.heading_rad) - float(self.launch_heading_rad)),
        )
        return abs(delta) > 1e-12

    def active_run_duration_s(self) -> float:
        """Return time available for post-launch motion under current semantics."""

        return max(0.0, float(self.max_run_time) - float(self.launch_delay))

    def gyro_turn_time_s(self) -> float:
        """Return absolute time when the gyro deflection is applied."""

        if not self.uses_gyro_turn() or self.speed <= 0.0:
            return float(self.launch_delay)
        straight_run_s = float(self.gyro_turn_distance_m) / float(self.speed)
        straight_run_s = min(straight_run_s, self.active_run_duration_s())
        return float(self.launch_delay + straight_run_s)

    def velocity_vec(self, time_s: float | None = None) -> Vec2:
        """Return the torpedo velocity vector for the requested phase."""

        heading = float(self.heading_rad)
        if time_s is not None and self.uses_gyro_turn() and time_s < self.gyro_turn_time_s():
            heading = self.initial_heading_rad()
        vx = math.cos(heading) * self.speed
        vy = math.sin(heading) * self.speed
        return as_vec(vx, vy)

    def position_at(self, time_s: float) -> Vec2:
        """Return the torpedo position after ``time_s`` seconds."""

        if time_s <= self.launch_delay:
            return self.launch_position
        effective_t = min(time_s - self.launch_delay, self.active_run_duration_s())
        if not self.uses_gyro_turn():
            return step_position(self.launch_position, self.velocity_vec(), effective_t)

        straight_run_s = min(
            float(self.gyro_turn_distance_m) / max(float(self.speed), 1e-12),
            self.active_run_duration_s(),
        )
        if effective_t <= straight_run_s:
            return step_position(self.launch_position, self.velocity_vec(self.launch_delay), effective_t)

        turn_position = step_position(
            self.launch_position,
            self.velocity_vec(self.launch_delay),
            straight_run_s,
        )
        return step_position(turn_position, self.velocity_vec(self.gyro_turn_time_s()), effective_t - straight_run_s)

    def motion_segments(self, t_end: float) -> list[tuple[float, float, Vec2, Vec2]]:
        """Return absolute-time motion segments up to ``t_end``.

        Each segment is `(start_t, end_t, start_pos, velocity_vec)`.
        """

        window_end = min(float(t_end), float(self.max_run_time))
        if window_end <= 0.0:
            return []

        segments: list[tuple[float, float, Vec2, Vec2]] = []
        delay_end = min(window_end, float(self.launch_delay))
        if delay_end > 0.0:
            segments.append((0.0, delay_end, self.launch_position, as_vec(0.0, 0.0)))
        if window_end <= float(self.launch_delay):
            return segments

        if not self.uses_gyro_turn():
            segments.append(
                (
                    float(self.launch_delay),
                    window_end,
                    self.launch_position,
                    self.velocity_vec(),
                )
            )
            return segments

        turn_time = min(self.gyro_turn_time_s(), window_end)
        launch_velocity = self.velocity_vec(self.launch_delay)
        if turn_time > float(self.launch_delay):
            segments.append(
                (
                    float(self.launch_delay),
                    float(turn_time),
                    self.launch_position,
                    launch_velocity,
                )
            )
        if window_end > turn_time:
            turn_position = self.position_at(turn_time)
            segments.append(
                (
                    float(turn_time),
                    window_end,
                    turn_position,
                    self.velocity_vec(turn_time),
                )
            )
        return segments

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


def _earliest_entry_time(
    relative_pos: Vec2,
    relative_vel: Vec2,
    radius: float,
    duration_s: float,
) -> float | None:
    """Return earliest local time when distance falls within ``radius``."""

    radius_sq = float(radius) * float(radius)
    c = float(np.dot(relative_pos, relative_pos) - radius_sq)
    if c <= 0.0:
        return 0.0
    a = float(np.dot(relative_vel, relative_vel))
    if a <= 1e-12:
        return None
    b = 2.0 * float(np.dot(relative_pos, relative_vel))
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    if t1 < 0.0 or t0 > duration_s:
        return None
    return float(max(0.0, t0))


def torpedo_hit_time(ship: Ship, torpedo: Torpedo, t_max: float, safety_margin: float = 0.0) -> float | None:
    """Return earliest hit time if torpedo track intersects ship within ``t_max``."""

    if t_max <= 0.0 or torpedo.is_dud:
        return None
    ship_radius = ship.effective_hit_radius() + float(safety_margin)
    for start_t, end_t, torp_start, torp_velocity in torpedo.motion_segments(float(t_max)):
        duration = float(end_t - start_t)
        if duration <= 0.0:
            continue
        ship_start = ship.position_at(start_t)
        rel_pos = np.asarray(torp_start, dtype=float) - np.asarray(ship_start, dtype=float)
        rel_vel = np.asarray(torp_velocity, dtype=float) - np.asarray(ship.velocity_vec(), dtype=float)
        hit_local_t = _earliest_entry_time(rel_pos, rel_vel, ship_radius, duration)
        if hit_local_t is not None:
            return float(start_t + hit_local_t)
    return None


def torpedo_hits_ship(ship: Ship, torpedo: Torpedo, t_max: float, safety_margin: float = 0.0) -> bool:
    """Return True if the torpedo track intersects the ship footprint within ``t_max`` seconds."""

    return torpedo_hit_time(ship, torpedo, t_max, safety_margin=safety_margin) is not None
