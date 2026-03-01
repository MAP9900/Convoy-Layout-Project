"""Convoy-level temporal dynamics scaffolding for route legs and zig-zag plans.

We model heading changes as piecewise-constant route legs plus an optional smooth
zig-zag perturbation. This is a convoy-level abstraction shared by ships, not a
per-ship maneuvering model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from convoy_sim.entities import Ship
from convoy_sim.geometry import Vec2, as_vec


def validate_dt(dt: float) -> float:
    """Return ``dt`` as float, raising if non-positive."""

    value = float(dt)
    if value <= 0.0:
        raise ValueError("dt must be > 0")
    return value


def _vec(value: Vec2) -> Vec2:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("Expected a 2D vector with shape (2,)")
    return arr


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)


def _rotation_matrix_inv(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=float)

@dataclass(frozen=True)
class RouteLeg:
    """Single leg with constant heading over a duration."""

    duration_s: float
    heading_rad: float
    speed: float | None = None


@dataclass(frozen=True)
class RoutePlan:
    """Ordered sequence of route legs defining a piecewise-constant heading."""

    legs: list[RouteLeg]

    def heading_at(self, t: float) -> float:
        """Return the heading (rad) at time ``t`` along the route."""

        if not self.legs:
            raise ValueError("RoutePlan.legs must contain at least one leg")
        elapsed = 0.0
        for leg in self.legs:
            elapsed += float(leg.duration_s)
            if t < elapsed:
                return float(leg.heading_rad)
        return float(self.legs[-1].heading_rad)

    def speed_at(self, t: float, default_speed: float) -> float:
        """Return the speed at time ``t``, falling back to ``default_speed``."""

        if not self.legs:
            raise ValueError("RoutePlan.legs must contain at least one leg")
        elapsed = 0.0
        for leg in self.legs:
            elapsed += float(leg.duration_s)
            if t < elapsed:
                return float(default_speed if leg.speed is None else leg.speed)
        last_leg = self.legs[-1]
        return float(default_speed if last_leg.speed is None else last_leg.speed)


@dataclass(frozen=True)
class ZigZagPlan:
    """Convoy-level zig-zag plan that perturbs heading around the base course."""

    enabled: bool = False
    amplitude_rad: float = 0.0
    period_s: float = 0.0
    phase_s: float = 0.0
    waveform: Literal["sine", "triangle"] = "sine"

    def delta_heading_at(self, t: float) -> float:
        """Return the heading offset (rad) at time ``t`` from the zig-zag."""

        if not self.enabled or self.amplitude_rad == 0.0:
            return 0.0
        if self.period_s <= 0.0:
            raise ValueError("ZigZagPlan.period_s must be > 0 when enabled")
        phase = (t + self.phase_s) / self.period_s
        frac = phase - math.floor(phase)
        if self.waveform == "triangle":
            wave = 2.0 * abs(2.0 * frac - 1.0) - 1.0
        else:
            wave = math.sin(2.0 * math.pi * frac)
        return float(self.amplitude_rad) * float(wave)


@dataclass(frozen=True)
class ConvoyKinematics:
    """Convoy-level kinematics from a route plan plus optional zig-zag."""

    route: RoutePlan | None = None
    zigzag: ZigZagPlan | None = None

    def convoy_heading_at(self, t: float, base_heading: float) -> float:
        """Return the convoy heading at time ``t`` (rad)."""

        heading = self.route.heading_at(t) if self.route else float(base_heading)
        if self.zigzag:
            heading += self.zigzag.delta_heading_at(t)
        return float(heading)

    def convoy_speed_at(self, t: float, base_speed: float) -> float:
        """Return the convoy speed at time ``t``."""

        if self.route:
            return self.route.speed_at(t, base_speed)
        return float(base_speed)


@dataclass(frozen=True)
class ConvoyFormation:
    """Convoy formation snapshot that preserves ship offsets in convoy frame."""

    ships0: list[Ship]
    convoy_origin0: Vec2
    convoy_heading0: float
    offsets_convoy_frame: list[Vec2] = field(init=False)

    def __post_init__(self) -> None:
        origin = _vec(self.convoy_origin0)
        rot_inv = _rotation_matrix_inv(self.convoy_heading0)
        offsets: list[Vec2] = []
        for ship in self.ships0:
            relative = _vec(ship.position) - origin
            offsets.append(rot_inv @ relative)
        object.__setattr__(self, "offsets_convoy_frame", offsets)

    def offsets_at_t0(self) -> list[Vec2]:
        """Return fixed ship offsets expressed in the convoy frame at t=0."""

        return list(self.offsets_convoy_frame)


def convoy_pose_at(
    t: float,
    origin0: Vec2,
    heading0: float,
    speed0: float,
    kin: ConvoyKinematics,
    dt: float = 1.0,
) -> tuple[Vec2, float, float]:
    """Integrate convoy translation to time ``t`` using piecewise-constant heading."""

    dt = validate_dt(dt)
    origin = _vec(origin0).copy()
    if t <= 0.0:
        heading_t = kin.convoy_heading_at(0.0, heading0)
        speed_t = kin.convoy_speed_at(0.0, speed0)
        return origin, heading_t, speed_t
    time = 0.0
    while time < t:
        step = min(dt, t - time)
        heading = kin.convoy_heading_at(time, heading0)
        speed = kin.convoy_speed_at(time, speed0)
        direction = as_vec(math.cos(heading), math.sin(heading))
        origin = origin + direction * float(speed) * float(step)
        time += step
    heading_t = kin.convoy_heading_at(t, heading0)
    speed_t = kin.convoy_speed_at(t, speed0)
    return origin, heading_t, speed_t


def ship_positions_at(
    t: float,
    formation: ConvoyFormation,
    kin: ConvoyKinematics,
    dt: float = 1.0,
    *,
    motion: Literal["independent", "rigid"] = "independent",
    speed_factors: dict[str, float] | None = None,
    hit_time_by_ship: dict[str, float] | None = None,
    hit_decay_rate: float | None = None,
    hit_min_factor: float = 0.3,
) -> list[Vec2]:
    """Return ship positions at time ``t`` using convoy-level kinematics."""

    dt = validate_dt(dt)
    if motion == "rigid":
        if formation.ships0:
            base_speed = float(sum(ship.speed for ship in formation.ships0) / len(formation.ships0))
        else:
            base_speed = 0.0
        origin_t, heading_t, _speed_t = convoy_pose_at(
            t,
            formation.convoy_origin0,
            formation.convoy_heading0,
            base_speed,
            kin,
            dt=dt,
        )
        rotation = _rotation_matrix(heading_t)
        return [origin_t + rotation @ offset for offset in formation.offsets_convoy_frame]

    dt = validate_dt(dt)
    positions: list[Vec2] = []
    for ship in formation.ships0:
        pos = _vec(ship.position).copy()
        time = 0.0
        while time < t:
            step = min(dt, t - time)
            heading = kin.convoy_heading_at(time, ship.heading_rad)
            speed = float(ship.speed)
            if hit_time_by_ship and ship.id in hit_time_by_ship and hit_decay_rate is not None:
                hit_time = float(hit_time_by_ship[ship.id])
                if time >= hit_time:
                    elapsed = max(0.0, time - hit_time)
                    factor = float(math.exp(-float(hit_decay_rate) * elapsed))
                    speed *= max(float(hit_min_factor), factor)
            elif speed_factors:
                speed *= float(speed_factors.get(ship.id, 1.0))
            direction = as_vec(math.cos(heading), math.sin(heading))
            pos = pos + direction * speed * float(step)
            time += step
        positions.append(pos)
    return positions
