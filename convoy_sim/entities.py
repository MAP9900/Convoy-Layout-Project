"""Convoy and weapon entity definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .geometry import Point2D


@dataclass
class Ship:
    """Simple surface vessel model with straight-line kinematics."""

    name: str
    position: Point2D
    heading_deg: float
    speed_mps: float
    length_m: float
    beam_m: float

    def advance(self, delta_t: float) -> Point2D:
        """Project the ship forward by ``delta_t`` seconds at constant speed."""

        raise NotImplementedError

    def outline(self) -> Sequence[Point2D]:
        """Return representative hull vertices for coarse collision checks."""

        raise NotImplementedError


@dataclass
class Torpedo:
    """Straight-running torpedo simplified to a 2D kinematic track."""

    origin: Point2D
    heading_deg: float
    speed_mps: float
    warhead_radius_m: float

    def position_at(self, time_s: float) -> Point2D:
        """Return the torpedo position after ``time_s`` seconds."""

        raise NotImplementedError

    def time_to_intercept(self, ship: Ship) -> float:
        """Return time-to-intercept with ``ship`` if on a collision course."""

        raise NotImplementedError


@dataclass
class Convoy:
    """Logical convoy composed of individual ships."""

    ships: list[Ship] = field(default_factory=list)
    layout_name: str | None = None
    spacing_m: float | None = None

    def assign_layout(self, anchor_points: Sequence[Point2D]) -> None:
        """Attach ships to layout anchor positions in order."""

        raise NotImplementedError

    def advance(self, delta_t: float) -> None:
        """Advance every ship by ``delta_t`` seconds."""

        raise NotImplementedError

    def as_dict(self) -> dict:
        """Return a JSON-serializable snapshot for downstream analytics."""

        raise NotImplementedError
