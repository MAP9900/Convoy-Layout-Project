"""Attack feasibility and detection constraint scaffolding.

These dataclasses describe optional geometric and environmental constraints
that can filter attacker proposals before torpedoes are sampled. Hard exclusion
zones are no-go regions for attack initiation. Soft risk represents a numeric
penalty that can be compared against a maximum threshold when enabled.
Approach modes capture the relative geometry between attacker bearing and the
convoy's heading (abeam, bow-on, stern chase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


class ApproachMode(str, Enum):
    """Relative geometry between attacker bearing and convoy heading."""

    ABEAM = "abeam"
    BOW_ON = "bow_on"
    STERN_CHASE = "stern_chase"


@dataclass(frozen=True)
class Environment:
    """Environmental conditions that can influence detection and feasibility."""

    time_of_day: Literal["day", "night"]
    visibility_m: float
    sea_state: int
    detection_risk_scale: float = 1.0


@dataclass(frozen=True)
class EscortZone:
    """Circular escort zone that may exclude or penalize attack positions."""

    center: np.ndarray
    radius: float
    hard_exclusion: bool
    risk_weight: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,):
            raise ValueError("center must be a 2D vector")
        object.__setattr__(self, "center", center)


@dataclass
class AttackConstraints:
    """Optional constraints applied to attack proposals.

    Defaults are permissive to preserve backward compatibility.
    """

    min_range: float = 0.0
    max_range: float = float("inf")
    allowed_modes: set[ApproachMode] = field(
        default_factory=lambda: {
            ApproachMode.ABEAM,
            ApproachMode.BOW_ON,
            ApproachMode.STERN_CHASE,
        }
    )
    escort_zones: list[EscortZone] = field(default_factory=list)
    enable_soft_risk: bool = False
    max_allowed_risk: float = float("inf")
    notes: str = ""


@dataclass(frozen=True)
class AttackProposal:
    """Attack intent description prior to torpedo sampling."""

    u_boat_pos: np.ndarray
    target_point: np.ndarray
    bearing_rad: float
    approach_mode: ApproachMode
    salvo_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        u_boat_pos = np.asarray(self.u_boat_pos, dtype=float)
        target_point = np.asarray(self.target_point, dtype=float)
        if u_boat_pos.shape != (2,) or target_point.shape != (2,):
            raise ValueError("u_boat_pos and target_point must be 2D vectors")
        object.__setattr__(self, "u_boat_pos", u_boat_pos)
        object.__setattr__(self, "target_point", target_point)
