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

from convoy_sim.entities import Ship
from convoy_sim.geometry import Vec2, as_vec, distance


APPROACH_CONE_HALF_WIDTH_DEG = 30.0
VIS_REF_M = 5000.0
SEA_STATE_REF = 5.0
SEA_STATE_FACTOR_SCALE = 0.05
ESCORT_DECAY_LEN_M = 2000.0


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


def compute_convoy_reference(ships: list[Ship], t_ref: float = 0.0) -> dict[str, Any]:
    """Return convoy centroid, mean heading/speed, and bounding box extents."""

    if not ships:
        raise ValueError("ships must be non-empty")
    positions = np.array([ship.position_at(t_ref) for ship in ships], dtype=float)
    centroid = np.mean(positions, axis=0)
    mean_speed = float(np.mean([ship.speed for ship in ships]))
    headings = np.array([ship.heading_rad for ship in ships], dtype=float)
    mean_heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    cos_h = np.cos(-mean_heading)
    sin_h = np.sin(-mean_heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
    rotated = (rotation @ (positions - centroid).T).T
    bbox_along = float(np.max(rotated[:, 0]) - np.min(rotated[:, 0]))
    bbox_across = float(np.max(rotated[:, 1]) - np.min(rotated[:, 1]))
    return {
        "centroid": centroid,
        "heading_rad": mean_heading,
        "speed": mean_speed,
        "bbox_along": bbox_along,
        "bbox_across": bbox_across,
    }


def attack_in_range(
    proposal: AttackProposal,
    convoy_centroid: Vec2,
    constraints: AttackConstraints,
) -> bool:
    """Check if the proposal launch point is within range of the convoy centroid."""

    range_m = distance(proposal.u_boat_pos, convoy_centroid)
    return constraints.min_range <= range_m <= constraints.max_range


def violates_escort_exclusion(proposal: AttackProposal, zones: list[EscortZone]) -> bool:
    """Return True if the proposal starts inside any hard exclusion zone."""

    for zone in zones:
        if zone.hard_exclusion:
            if distance(proposal.u_boat_pos, zone.center) <= zone.radius:
                return True
    return False


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def _angle_close(angle_a: float, angle_b: float, tolerance_rad: float) -> bool:
    return abs(_wrap_angle(angle_a - angle_b)) <= tolerance_rad


def approach_mode_feasible(
    proposal: AttackProposal,
    convoy_heading_rad: float,
    constraints: AttackConstraints,
) -> bool:
    """Check if the proposal bearing fits allowed approach cones."""

    if proposal.approach_mode not in constraints.allowed_modes:
        return False
    tolerance = np.deg2rad(APPROACH_CONE_HALF_WIDTH_DEG)
    bearing = proposal.bearing_rad
    if proposal.approach_mode == ApproachMode.BOW_ON:
        target = _wrap_angle(convoy_heading_rad + np.pi)
        return _angle_close(bearing, target, tolerance)
    if proposal.approach_mode == ApproachMode.STERN_CHASE:
        return _angle_close(bearing, convoy_heading_rad, tolerance)
    if proposal.approach_mode == ApproachMode.ABEAM:
        starboard = _wrap_angle(convoy_heading_rad + np.pi / 2.0)
        port = _wrap_angle(convoy_heading_rad - np.pi / 2.0)
        return _angle_close(bearing, starboard, tolerance) or _angle_close(bearing, port, tolerance)
    return False


def is_attack_feasible(
    ships: list[Ship],
    proposal: AttackProposal,
    constraints: AttackConstraints,
    env: Environment | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate proposal feasibility and return (feasible, details)."""

    reference = compute_convoy_reference(ships)
    failed_checks: list[str] = []
    range_m = distance(proposal.u_boat_pos, reference["centroid"])
    risk_score = None
    if not attack_in_range(proposal, reference["centroid"], constraints):
        failed_checks.append("range")
    if violates_escort_exclusion(proposal, constraints.escort_zones):
        failed_checks.append("escort_exclusion")
    if not approach_mode_feasible(proposal, reference["heading_rad"], constraints):
        failed_checks.append("approach_mode")
    if env is not None:
        risk_score = detection_risk_score(proposal, constraints.escort_zones, env)
        if constraints.enable_soft_risk and risk_score > constraints.max_allowed_risk:
            failed_checks.append("risk")
    details = {
        "failed_checks": failed_checks,
        "range_m": range_m,
        "convoy_reference": reference,
        "detection_risk": risk_score,
        "env": env,
    }
    return len(failed_checks) == 0, details


def detection_risk_score(
    proposal: AttackProposal,
    escort_zones: list[EscortZone],
    env: Environment,
) -> float:
    """Compute a simple detection risk score for an attack proposal.

    Sea state adjustment assumes rougher seas reduce detection slightly.
    """

    base = float(env.detection_risk_scale)
    visibility_factor = float(np.clip(env.visibility_m / VIS_REF_M, 0.25, 4.0))
    time_of_day_factor = 1.5 if env.time_of_day == "day" else 0.8
    sea_state_delta = SEA_STATE_FACTOR_SCALE * (env.sea_state - SEA_STATE_REF)
    sea_state_factor = float(np.clip(1.0 + sea_state_delta, 0.5, 1.5))

    escort_proximity = 0.0
    for zone in escort_zones:
        d = distance(proposal.u_boat_pos, zone.center)
        escort_proximity += zone.risk_weight * float(np.exp(-d / ESCORT_DECAY_LEN_M))
    return base * visibility_factor * time_of_day_factor * sea_state_factor * (1.0 + escort_proximity)
