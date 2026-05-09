"""Geometry plausibility audit helpers for attack profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship


@dataclass(frozen=True)
class AuditThresholds:
    """Thresholds for profile plausibility rules."""

    near_range_m: float = 2500.0
    mid_range_m: float = 4500.0
    near_max_error_deg: float = 15.0
    mid_max_error_deg: float = 30.0
    far_max_error_deg: float = 45.0
    fan_margin_deg: float = 3.0


def _wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _convoy_centroid(ships: Iterable[Ship]) -> np.ndarray:
    positions = np.array([ship.position for ship in ships], dtype=float)
    if positions.size == 0:
        raise ValueError("ships must be non-empty")
    return np.mean(positions, axis=0)


def _error_threshold_deg(range_m: float, cfg: AuditThresholds) -> float:
    if range_m <= cfg.near_range_m:
        return cfg.near_max_error_deg
    if range_m <= cfg.mid_range_m:
        return cfg.mid_max_error_deg
    return cfg.far_max_error_deg


def audit_attack_profiles(
    profiles: list[AttackProfile],
    ships: list[Ship],
    *,
    intents: Sequence[Mapping[str, Any]] | None = None,
    thresholds: AuditThresholds | None = None,
) -> list[dict[str, Any]]:
    """Return plausibility audit rows for attack profiles against a convoy layout."""

    cfg = thresholds or AuditThresholds()
    centroid = _convoy_centroid(ships)
    if intents is not None and len(intents) != len(profiles):
        raise ValueError("intents length must match profiles length")
    rows: list[dict[str, Any]] = []

    for index, profile in enumerate(profiles):
        u_pos = np.asarray(profile.u_pos, dtype=float)
        dx_centroid = float(centroid[0] - u_pos[0])
        dy_centroid = float(centroid[1] - u_pos[1])
        centroid_range_m = float(np.hypot(dx_centroid, dy_centroid))
        centroid_bearing_rad = float(np.arctan2(dy_centroid, dx_centroid))

        intent = dict(intents[index]) if intents is not None else None
        if intent is None:
            target_point = centroid
            target_zone_id = ""
            target_zone_kind = ""
            approach_lane = ""
            approach_side = ""
            intended_label = ""
        else:
            target_point = np.asarray(intent["target_point"], dtype=float)
            if target_point.shape != (2,):
                raise ValueError("intent target_point must be a 2D vector")
            target_zone_id = str(intent.get("target_zone_id", ""))
            target_zone_kind = str(intent.get("target_zone_kind", ""))
            approach_lane = str(intent.get("approach_lane", ""))
            approach_side = str(intent.get("approach_side", ""))
            intended_label = str(intent.get("intended_label", ""))

        dx = float(target_point[0] - u_pos[0])
        dy = float(target_point[1] - u_pos[1])
        range_to_target_m = float(np.hypot(dx, dy))
        intent_bearing_rad = float(np.arctan2(dy, dx))

        if profile.mode == "fan":
            active_bearing = float(profile.base_bearing_rad)
        else:
            active_bearing = float(profile.bearing_rad)

        bearing_error_rad = _wrap_to_pi(active_bearing - intent_bearing_rad)
        bearing_error_deg = float(np.degrees(abs(bearing_error_rad)))

        threshold_deg = _error_threshold_deg(range_to_target_m, cfg)
        flags: list[str] = []

        if bearing_error_deg > threshold_deg:
            if range_to_target_m <= cfg.near_range_m:
                flags.append("near_range_large_error")
            elif range_to_target_m <= cfg.mid_range_m:
                flags.append("mid_range_large_error")
            else:
                flags.append("far_range_large_error")

        if profile.mode == "fan":
            half_spread_deg = 0.5 * float(np.degrees(profile.spread_rad))
            if bearing_error_deg > (half_spread_deg + cfg.fan_margin_deg):
                flags.append("fan_not_covering_target")

        if range_to_target_m <= cfg.near_range_m and bearing_error_deg >= 90.0:
            flags.append("near_range_opposite_direction")

        if "near_range_opposite_direction" in flags:
            suggested_label = "implausible_geometry"
        elif "near_range_large_error" in flags or "fan_not_covering_target" in flags:
            suggested_label = "implausible_geometry"
        elif bearing_error_deg <= 8.0:
            suggested_label = "credible_hit_threat"
        else:
            suggested_label = "credible_near_miss"

        severity = min(
            100.0,
            bearing_error_deg * 1.2
            + (20.0 if "fan_not_covering_target" in flags else 0.0)
            + (40.0 if "near_range_opposite_direction" in flags else 0.0),
        )

        rows.append(
            {
                "profile_id": profile.profile_id,
                "name": profile.name,
                "mode": profile.mode,
                "u_pos_x": float(u_pos[0]),
                "u_pos_y": float(u_pos[1]),
                "range_to_centroid_m": centroid_range_m,
                "centroid_bearing_rad": centroid_bearing_rad,
                "range_to_target_m": range_to_target_m,
                "target_bearing_rad": intent_bearing_rad,
                "intent_bearing_rad": intent_bearing_rad,
                "active_bearing_rad": active_bearing,
                "bearing_error_deg": bearing_error_deg,
                "target_bearing_error_deg": bearing_error_deg,
                "spread_deg": float(np.degrees(profile.spread_rad)),
                "flag_count": int(len(flags)),
                "flags": flags,
                "severity": float(severity),
                "suggested_label": suggested_label,
                "target_zone_id": target_zone_id,
                "target_zone_kind": target_zone_kind,
                "approach_lane": approach_lane,
                "approach_side": approach_side,
                "intended_label": intended_label,
            }
        )

    rows.sort(key=lambda r: (-r["severity"], r["profile_id"]))
    return rows
