"""Dynamic outcome audit helpers for generated attack-profile datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.dynamics import (
    ConvoyFormation,
    ConvoyKinematics,
    RouteLeg,
    RoutePlan,
    ZigZagPlan,
)
from convoy_sim.entities import Ship, Torpedo


@dataclass(frozen=True)
class OutcomeAuditConfig:
    """Runtime settings for profile outcome audit."""

    t_max_s: float = 600.0
    hit_dt_s: float = 0.5
    near_miss_margin_m: float = 250.0
    max_hits_per_torpedo: int | None = 1
    zigzag_enabled: bool = True
    zigzag_amplitude_rad: float = 0.12
    zigzag_period_s: float = 60.0
    zigzag_phase_s: float = 0.0
    zigzag_waveform: str = "sine"


def build_standard_evasive_kinematics(
    ships: Sequence[Ship],
    *,
    t_max_s: float,
    cfg: OutcomeAuditConfig | None = None,
) -> tuple[ConvoyFormation, ConvoyKinematics]:
    """Build the standard moving convoy + zig-zag kinematics used for profile QA."""

    if not ships:
        raise ValueError("ships must be non-empty")
    audit_cfg = cfg or OutcomeAuditConfig()
    positions = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
    origin = np.mean(positions, axis=0)
    headings = np.asarray([float(ship.heading_rad) for ship in ships], dtype=float)
    mean_heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    formation = ConvoyFormation(
        ships0=list(ships),
        convoy_origin0=origin,
        convoy_heading0=mean_heading,
    )
    route = RoutePlan(legs=[RouteLeg(duration_s=max(120.0, float(t_max_s)), heading_rad=mean_heading)])
    zigzag = ZigZagPlan(
        enabled=bool(audit_cfg.zigzag_enabled),
        amplitude_rad=float(audit_cfg.zigzag_amplitude_rad),
        period_s=float(audit_cfg.zigzag_period_s),
        phase_s=float(audit_cfg.zigzag_phase_s),
        waveform="triangle" if str(audit_cfg.zigzag_waveform) == "triangle" else "sine",
    )
    return formation, ConvoyKinematics(route=route, zigzag=zigzag)


def _ship_position_cube(
    *,
    formation: ConvoyFormation,
    kinematics: ConvoyKinematics,
    times: np.ndarray,
    dt: float,
) -> np.ndarray:
    if times.size == 0:
        return np.empty((0, len(formation.ships0), 2), dtype=float)
    positions = np.empty((int(times.size), len(formation.ships0), 2), dtype=float)
    positions[0] = np.asarray([np.asarray(ship.position, dtype=float) for ship in formation.ships0], dtype=float)
    for time_idx in range(1, int(times.size)):
        prev_time = float(times[time_idx - 1])
        step = float(times[time_idx] - times[time_idx - 1])
        for ship_idx, ship in enumerate(formation.ships0):
            heading = float(kinematics.convoy_heading_at(prev_time, float(ship.heading_rad)))
            direction = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
            positions[time_idx, ship_idx] = positions[time_idx - 1, ship_idx] + direction * float(ship.speed) * step
    return positions


def _closest_torpedo_ship_passes(
    *,
    ships: Sequence[Ship],
    torpedoes: Sequence[Torpedo],
    times: np.ndarray,
    ship_positions: np.ndarray,
    t_max_s: float,
    max_hits_per_torpedo: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hit_events: list[dict[str, Any]] = []
    radii = np.asarray([float(ship.effective_hit_radius()) for ship in ships], dtype=float)
    for torp_idx, torpedo in enumerate(torpedoes):
        active_mask = (times >= float(torpedo.launch_delay)) & (times <= min(float(t_max_s), torpedo.end_time_s()))
        active_times = times[active_mask]
        if active_times.size == 0:
            rows.append(
                {
                    "torpedo_id": str(torpedo.id),
                    "closest_ship_id": "",
                    "closest_distance_m": float("inf"),
                    "closest_time_s": float("nan"),
                    "closest_ship_radius_m": float("nan"),
                }
            )
            continue
        torpedo_xy = np.asarray([torpedo.position_at(float(time)) for time in active_times], dtype=float)
        active_ship_positions = ship_positions[active_mask]
        distances = np.linalg.norm(active_ship_positions - torpedo_xy[:, None, :], axis=2)
        flat_idx = int(np.argmin(distances))
        time_idx, ship_idx = np.unravel_index(flat_idx, distances.shape)
        ship = ships[int(ship_idx)]
        rows.append(
            {
                "torpedo_id": str(torpedo.id),
                "closest_ship_id": str(ship.id),
                "closest_distance_m": float(distances[time_idx, ship_idx]),
                "closest_time_s": float(active_times[time_idx]),
                "closest_ship_radius_m": float(ship.effective_hit_radius()),
            }
        )
        hit_mask = distances <= radii[None, :]
        if not np.any(hit_mask):
            continue
        if max_hits_per_torpedo == 1:
            hit_time_indices = np.where(np.any(hit_mask, axis=1))[0]
            first_time_idx = int(hit_time_indices[0])
            first_ship_idx = int(np.where(hit_mask[first_time_idx])[0][0])
            hit_events.append(
                {
                    "torpedo_index": int(torp_idx),
                    "torpedo_id": str(torpedo.id),
                    "ship_index": int(first_ship_idx),
                    "ship_id": str(ships[first_ship_idx].id),
                    "time_s": float(active_times[first_time_idx]),
                    "hit_x": float(torpedo_xy[first_time_idx, 0]),
                    "hit_y": float(torpedo_xy[first_time_idx, 1]),
                }
            )
        else:
            for ship_idx, ship in enumerate(ships):
                ship_hit_time_indices = np.where(hit_mask[:, ship_idx])[0]
                if ship_hit_time_indices.size == 0:
                    continue
                first_time_idx = int(ship_hit_time_indices[0])
                hit_events.append(
                    {
                        "torpedo_index": int(torp_idx),
                        "torpedo_id": str(torpedo.id),
                        "ship_index": int(ship_idx),
                        "ship_id": str(ship.id),
                        "time_s": float(active_times[first_time_idx]),
                        "hit_x": float(torpedo_xy[first_time_idx, 0]),
                        "hit_y": float(torpedo_xy[first_time_idx, 1]),
                    }
                )
    return rows, sorted(hit_events, key=lambda item: (float(item["time_s"]), int(item["torpedo_index"])))


def _min_distance_for_ship_ids(
    closest_rows: Sequence[dict[str, Any]],
    target_ship_ids: set[str],
) -> float:
    distances = [
        float(row["closest_distance_m"])
        for row in closest_rows
        if str(row.get("closest_ship_id", "")) in target_ship_ids
    ]
    return float(min(distances)) if distances else float("inf")


def _actual_outcome_label(
    *,
    intended_target_hit: bool,
    any_ship_hit: bool,
    closest_intended_target_distance_m: float,
    closest_any_ship_distance_m: float,
    nearest_relevant_radius_m: float,
    near_miss_margin_m: float,
) -> str:
    if intended_target_hit:
        return "credible_hit_threat"
    if any_ship_hit:
        return "hit_other_ship"
    near_threshold = float(nearest_relevant_radius_m) + float(near_miss_margin_m)
    if min(float(closest_intended_target_distance_m), float(closest_any_ship_distance_m)) <= near_threshold:
        return "credible_near_miss"
    return "miss"


def audit_profile_outcome(
    profile: AttackProfile,
    ships: Sequence[Ship],
    *,
    intent: Mapping[str, Any] | None = None,
    rng_seed: int = 1945,
    profile_index: int = 0,
    cfg: OutcomeAuditConfig | None = None,
) -> dict[str, Any]:
    """Run one profile through the standard dynamic sim path and summarize outcome."""

    audit_cfg = cfg or OutcomeAuditConfig()
    formation, kinematics = build_standard_evasive_kinematics(
        ships,
        t_max_s=float(audit_cfg.t_max_s),
        cfg=audit_cfg,
    )
    times = np.arange(0.0, float(audit_cfg.t_max_s) + 1e-9, float(audit_cfg.hit_dt_s), dtype=float)
    ship_position_cube = _ship_position_cube(
        formation=formation,
        kinematics=kinematics,
        times=times,
        dt=float(audit_cfg.hit_dt_s),
    )
    return _audit_profile_outcome_precomputed(
        profile,
        ships,
        intent=intent,
        rng_seed=int(rng_seed),
        profile_index=int(profile_index),
        cfg=audit_cfg,
        times=times,
        ship_position_cube=ship_position_cube,
    )


def _audit_profile_outcome_precomputed(
    profile: AttackProfile,
    ships: Sequence[Ship],
    *,
    intent: Mapping[str, Any] | None,
    rng_seed: int,
    profile_index: int,
    cfg: OutcomeAuditConfig,
    times: np.ndarray,
    ship_position_cube: np.ndarray,
) -> dict[str, Any]:
    """Audit one profile using precomputed moving-convoy positions."""

    rng = np.random.default_rng(int(rng_seed) + int(profile_index))
    torpedoes = profile.build_torpedoes(rng=rng, ships=list(ships))
    closest_rows, hit_events = _closest_torpedo_ship_passes(
        ships=ships,
        torpedoes=torpedoes,
        times=times,
        ship_positions=ship_position_cube,
        t_max_s=float(cfg.t_max_s),
        max_hits_per_torpedo=cfg.max_hits_per_torpedo,
    )
    hit_ship_ids = [str(event["ship_id"]) for event in hit_events]
    unique_hit_ship_ids = sorted(set(hit_ship_ids))
    target_ship_ids = {str(value) for value in (intent or {}).get("target_ship_ids", [])}
    intended_label = str((intent or {}).get("intended_label", ""))
    intended_target_hit = bool(target_ship_ids.intersection(unique_hit_ship_ids))
    any_ship_hit = bool(unique_hit_ship_ids)

    closest_any = min(closest_rows, key=lambda row: float(row["closest_distance_m"])) if closest_rows else {}
    closest_any_distance = float(closest_any.get("closest_distance_m", float("inf")))
    closest_any_radius = float(closest_any.get("closest_ship_radius_m", 0.0))
    closest_intended_distance = _min_distance_for_ship_ids(closest_rows, target_ship_ids)
    target_radii = [float(ship.effective_hit_radius()) for ship in ships if str(ship.id) in target_ship_ids]
    nearest_relevant_radius = float(max(target_radii) if target_radii else closest_any_radius)
    actual_label = _actual_outcome_label(
        intended_target_hit=intended_target_hit,
        any_ship_hit=any_ship_hit,
        closest_intended_target_distance_m=closest_intended_distance,
        closest_any_ship_distance_m=closest_any_distance,
        nearest_relevant_radius_m=nearest_relevant_radius,
        near_miss_margin_m=float(cfg.near_miss_margin_m),
    )
    outcome_matches_intent = (
        (intended_label == "credible_hit_threat" and intended_target_hit)
        or (intended_label == "credible_near_miss" and actual_label == "credible_near_miss")
    )
    first_hit = hit_events[0] if hit_events else None
    return {
        "profile_id": str(profile.profile_id),
        "name": str(profile.name),
        "intended_label": intended_label,
        "actual_outcome_label": actual_label,
        "outcome_matches_intent": bool(outcome_matches_intent),
        "target_ship_ids": sorted(target_ship_ids),
        "n_torpedoes": int(len(torpedoes)),
        "n_hits": int(len(hit_events)),
        "unique_ships_hit": int(len(unique_hit_ship_ids)),
        "hit_ship_ids": unique_hit_ship_ids,
        "any_ship_hit": any_ship_hit,
        "intended_target_hit": intended_target_hit,
        "first_hit_ship_id": "" if first_hit is None else str(first_hit["ship_id"]),
        "first_hit_torpedo_id": "" if first_hit is None else str(first_hit["torpedo_id"]),
        "first_hit_time_s": float("nan") if first_hit is None else float(first_hit["time_s"]),
        "closest_intended_target_distance_m": closest_intended_distance,
        "closest_any_ship_distance_m": closest_any_distance,
        "closest_any_ship_id": str(closest_any.get("closest_ship_id", "")),
        "closest_any_time_s": float(closest_any.get("closest_time_s", float("nan"))),
        "spawn_region": str((intent or {}).get("spawn_region", "")),
        "approach_side": str((intent or {}).get("approach_side", "")),
        "target_zone_kind": str((intent or {}).get("target_zone_kind", "")),
        "spread_doctrine": str(profile.spread_doctrine),
        "u_boat_mode": str(profile.u_boat_mode),
        "convoy_motion": "zigzag" if cfg.zigzag_enabled else "straight",
        "hit_dt_s": float(cfg.hit_dt_s),
        "t_max_s": float(cfg.t_max_s),
    }


def audit_dataset_outcomes(
    records: Sequence[Mapping[str, Any]],
    *,
    ships: Sequence[Ship],
    rng_seed: int = 1945,
    cfg: OutcomeAuditConfig | None = None,
) -> list[dict[str, Any]]:
    """Audit JSONL-style generated dataset records against dynamic sim outcomes."""

    audit_cfg = cfg or OutcomeAuditConfig()
    formation, kinematics = build_standard_evasive_kinematics(
        ships,
        t_max_s=float(audit_cfg.t_max_s),
        cfg=audit_cfg,
    )
    times = np.arange(0.0, float(audit_cfg.t_max_s) + 1e-9, float(audit_cfg.hit_dt_s), dtype=float)
    ship_position_cube = _ship_position_cube(
        formation=formation,
        kinematics=kinematics,
        times=times,
        dt=float(audit_cfg.hit_dt_s),
    )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        profile = AttackProfile.from_dict(dict(record["profile"]))
        rows.append(
            _audit_profile_outcome_precomputed(
                profile,
                ships,
                intent=dict(record.get("intent", {})),
                rng_seed=int(rng_seed),
                profile_index=index,
                cfg=audit_cfg,
                times=times,
                ship_position_cube=ship_position_cube,
            )
        )
    return rows


def enrich_dataset_records_with_outcomes(
    records: Sequence[Mapping[str, Any]],
    *,
    ships: Sequence[Ship],
    rng_seed: int = 1945,
    cfg: OutcomeAuditConfig | None = None,
    outcome_key: str = "outcome",
) -> list[dict[str, Any]]:
    """Return JSONL-style records with dynamic outcome audit payloads attached."""

    outcome_rows = audit_dataset_outcomes(records, ships=ships, rng_seed=int(rng_seed), cfg=cfg)
    enriched: list[dict[str, Any]] = []
    for record, outcome in zip(records, outcome_rows):
        item = dict(record)
        item[str(outcome_key)] = dict(outcome)
        enriched.append(item)
    return enriched


def summarize_outcome_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact aggregate diagnostics for outcome-audit rows."""

    if not rows:
        return {
            "profile_count": 0,
            "actual_outcome_labels": {},
            "intended_labels": {},
            "outcome_match_rate": 0.0,
        }
    actual_counter = Counter(str(row["actual_outcome_label"]) for row in rows)
    intended_counter = Counter(str(row["intended_label"]) for row in rows)
    finite_target_distances = [
        float(row["closest_intended_target_distance_m"])
        for row in rows
        if np.isfinite(float(row["closest_intended_target_distance_m"]))
    ]
    return {
        "profile_count": int(len(rows)),
        "actual_outcome_labels": dict(actual_counter),
        "intended_labels": dict(intended_counter),
        "outcome_match_rate": float(mean(1.0 if bool(row["outcome_matches_intent"]) else 0.0 for row in rows)),
        "intended_target_hit_rate": float(mean(1.0 if bool(row["intended_target_hit"]) else 0.0 for row in rows)),
        "any_ship_hit_rate": float(mean(1.0 if bool(row["any_ship_hit"]) else 0.0 for row in rows)),
        "mean_hits": float(mean(float(row["n_hits"]) for row in rows)),
        "mean_unique_ships_hit": float(mean(float(row["unique_ships_hit"]) for row in rows)),
        "mean_closest_intended_target_distance_m": float(
            mean(finite_target_distances) if finite_target_distances else float("inf")
        ),
        "mean_closest_any_ship_distance_m": float(mean(float(row["closest_any_ship_distance_m"]) for row in rows)),
    }
