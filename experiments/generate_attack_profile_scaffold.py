from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.profile_audit import AuditThresholds, audit_attack_profiles
from convoy_sim.profile_outcome_audit import OutcomeAuditConfig, OutcomeAuditContext
from convoy_sim.target_zones import (
    AttackIntent,
    build_curated_attack_intents,
    sample_random_attack_intent,
    sample_random_tactical_attack_intent,
    spawn_world_from_intent,
)
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


@dataclass(frozen=True)
class ApproachPreset:
    name: str
    angle_deg: float


@dataclass(frozen=True)
class RangePreset:
    name: str
    radius_m: float


@dataclass(frozen=True)
class SalvoPreset:
    name: str
    n: int
    spread_doctrine: str
    spread_rad: float
    per_torpedo_heading_offsets_rad: tuple[float, ...]
    launch_delay_s: float
    salvo_interval_s: float
    gyro_straight_run_m: float


APPROACH_PRESETS: tuple[ApproachPreset, ...] = (
    ApproachPreset("east_abeam", 0.0),
    ApproachPreset("north_east_quarter", 55.0),
    ApproachPreset("south_east_quarter", -55.0),
    ApproachPreset("west_abeam", 180.0),
    ApproachPreset("north_west_quarter", 125.0),
    ApproachPreset("south_west_quarter", -125.0),
)

RANGE_PRESETS: tuple[RangePreset, ...] = (
    RangePreset("close", 1200.0),
    RangePreset("medium", 2200.0),
    RangePreset("long", 3200.0),
)

SALVO_PRESETS: tuple[SalvoPreset, ...] = (
    SalvoPreset(
        "uniform_narrow_4",
        n=4,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(4.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=0.5,
        salvo_interval_s=2.0,
        gyro_straight_run_m=10.0,
    ),
    SalvoPreset(
        "uniform_medium_4",
        n=4,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(5.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=0.9,
        salvo_interval_s=2.0,
        gyro_straight_run_m=10.0,
    ),
    SalvoPreset(
        "uniform_wide_4",
        n=4,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(6.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=1.4,
        salvo_interval_s=3.0,
        gyro_straight_run_m=10.0,
    ),
)

TORPEDO_SPEED_MPS = 15.4333
TORPEDO_MAX_RUN_TIME_S = 486.0
DEFAULT_WEIGHT = 1.0
DEFAULT_U_BOAT_SPEED_OPTIONS_MPS = tuple(round(v, 1) for v in np.arange(1.0, 2.01, 0.1))
DEFAULT_SUB_LENGTH_M = 67.0
DEFAULT_SUB_BEAM_M = 6.5
DEFAULT_MAX_BOW_OFFSET_DEG = 15.0
DEFAULT_ACCEPTED_LABELS = ("credible_hit_threat", "credible_near_miss")
HIT_THREAT_LABEL = "credible_hit_threat"
NEAR_MISS_LABEL = "credible_near_miss"
INTENTIONAL_MISS_LABEL = "intentional_miss"
INTENT_LABELS = (HIT_THREAT_LABEL, NEAR_MISS_LABEL)
MIN_SPAWN_CLEARANCE_M = 250.0
RANGE_BAND_TOLERANCE_M = 250.0
STANDARD_ZIGZAG_AMPLITUDE_RAD = 0.12
STANDARD_ZIGZAG_PERIOD_S = 60.0
STANDARD_LEAD_DT_S = 1.0
TACTICAL_OUTCOME_GATE_T_MAX_S = 600.0
TACTICAL_OUTCOME_GATE_HIT_DT_S = 0.5
GENERATOR_SOURCE = "generate_attack_profile_scaffold"
GENERATOR_VERSION = "v2"
TARGET_ZONE_GENERATOR_VERSION = "v3"
TACTICAL_GENERATOR_VERSION = "v4"
DATASET_HIT_THREAT_FRACTION = 0.75
TACTICAL_HIT_THREAT_FRACTION = 0.65
TACTICAL_NEAR_MISS_FRACTION = 0.25
LEGACY_MODES = {"curated", "dataset"}
TARGET_ZONE_MODES = {"curated_zones", "random_zones"}
TACTICAL_MODES = {"random_tactical_v4"}
INTENT_MODES = TARGET_ZONE_MODES | TACTICAL_MODES
ALL_MODES = LEGACY_MODES | INTENT_MODES


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _all_profile_specs() -> list[tuple[ApproachPreset, RangePreset, SalvoPreset]]:
    specs: list[tuple[ApproachPreset, RangePreset, SalvoPreset]] = []
    for approach in APPROACH_PRESETS:
        for range_preset in RANGE_PRESETS:
            for salvo in SALVO_PRESETS:
                specs.append((approach, range_preset, salvo))
    return specs


def _build_attack_profile(
    *,
    profile_id: str,
    name: str,
    weight: float,
    u_pos: tuple[float, float],
    n: int,
    base_bearing_rad: float,
    spread_doctrine: str,
    spread_rad: float,
    per_torpedo_heading_offsets_rad: Sequence[float],
    launch_delay_s: float,
    salvo_interval_s: float,
    u_boat_initial_speed_mps: float,
    gyro_straight_run_m: float,
) -> AttackProfile:
    return AttackProfile(
        profile_id=profile_id,
        name=name,
        weight=float(weight),
        mode="fan",
        u_pos=u_pos,
        n=int(n),
        speed=TORPEDO_SPEED_MPS,
        max_run_time=TORPEDO_MAX_RUN_TIME_S,
        base_bearing_rad=float(base_bearing_rad),
        spread_rad=float(spread_rad),
        spread_doctrine=str(spread_doctrine),
        per_torpedo_heading_offsets_rad=tuple(float(v) for v in per_torpedo_heading_offsets_rad),
        launch_delay_s=float(launch_delay_s),
        salvo_interval_s=float(salvo_interval_s),
        u_boat_mode="moving",
        u_boat_initial_heading_rad=float(base_bearing_rad),
        u_boat_initial_speed_mps=float(u_boat_initial_speed_mps),
        sub_length_m=DEFAULT_SUB_LENGTH_M,
        sub_beam_m=DEFAULT_SUB_BEAM_M,
        launch_from="bow",
        max_bow_offset_deg=DEFAULT_MAX_BOW_OFFSET_DEG,
        gyro_straight_run_m=float(gyro_straight_run_m),
    )


def _range_band_bounds_m(range_preset: RangePreset) -> tuple[float, float]:
    center = float(range_preset.radius_m)
    tol = float(RANGE_BAND_TOLERANCE_M)
    return center - tol, center + tol


def _spawn_is_feasible(
    u_pos: tuple[float, float],
    ships: Sequence[object],
    *,
    range_preset: RangePreset,
) -> bool:
    u_boat_pos = np.asarray(u_pos, dtype=float)
    ship_positions = np.asarray([np.asarray(getattr(ship, "position"), dtype=float) for ship in ships], dtype=float)
    if ship_positions.size == 0:
        return False
    distances = np.linalg.norm(ship_positions - u_boat_pos, axis=1)
    if float(np.min(distances)) < float(MIN_SPAWN_CLEARANCE_M):
        return False
    centroid = np.mean(ship_positions, axis=0)
    centroid_range = float(np.linalg.norm(u_boat_pos - centroid))
    min_range, max_range = _range_band_bounds_m(range_preset)
    return min_range <= centroid_range <= max_range


def _spawn_has_clearance(
    u_pos: tuple[float, float],
    ships: Sequence[object],
    *,
    min_clearance_m: float = MIN_SPAWN_CLEARANCE_M,
) -> bool:
    u_boat_pos = np.asarray(u_pos, dtype=float)
    ship_positions = np.asarray([np.asarray(getattr(ship, "position"), dtype=float) for ship in ships], dtype=float)
    if ship_positions.size == 0:
        return False
    distances = np.linalg.norm(ship_positions - u_boat_pos, axis=1)
    return float(np.min(distances)) >= float(min_clearance_m)


def _target_zone_spread_deg(target_label: str, rng: np.random.Generator) -> float:
    if target_label == HIT_THREAT_LABEL:
        return float(rng.choice(np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0], dtype=float)))
    return float(rng.choice(np.array([12.0, 14.0, 16.0], dtype=float)))


def _tactical_spread_deg(target_label: str, rng: np.random.Generator) -> float:
    if target_label == HIT_THREAT_LABEL:
        return float(rng.choice(np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], dtype=float)))
    if target_label == NEAR_MISS_LABEL:
        return float(rng.choice(np.array([2.0, 2.5, 3.0, 3.5], dtype=float)))
    if target_label == INTENTIONAL_MISS_LABEL:
        return float(rng.choice(np.array([2.5, 3.0, 3.5, 4.0], dtype=float)))
    raise ValueError(f"Unsupported tactical target label: {target_label}")


def _standard_evasive_displacement(
    *,
    speed_mps: float,
    base_heading_rad: float,
    t_s: float,
    dt_s: float = STANDARD_LEAD_DT_S,
) -> np.ndarray:
    """Approximate standard zig-zag convoy displacement over ``t_s`` seconds."""

    duration = float(t_s)
    if duration <= 0.0 or speed_mps == 0.0:
        return np.zeros(2, dtype=float)
    dt = max(0.05, float(dt_s))
    times = np.arange(0.0, duration, dt, dtype=float)
    if times.size == 0:
        times = np.asarray([0.0], dtype=float)
    steps = np.full(times.shape, dt, dtype=float)
    overshoot = float(np.sum(steps) - duration)
    if overshoot > 0.0:
        steps[-1] = max(0.0, steps[-1] - overshoot)
    headings = (
        float(base_heading_rad)
        + float(STANDARD_ZIGZAG_AMPLITUDE_RAD)
        * np.sin((2.0 * np.pi * times) / float(STANDARD_ZIGZAG_PERIOD_S))
    )
    directions = np.column_stack([np.cos(headings), np.sin(headings)])
    return np.sum(directions * steps[:, None] * float(speed_mps), axis=0)


def _ship_for_intent(ships: Sequence[object], intent: AttackIntent) -> object | None:
    target_ids = {str(ship_id) for ship_id in intent.target_ship_ids}
    for ship in ships:
        if str(getattr(ship, "id", "")) in target_ids:
            return ship
    return None


def _lead_solution_for_standard_evasive_target(
    *,
    u_pos: tuple[float, float],
    target_point: np.ndarray,
    target_ship: object | None,
    launch_delay_s: float,
    torpedo_speed_mps: float,
    torpedo_max_run_time_s: float,
    iterations: int = 8,
) -> dict[str, object]:
    """Return a lead aim point for the standard moving zig-zag convoy audit."""

    if target_ship is None:
        return {
            "aim_point": [float(target_point[0]), float(target_point[1])],
            "aim_solution_kind": "direct_no_target_ship",
            "aim_intercept_time_s": float(launch_delay_s),
            "aim_lead_distance_m": 0.0,
        }
    u_arr = np.asarray(u_pos, dtype=float)
    target_arr = np.asarray(target_point, dtype=float)
    speed = max(float(torpedo_speed_mps), 1e-9)
    travel_time = min(float(torpedo_max_run_time_s), float(np.linalg.norm(target_arr - u_arr) / speed))
    aim_point = target_arr.copy()
    for _ in range(int(iterations)):
        intercept_t = float(launch_delay_s) + float(travel_time)
        displacement = _standard_evasive_displacement(
            speed_mps=float(getattr(target_ship, "speed", 0.0)),
            base_heading_rad=float(getattr(target_ship, "heading_rad", 0.0)),
            t_s=intercept_t,
        )
        aim_point = target_arr + displacement
        travel_time = min(float(torpedo_max_run_time_s), float(np.linalg.norm(aim_point - u_arr) / speed))
    intercept_t = float(launch_delay_s) + float(travel_time)
    lead_distance = float(np.linalg.norm(aim_point - target_arr))
    return {
        "aim_point": [float(aim_point[0]), float(aim_point[1])],
        "aim_solution_kind": "standard_zigzag_lead",
        "aim_intercept_time_s": float(intercept_t),
        "aim_lead_distance_m": lead_distance,
    }


def _ship_hit_radius_m(ship: object | None) -> float:
    if ship is None:
        return 80.0
    radius_fn = getattr(ship, "effective_hit_radius", None)
    if callable(radius_fn):
        return float(radius_fn())
    length = float(getattr(ship, "length", 120.0))
    beam = float(getattr(ship, "beam", 20.0))
    return float(0.5 * np.hypot(length, beam))


def _offset_aim_payload_for_tactical_label(
    aim_payload: dict[str, object],
    *,
    u_pos: tuple[float, float],
    target_ship: object | None,
    convoy_centroid: np.ndarray,
    target_label: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Apply deliberate lateral aim offsets for training-data near/miss labels."""

    if target_label == HIT_THREAT_LABEL:
        enriched = dict(aim_payload)
        enriched.update(
            {
                "aim_offset_kind": "none",
                "aim_lateral_offset_m": 0.0,
                "aim_offset_side": 0,
            }
        )
        return enriched

    aim_point = np.asarray(aim_payload["aim_point"], dtype=float)
    u_arr = np.asarray(u_pos, dtype=float)
    shot_vec = aim_point - u_arr
    shot_len = float(np.linalg.norm(shot_vec))
    if shot_len <= 1e-9:
        enriched = dict(aim_payload)
        enriched.update(
            {
                "aim_offset_kind": "none_degenerate",
                "aim_lateral_offset_m": 0.0,
                "aim_offset_side": 0,
            }
        )
        return enriched

    ship_radius_m = _ship_hit_radius_m(target_ship)
    if target_label == NEAR_MISS_LABEL:
        lateral_offset_m = float(ship_radius_m + rng.uniform(260.0, 420.0))
        offset_kind = "near_miss_lateral_offset"
    elif target_label == INTENTIONAL_MISS_LABEL:
        outward = aim_point - np.asarray(convoy_centroid, dtype=float)
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm <= 1e-9:
            outward = shot_vec
            outward_norm = shot_len
        outward_unit = outward / outward_norm
        lateral_offset_m = float(ship_radius_m + rng.uniform(900.0, 1700.0))
        shifted = aim_point + lateral_offset_m * outward_unit
        enriched = dict(aim_payload)
        base_kind = str(enriched.get("aim_solution_kind", "direct"))
        enriched.update(
            {
                "aim_point": [float(shifted[0]), float(shifted[1])],
                "aim_solution_kind": f"{base_kind}_intentional_miss_clear_water_offset",
                "aim_offset_kind": "intentional_miss_clear_water_offset",
                "aim_lateral_offset_m": lateral_offset_m,
                "aim_offset_side": 0,
            }
        )
        return enriched
    else:
        raise ValueError(f"Unsupported tactical target label: {target_label}")

    side = -1 if float(rng.random()) < 0.5 else 1
    unit = shot_vec / shot_len
    perp = np.asarray([-unit[1], unit[0]], dtype=float)
    shifted = aim_point + float(side) * lateral_offset_m * perp
    enriched = dict(aim_payload)
    base_kind = str(enriched.get("aim_solution_kind", "direct"))
    enriched.update(
        {
            "aim_point": [float(shifted[0]), float(shifted[1])],
            "aim_solution_kind": f"{base_kind}_{offset_kind}",
            "aim_offset_kind": offset_kind,
            "aim_lateral_offset_m": lateral_offset_m,
            "aim_offset_side": int(side),
        }
    )
    return enriched


def _target_zone_label_targets(count: int) -> dict[str, int]:
    hit_target = int(round(count * DATASET_HIT_THREAT_FRACTION))
    return {
        HIT_THREAT_LABEL: hit_target,
        NEAR_MISS_LABEL: int(count - hit_target),
    }


def _tactical_label_targets(count: int) -> dict[str, int]:
    hit_target = int(round(count * TACTICAL_HIT_THREAT_FRACTION))
    near_target = int(round(count * TACTICAL_NEAR_MISS_FRACTION))
    miss_target = int(count - hit_target - near_target)
    return {
        HIT_THREAT_LABEL: hit_target,
        NEAR_MISS_LABEL: near_target,
        INTENTIONAL_MISS_LABEL: miss_target,
    }


def _choose_remaining_target_label(
    *,
    rng: np.random.Generator,
    targets: dict[str, int],
    counts: Counter[str],
) -> str:
    remaining = {
        label: int(target - counts[label])
        for label, target in targets.items()
        if int(target - counts[label]) > 0
    }
    if not remaining:
        raise StopIteration
    labels = list(remaining)
    weights = np.asarray([float(remaining[label]) for label in labels], dtype=float)
    weights = weights / float(np.sum(weights))
    return str(rng.choice(np.asarray(labels, dtype=object), p=weights))


def _generator_version_for_mode(mode: str) -> str:
    if mode in TACTICAL_MODES:
        return TACTICAL_GENERATOR_VERSION
    if mode in TARGET_ZONE_MODES:
        return TARGET_ZONE_GENERATOR_VERSION
    return GENERATOR_VERSION


def _explicit_profile_dict(profile: AttackProfile) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "name": profile.name,
        "weight": float(profile.weight),
        "mode": profile.mode,
        "u_pos": [float(v) for v in profile.u_pos],
        "n": int(profile.n),
        "speed": float(profile.speed),
        "max_run_time": float(profile.max_run_time),
        "base_bearing_rad": float(profile.base_bearing_rad),
        "spread_doctrine": str(profile.spread_doctrine),
        "spread_rad": float(profile.spread_rad),
        "per_torpedo_heading_offsets_rad": [float(v) for v in profile.per_torpedo_heading_offsets_rad],
        "launch_delay_s": float(profile.launch_delay_s),
        "salvo_interval_s": float(profile.salvo_interval_s),
        "u_boat_mode": str(profile.u_boat_mode),
        "u_boat_initial_heading_rad": float(profile.u_boat_initial_heading_rad),
        "u_boat_initial_speed_mps": float(profile.u_boat_initial_speed_mps),
        "sub_length_m": float(profile.sub_length_m),
        "sub_beam_m": float(profile.sub_beam_m),
        "launch_from": str(profile.launch_from),
        "max_bow_offset_deg": float(profile.max_bow_offset_deg),
        "gyro_straight_run_m": float(profile.gyro_straight_run_m),
    }


def generate_attack_profile_scaffolds(
    *,
    start_index: int = 31,
    count: int = 30,
    seed: int = 1945,
    weight: float = DEFAULT_WEIGHT,
    convoy_profile: str = "convoy_layout_1",
    accepted_labels: Sequence[str] = DEFAULT_ACCEPTED_LABELS,
    thresholds: AuditThresholds | None = None,
    mode: str = "curated",
) -> tuple[list[AttackProfile], list[dict[str, object]]]:
    if mode not in ALL_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(ALL_MODES))}")
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    specs = _all_profile_specs()
    rng = np.random.default_rng(seed)
    ships = get_convoy_layout_profile(convoy_profile).build_ships()
    convoy_centroid = np.mean(
        np.asarray([np.asarray(getattr(ship, "position"), dtype=float) for ship in ships]),
        axis=0,
    )
    outcome_context = (
        OutcomeAuditContext.from_ships(
            ships,
            cfg=OutcomeAuditConfig(
                t_max_s=TACTICAL_OUTCOME_GATE_T_MAX_S,
                hit_dt_s=TACTICAL_OUTCOME_GATE_HIT_DT_S,
                near_miss_margin_m=250.0,
                max_hits_per_torpedo=1,
                zigzag_enabled=True,
                zigzag_amplitude_rad=STANDARD_ZIGZAG_AMPLITUDE_RAD,
                zigzag_period_s=STANDARD_ZIGZAG_PERIOD_S,
            ),
        )
        if mode in TACTICAL_MODES
        else None
    )
    accepted = set(accepted_labels)
    if not accepted:
        raise ValueError("accepted_labels must not be empty")

    max_attempts = max(count * (20 if mode == "curated" else 300), len(specs) * 2)
    profiles: list[AttackProfile] = []
    accepted_audit_rows: list[dict[str, object]] = []
    used_signatures: set[tuple[str, ...]] = set()
    next_index = start_index
    attempts = 0
    dataset_label_targets: dict[str, int] | None = None
    dataset_label_counts: Counter[str] | None = None
    curated_zone_intents: list[AttackIntent] = []
    if mode in {"dataset", "random_zones", "random_tactical_v4"}:
        if accepted != set(INTENT_LABELS):
            raise ValueError(
                f"{mode} mode currently expects accepted_labels to be exactly "
                "{credible_hit_threat, credible_near_miss}"
            )
        dataset_label_targets = (
            _tactical_label_targets(count) if mode in TACTICAL_MODES else _target_zone_label_targets(count)
        )
        dataset_label_counts = Counter()
    if mode == "curated_zones":
        curated_zone_intents = build_curated_attack_intents(
            ships,
            count=max(count * 3, count + len(ships)),
            start_index=start_index,
        )

    while len(profiles) < count:
        if attempts >= max_attempts:
            raise ValueError(
                f"Unable to generate {count} plausible profiles after {attempts} attempts; "
                "broaden presets or relax audit thresholds."
            )
        attempts += 1
        if mode in INTENT_MODES:
            intent: AttackIntent
            if mode == "curated_zones":
                intent = curated_zone_intents[(attempts - 1) % len(curated_zone_intents)]
                target_label = intent.intended_label
                profile_id = f"P{next_index:02d}"
                profile_name = (
                    f"profile_{next_index:02d}_{intent.target_zone_kind}_"
                    f"{intent.approach_side}_{target_label}"
                )
            elif mode == "random_zones":
                assert dataset_label_targets is not None
                assert dataset_label_counts is not None
                try:
                    target_label = _choose_remaining_target_label(
                        rng=rng,
                        targets=dataset_label_targets,
                        counts=dataset_label_counts,
                    )
                except StopIteration:
                    break
                intent = sample_random_attack_intent(
                    ships,
                    rng=rng,
                    sequence_id=next_index,
                    intended_label=target_label,
                )
                profile_id = f"Z{next_index:06d}"
                profile_name = (
                    f"zone_dataset_{next_index:06d}_{intent.target_zone_kind}_"
                    f"{intent.approach_side}_{target_label}"
                )
            else:
                assert dataset_label_targets is not None
                assert dataset_label_counts is not None
                try:
                    target_label = _choose_remaining_target_label(
                        rng=rng,
                        targets=dataset_label_targets,
                        counts=dataset_label_counts,
                    )
                except StopIteration:
                    break
                intent = sample_random_tactical_attack_intent(
                    ships,
                    rng=rng,
                    sequence_id=next_index,
                    intended_label=target_label,
                    min_clearance_m=MIN_SPAWN_CLEARANCE_M,
                )
                profile_id = f"T{next_index:06d}"
                profile_name = (
                    f"tactical_dataset_{next_index:06d}_{intent.spawn_region}_"
                    f"{intent.target_zone_kind}_{target_label}"
                )
            signature = (
                intent.target_zone_id,
                intent.approach_side,
                intent.approach_lane,
                str(target_label),
                f"{rng.integers(0, 1_000_000)}",
            )
            if signature in used_signatures:
                continue
            u_boat_initial_speed_mps = float(rng.choice(DEFAULT_U_BOAT_SPEED_OPTIONS_MPS))
            u_pos = spawn_world_from_intent(intent, ships)
            if not _spawn_has_clearance(u_pos, ships):
                continue
            target_point = np.asarray(intent.target_point, dtype=float)
            u_arr = np.asarray(u_pos, dtype=float)
            launch_delay_s = float(rng.choice(np.round(np.arange(0.5, 2.51, 0.1), 1)))
            salvo_interval_s = float(rng.choice(np.array([1.5, 2.0, 2.5, 3.0], dtype=float)))
            aim_payload: dict[str, object] = {
                "aim_point": [float(target_point[0]), float(target_point[1])],
                "aim_solution_kind": "direct",
                "aim_intercept_time_s": float(launch_delay_s),
                "aim_lead_distance_m": 0.0,
            }
            if mode == "random_tactical_v4":
                target_ship = _ship_for_intent(ships, intent)
                aim_payload = _lead_solution_for_standard_evasive_target(
                    u_pos=u_pos,
                    target_point=target_point,
                    target_ship=target_ship,
                    launch_delay_s=launch_delay_s,
                    torpedo_speed_mps=TORPEDO_SPEED_MPS,
                    torpedo_max_run_time_s=TORPEDO_MAX_RUN_TIME_S,
                )
                aim_payload = _offset_aim_payload_for_tactical_label(
                    aim_payload,
                    u_pos=u_pos,
                    target_ship=target_ship,
                    convoy_centroid=convoy_centroid,
                    target_label=target_label,
                    rng=rng,
                )
            aim_point = np.asarray(aim_payload["aim_point"], dtype=float)
            bearing_to_target = float(np.arctan2(aim_point[1] - u_arr[1], aim_point[0] - u_arr[0]))
            base_bearing_rad = _wrap_angle_rad(
                bearing_to_target + float(np.deg2rad(intent.planned_bearing_error_deg))
            )
            spread_deg = (
                _tactical_spread_deg(target_label, rng)
                if mode == "random_tactical_v4"
                else _target_zone_spread_deg(target_label, rng)
            )
            spread_rad = float(np.deg2rad(spread_deg))
            profile = _build_attack_profile(
                profile_id=profile_id,
                name=profile_name,
                weight=float(weight),
                u_pos=u_pos,
                n=4,
                base_bearing_rad=base_bearing_rad,
                spread_doctrine="uniform_divergent",
                spread_rad=spread_rad,
                per_torpedo_heading_offsets_rad=(),
                launch_delay_s=launch_delay_s,
                salvo_interval_s=salvo_interval_s,
                u_boat_initial_speed_mps=u_boat_initial_speed_mps,
                gyro_straight_run_m=10.0,
            )
            intent_dict = intent.to_dict()
            intent_dict.update(aim_payload)
            audit_row = audit_attack_profiles(
                [profile],
                ships,
                intents=[intent_dict],
                thresholds=thresholds,
            )[0]
            audit_label = str(audit_row["suggested_label"])
            accepted_for_mode = accepted | (
                {INTENTIONAL_MISS_LABEL} if mode in TACTICAL_MODES else set()
            )
            if audit_label not in accepted_for_mode:
                continue
            if mode in {"random_zones", "random_tactical_v4"} and audit_label != target_label:
                continue
            if mode == "random_tactical_v4":
                assert outcome_context is not None
                outcome_row = outcome_context.audit_profile(
                    profile,
                    intent=intent_dict,
                    rng_seed=int(seed),
                    profile_index=len(profiles),
                )
                if not bool(outcome_row["outcome_matches_intent"]):
                    continue
                audit_row["outcome_gate"] = {
                    "actual_outcome_label": str(outcome_row["actual_outcome_label"]),
                    "passes_outcome_gate": bool(outcome_row["passes_outcome_gate"]),
                    "intended_target_hit": bool(outcome_row["intended_target_hit"]),
                    "any_ship_hit": bool(outcome_row["any_ship_hit"]),
                    "closest_intended_target_distance_m": float(
                        outcome_row["closest_intended_target_distance_m"]
                    ),
                    "closest_any_ship_distance_m": float(outcome_row["closest_any_ship_distance_m"]),
                }
            audit_row["intent"] = intent_dict
            used_signatures.add(signature)
            profiles.append(profile)
            accepted_audit_rows.append(audit_row)
            if dataset_label_counts is not None:
                dataset_label_counts[audit_label] += 1
            next_index += 1
            continue

        approach, range_preset, salvo = specs[int(rng.integers(0, len(specs)))]
        u_boat_initial_speed_mps = float(rng.choice(DEFAULT_U_BOAT_SPEED_OPTIONS_MPS))
        target_label: str | None = None
        if mode == "curated":
            signature = (approach.name, range_preset.name, salvo.name, u_boat_initial_speed_mps)
            if signature in used_signatures:
                continue
            angle_jitter_deg = 4.0
            radius_jitter_m = 180.0
            spread_rad = float(salvo.spread_rad)
            launch_delay_s = float(salvo.launch_delay_s)
            salvo_interval_s = float(salvo.salvo_interval_s)
            n = int(salvo.n)
            spread_doctrine = str(salvo.spread_doctrine)
            per_offsets = tuple(float(v) for v in salvo.per_torpedo_heading_offsets_rad)
            profile_id = f"P{next_index:02d}"
            profile_name = f"profile_{next_index:02d}_{approach.name}_{range_preset.name}_{salvo.name}"
        else:
            assert dataset_label_targets is not None
            assert dataset_label_counts is not None
            remaining_hit = dataset_label_targets[HIT_THREAT_LABEL] - dataset_label_counts[HIT_THREAT_LABEL]
            remaining_near = dataset_label_targets[NEAR_MISS_LABEL] - dataset_label_counts[NEAR_MISS_LABEL]
            if remaining_hit <= 0 and remaining_near <= 0:
                break
            if remaining_hit > 0 and remaining_near > 0:
                total_remaining = remaining_hit + remaining_near
                target_label = (
                    HIT_THREAT_LABEL
                    if float(rng.random()) < (remaining_hit / total_remaining)
                    else NEAR_MISS_LABEL
                )
            elif remaining_hit > 0:
                target_label = HIT_THREAT_LABEL
            else:
                target_label = NEAR_MISS_LABEL
            signature = (
                approach.name,
                range_preset.name,
                str(target_label),
                f"{u_boat_initial_speed_mps:.1f}",
                f"{rng.integers(0, 1_000_000)}",
            )
            angle_jitter_deg = 10.0
            radius_jitter_m = 350.0
            if target_label == HIT_THREAT_LABEL:
                spread_deg = float(rng.choice(np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0], dtype=float)))
            else:
                spread_deg = float(rng.choice(np.array([12.0, 14.0, 16.0], dtype=float)))
            spread_rad = float(np.deg2rad(spread_deg))
            launch_delay_s = float(rng.choice(np.round(np.arange(0.5, 2.51, 0.1), 1)))
            salvo_interval_s = float(rng.choice(np.array([1.5, 2.0, 2.5, 3.0], dtype=float)))
            n = 4 #Max torpedo salvo size. Assumes max number is always fired
            spread_doctrine = "uniform_divergent"
            per_offsets = ()
            profile_id = f"D{next_index:06d}"
            profile_name = f"dataset_{next_index:06d}_{approach.name}_{range_preset.name}_uniform_random"
        if signature in used_signatures:
            continue
        angle_rad = float(np.deg2rad(approach.angle_deg + rng.uniform(-angle_jitter_deg, angle_jitter_deg)))
        radius_m = float(range_preset.radius_m + rng.uniform(-radius_jitter_m, radius_jitter_m))
        u_pos = (
            float(radius_m * np.cos(angle_rad)),
            float(radius_m * np.sin(angle_rad)),
        )
        if not _spawn_is_feasible(u_pos, ships, range_preset=range_preset):
            continue
        bearing_to_origin = float(np.arctan2(-u_pos[1], -u_pos[0]))
        if mode == "curated":
            base_bearing_offset_deg = float(rng.uniform(-4.0, 4.0))
        else:
            assert target_label is not None
            if target_label == HIT_THREAT_LABEL:
                base_bearing_offset_deg = float(rng.uniform(-5.0, 5.0))
            else:
                miss_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
                base_bearing_offset_deg = float(miss_sign * rng.uniform(8.4, 10.4))
        base_bearing_rad = _wrap_angle_rad(
            bearing_to_origin + float(np.deg2rad(base_bearing_offset_deg))
        )
        profile = _build_attack_profile(
            profile_id=profile_id,
            name=profile_name,
            weight=float(weight),
            u_pos=u_pos,
            n=n,
            base_bearing_rad=base_bearing_rad,
            spread_doctrine=spread_doctrine,
            spread_rad=spread_rad,
            per_torpedo_heading_offsets_rad=per_offsets,
            launch_delay_s=launch_delay_s,
            salvo_interval_s=salvo_interval_s,
            u_boat_initial_speed_mps=u_boat_initial_speed_mps,
            gyro_straight_run_m=float(salvo.gyro_straight_run_m),
        )
        audit_row = audit_attack_profiles([profile], ships, thresholds=thresholds)[0]
        audit_label = str(audit_row["suggested_label"])
        if audit_label not in accepted:
            continue
        if mode == "dataset" and audit_label != target_label:
            continue
        used_signatures.add(signature)
        profiles.append(profile)
        accepted_audit_rows.append(audit_row)
        if dataset_label_counts is not None:
            dataset_label_counts[audit_label] += 1
        next_index += 1
    return profiles, accepted_audit_rows


def _format_float(value: float) -> str:
    text = f"{float(value):.6f}"
    text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _format_tuple(values: Sequence[float]) -> str:
    rendered = ", ".join(_format_float(v) for v in values)
    if len(values) == 1:
        rendered += ","
    return f"({rendered})"


def render_profiles_as_python(profiles: Sequence[AttackProfile]) -> str:
    lines = [
        "# Generated by `python -m experiments.generate_attack_profile_scaffold ...`.",
        "# Paste entries into the `profiles = [...]` block inside",
        "# `build_scaffolded_attack_profile_library()` after the local `_scaffolded_fan_profile(...)` helper.",
        "GENERATED_ATTACK_PROFILE_CALLS = [",
    ]
    for profile in profiles:
        lines.extend(
            [
                "    _scaffolded_fan_profile(",
                f'        profile_id="{profile.profile_id}",',
                f'        name="{profile.name}",',
                f"        u_pos={_format_tuple(profile.u_pos)},",
                f"        base_bearing_rad={_format_float(profile.base_bearing_rad)},",
                f"        spread_rad={_format_float(profile.spread_rad)},",
                f"        launch_delay_s={_format_float(profile.launch_delay_s)},",
                f"        salvo_interval_s={_format_float(profile.salvo_interval_s)},",
                f"        u_boat_initial_speed_mps={_format_float(profile.u_boat_initial_speed_mps)},",
                "    ),",
            ]
        )
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def render_profiles_as_json(
    profiles: Sequence[AttackProfile],
    *,
    audit_rows: Sequence[dict[str, object]],
    seed: int,
    convoy_profile: str,
    accepted_labels: Sequence[str],
    mode: str = "curated",
) -> str:
    is_intent_mode = mode in INTENT_MODES
    payload = {
        "generator_meta": {
            "seed": int(seed),
            "convoy_profile": str(convoy_profile),
            "accepted_labels": [str(label) for label in accepted_labels],
            "mode": str(mode),
            "source": GENERATOR_SOURCE,
            "generator_version": _generator_version_for_mode(mode),
            "u_boat_initial_speed_mps_options": [float(v) for v in DEFAULT_U_BOAT_SPEED_OPTIONS_MPS],
            "profile_count": len(profiles),
        },
        "profiles": [_explicit_profile_dict(profile) for profile in profiles],
        "audit_rows": list(audit_rows),
    }
    if is_intent_mode:
        payload["intents"] = [dict(row["intent"]) for row in audit_rows if "intent" in row]
    return json.dumps(payload, indent=2) + "\n"


def render_profiles_as_jsonl(
    profiles: Sequence[AttackProfile],
    *,
    audit_rows: Sequence[dict[str, object]],
    seed: int,
    convoy_profile: str,
    accepted_labels: Sequence[str],
    mode: str = "dataset",
) -> str:
    lines: list[str] = []
    for profile, audit_row in zip(profiles, audit_rows):
        audit_payload = dict(audit_row)
        intent_payload = audit_payload.pop("intent", None)
        record = {
            "profile": _explicit_profile_dict(profile),
            "audit": audit_payload,
            "generator_meta": {
                "mode": str(mode),
                "seed": int(seed),
                "convoy_profile": str(convoy_profile),
                "accepted_labels": [str(label) for label in accepted_labels],
                "source": GENERATOR_SOURCE,
                "generator_version": _generator_version_for_mode(mode),
            },
        }
        if intent_payload is not None:
            record["intent"] = intent_payload
        lines.append(json.dumps(record))
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic AttackProfile scaffolds from bounded convoy-attack presets."
    )
    parser.add_argument("--start-index", type=int, default=31, help="Starting numeric profile id (default: 31).")
    parser.add_argument("--count", type=int, default=30, help="Number of profiles to generate (default: 30).")
    parser.add_argument("--seed", type=int, default=1945, help="RNG seed used to shuffle preset combinations.")
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT, help="Profile sampling weight (default: 1.0).")
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(ALL_MODES)),
        default="curated",
        help=(
            "Generation mode. `curated` and `dataset` preserve the current centroid workflow; "
            "`curated_zones` and `random_zones` use v3 target-zone intents; "
            "`random_tactical_v4` samples spawn-first tactical intents."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("python", "json", "jsonl"),
        default=None,
        help="Output format. Defaults to `json` for curated modes and `jsonl` for dataset/random modes.",
    )
    parser.add_argument(
        "--convoy-profile",
        choices=list_convoy_layout_profiles(),
        default="convoy_layout_1",
        help="Convoy profile used for generation-time geometry plausibility audit.",
    )
    parser.add_argument(
        "--accepted-labels",
        type=str,
        default="credible_hit_threat,credible_near_miss",
        help="Comma-separated audit labels accepted during generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. If omitted, content is printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    accepted_labels = tuple(item.strip() for item in str(args.accepted_labels).split(",") if item.strip())
    profiles, audit_rows = generate_attack_profile_scaffolds(
        start_index=int(args.start_index),
        count=int(args.count),
        seed=int(args.seed),
        weight=float(args.weight),
        convoy_profile=str(args.convoy_profile),
        accepted_labels=accepted_labels,
        mode=str(args.mode),
    )
    default_json_modes = {"curated", "curated_zones"}
    output_format = str(args.format) if args.format is not None else ("json" if args.mode in default_json_modes else "jsonl")
    if args.mode in {"dataset", "random_zones", "random_tactical_v4"} and output_format == "python":
        raise ValueError(f"{args.mode} mode does not support --format python; use jsonl or json")
    if output_format == "json":
        rendered = render_profiles_as_json(
            profiles,
            audit_rows=audit_rows,
            seed=int(args.seed),
            convoy_profile=str(args.convoy_profile),
            accepted_labels=accepted_labels,
            mode=str(args.mode),
        )
    elif output_format == "jsonl":
        rendered = render_profiles_as_jsonl(
            profiles,
            audit_rows=audit_rows,
            seed=int(args.seed),
            convoy_profile=str(args.convoy_profile),
            accepted_labels=accepted_labels,
            mode=str(args.mode),
        )
    else:
        rendered = render_profiles_as_python(profiles)
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(profiles)} profiles to {args.output}")


if __name__ == "__main__":
    main()
