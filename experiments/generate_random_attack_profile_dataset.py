"""Profile-first random attack-profile dataset generator.

This module is the random baseline counterpart to ``random_tactical_v4``.  It
samples U-boat spawn, bearing, spread, and timing directly, then labels accepted
records from the moving zig-zag dynamic outcome audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.profile_audit import AuditThresholds, audit_attack_profiles
from convoy_sim.profile_outcome_audit import OutcomeAuditConfig, OutcomeAuditContext
from convoy_sim.target_zones import ConvoyEnvelope, ConvoyFrame, convoy_frame_and_envelope
from experiments.generate_attack_profile_scaffold import (
    DEFAULT_MAX_BOW_OFFSET_DEG,
    DEFAULT_SUB_BEAM_M,
    DEFAULT_SUB_LENGTH_M,
    DEFAULT_U_BOAT_SPEED_OPTIONS_MPS,
    HIT_THREAT_LABEL,
    INTENTIONAL_MISS_LABEL,
    MIN_SPAWN_CLEARANCE_M,
    NEAR_MISS_LABEL,
    TORPEDO_MAX_RUN_TIME_S,
    TORPEDO_SPEED_MPS,
)
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


RANDOM_BASELINE_MODE = "random_profile_v1"
RANDOM_BASELINE_SOURCE = "generate_random_attack_profile_dataset"
RANDOM_BASELINE_VERSION = "v1"
RANDOM_LABELS = (HIT_THREAT_LABEL, NEAR_MISS_LABEL, INTENTIONAL_MISS_LABEL)
RANDOM_HIT_THREAT_FRACTION = 0.65
RANDOM_NEAR_MISS_FRACTION = 0.25
RANDOM_OUTCOME_T_MAX_S = 600.0
RANDOM_OUTCOME_HIT_DT_S = 0.5
RANDOM_OUTCOME_NEAR_MISS_MARGIN_M = 250.0


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _label_targets(count: int) -> dict[str, int]:
    hit_count = int(round(int(count) * RANDOM_HIT_THREAT_FRACTION))
    near_count = int(round(int(count) * RANDOM_NEAR_MISS_FRACTION))
    return {
        HIT_THREAT_LABEL: hit_count,
        NEAR_MISS_LABEL: near_count,
        INTENTIONAL_MISS_LABEL: int(count) - hit_count - near_count,
    }


def _choose_remaining_label(rng: np.random.Generator, targets: Mapping[str, int], counts: Counter[str]) -> str:
    remaining = {
        label: int(targets[label] - counts[label])
        for label in RANDOM_LABELS
        if int(targets[label] - counts[label]) > 0
    }
    if not remaining:
        raise StopIteration
    labels = list(remaining)
    weights = np.asarray([float(remaining[label]) for label in labels], dtype=float)
    weights = weights / float(np.sum(weights))
    return str(rng.choice(np.asarray(labels, dtype=object), p=weights))


def _inside_envelope(local: np.ndarray, envelope: ConvoyEnvelope) -> bool:
    return bool(
        envelope.min_x <= float(local[0]) <= envelope.max_x
        and envelope.min_y <= float(local[1]) <= envelope.max_y
    )


def _nearest_ship_clearance_m(local: np.ndarray, local_positions: np.ndarray) -> float:
    distances = np.linalg.norm(local_positions - np.asarray(local, dtype=float), axis=1)
    return float(np.min(distances))


def _spawn_region(local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    if _inside_envelope(local, envelope):
        return "inside_convoy_envelope"
    dx_ahead = float(local[0] - envelope.max_x)
    dx_astern = float(envelope.min_x - local[0])
    dy_port = float(local[1] - envelope.max_y)
    dy_starboard = float(envelope.min_y - local[1])
    distances = {
        "ahead_random": dx_ahead,
        "astern_random": dx_astern,
        "port_random": dy_port,
        "starboard_random": dy_starboard,
    }
    return max(distances.items(), key=lambda item: item[1])[0]


def _approach_side(local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    region = _spawn_region(local, envelope)
    if region == "inside_convoy_envelope":
        return "inside"
    return region.replace("_random", "")


def _sample_spawn_local(
    rng: np.random.Generator,
    envelope: ConvoyEnvelope,
    local_positions: np.ndarray,
    *,
    min_clearance_m: float,
    max_attempts: int = 300,
) -> tuple[np.ndarray, str, float]:
    x_pad = max(4500.0, 2.0 * float(envelope.spacing_x_m))
    y_pad = max(6500.0, 3.0 * float(envelope.spacing_y_m))
    for _ in range(int(max_attempts)):
        if float(rng.random()) < 0.18:
            local = np.asarray(
                [
                    float(rng.uniform(envelope.min_x, envelope.max_x)),
                    float(rng.uniform(envelope.min_y, envelope.max_y)),
                ],
                dtype=float,
            )
        else:
            local = np.asarray(
                [
                    float(rng.uniform(envelope.min_x - x_pad, envelope.max_x + x_pad)),
                    float(rng.uniform(envelope.min_y - y_pad, envelope.max_y + y_pad)),
                ],
                dtype=float,
            )
        clearance_m = _nearest_ship_clearance_m(local, local_positions)
        if clearance_m >= float(min_clearance_m):
            return local, _spawn_region(local, envelope), clearance_m
    raise ValueError("Unable to sample random spawn satisfying minimum clearance")


def _sample_reference_point(
    rng: np.random.Generator,
    ships: Sequence[object],
    frame: ConvoyFrame,
    envelope: ConvoyEnvelope,
) -> tuple[np.ndarray, str]:
    draw = float(rng.random())
    if draw < 0.55:
        ship = ships[int(rng.integers(0, len(ships)))]
        point = np.asarray(getattr(ship, "position"), dtype=float)
        return point, "random_ship"
    if draw < 0.85:
        local = np.asarray(
            [
                float(rng.uniform(envelope.min_x, envelope.max_x)),
                float(rng.uniform(envelope.min_y, envelope.max_y)),
            ],
            dtype=float,
        )
        return frame.local_to_world(local), "random_convoy_box"
    center = frame.local_to_world(
        np.asarray(
            [
                0.5 * (float(envelope.min_x) + float(envelope.max_x)),
                0.5 * (float(envelope.min_y) + float(envelope.max_y)),
            ],
            dtype=float,
        )
    )
    return center, "convoy_center"


def _profile_params_for_hint(label_hint: str, rng: np.random.Generator) -> tuple[float, float]:
    if label_hint == HIT_THREAT_LABEL:
        bearing_error_deg = float(rng.normal(0.0, 7.5))
        spread_deg = float(rng.uniform(2.5, 8.0))
    elif label_hint == NEAR_MISS_LABEL:
        sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        bearing_error_deg = float(sign * rng.uniform(7.5, 22.0))
        spread_deg = float(rng.uniform(1.5, 5.0))
    elif label_hint == INTENTIONAL_MISS_LABEL:
        sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        bearing_error_deg = float(sign * rng.uniform(25.0, 85.0))
        spread_deg = float(rng.uniform(1.0, 4.0))
    else:
        raise ValueError(f"Unsupported label hint: {label_hint}")
    return bearing_error_deg, spread_deg


def _outcome_to_dataset_label(actual_label: str) -> str:
    if actual_label == "credible_hit_threat":
        return HIT_THREAT_LABEL
    if actual_label == "credible_near_miss":
        return NEAR_MISS_LABEL
    if actual_label == "miss":
        return INTENTIONAL_MISS_LABEL
    return ""


def _build_profile(
    *,
    profile_id: str,
    name: str,
    u_pos: tuple[float, float],
    base_bearing_rad: float,
    spread_rad: float,
    launch_delay_s: float,
    salvo_interval_s: float,
    u_boat_initial_speed_mps: float,
) -> AttackProfile:
    return AttackProfile(
        profile_id=profile_id,
        name=name,
        weight=1.0,
        mode="fan",
        u_pos=u_pos,
        n=4,
        speed=TORPEDO_SPEED_MPS,
        max_run_time=TORPEDO_MAX_RUN_TIME_S,
        base_bearing_rad=float(base_bearing_rad),
        spread_rad=float(spread_rad),
        spread_doctrine="uniform_divergent",
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=float(launch_delay_s),
        salvo_interval_s=float(salvo_interval_s),
        u_boat_mode="moving",
        u_boat_initial_heading_rad=float(base_bearing_rad),
        u_boat_initial_speed_mps=float(u_boat_initial_speed_mps),
        sub_length_m=DEFAULT_SUB_LENGTH_M,
        sub_beam_m=DEFAULT_SUB_BEAM_M,
        launch_from="bow",
        max_bow_offset_deg=DEFAULT_MAX_BOW_OFFSET_DEG,
        gyro_straight_run_m=10.0,
    )


def _profile_to_dict(profile: AttackProfile) -> dict[str, Any]:
    payload = profile.to_dict()
    payload["spread_doctrine"] = str(profile.spread_doctrine)
    payload["per_torpedo_heading_offsets_rad"] = [
        float(value) for value in profile.per_torpedo_heading_offsets_rad
    ]
    return payload


def generate_random_baseline_profiles(
    *,
    count: int,
    seed: int = 1945,
    start_index: int = 1,
    convoy_profile: str = "convoy_layout_1",
    min_clearance_m: float = MIN_SPAWN_CLEARANCE_M,
    max_attempts: int | None = None,
) -> tuple[list[AttackProfile], list[dict[str, Any]], dict[str, Any]]:
    """Generate profile-first random records with outcome-derived labels."""

    if count <= 0:
        raise ValueError("count must be positive")
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    rng = np.random.default_rng(int(seed))
    ships = get_convoy_layout_profile(convoy_profile).build_ships()
    frame, envelope, local_positions = convoy_frame_and_envelope(ships)
    context = OutcomeAuditContext.from_ships(
        ships,
        cfg=OutcomeAuditConfig(
            t_max_s=RANDOM_OUTCOME_T_MAX_S,
            hit_dt_s=RANDOM_OUTCOME_HIT_DT_S,
            near_miss_margin_m=RANDOM_OUTCOME_NEAR_MISS_MARGIN_M,
            max_hits_per_torpedo=1,
            zigzag_enabled=True,
        ),
    )
    targets = _label_targets(int(count))
    counts: Counter[str] = Counter()
    attempts = 0
    max_attempts_resolved = int(max_attempts or max(int(count) * 4000, 10_000))
    profiles: list[AttackProfile] = []
    audit_rows: list[dict[str, Any]] = []
    next_index = int(start_index)

    while len(profiles) < int(count):
        if attempts >= max_attempts_resolved:
            raise ValueError(
                f"Unable to generate {count} random baseline profiles after {attempts} attempts; "
                f"current label counts: {dict(counts)}"
            )
        attempts += 1
        try:
            label_hint = _choose_remaining_label(rng, targets, counts)
        except StopIteration:
            break
        spawn_local, spawn_region, clearance_m = _sample_spawn_local(
            rng,
            envelope,
            local_positions,
            min_clearance_m=float(min_clearance_m),
        )
        u_pos_arr = frame.local_to_world(spawn_local)
        reference_point, reference_kind = _sample_reference_point(rng, ships, frame, envelope)
        bearing_to_reference = float(np.arctan2(reference_point[1] - u_pos_arr[1], reference_point[0] - u_pos_arr[0]))
        bearing_error_deg, spread_deg = _profile_params_for_hint(label_hint, rng)
        base_bearing_rad = _wrap_angle_rad(bearing_to_reference + float(np.deg2rad(bearing_error_deg)))
        profile_id = f"R{next_index:06d}"
        profile = _build_profile(
            profile_id=profile_id,
            name=f"random_profile_{next_index:06d}_{spawn_region}_{reference_kind}",
            u_pos=(float(u_pos_arr[0]), float(u_pos_arr[1])),
            base_bearing_rad=base_bearing_rad,
            spread_rad=float(np.deg2rad(spread_deg)),
            launch_delay_s=float(rng.choice(np.round(np.arange(0.5, 2.51, 0.1), 1))),
            salvo_interval_s=float(rng.choice(np.array([1.5, 2.0, 2.5, 3.0], dtype=float))),
            u_boat_initial_speed_mps=float(rng.choice(DEFAULT_U_BOAT_SPEED_OPTIONS_MPS)),
        )
        base_intent = {
            "target_zone_id": f"RB{next_index:06d}_{spawn_region}_{reference_kind}",
            "target_zone_kind": "profile_first_random",
            "target_ship_ids": [],
            "target_point": [float(reference_point[0]), float(reference_point[1])],
            "target_local": [float(v) for v in frame.world_to_local(reference_point)],
            "spawn_local": [float(spawn_local[0]), float(spawn_local[1])],
            "approach_side": _approach_side(spawn_local, envelope),
            "approach_lane": f"{spawn_region}:{reference_kind}",
            "range_to_target_m": float(np.linalg.norm(reference_point - u_pos_arr)),
            "planned_bearing_error_deg": float(bearing_error_deg),
            "intended_label": "",
            "convoy_heading_rad": float(frame.heading_rad),
            "spawn_region": spawn_region,
            "inside_convoy_envelope": _inside_envelope(spawn_local, envelope),
            "nearest_ship_clearance_m": float(clearance_m),
            "random_reference_kind": reference_kind,
            "proposal_label_hint": label_hint,
            "profile_first_outcome_label": True,
        }
        outcome_row = context.audit_profile(
            profile,
            intent=base_intent,
            rng_seed=int(seed),
            profile_index=len(profiles),
        )
        dataset_label = _outcome_to_dataset_label(str(outcome_row["actual_outcome_label"]))
        if dataset_label != label_hint:
            continue
        if counts[dataset_label] >= targets[dataset_label]:
            continue

        hit_ship_ids = list(outcome_row.get("hit_ship_ids", []))
        intent = dict(base_intent)
        intent["intended_label"] = dataset_label
        intent["target_ship_ids"] = hit_ship_ids if dataset_label == HIT_THREAT_LABEL else []
        static_audit = audit_attack_profiles(
            [profile],
            ships,
            intents=[intent],
            thresholds=AuditThresholds(),
        )[0]
        static_suggested_label = str(static_audit["suggested_label"])
        static_audit["random_static_suggested_label"] = static_suggested_label
        static_audit["suggested_label"] = dataset_label
        static_audit["intent"] = intent
        outcome_payload = dict(outcome_row)
        outcome_payload["intended_label"] = dataset_label
        outcome_payload["target_ship_ids"] = list(intent["target_ship_ids"])
        outcome_payload["outcome_matches_intent"] = True
        outcome_payload["passes_outcome_gate"] = True
        static_audit["outcome"] = outcome_payload
        profiles.append(profile)
        audit_rows.append(static_audit)
        counts[dataset_label] += 1
        next_index += 1

    stats = {
        "requested_count": int(count),
        "accepted_count": int(len(profiles)),
        "attempts": int(attempts),
        "acceptance_rate": float(len(profiles) / max(attempts, 1)),
        "label_counts": dict(counts),
        "label_targets": dict(targets),
        "mode": RANDOM_BASELINE_MODE,
        "generator_version": RANDOM_BASELINE_VERSION,
    }
    return profiles, audit_rows, stats


def render_random_baseline_jsonl(
    profiles: Sequence[AttackProfile],
    *,
    audit_rows: Sequence[dict[str, Any]],
    seed: int,
    convoy_profile: str,
    stats: Mapping[str, Any] | None = None,
) -> str:
    """Render generated random baseline profiles as JSONL dataset records."""

    lines: list[str] = []
    for profile, audit_row in zip(profiles, audit_rows):
        audit_payload = dict(audit_row)
        intent_payload = audit_payload.pop("intent", None)
        outcome_payload = audit_payload.pop("outcome", None)
        record = {
            "profile": _profile_to_dict(profile),
            "audit": audit_payload,
            "generator_meta": {
                "mode": RANDOM_BASELINE_MODE,
                "seed": int(seed),
                "convoy_profile": str(convoy_profile),
                "accepted_labels": list(RANDOM_LABELS),
                "source": RANDOM_BASELINE_SOURCE,
                "generator_version": RANDOM_BASELINE_VERSION,
            },
        }
        if stats is not None:
            record["generator_meta"]["acceptance_rate"] = float(stats.get("acceptance_rate", 0.0))
        if intent_payload is not None:
            record["intent"] = intent_payload
        if outcome_payload is not None:
            record["outcome"] = outcome_payload
        lines.append(json.dumps(record))
    return "\n".join(lines) + ("\n" if lines else "")


def generate_random_baseline_records(
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate random baseline JSONL-style records in memory."""

    profiles, audit_rows, stats = generate_random_baseline_profiles(**kwargs)
    text = render_random_baseline_jsonl(
        profiles,
        audit_rows=audit_rows,
        seed=int(kwargs.get("seed", 1945)),
        convoy_profile=str(kwargs.get("convoy_profile", "convoy_layout_1")),
        stats=stats,
    )
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate profile-first random baseline attack-profile JSONL.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1945)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--convoy-profile", choices=list_convoy_layout_profiles(), default="convoy_layout_1")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profiles, audit_rows, stats = generate_random_baseline_profiles(
        count=int(args.count),
        seed=int(args.seed),
        start_index=int(args.start_index),
        convoy_profile=str(args.convoy_profile),
    )
    text = render_random_baseline_jsonl(
        profiles,
        audit_rows=audit_rows,
        seed=int(args.seed),
        convoy_profile=str(args.convoy_profile),
        stats=stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
