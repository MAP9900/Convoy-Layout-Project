from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.profile_audit import AuditThresholds, audit_attack_profiles
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
MIN_SPAWN_CLEARANCE_M = 250.0
RANGE_BAND_TOLERANCE_M = 250.0
GENERATOR_SOURCE = "generate_attack_profile_scaffold"
GENERATOR_VERSION = "v2"


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
    if mode not in {"curated", "dataset"}:
        raise ValueError("mode must be one of: curated, dataset")
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    specs = _all_profile_specs()
    rng = np.random.default_rng(seed)
    ships = get_convoy_layout_profile(convoy_profile).build_ships()
    accepted = set(accepted_labels)
    if not accepted:
        raise ValueError("accepted_labels must not be empty")

    max_attempts = max(count * 20, len(specs) * 2)
    profiles: list[AttackProfile] = []
    accepted_audit_rows: list[dict[str, object]] = []
    used_signatures: set[tuple[str, str, str, float]] = set()
    next_index = start_index
    attempts = 0

    while len(profiles) < count:
        if attempts >= max_attempts:
            raise ValueError(
                f"Unable to generate {count} plausible profiles after {attempts} attempts; "
                "broaden presets or relax audit thresholds."
            )
        attempts += 1
        approach, range_preset, salvo = specs[int(rng.integers(0, len(specs)))]
        u_boat_initial_speed_mps = float(rng.choice(DEFAULT_U_BOAT_SPEED_OPTIONS_MPS))
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
            signature = (
                approach.name,
                range_preset.name,
                f"{u_boat_initial_speed_mps:.1f}",
                f"{rng.integers(0, 1_000_000)}",
            )
            angle_jitter_deg = 10.0
            radius_jitter_m = 350.0
            spread_deg = float(rng.choice(np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0], dtype=float)))
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
        base_bearing_jitter_deg = 4.0 if mode == "curated" else 8.0
        base_bearing_rad = _wrap_angle_rad(
            bearing_to_origin + float(np.deg2rad(rng.uniform(-base_bearing_jitter_deg, base_bearing_jitter_deg)))
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
        if str(audit_row["suggested_label"]) not in accepted:
            continue
        used_signatures.add(signature)
        profiles.append(profile)
        accepted_audit_rows.append(audit_row)
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
) -> str:
    payload = {
        "generator_meta": {
            "seed": int(seed),
            "convoy_profile": str(convoy_profile),
            "accepted_labels": [str(label) for label in accepted_labels],
            "mode": "curated",
            "source": GENERATOR_SOURCE,
            "generator_version": GENERATOR_VERSION,
            "u_boat_initial_speed_mps_options": [float(v) for v in DEFAULT_U_BOAT_SPEED_OPTIONS_MPS],
            "profile_count": len(profiles),
        },
        "profiles": [_explicit_profile_dict(profile) for profile in profiles],
        "audit_rows": list(audit_rows),
    }
    return json.dumps(payload, indent=2) + "\n"


def render_profiles_as_jsonl(
    profiles: Sequence[AttackProfile],
    *,
    audit_rows: Sequence[dict[str, object]],
    seed: int,
    convoy_profile: str,
    accepted_labels: Sequence[str],
) -> str:
    lines: list[str] = []
    for profile, audit_row in zip(profiles, audit_rows):
        record = {
            "profile": _explicit_profile_dict(profile),
            "audit": dict(audit_row),
            "generator_meta": {
                "mode": "dataset",
                "seed": int(seed),
                "convoy_profile": str(convoy_profile),
                "accepted_labels": [str(label) for label in accepted_labels],
                "source": GENERATOR_SOURCE,
                "generator_version": GENERATOR_VERSION,
            },
        }
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
        choices=("curated", "dataset"),
        default="curated",
        help="Generation mode. `curated` preserves the current helper-style library workflow; `dataset` emits synthetic training records.",
    )
    parser.add_argument(
        "--format",
        choices=("python", "json", "jsonl"),
        default=None,
        help="Output format. Defaults to `json` for curated mode and `jsonl` for dataset mode.",
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
    output_format = str(args.format) if args.format is not None else ("json" if args.mode == "curated" else "jsonl")
    if args.mode == "dataset" and output_format == "python":
        raise ValueError("dataset mode does not support --format python; use jsonl or json")
    if output_format == "json":
        rendered = render_profiles_as_json(
            profiles,
            audit_rows=audit_rows,
            seed=int(args.seed),
            convoy_profile=str(args.convoy_profile),
            accepted_labels=accepted_labels,
        )
    elif output_format == "jsonl":
        rendered = render_profiles_as_jsonl(
            profiles,
            audit_rows=audit_rows,
            seed=int(args.seed),
            convoy_profile=str(args.convoy_profile),
            accepted_labels=accepted_labels,
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
