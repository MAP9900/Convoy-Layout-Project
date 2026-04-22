from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile


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
        gyro_straight_run_m=30.0,
    ),
    SalvoPreset(
        "uniform_medium_4",
        n=4,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(5.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=0.9,
        salvo_interval_s=2.0,
        gyro_straight_run_m=30.0,
    ),
    SalvoPreset(
        "uniform_wide_4",
        n=4,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(6.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=1.4,
        salvo_interval_s=3.0,
        gyro_straight_run_m=30.0,
    ),
    SalvoPreset(
        "explicit_asym_4",
        n=4,
        spread_doctrine="explicit_divergent",
        spread_rad=0.0,
        per_torpedo_heading_offsets_rad=tuple(float(np.deg2rad(v)) for v in (-3.0, -1.0, 1.5, 4.0)),
        launch_delay_s=1.0,
        salvo_interval_s=2.0,
        gyro_straight_run_m=30.0,
    ),
    SalvoPreset(
        "heavy_uniform_6",
        n=6,
        spread_doctrine="uniform_divergent",
        spread_rad=float(np.deg2rad(6.0)),
        per_torpedo_heading_offsets_rad=(),
        launch_delay_s=0.8,
        salvo_interval_s=1.5,
        gyro_straight_run_m=30.0,
    ),
)

TORPEDO_SPEED_MPS = 15.4333
TORPEDO_MAX_RUN_TIME_S = 486.0
DEFAULT_WEIGHT = 1.0
DEFAULT_U_BOAT_SPEED_MPS = 2.0
DEFAULT_SUB_LENGTH_M = 67.0
DEFAULT_SUB_BEAM_M = 6.5
DEFAULT_MAX_BOW_OFFSET_DEG = 15.0


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _all_profile_specs() -> list[tuple[ApproachPreset, RangePreset, SalvoPreset]]:
    specs: list[tuple[ApproachPreset, RangePreset, SalvoPreset]] = []
    for approach in APPROACH_PRESETS:
        for range_preset in RANGE_PRESETS:
            for salvo in SALVO_PRESETS:
                specs.append((approach, range_preset, salvo))
    return specs


def generate_attack_profile_scaffolds(
    *,
    start_index: int = 31,
    count: int = 30,
    seed: int = 1945,
    weight: float = DEFAULT_WEIGHT,
) -> list[AttackProfile]:
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    specs = _all_profile_specs()
    if count > len(specs):
        raise ValueError(
            f"count={count} exceeds available scaffold combinations ({len(specs)}); add more presets first"
        )

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(specs))[:count])
    profiles: list[AttackProfile] = []
    for idx, spec_index in enumerate(order, start=start_index):
        approach, range_preset, salvo = specs[int(spec_index)]
        angle_rad = float(np.deg2rad(approach.angle_deg + rng.uniform(-4.0, 4.0)))
        radius_m = float(range_preset.radius_m + rng.uniform(-180.0, 180.0))
        u_pos = (
            float(radius_m * np.cos(angle_rad)),
            float(radius_m * np.sin(angle_rad)),
        )
        bearing_to_origin = float(np.arctan2(-u_pos[1], -u_pos[0]))
        base_bearing_rad = _wrap_angle_rad(bearing_to_origin + float(np.deg2rad(rng.uniform(-4.0, 4.0))))
        profile = AttackProfile(
            profile_id=f"P{idx:02d}",
            name=f"profile_{idx:02d}_{approach.name}_{range_preset.name}_{salvo.name}",
            weight=float(weight),
            mode="fan",
            u_pos=u_pos,
            n=int(salvo.n),
            speed=TORPEDO_SPEED_MPS,
            max_run_time=TORPEDO_MAX_RUN_TIME_S,
            base_bearing_rad=base_bearing_rad,
            spread_rad=float(salvo.spread_rad),
            spread_doctrine=str(salvo.spread_doctrine),
            per_torpedo_heading_offsets_rad=tuple(float(v) for v in salvo.per_torpedo_heading_offsets_rad),
            launch_delay_s=float(salvo.launch_delay_s),
            salvo_interval_s=float(salvo.salvo_interval_s),
            u_boat_mode="moving",
            u_boat_initial_heading_rad=base_bearing_rad,
            u_boat_initial_speed_mps=DEFAULT_U_BOAT_SPEED_MPS,
            sub_length_m=DEFAULT_SUB_LENGTH_M,
            sub_beam_m=DEFAULT_SUB_BEAM_M,
            launch_from="bow",
            max_bow_offset_deg=DEFAULT_MAX_BOW_OFFSET_DEG,
            gyro_straight_run_m=float(salvo.gyro_straight_run_m),
        )
        profiles.append(profile)
    return profiles


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
        "from convoy_sim.attack_profiles import AttackProfile",
        "",
        "# Generated by `python -m experiments.generate_attack_profile_scaffold ...`.",
        "# Paste entries into `build_scaffolded_attack_profile_library()` or adapt as needed.",
        "GENERATED_ATTACK_PROFILES = [",
    ]
    for profile in profiles:
        lines.extend(
            [
                "    AttackProfile(",
                f'        profile_id="{profile.profile_id}",',
                f'        name="{profile.name}",',
                f"        weight={_format_float(profile.weight)},",
                f'        mode="{profile.mode}",',
                f"        u_pos={_format_tuple(profile.u_pos)},",
                f"        n={profile.n},",
                f"        speed={_format_float(profile.speed)},",
                f"        max_run_time={_format_float(profile.max_run_time)},",
                f"        base_bearing_rad={_format_float(profile.base_bearing_rad)},",
                f'        spread_doctrine="{profile.spread_doctrine}",',
                f"        spread_rad={_format_float(profile.spread_rad)},",
            ]
        )
        if profile.per_torpedo_heading_offsets_rad:
            lines.append(
                f"        per_torpedo_heading_offsets_rad={_format_tuple(profile.per_torpedo_heading_offsets_rad)},"
            )
        lines.extend(
            [
                f"        launch_delay_s={_format_float(profile.launch_delay_s)},",
                f"        salvo_interval_s={_format_float(profile.salvo_interval_s)},",
                f'        u_boat_mode="{profile.u_boat_mode}",',
                f"        u_boat_initial_heading_rad={_format_float(profile.u_boat_initial_heading_rad)},",
                f"        u_boat_initial_speed_mps={_format_float(profile.u_boat_initial_speed_mps)},",
                f"        sub_length_m={_format_float(profile.sub_length_m)},",
                f"        sub_beam_m={_format_float(profile.sub_beam_m)},",
                f'        launch_from="{profile.launch_from}",',
                f"        max_bow_offset_deg={_format_float(profile.max_bow_offset_deg)},",
                f"        gyro_straight_run_m={_format_float(profile.gyro_straight_run_m)},",
                "    ),",
            ]
        )
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def render_profiles_as_json(profiles: Sequence[AttackProfile]) -> str:
    payload = {"profiles": [profile.to_dict() for profile in profiles]}
    return json.dumps(payload, indent=2) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic AttackProfile scaffolds from bounded convoy-attack presets."
    )
    parser.add_argument("--start-index", type=int, default=31, help="Starting numeric profile id (default: 31).")
    parser.add_argument("--count", type=int, default=30, help="Number of profiles to generate (default: 30).")
    parser.add_argument("--seed", type=int, default=1945, help="RNG seed used to shuffle preset combinations.")
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT, help="Profile sampling weight (default: 1.0).")
    parser.add_argument(
        "--format",
        choices=("python", "json"),
        default="python",
        help="Output format. `python` is paste-friendly for attack_profiles.py; `json` is machine-friendly.",
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
    profiles = generate_attack_profile_scaffolds(
        start_index=int(args.start_index),
        count=int(args.count),
        seed=int(args.seed),
        weight=float(args.weight),
    )
    if args.format == "json":
        rendered = render_profiles_as_json(profiles)
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
