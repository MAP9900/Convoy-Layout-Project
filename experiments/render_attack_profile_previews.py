"""Render attack profile preview frames (first/middle/last) with optional parallel workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.attack_profiles import AttackProfile, DEFAULT_ATTACK_PROFILE_LIBRARY
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan, ZigZagPlan
from convoy_sim.profile_audit import audit_attack_profiles
from convoy_sim.simulation import HitSlowdownSpec, init_dynamic_hit_state
from convoy_sim import viz_attack
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render first/middle/last profile preview frames")
    parser.add_argument(
        "--convoy-profile",
        choices=list_convoy_layout_profiles(),
        default="convoy_layout_1",
        help="Convoy layout profile used for rendering/audit",
    )
    parser.add_argument("--run-mode", choices=["fast", "verify"], default="verify")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--sim-duration-s", type=float, default=None)
    parser.add_argument("--hit-dt", type=float, default=None)
    parser.add_argument(
        "--select-profile-ids",
        type=str,
        default=None,
        help="Comma-separated profile IDs to render (e.g. P15,P19,P21)",
    )
    parser.add_argument("--profile-limit", type=int, default=None)
    parser.add_argument("--exclude-implausible", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=1945)
    parser.add_argument("--workers", type=int, default=1, help="Number of process workers (1 = serial)")
    parser.add_argument("--trail-length-s", type=float, default=20.0)
    parser.add_argument("--trail-linewidth", type=float, default=0.8)
    parser.add_argument("--trail-alpha", type=float, default=0.6)
    parser.add_argument("--trail-antialiased", action="store_true")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--view-bounds",
        type=float,
        nargs=4,
        default=(-3500.0, 7000.0, -4500.0, 4500.0),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/frames/attack_profile_previews"),
        help="Root directory for output frame PNGs",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("results/diag/attack_profile_geometry_audit.csv"),
        help="Audit CSV output path",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("results/diag/attack_profile_geometry_audit.json"),
        help="Audit JSON output path",
    )
    parser.add_argument(
        "--hit-report-csv",
        type=Path,
        default=Path("results/diag/attack_profile_hit_report.csv"),
        help="Per-profile hit summary CSV output path",
    )
    return parser.parse_args()


def _build_demo_dynamics(ships_t0: list[Any], t_end_s: float) -> tuple[ConvoyFormation, ConvoyKinematics]:
    formation = ConvoyFormation(
        ships0=ships_t0,
        convoy_origin0=np.array([0.0, 0.0], dtype=float),
        convoy_heading0=0.0,
    )
    route = RoutePlan(legs=[RouteLeg(duration_s=max(120.0, float(t_end_s)), heading_rad=0.0)])
    zigzag = ZigZagPlan(enabled=True, amplitude_rad=0.12, period_s=60.0, phase_s=0.0, waveform="sine")
    kin = ConvoyKinematics(route=route, zigzag=zigzag)
    return formation, kin


def _to_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["flags"] = list(row.get("flags", []))
        payload.append(item)
    return payload


def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile_id",
        "name",
        "mode",
        "u_pos_x",
        "u_pos_y",
        "range_to_centroid_m",
        "intent_bearing_rad",
        "active_bearing_rad",
        "bearing_error_deg",
        "spread_deg",
        "flag_count",
        "flags",
        "severity",
        "suggested_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["flags"] = ";".join(row.get("flags", []))
            writer.writerow(out)


def _frame_indices(sim_duration_s: float, fps: int) -> list[int]:
    last_frame = int(float(sim_duration_s) * int(fps))
    return [1, last_frame // 2, last_frame]


def _write_hit_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile_id",
        "total_hits",
        "ships_hit",
        "torpedoes_that_hit",
        "sim_duration_s",
        "hit_dt",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_one_profile(
    profile: AttackProfile,
    profile_index: int,
    *,
    convoy_profile: str,
    sim_duration_s: float,
    fps: int,
    hit_dt: float,
    frame_indices: list[int],
    output_root: Path,
    rng_seed: int,
    view_bounds: tuple[float, float, float, float] | None,
    trail_length_s: float,
    trail_linewidth: float,
    trail_alpha: float,
    trail_antialiased: bool,
    dpi: int,
) -> tuple[str, list[str], dict[str, Any]]:
    # Ensure headless rendering in worker processes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    ships_t0 = get_convoy_layout_profile(convoy_profile).build_ships()
    dynamics = _build_demo_dynamics(ships_t0, sim_duration_s)

    rng = np.random.default_rng(rng_seed + profile_index)
    torpedoes = profile.build_torpedoes(rng=rng)
    hit_state = init_dynamic_hit_state(0.0)
    u_pos = np.asarray(profile.u_pos, dtype=float)

    out_dir = output_root / profile.profile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written_frames: list[str] = []
    for frame_idx in sorted(frame_indices):
        frame_name = f"frame_{frame_idx:04d}.png"
        frame_time_s = float(frame_idx) / float(fps)
        out_path = out_dir / frame_name

        fig, ax = plt.subplots(figsize=(6, 6), facecolor="lightgrey")
        viz_attack.render_attack_frame(
            ships_t0=ships_t0,
            torpedoes=torpedoes,
            t_global=frame_time_s,
            t_max=sim_duration_s,
            dynamics=dynamics,
            ax=ax,
            color_by="class",
            show_trails=True,
            trail_length_s=float(trail_length_s),
            trail_color="red",
            trail_linewidth=float(trail_linewidth),
            trail_alpha=float(trail_alpha),
            trail_antialiased=bool(trail_antialiased),
            show_footprint=False,
            ship_marker="ship",
            rotate_by_heading=True,
            use_hull_dimensions=True,
            legend_bbox_to_anchor=(0.5, -0.50),
            view_bounds=view_bounds,
            hide_spines=True,
            hit_state=hit_state,
            hit_dt=float(hit_dt),
            apply_hit_slowdown=False,
            hit_slowdown=HitSlowdownSpec(enabled=False, decay_rate=0.02, min_factor=0.4),
            figure_facecolor="lightgrey",
            show_u_boat=True,
            u_boat_position=u_pos,
            u_boat_marker="o",
            u_boat_color="black",
            u_boat_size=24.0,
            u_boat_label="U-boat",
        )
        fig.subplots_adjust(bottom=0.32)
        fig.savefig(out_path, dpi=int(dpi))
        plt.close(fig)
        written_frames.append(frame_name)

    total_hits = int(sum(int(v) for v in hit_state.hit_counts.values()))
    ships_hit = int(len(hit_state.hit_counts))
    torpedoes_that_hit = int(len(hit_state.torpedo_hit_times))
    return (
        profile.profile_id,
        written_frames,
        {
            "profile_id": profile.profile_id,
            "total_hits": total_hits,
            "ships_hit": ships_hit,
            "torpedoes_that_hit": torpedoes_that_hit,
            "sim_duration_s": float(sim_duration_s),
            "hit_dt": float(hit_dt),
        },
    )


def _iter_profiles(args: argparse.Namespace) -> list[AttackProfile]:
    profiles = list(DEFAULT_ATTACK_PROFILE_LIBRARY.profiles)
    if args.select_profile_ids:
        selected = {item.strip() for item in args.select_profile_ids.split(",") if item.strip()}
        profiles = [p for p in profiles if p.profile_id in selected]
    if args.profile_limit is not None:
        profiles = profiles[: int(args.profile_limit)]
    return profiles


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")

    if args.run_mode == "fast":
        sim_duration_s = 300.0 if args.sim_duration_s is None else float(args.sim_duration_s)
        hit_dt = 0.5 if args.hit_dt is None else float(args.hit_dt)
    else:
        sim_duration_s = 600.0 if args.sim_duration_s is None else float(args.sim_duration_s)
        hit_dt = (1.0 / float(args.fps)) if args.hit_dt is None else float(args.hit_dt)

    profiles = _iter_profiles(args)
    ships_for_audit = get_convoy_layout_profile(args.convoy_profile).build_ships()
    audit_rows = audit_attack_profiles(profiles, ships_for_audit)
    audit_by_id = {row["profile_id"]: row for row in audit_rows}

    _write_audit_csv(args.audit_csv, audit_rows)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(_to_jsonable(audit_rows), indent=2), encoding="utf-8")

    if args.exclude_implausible:
        bad = {row["profile_id"] for row in audit_rows if row["suggested_label"] == "implausible_geometry"}
        profiles = [p for p in profiles if p.profile_id not in bad]
        print(f"Excluded {len(bad)} implausible profiles from render pass.")

    frame_indices = _frame_indices(sim_duration_s, int(args.fps))
    view_bounds = tuple(float(v) for v in args.view_bounds) if args.view_bounds is not None else None
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(profiles)} profiles with workers={args.workers} ...")
    saved = 0
    hit_rows: list[dict[str, Any]] = []

    if args.workers == 1:
        for idx, profile in enumerate(profiles):
            profile_id, frame_names, hit_row = _render_one_profile(
                profile,
                idx,
                convoy_profile=args.convoy_profile,
                sim_duration_s=sim_duration_s,
                fps=int(args.fps),
                hit_dt=hit_dt,
                frame_indices=frame_indices,
                output_root=args.output_root,
                rng_seed=int(args.rng_seed),
                view_bounds=view_bounds,
                trail_length_s=float(args.trail_length_s),
                trail_linewidth=float(args.trail_linewidth),
                trail_alpha=float(args.trail_alpha),
                trail_antialiased=bool(args.trail_antialiased),
                dpi=int(args.dpi),
            )
            row = audit_by_id.get(profile_id)
            label = row["suggested_label"] if row is not None else "n/a"
            err = row["bearing_error_deg"] if row is not None else float("nan")
            print(
                f"Completed {profile_id} [{label}, err={err:.1f}deg, "
                f"hits={hit_row['total_hits']}, ships_hit={hit_row['ships_hit']}]: "
                f"{', '.join(frame_names)}"
            )
            saved += len(frame_names)
            hit_rows.append(hit_row)
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                futures = [
                    pool.submit(
                        _render_one_profile,
                        profile,
                        idx,
                        convoy_profile=args.convoy_profile,
                        sim_duration_s=sim_duration_s,
                        fps=int(args.fps),
                        hit_dt=hit_dt,
                        frame_indices=frame_indices,
                        output_root=args.output_root,
                        rng_seed=int(args.rng_seed),
                        view_bounds=view_bounds,
                        trail_length_s=float(args.trail_length_s),
                        trail_linewidth=float(args.trail_linewidth),
                        trail_alpha=float(args.trail_alpha),
                        trail_antialiased=bool(args.trail_antialiased),
                        dpi=int(args.dpi),
                    )
                    for idx, profile in enumerate(profiles)
                ]
                for future in as_completed(futures):
                    profile_id, frame_names, hit_row = future.result()
                    row = audit_by_id.get(profile_id)
                    label = row["suggested_label"] if row is not None else "n/a"
                    err = row["bearing_error_deg"] if row is not None else float("nan")
                    print(
                        f"Completed {profile_id} [{label}, err={err:.1f}deg, "
                        f"hits={hit_row['total_hits']}, ships_hit={hit_row['ships_hit']}]: "
                        f"{', '.join(frame_names)}"
                    )
                    saved += len(frame_names)
                    hit_rows.append(hit_row)
        except (PermissionError, OSError) as exc:
            print(
                "Parallel worker startup failed; falling back to serial rendering. "
                f"Reason: {exc}"
            )
            for idx, profile in enumerate(profiles):
                profile_id, frame_names, hit_row = _render_one_profile(
                    profile,
                    idx,
                    convoy_profile=args.convoy_profile,
                    sim_duration_s=sim_duration_s,
                    fps=int(args.fps),
                    hit_dt=hit_dt,
                    frame_indices=frame_indices,
                    output_root=args.output_root,
                    rng_seed=int(args.rng_seed),
                    view_bounds=view_bounds,
                    trail_length_s=float(args.trail_length_s),
                    trail_linewidth=float(args.trail_linewidth),
                    trail_alpha=float(args.trail_alpha),
                    trail_antialiased=bool(args.trail_antialiased),
                    dpi=int(args.dpi),
                )
                row = audit_by_id.get(profile_id)
                label = row["suggested_label"] if row is not None else "n/a"
                err = row["bearing_error_deg"] if row is not None else float("nan")
                print(
                    f"Completed {profile_id} [{label}, err={err:.1f}deg, "
                    f"hits={hit_row['total_hits']}, ships_hit={hit_row['ships_hit']}]: "
                    f"{', '.join(frame_names)}"
                )
                saved += len(frame_names)
                hit_rows.append(hit_row)

    n_implausible = sum(1 for row in audit_rows if row["suggested_label"] == "implausible_geometry")
    n_near_miss = sum(1 for row in audit_rows if row["suggested_label"] == "credible_near_miss")
    n_hit_threat = sum(1 for row in audit_rows if row["suggested_label"] == "credible_hit_threat")

    print(f"Wrote CSV: {args.audit_csv}")
    print(f"Wrote JSON: {args.audit_json}")
    _write_hit_report_csv(args.hit_report_csv, sorted(hit_rows, key=lambda row: row["profile_id"]))
    print(f"Wrote hit report CSV: {args.hit_report_csv}")
    print(
        "Summary:",
        {
            "profiles": len(audit_rows),
            "credible_hit_threat": n_hit_threat,
            "credible_near_miss": n_near_miss,
            "implausible_geometry": n_implausible,
        },
    )
    print(f"Rendered {saved} total frame images across {len(profiles)} profiles.")


if __name__ == "__main__":
    # Avoid over-subscribing BLAS worker threads inside child processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
