"""Audit attack profile geometry plausibility against a selected convoy layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from convoy_sim.attack_profiles import DEFAULT_ATTACK_PROFILE_LIBRARY
from convoy_sim.profile_audit import AuditThresholds, audit_attack_profiles
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit attack profile geometry plausibility")
    parser.add_argument(
        "--convoy-profile",
        choices=list_convoy_layout_profiles(),
        default="convoy_layout_1",
        help="Convoy layout profile used for centroid/geometry checks",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/diag/attack_profile_geometry_audit.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/diag/attack_profile_geometry_audit.json"),
        help="JSON output path",
    )
    parser.add_argument("--near-range-m", type=float, default=2500.0)
    parser.add_argument("--mid-range-m", type=float, default=4500.0)
    parser.add_argument("--near-max-error-deg", type=float, default=15.0)
    parser.add_argument("--mid-max-error-deg", type=float, default=30.0)
    parser.add_argument("--far-max-error-deg", type=float, default=45.0)
    parser.add_argument("--fan-margin-deg", type=float, default=3.0)
    return parser.parse_args()


def _to_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["flags"] = list(row.get("flags", []))
        payload.append(item)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def main() -> None:
    args = parse_args()

    ships = get_convoy_layout_profile(args.convoy_profile).build_ships()
    thresholds = AuditThresholds(
        near_range_m=float(args.near_range_m),
        mid_range_m=float(args.mid_range_m),
        near_max_error_deg=float(args.near_max_error_deg),
        mid_max_error_deg=float(args.mid_max_error_deg),
        far_max_error_deg=float(args.far_max_error_deg),
        fan_margin_deg=float(args.fan_margin_deg),
    )

    rows = audit_attack_profiles(DEFAULT_ATTACK_PROFILE_LIBRARY.profiles, ships, thresholds=thresholds)

    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_to_jsonable(rows), indent=2), encoding="utf-8")

    n_implausible = sum(1 for row in rows if row["suggested_label"] == "implausible_geometry")
    n_near_miss = sum(1 for row in rows if row["suggested_label"] == "credible_near_miss")
    n_hit_threat = sum(1 for row in rows if row["suggested_label"] == "credible_hit_threat")

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSON: {args.output_json}")
    print(
        "Summary:",
        {
            "profiles": len(rows),
            "credible_hit_threat": n_hit_threat,
            "credible_near_miss": n_near_miss,
            "implausible_geometry": n_implausible,
        },
    )
    print("Top 5 highest-severity profiles:")
    for row in rows[:5]:
        print(
            f"  {row['profile_id']}: severity={row['severity']:.1f}, "
            f"error={row['bearing_error_deg']:.1f}deg, flags={row['flags']}"
        )


if __name__ == "__main__":
    main()
