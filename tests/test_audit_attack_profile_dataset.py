from __future__ import annotations

import json
from pathlib import Path

from experiments.audit_attack_profile_dataset import (
    flatten_dataset_records,
    load_dataset_records,
    summarize_flat_rows,
    write_dataset_audit_outputs,
)


def _sample_record(
    *,
    profile_id: str,
    name: str,
    spread_deg: float,
    speed: float,
    label: str,
) -> dict[str, object]:
    return {
        "profile": {
            "profile_id": profile_id,
            "name": name,
            "weight": 1.0,
            "mode": "fan",
            "u_pos": [1000.0, 2000.0],
            "n": 4,
            "speed": 15.4333,
            "max_run_time": 486.0,
            "base_bearing_rad": 1.2,
            "spread_doctrine": "uniform_divergent",
            "spread_rad": 0.1,
            "per_torpedo_heading_offsets_rad": [],
            "launch_delay_s": 0.9,
            "salvo_interval_s": 2.0,
            "u_boat_mode": "moving",
            "u_boat_initial_heading_rad": 1.2,
            "u_boat_initial_speed_mps": speed,
            "sub_length_m": 67.0,
            "sub_beam_m": 6.5,
            "launch_from": "bow",
            "max_bow_offset_deg": 15.0,
            "gyro_straight_run_m": 10.0,
        },
        "audit": {
            "profile_id": profile_id,
            "name": name,
            "mode": "fan",
            "u_pos_x": 1000.0,
            "u_pos_y": 2000.0,
            "range_to_centroid_m": 2236.0,
            "intent_bearing_rad": 1.1,
            "active_bearing_rad": 1.2,
            "bearing_error_deg": 5.0,
            "spread_deg": spread_deg,
            "flag_count": 0,
            "flags": [],
            "severity": 5.0,
            "suggested_label": label,
        },
        "generator_meta": {
            "mode": "dataset",
            "seed": 1945,
            "convoy_profile": "convoy_layout_1",
            "accepted_labels": ["credible_hit_threat", "credible_near_miss"],
            "source": "generate_attack_profile_scaffold",
            "generator_version": "v2",
        },
    }


def test_dataset_audit_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "profiles.jsonl"
    rows = [
        _sample_record(
            profile_id="D000001",
            name="dataset_000001_east_abeam_medium_uniform_random",
            spread_deg=4.0,
            speed=1.5,
            label="credible_hit_threat",
        ),
        _sample_record(
            profile_id="D000002",
            name="dataset_000002_south_west_quarter_long_uniform_random",
            spread_deg=6.5,
            speed=1.7,
            label="credible_near_miss",
        ),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = load_dataset_records(path)
    flat_rows = flatten_dataset_records(loaded)
    summary = summarize_flat_rows(flat_rows)
    outputs = write_dataset_audit_outputs(output_dir=tmp_path / "audit", flat_rows=flat_rows, summary=summary)

    assert len(flat_rows) == 2
    assert flat_rows[0]["approach_family"] == "east_abeam"
    assert flat_rows[0]["range_band"] == "medium"
    assert summary["profile_count"] == 2
    assert summary["labels"]["credible_hit_threat"] == 1
    assert summary["range_bands"]["long"] == 1
    assert outputs["summary_json"].exists()
    assert outputs["profiles_flat_csv"].exists()
    assert outputs["counts_by_label_csv"].exists()
