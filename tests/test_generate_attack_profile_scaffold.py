from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.generate_attack_profile_scaffold import (
    MIN_SPAWN_CLEARANCE_M,
    generate_attack_profile_scaffolds,
)
from scenarios.convoy_profiles import get_convoy_layout_profile


def test_generate_attack_profile_scaffold_python_output(tmp_path: Path) -> None:
    output = tmp_path / "profiles.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_attack_profile_scaffold",
            "--start-index",
            "31",
            "--count",
            "3",
            "--seed",
            "1945",
            "--format",
            "python",
            "--output",
            str(output),
        ],
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "GENERATED_ATTACK_PROFILE_CALLS = [" in text
    assert "_scaffolded_fan_profile(" in text
    assert 'profile_id="P31"' in text
    assert 'profile_id="P33"' in text
    assert "u_boat_initial_speed_mps=" in text
    assert 'spread_doctrine=' not in text


def test_generate_attack_profile_scaffold_json_output(tmp_path: Path) -> None:
    output = tmp_path / "profiles.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_attack_profile_scaffold",
            "--start-index",
            "61",
            "--count",
            "2",
            "--seed",
            "1945",
            "--format",
            "json",
            "--output",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generator_meta"]["convoy_profile"] == "convoy_layout_1"
    assert payload["generator_meta"]["accepted_labels"] == ["credible_hit_threat", "credible_near_miss"]
    assert len(payload["profiles"]) == 2
    assert len(payload["audit_rows"]) == 2
    assert payload["profiles"][0]["profile_id"] == "P61"
    assert payload["profiles"][0]["u_boat_mode"] == "moving"
    assert payload["profiles"][0]["spread_doctrine"] == "uniform_divergent"
    assert payload["profiles"][0]["n"] == 4
    assert 1.0 <= payload["profiles"][0]["u_boat_initial_speed_mps"] <= 2.0
    assert round(payload["profiles"][0]["u_boat_initial_speed_mps"], 1) == payload["profiles"][0]["u_boat_initial_speed_mps"]
    assert payload["audit_rows"][0]["suggested_label"] in {"credible_hit_threat", "credible_near_miss"}
    assert payload["generator_meta"]["mode"] == "curated"


def test_generated_profiles_respect_spawn_clearance() -> None:
    profiles, _audit_rows = generate_attack_profile_scaffolds(start_index=31, count=6, seed=1945)
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    ship_positions = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
    for profile in profiles:
        u_pos = np.asarray(profile.u_pos, dtype=float)
        min_distance = float(np.min(np.linalg.norm(ship_positions - u_pos, axis=1)))
        assert min_distance >= MIN_SPAWN_CLEARANCE_M


def test_generate_attack_profile_scaffold_dataset_jsonl_output(tmp_path: Path) -> None:
    output = tmp_path / "profiles.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_attack_profile_scaffold",
            "--mode",
            "dataset",
            "--start-index",
            "1",
            "--count",
            "3",
            "--seed",
            "1945",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert rows[0]["generator_meta"]["mode"] == "dataset"
    assert rows[0]["generator_meta"]["source"] == "generate_attack_profile_scaffold"
    assert rows[0]["profile"]["profile_id"].startswith("D")
    assert rows[0]["profile"]["spread_doctrine"] == "uniform_divergent"
    assert rows[0]["audit"]["suggested_label"] in {"credible_hit_threat", "credible_near_miss"}


def test_generate_attack_profile_scaffold_dataset_targets_75_25_mix() -> None:
    profiles, audit_rows = generate_attack_profile_scaffolds(
        start_index=1,
        count=20,
        seed=1945,
        mode="dataset",
    )
    labels = [str(row["suggested_label"]) for row in audit_rows]
    assert len(profiles) == 20
    assert labels.count("credible_hit_threat") == 15
    assert labels.count("credible_near_miss") == 5


def test_generate_attack_profile_scaffold_random_zones_jsonl_output(tmp_path: Path) -> None:
    output = tmp_path / "profiles_v3.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_attack_profile_scaffold",
            "--mode",
            "random_zones",
            "--start-index",
            "1",
            "--count",
            "8",
            "--seed",
            "1945",
            "--output",
            str(output),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    assert rows[0]["generator_meta"]["mode"] == "random_zones"
    assert rows[0]["generator_meta"]["generator_version"] == "v3"
    assert rows[0]["profile"]["profile_id"].startswith("Z")
    assert rows[0]["intent"]["target_zone_id"].startswith("TZ")
    assert rows[0]["audit"]["target_zone_id"] == rows[0]["intent"]["target_zone_id"]
    assert rows[0]["audit"]["suggested_label"] in {"credible_hit_threat", "credible_near_miss"}


def test_generate_attack_profile_scaffold_random_zones_targets_75_25_mix() -> None:
    profiles, audit_rows = generate_attack_profile_scaffolds(
        start_index=1,
        count=20,
        seed=1945,
        mode="random_zones",
    )
    labels = [str(row["suggested_label"]) for row in audit_rows]
    target_zone_kinds = {str(row["target_zone_kind"]) for row in audit_rows}
    approach_sides = {str(row["approach_side"]) for row in audit_rows}

    assert len(profiles) == 20
    assert labels.count("credible_hit_threat") == 15
    assert labels.count("credible_near_miss") == 5
    assert len(target_zone_kinds) > 1
    assert len(approach_sides) > 1
    assert all("intent" in row for row in audit_rows)


def test_generate_attack_profile_scaffold_random_tactical_v4_jsonl_output(tmp_path: Path) -> None:
    output = tmp_path / "profiles_v4.jsonl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_attack_profile_scaffold",
            "--mode",
            "random_tactical_v4",
            "--start-index",
            "1",
            "--count",
            "8",
            "--seed",
            "1945",
            "--output",
            str(output),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    assert rows[0]["generator_meta"]["mode"] == "random_tactical_v4"
    assert rows[0]["generator_meta"]["generator_version"] == "v4"
    assert rows[0]["profile"]["profile_id"].startswith("T")
    assert rows[0]["intent"]["target_zone_id"].startswith("TV4")
    assert rows[0]["intent"]["spawn_region"]
    assert rows[0]["intent"]["nearest_ship_clearance_m"] >= MIN_SPAWN_CLEARANCE_M
    assert rows[0]["audit"]["target_zone_id"] == rows[0]["intent"]["target_zone_id"]


def test_generate_attack_profile_scaffold_random_tactical_v4_targets_75_25_mix() -> None:
    profiles, audit_rows = generate_attack_profile_scaffolds(
        start_index=1,
        count=20,
        seed=1945,
        mode="random_tactical_v4",
    )
    labels = [str(row["suggested_label"]) for row in audit_rows]
    spawn_regions = {str(row["intent"]["spawn_region"]) for row in audit_rows}
    inside_count = sum(1 for row in audit_rows if bool(row["intent"]["inside_convoy_envelope"]))

    assert len(profiles) == 20
    assert labels.count("credible_hit_threat") == 15
    assert labels.count("credible_near_miss") == 5
    assert len(spawn_regions) > 1
    assert inside_count > 0
    assert all(float(row["intent"]["nearest_ship_clearance_m"]) >= MIN_SPAWN_CLEARANCE_M for row in audit_rows)
