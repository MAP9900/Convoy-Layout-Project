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


def test_generated_profiles_respect_spawn_clearance() -> None:
    profiles, _audit_rows = generate_attack_profile_scaffolds(start_index=31, count=6, seed=1945)
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    ship_positions = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
    for profile in profiles:
        u_pos = np.asarray(profile.u_pos, dtype=float)
        min_distance = float(np.min(np.linalg.norm(ship_positions - u_pos, axis=1)))
        assert min_distance >= MIN_SPAWN_CLEARANCE_M
