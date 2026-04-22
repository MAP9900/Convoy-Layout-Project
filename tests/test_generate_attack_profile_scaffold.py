from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
    assert "GENERATED_ATTACK_PROFILES = [" in text
    assert 'profile_id="P31"' in text
    assert 'profile_id="P33"' in text
    assert 'u_boat_mode="moving"' in text
    assert "u_boat_initial_speed_mps=" in text


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
    assert 1.0 <= payload["profiles"][0]["u_boat_initial_speed_mps"] <= 2.0
    assert round(payload["profiles"][0]["u_boat_initial_speed_mps"], 1) == payload["profiles"][0]["u_boat_initial_speed_mps"]
    assert payload["audit_rows"][0]["suggested_label"] in {"credible_hit_threat", "credible_near_miss"}
