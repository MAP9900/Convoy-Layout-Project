from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from convoy_sim.vae import AttackProfileVAE, load_vae_dataset
from experiments.generate_vae_candidate_pool import build_vae_candidate_records, render_vae_candidate_jsonl


def _record(profile_id: str, x: float, y: float, bearing: float) -> dict:
    return {
        "profile": {
            "profile_id": profile_id,
            "name": f"dataset_{profile_id}",
            "weight": 1.0,
            "mode": "fan",
            "u_pos": [x, y],
            "n": 4,
            "speed": 15.4333,
            "max_run_time": 486.0,
            "base_bearing_rad": bearing,
            "spread_doctrine": "uniform_divergent",
            "spread_rad": 0.10,
            "per_torpedo_heading_offsets_rad": [],
            "launch_delay_s": 1.0,
            "salvo_interval_s": 2.0,
            "u_boat_mode": "moving",
            "u_boat_initial_heading_rad": bearing,
            "u_boat_initial_speed_mps": 1.4,
            "sub_length_m": 67.0,
            "sub_beam_m": 6.5,
            "launch_from": "bow",
            "max_bow_offset_deg": 15.0,
            "gyro_straight_run_m": 10.0,
        },
        "audit": {"suggested_label": "credible_hit_threat"},
        "generator_meta": {"mode": "unit_test", "seed": 1945, "convoy_profile": "convoy_layout_1"},
    }


def test_build_vae_candidate_records_writes_dataset_shaped_candidates(tmp_path) -> None:
    records = [
        _record("D000001", -1800.0, 800.0, 0.0),
        _record("D000002", 1800.0, -800.0, 3.14),
    ]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    _, preprocessor = load_vae_dataset(train_path)
    model = AttackProfileVAE(latent_dim=2, hidden_dim=8)

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    manifest = {
        "preprocessor": preprocessor.to_dict(),
        "hyperparameters": {"latent_dim": 2, "hidden_dim": 8},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "preprocessor": preprocessor.to_dict(),
            "hyperparameters": {"latent_dim": 2, "hidden_dim": 8},
        },
        checkpoint_dir / "model_best.pt",
    )
    candidates, summary = build_vae_candidate_records(
        run_dir=run_dir,
        train_path=train_path,
        sample_count=3,
        seed=1945,
        device="cpu",
        min_clearance_m=0.0,
        accepted_outcomes=("credible_hit_threat", "credible_near_miss", "miss"),
    )
    rendered = render_vae_candidate_jsonl(candidates)
    parsed = [json.loads(line) for line in rendered.splitlines() if line.strip()]

    assert summary["sample_count"] == 3
    assert summary["accepted_count"] == len(candidates)
    assert len(parsed) == len(candidates)
    assert parsed
    assert {"profile", "audit", "intent", "outcome", "generator_meta"} <= set(parsed[0])
    assert parsed[0]["generator_meta"]["mode"] == "vae_candidate_pool_v1"
    assert parsed[0]["intent"]["derived_from_vae_sample"] is True
    assert parsed[0]["outcome"]["passes_outcome_gate"] is True
