from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from convoy_sim.vae import (
    AttackProfileVAE,
    AttackProfileVAEDataset,
    AttackProfileVAEPreprocessor,
    FEATURE_NAMES,
    vae_loss,
)


def _record(profile_id: str, speed: float, spread_rad: float) -> dict:
    return {
        "profile": {
            "profile_id": profile_id,
            "name": f"dataset_{profile_id}_east_abeam_medium_uniform_random",
            "weight": 1.0,
            "mode": "fan",
            "u_pos": [2200.0, 50.0],
            "n": 4,
            "speed": 15.4333,
            "max_run_time": 486.0,
            "base_bearing_rad": 3.10,
            "spread_doctrine": "uniform_divergent",
            "spread_rad": spread_rad,
            "per_torpedo_heading_offsets_rad": [],
            "launch_delay_s": 0.9,
            "salvo_interval_s": 2.0,
            "u_boat_mode": "moving",
            "u_boat_initial_heading_rad": 3.10,
            "u_boat_initial_speed_mps": speed,
            "sub_length_m": 67.0,
            "sub_beam_m": 6.5,
            "launch_from": "bow",
            "max_bow_offset_deg": 15.0,
            "gyro_straight_run_m": 10.0,
        },
        "audit": {
            "suggested_label": "credible_hit_threat",
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


def test_vae_preprocessor_dataset_and_model_roundtrip() -> None:
    records = [
        _record("D000001", speed=1.4, spread_rad=0.09),
        _record("D000002", speed=1.8, spread_rad=0.11),
    ]
    preprocessor = AttackProfileVAEPreprocessor.fit(records)
    assert preprocessor.feature_names == FEATURE_NAMES

    dataset = AttackProfileVAEDataset(records, preprocessor=preprocessor)
    assert len(dataset) == 2
    assert dataset[0].shape == (len(FEATURE_NAMES),)

    model = AttackProfileVAE(input_dim=len(FEATURE_NAMES), latent_dim=4, hidden_dim=16)
    batch = torch.stack([dataset[0], dataset[1]], dim=0)
    recon, mu, logvar = model(batch)
    assert recon.shape == batch.shape
    assert mu.shape == (2, 4)
    assert logvar.shape == (2, 4)

    loss, stats = vae_loss(recon, batch, mu, logvar, beta=0.05)
    assert loss.ndim == 0
    assert stats["loss"] >= 0.0
    assert stats["recon_loss"] >= 0.0
    assert stats["kl_loss"] >= 0.0

    decoded = preprocessor.decode_profile_fields(recon[0], profile_id="SAMPLE", name="sample")
    assert decoded["profile_id"] == "SAMPLE"
    assert decoded["n"] == 4
    assert decoded["spread_doctrine"] == "uniform_divergent"
    assert 1.0 <= decoded["u_boat_initial_speed_mps"] <= 2.0
