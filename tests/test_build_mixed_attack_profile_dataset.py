from __future__ import annotations

from experiments.build_mixed_attack_profile_dataset import build_mixed_split


def _record(profile_id: str, *, source: str, label: str = "credible_hit_threat") -> dict:
    return {
        "profile": {
            "profile_id": profile_id,
            "name": profile_id.lower(),
            "u_pos": [0.0, 0.0],
            "base_bearing_rad": 0.0,
            "spread_rad": 0.05,
            "launch_delay_s": 1.0,
            "salvo_interval_s": 2.0,
            "u_boat_initial_speed_mps": 1.5,
        },
        "audit": {
            "profile_id": profile_id,
            "name": profile_id.lower(),
            "suggested_label": label,
        },
        "generator_meta": {
            "mode": source,
            "source": source,
            "generator_version": "test",
            "seed": 1,
            "convoy_profile": "convoy_layout_1",
        },
    }


def test_build_mixed_split_uses_requested_ratio_and_rewrites_ids() -> None:
    curated = [_record(f"C{i}", source="curated") for i in range(10)]
    random = [_record(f"R{i}", source="random") for i in range(10)]

    records, summary = build_mixed_split(
        curated_records=curated,
        random_records=random,
        total_count=10,
        curated_fraction=0.7,
        split="train",
        seed=1945,
    )

    assert len(records) == 10
    assert summary["source_counts"] == {"curated_v4": 7, "random_profile_v1": 3}
    assert records[0]["profile"]["profile_id"].startswith("MIXTRN")
    assert records[0]["audit"]["profile_id"] == records[0]["profile"]["profile_id"]
    assert "source_profile_id" in records[0]["mixture_meta"]
    assert records[0]["generator_meta"]["mode"] == "mixed_curated_random_v1"


def test_build_mixed_split_rejects_insufficient_source_records() -> None:
    curated = [_record("C0", source="curated")]
    random = [_record("R0", source="random")]

    try:
        build_mixed_split(
            curated_records=curated,
            random_records=random,
            total_count=10,
            curated_fraction=0.7,
            split="train",
            seed=1945,
        )
    except ValueError as exc:
        assert "curated_v4" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected insufficient source records to raise")
