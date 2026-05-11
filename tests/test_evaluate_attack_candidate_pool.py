from __future__ import annotations

import csv
import json
from pathlib import Path

from convoy_sim.attack_profiles import AttackProfile
from experiments.evaluate_attack_candidate_pool import evaluate_candidate_pool, load_candidate_records


def _candidate_record(profile_id: str, y: float) -> dict:
    profile = AttackProfile(
        profile_id=profile_id,
        name=f"candidate_{profile_id}",
        weight=1.0,
        mode="fan",
        u_pos=(-1800.0, y),
        n=1,
        speed=15.4333,
        max_run_time=250.0,
        base_bearing_rad=0.0,
        spread_rad=0.0,
        spread_doctrine="uniform_divergent",
        u_boat_mode="static",
        launch_from="center",
    )
    return {
        "profile": profile.to_dict(),
        "audit": {
            "suggested_label": "credible_hit_threat",
            "actual_outcome_label": "credible_hit_threat",
            "clearance_m": 1000.0,
            "n_hits": 1,
            "unique_ships_hit": 1,
            "closest_any_ship_distance_m": 5.0,
            "closest_any_ship_id": "S1",
        },
        "intent": {
            "spawn_region": "unit_test",
            "approach_side": "astern",
            "inside_convoy_envelope": False,
        },
        "outcome": {
            "actual_outcome_label": "credible_hit_threat",
        },
        "generator_meta": {
            "mode": "unit_test_candidate_pool",
            "convoy_profile": "convoy_layout_1",
        },
    }


def test_evaluate_candidate_pool_writes_ranked_artifacts(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    records = [_candidate_record("C001", 0.0), _candidate_record("C002", 1371.6)]
    candidate_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    loaded = load_candidate_records(candidate_path)
    assert len(loaded) == 2

    run_dir = evaluate_candidate_pool(
        candidate_path=candidate_path,
        project_root=tmp_path,
        output_root=Path("runs"),
        run_name="smoke",
        convoy_profile="convoy_layout_1",
        max_profiles=2,
        top_k=1,
        seeds=[7],
        n_trials_per_seed=1,
        t_max=120.0,
        max_hits_per_torpedo=1,
        objective_cfg={},
        ship_movement_realism=None,
    )

    assert (run_dir / "ranked_candidates.csv").exists()
    assert (run_dir / "metrics_summary.json").exists()
    assert (run_dir / "top_candidates.json").exists()
    assert (run_dir / "run_manifest.json").exists()

    with (run_dir / "ranked_candidates.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
    assert rows[0]["profile_id"] in {"C001", "C002"}
    assert "attacker_score" in rows[0]

    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["selection"]["method"] == "full_state_attacker_rank"
    assert metrics["selection"]["score"] == "expected_hits"
    assert metrics["best_candidate"]["rank"] == 1
