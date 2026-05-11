from __future__ import annotations

import json
from collections import Counter

from convoy_sim.profile_outcome_audit import OutcomeAuditConfig, audit_dataset_outcomes
from experiments.generate_attack_profile_scaffold import MIN_SPAWN_CLEARANCE_M
from experiments.generate_random_attack_profile_dataset import (
    RANDOM_BASELINE_MODE,
    generate_random_baseline_profiles,
    generate_random_baseline_records,
    render_random_baseline_jsonl,
)
from scenarios.convoy_profiles import get_convoy_layout_profile


def test_generate_random_baseline_records_are_profile_first_and_outcome_labeled() -> None:
    records, stats = generate_random_baseline_records(count=10, seed=1945, start_index=1)
    labels = Counter(str(record["audit"]["suggested_label"]) for record in records)
    outcomes = Counter(str(record["outcome"]["actual_outcome_label"]) for record in records)

    assert len(records) == 10
    assert stats["accepted_count"] == 10
    assert labels == {"credible_hit_threat": 6, "credible_near_miss": 2, "intentional_miss": 2}
    assert outcomes == {"credible_hit_threat": 6, "credible_near_miss": 2, "miss": 2}
    assert all(record["generator_meta"]["mode"] == RANDOM_BASELINE_MODE for record in records)
    assert all(record["profile"]["u_boat_mode"] == "moving" for record in records)
    assert all(record["profile"]["spread_doctrine"] == "uniform_divergent" for record in records)
    assert all(record["intent"]["profile_first_outcome_label"] is True for record in records)
    assert all(float(record["intent"]["nearest_ship_clearance_m"]) >= MIN_SPAWN_CLEARANCE_M for record in records)


def test_render_random_baseline_jsonl_and_reaudit_outcomes() -> None:
    profiles, audit_rows, stats = generate_random_baseline_profiles(count=8, seed=2945, start_index=101)
    text = render_random_baseline_jsonl(
        profiles,
        audit_rows=audit_rows,
        seed=2945,
        convoy_profile="convoy_layout_1",
        stats=stats,
    )
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    outcome_rows = audit_dataset_outcomes(
        records,
        ships=ships,
        rng_seed=2945,
        cfg=OutcomeAuditConfig(t_max_s=600.0, hit_dt_s=0.5),
    )

    assert len(records) == 8
    assert records[0]["profile"]["profile_id"].startswith("R")
    assert records[0]["outcome"]["passes_outcome_gate"] is True
    assert all(row["profile_first_outcome_label"] is True for row in outcome_rows)
    assert all(row["outcome_matches_intent"] is True for row in outcome_rows)
