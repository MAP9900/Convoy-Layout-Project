from __future__ import annotations

import json

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.profile_outcome_audit import (
    OutcomeAuditConfig,
    audit_dataset_outcomes,
    audit_profile_outcome,
    enrich_dataset_records_with_outcomes,
    filter_records_by_outcome_gate,
    summarize_outcome_rows,
)
from experiments.generate_attack_profile_scaffold import generate_attack_profile_scaffolds, render_profiles_as_jsonl
from scenarios.convoy_profiles import get_convoy_layout_profile


def test_profile_outcome_audit_detects_intended_target_hit() -> None:
    ship = Ship(
        id="S1",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
        ship_class=ShipClass.FREIGHTER,
    )
    profile = AttackProfile(
        profile_id="T001",
        name="direct_hit",
        mode="fan",
        u_pos=(-1000.0, 0.0),
        n=1,
        speed=20.0,
        max_run_time=100.0,
        base_bearing_rad=0.0,
        spread_rad=0.0,
        spread_doctrine="uniform_divergent",
        u_boat_mode="static",
        u_boat_initial_heading_rad=0.0,
        launch_from="center",
    )
    row = audit_profile_outcome(
        profile,
        [ship],
        intent={
            "target_ship_ids": ["S1"],
            "intended_label": "credible_hit_threat",
            "spawn_region": "unit_test",
        },
        cfg=OutcomeAuditConfig(t_max_s=100.0, hit_dt_s=0.25, zigzag_enabled=False),
    )

    assert row["actual_outcome_label"] == "credible_hit_threat"
    assert row["intended_target_hit"] is True
    assert row["outcome_matches_intent"] is True
    assert row["spread_doctrine"] == "uniform_divergent"


def test_profile_outcome_audit_accepts_intentional_miss() -> None:
    ship = Ship(
        id="S1",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
        ship_class=ShipClass.FREIGHTER,
    )
    profile = AttackProfile(
        profile_id="T002",
        name="deliberate_miss",
        mode="fan",
        u_pos=(-1000.0, 0.0),
        n=1,
        speed=20.0,
        max_run_time=100.0,
        base_bearing_rad=float(np.deg2rad(35.0)),
        spread_rad=0.0,
        spread_doctrine="uniform_divergent",
        u_boat_mode="static",
        u_boat_initial_heading_rad=0.0,
        launch_from="center",
    )
    row = audit_profile_outcome(
        profile,
        [ship],
        intent={
            "target_ship_ids": ["S1"],
            "intended_label": "intentional_miss",
            "spawn_region": "unit_test",
        },
        cfg=OutcomeAuditConfig(t_max_s=100.0, hit_dt_s=0.25, zigzag_enabled=False),
    )

    assert row["actual_outcome_label"] == "miss"
    assert row["passes_outcome_gate"] is True
    assert row["outcome_matches_intent"] is True


def test_dataset_outcome_audit_runs_generated_v4_through_standard_dynamic_pipeline() -> None:
    profiles, audit_rows = generate_attack_profile_scaffolds(
        mode="random_tactical_v4",
        count=6,
        seed=1945,
        start_index=1,
    )
    rendered = render_profiles_as_jsonl(
        profiles,
        audit_rows=audit_rows,
        seed=1945,
        convoy_profile="convoy_layout_1",
        accepted_labels=("credible_hit_threat", "credible_near_miss"),
        mode="random_tactical_v4",
    )
    records = [json.loads(line) for line in rendered.splitlines() if line.strip()]
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    rows = audit_dataset_outcomes(
        records,
        ships=ships,
        rng_seed=1945,
        cfg=OutcomeAuditConfig(t_max_s=120.0, hit_dt_s=1.0),
    )
    summary = summarize_outcome_rows(rows)

    assert len(rows) == 6
    assert summary["profile_count"] == 6
    assert {row["convoy_motion"] for row in rows} == {"zigzag"}
    assert {row["spread_doctrine"] for row in rows} == {"uniform_divergent"}
    assert {row["u_boat_mode"] for row in rows} == {"moving"}
    assert all(np.isfinite(float(row["closest_any_ship_distance_m"])) for row in rows)

    enriched = enrich_dataset_records_with_outcomes(
        records[:2],
        ships=ships,
        rng_seed=1945,
        cfg=OutcomeAuditConfig(t_max_s=120.0, hit_dt_s=1.0),
    )
    assert len(enriched) == 2
    assert "outcome" in enriched[0]
    assert "actual_outcome_label" in enriched[0]["outcome"]
    assert all("passes_outcome_gate" in row["outcome"] for row in enriched)
    assert len(filter_records_by_outcome_gate(enriched)) <= len(enriched)
