from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.feasibility import Environment
from convoy_sim.pomdp_candidate_selector import (
    build_candidate_observation_row,
    build_candidate_observation_rows,
    rank_candidate_observation_rows,
)
from convoy_sim.realism import AttackerObservationConfig, get_attacker_observation_config
from experiments.run_pomdp_candidate_selector import run_belief_selector
from scenarios.convoy_profiles import get_convoy_layout_profile


def _record(profile_id: str, *, outcome_hits: int = 1) -> dict:
    profile = AttackProfile(
        profile_id=profile_id,
        name=f"candidate_{profile_id}",
        weight=1.0,
        mode="fan",
        u_pos=(-1800.0, 0.0),
        n=4,
        speed=15.4333,
        max_run_time=486.0,
        base_bearing_rad=0.0,
        spread_rad=0.07,
        launch_delay_s=1.0,
        salvo_interval_s=2.0,
        spread_doctrine="uniform_divergent",
        u_boat_mode="moving",
        u_boat_initial_heading_rad=0.0,
        u_boat_initial_speed_mps=1.5,
        launch_from="bow",
    )
    return {
        "profile": profile.to_dict(),
        "audit": {
            "suggested_label": "credible_hit_threat",
            "actual_outcome_label": "credible_hit_threat",
            "clearance_m": 1000.0,
            "n_hits": int(outcome_hits),
            "unique_ships_hit": int(outcome_hits),
            "closest_any_ship_distance_m": 5.0,
            "closest_any_ship_id": "S1",
        },
        "intent": {
            "spawn_region": "astern_vae",
            "approach_side": "astern",
            "inside_convoy_envelope": False,
        },
        "outcome": {"n_hits": int(outcome_hits)},
        "generator_meta": {"mode": "unit_test_candidate_pool", "convoy_profile": "convoy_layout_1"},
    }


def test_candidate_observation_is_reproducible() -> None:
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    cfg = AttackerObservationConfig(range_sigma_m=0.0, bearing_sigma_rad=0.0, heading_sigma_rad=0.0)
    rows_a = build_candidate_observation_rows([_record("C001")], ships=ships, seed=123, observation_cfg=cfg)
    rows_b = build_candidate_observation_rows([_record("C001")], ships=ships, seed=123, observation_cfg=cfg)

    assert rows_a == rows_b
    assert rows_a[0]["profile_id"] == "C001"
    assert rows_a[0]["selector_method"] == "belief_limited_heuristic_v1"
    assert "belief_score" in rows_a[0]
    assert "estimated_formation_width_m" in rows_a[0]
    assert "formation_span_score" in rows_a[0]


def test_observation_presets_order_uncertainty() -> None:
    good = get_attacker_observation_config("good_contact")
    baseline = get_attacker_observation_config("baseline_night")
    poor = get_attacker_observation_config("poor_contact")

    assert good.range_sigma_m < baseline.range_sigma_m < poor.range_sigma_m
    assert good.formation_width_sigma_m < baseline.formation_width_sigma_m < poor.formation_width_sigma_m
    assert good.class_count_sigma < baseline.class_count_sigma < poor.class_count_sigma


def test_candidate_score_does_not_use_outcome_hit_count() -> None:
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    cfg = AttackerObservationConfig(range_sigma_m=0.0, bearing_sigma_rad=0.0, heading_sigma_rad=0.0)
    env = Environment(time_of_day="night", visibility_m=3500.0, sea_state=4)
    row_a = build_candidate_observation_row(
        _record("C001", outcome_hits=1),
        ships=ships,
        rng=np.random.default_rng(7),
        env=env,
        observation_cfg=cfg,
    )
    row_b = build_candidate_observation_row(
        _record("C001", outcome_hits=4),
        ships=ships,
        rng=np.random.default_rng(7),
        env=env,
        observation_cfg=cfg,
    )

    assert row_a["belief_score"] == row_b["belief_score"]
    assert "n_hits" not in row_a
    assert "expected_loss" not in row_a


def test_rank_candidate_observation_rows_assigns_belief_rank() -> None:
    rows = [
        {"profile_id": "low", "belief_score": 0.1, "estimated_range_m": 2000.0},
        {"profile_id": "high", "belief_score": 0.8, "estimated_range_m": 2000.0},
    ]
    ranked = rank_candidate_observation_rows(rows)
    assert ranked[0]["profile_id"] == "high"
    assert ranked[0]["belief_rank"] == 1


def test_run_belief_selector_writes_artifacts(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    records = [_record("C001"), _record("C002")]
    candidate_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    run_dir = run_belief_selector(
        candidate_path=candidate_path,
        project_root=tmp_path,
        output_root=Path("runs"),
        run_name="belief_smoke",
        max_profiles=2,
        top_k=1,
        seed=1945,
    )

    assert (run_dir / "belief_ranked_candidates.csv").exists()
    assert (run_dir / "top_belief_candidates.json").exists()
    assert (run_dir / "top_belief_candidate_pool.jsonl").exists()
    assert (run_dir / "metrics_summary.json").exists()
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["selection"]["method"] == "belief_limited_heuristic_v1"
    assert metrics["selection"]["observation_preset"] == "good_contact"
    assert metrics["top_k"]["profiles"] == 1
    selected = [
        json.loads(line)
        for line in (run_dir / "top_belief_candidate_pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(selected) == 1
    assert selected[0]["selection_meta"]["observation_preset"] == "good_contact"
