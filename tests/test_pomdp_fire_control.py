from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.pomdp_fire_control import rebuild_records_with_fire_control, write_fire_control_candidate_pool
from convoy_sim.realism import AttackerObservationConfig
from scenarios.convoy_profiles import get_convoy_layout_profile


def _record(profile_id: str) -> dict:
    profile = AttackProfile(
        profile_id=profile_id,
        name=f"candidate_{profile_id}",
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
        "audit": {"actual_outcome_label": "credible_hit_threat", "n_hits": 2},
        "intent": {"spawn_region": "astern_vae", "approach_side": "astern"},
        "outcome": {"n_hits": 2},
        "generator_meta": {"mode": "unit_test_candidate_pool", "convoy_profile": "convoy_layout_1"},
    }


def test_rebuild_records_with_fire_control_replaces_firing_solution() -> None:
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    source = _record("C001")
    source["profile"]["base_bearing_rad"] = 2.5
    rebuilt = rebuild_records_with_fire_control(
        [source],
        ships=ships,
        seed=123,
        observation_preset="good_contact",
        observation_cfg=AttackerObservationConfig(
            bearing_sigma_rad=0.0,
            range_sigma_m=0.0,
            heading_sigma_rad=0.0,
            speed_sigma_mps=0.0,
            contact_count_sigma=0.0,
            formation_width_sigma_m=0.0,
            formation_depth_sigma_m=0.0,
            class_count_sigma=0.0,
            contact_detection_fraction=1.0,
            contact_detection_fraction_sigma=0.0,
        ),
    )

    assert len(rebuilt) == 1
    record = rebuilt[0]
    profile = AttackProfile.from_dict(record["profile"])
    assert profile.profile_id == "POMDP_FC_0001"
    assert np.isclose(profile.u_pos[0], source["profile"]["u_pos"][0])
    assert not np.isclose(profile.base_bearing_rad, source["profile"]["base_bearing_rad"])
    assert profile.spread_rad > 0.0
    assert "audit" not in record
    assert "outcome" not in record
    assert record["intent"]["source_profile_id"] == "C001"
    assert record["selection_meta"]["method"] == "pomdp_fire_control_lite_v1"
    assert "solution" in record["fire_control_meta"]


def test_write_fire_control_candidate_pool_jsonl(tmp_path: Path) -> None:
    ships = get_convoy_layout_profile("convoy_layout_1").build_ships()
    records = rebuild_records_with_fire_control([_record("C001"), _record("C002")], ships=ships, seed=7)
    path = tmp_path / "fire_control_candidates.jsonl"

    write_fire_control_candidate_pool(path, records)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["profile"]["profile_id"] == "POMDP_FC_0001"
    assert rows[1]["profile"]["profile_id"] == "POMDP_FC_0002"
