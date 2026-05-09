"""Tests for attack profile geometry audit helpers."""

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.profile_audit import audit_attack_profiles
from convoy_sim.layouts import make_rectangular_convoy


def _ships() -> list:
    return make_rectangular_convoy(
        n_rows=1,
        n_cols=1,
        spacing_along=10.0,
        spacing_across=10.0,
        speed=0.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
        origin=np.array([0.0, 0.0]),
    )


def test_audit_marks_plausible_centroid_aim() -> None:
    profile = AttackProfile(
        profile_id="P01",
        name="plausible",
        mode="fan",
        u_pos=(1000.0, 0.0),
        base_bearing_rad=np.pi,
        spread_rad=0.0873,
        n=4,
        speed=15.0,
        max_run_time=500.0,
    )
    rows = audit_attack_profiles([profile], _ships())
    assert rows[0]["suggested_label"] in {"credible_hit_threat", "credible_near_miss"}
    assert rows[0]["flag_count"] == 0


def test_audit_marks_implausible_opposite_direction_near_range() -> None:
    profile = AttackProfile(
        profile_id="P01",
        name="implausible",
        mode="fan",
        u_pos=(1000.0, 0.0),
        base_bearing_rad=0.0,
        spread_rad=0.0698,
        n=4,
        speed=15.0,
        max_run_time=500.0,
    )
    rows = audit_attack_profiles([profile], _ships())
    assert rows[0]["suggested_label"] == "implausible_geometry"
    assert "near_range_opposite_direction" in rows[0]["flags"]


def test_audit_uses_target_intent_when_present() -> None:
    profile = AttackProfile(
        profile_id="P01",
        name="target_intent",
        mode="fan",
        u_pos=(1000.0, 1000.0),
        base_bearing_rad=-np.pi / 2.0,
        spread_rad=0.0873,
        n=4,
        speed=15.0,
        max_run_time=500.0,
    )
    intent = {
        "target_zone_id": "TZ000001_test",
        "target_zone_kind": "port_edge",
        "target_point": [1000.0, 0.0],
        "approach_side": "port",
        "approach_lane": "port_broadside",
        "intended_label": "credible_hit_threat",
    }

    rows = audit_attack_profiles([profile], _ships(), intents=[intent])

    assert rows[0]["target_zone_id"] == "TZ000001_test"
    assert rows[0]["range_to_target_m"] == 1000.0
    assert rows[0]["target_bearing_error_deg"] < 1e-6
    assert rows[0]["suggested_label"] == "credible_hit_threat"
