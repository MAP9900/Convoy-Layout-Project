from __future__ import annotations

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.profile_outcome_audit import OutcomeAuditConfig
from convoy_sim.vae_diagnostics import audit_decoded_vae_payloads, nearest_ship_clearance_m, summarize_decoded_vae_audit


def test_decoded_vae_audit_reports_dynamic_outcome_without_intent_metadata() -> None:
    ship = Ship(
        id="S1",
        position=as_vec(0.0, 0.0),
        speed=0.0,
        heading_rad=0.0,
        length=100.0,
        beam=20.0,
        ship_class=ShipClass.FREIGHTER,
    )
    payload = AttackProfile(
        profile_id="VAE_AUDIT_0001",
        name="decoded_direct_hit",
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
    ).to_dict()

    rows = audit_decoded_vae_payloads(
        [payload],
        [ship],
        min_clearance_m=250.0,
        outcome_cfg=OutcomeAuditConfig(t_max_s=100.0, hit_dt_s=0.25, zigzag_enabled=False),
    )
    summary = summarize_decoded_vae_audit(rows)

    assert nearest_ship_clearance_m(payload["u_pos"], [ship]) == 1000.0
    assert rows[0]["actual_outcome_label"] == "credible_hit_threat"
    assert rows[0]["clearance_ok"] is True
    assert rows[0]["passes_safety_gate"] is True
    assert rows[0]["centroid_static_label"] == "credible_hit_threat"
    assert summary["samples"] == 1
    assert summary["any_ship_hit_rate"] == 1.0
    assert np.isfinite(summary["mean_closest_any_ship_distance_m"])
