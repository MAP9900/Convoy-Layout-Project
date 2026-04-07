"""Tests for deterministic fire-control-lite helpers."""

from __future__ import annotations

import numpy as np

from convoy_sim.fire_control import (
    FireControlLiteConfig,
    build_attack_profile_from_fire_control,
    choose_g7a_speed_setting,
    solve_fire_control_lite,
)


def _observation(
    *,
    bearing_rad: float = 0.2,
    range_m: float = 1200.0,
    convoy_heading_rad: float = 0.0,
    convoy_speed_mps: float = 4.0,
    bearing_sigma_rad: float = 0.04,
    range_sigma_m: float = 120.0,
    heading_sigma_rad: float = 0.06,
) -> dict[str, object]:
    return {
        "estimated_bearing_rad": float(bearing_rad),
        "estimated_range_m": float(range_m),
        "estimated_convoy_heading_rad": float(convoy_heading_rad),
        "estimated_convoy_speed_mps": float(convoy_speed_mps),
        "observation_quality": {
            "bearing_sigma_rad": float(bearing_sigma_rad),
            "range_sigma_m": float(range_sigma_m),
            "heading_sigma_rad": float(heading_sigma_rad),
        },
    }


def test_fire_control_lite_is_deterministic() -> None:
    cfg = FireControlLiteConfig()
    obs = _observation()
    sol_a = solve_fire_control_lite(
        u_boat_position=np.array([-1000.0, 0.0]),
        u_boat_heading_rad=0.2,
        attacker_observation=obs,
        cfg=cfg,
    )
    sol_b = solve_fire_control_lite(
        u_boat_position=np.array([-1000.0, 0.0]),
        u_boat_heading_rad=0.2,
        attacker_observation=obs,
        cfg=cfg,
    )
    assert np.allclose(sol_a.target_point_est_m, sol_b.target_point_est_m)
    assert sol_a.as_metadata() == sol_b.as_metadata()


def test_choose_g7a_speed_setting_uses_range_bands() -> None:
    cfg = FireControlLiteConfig(fast_setting_max_range_m=1500.0, medium_setting_max_range_m=3000.0)
    assert choose_g7a_speed_setting(1200.0, cfg=cfg).name == "fast"
    assert choose_g7a_speed_setting(2200.0, cfg=cfg).name == "medium"
    assert choose_g7a_speed_setting(4200.0, cfg=cfg).name == "long_range"


def test_fire_control_lite_uncertainty_widens_spread() -> None:
    cfg = FireControlLiteConfig()
    low_unc = solve_fire_control_lite(
        u_boat_position=np.array([-1000.0, 0.0]),
        u_boat_heading_rad=0.2,
        attacker_observation=_observation(range_sigma_m=20.0, bearing_sigma_rad=0.01, heading_sigma_rad=0.01),
        cfg=cfg,
    )
    high_unc = solve_fire_control_lite(
        u_boat_position=np.array([-1000.0, 0.0]),
        u_boat_heading_rad=0.2,
        attacker_observation=_observation(range_sigma_m=250.0, bearing_sigma_rad=0.08, heading_sigma_rad=0.12),
        cfg=cfg,
    )
    assert high_unc.spread_rad > low_unc.spread_rad


def test_fire_control_lite_clips_centerline_to_bow_arc() -> None:
    cfg = FireControlLiteConfig(max_bow_centerline_offset_deg=15.0)
    sol = solve_fire_control_lite(
        u_boat_position=np.array([0.0, 0.0]),
        u_boat_heading_rad=0.0,
        attacker_observation=_observation(bearing_rad=np.deg2rad(70.0), convoy_heading_rad=np.deg2rad(90.0)),
        cfg=cfg,
    )
    assert np.isclose(sol.centerline_bearing_rad, np.deg2rad(15.0), atol=1e-6)
    assert np.isclose(sol.bow_offset_rad, np.deg2rad(15.0), atol=1e-6)


def test_build_attack_profile_from_fire_control_returns_coherent_profile() -> None:
    cfg = FireControlLiteConfig(desired_salvo_size=4)
    profile = build_attack_profile_from_fire_control(
        profile_id="FC01",
        name="fire_control",
        u_boat_position=np.array([-1200.0, 100.0]),
        u_boat_heading_rad=0.2,
        attacker_observation=_observation(),
        cfg=cfg,
        max_bow_offset_deg=15.0,
    )
    assert profile.profile_id == "FC01"
    assert profile.n == 4
    assert profile.u_boat_initial_heading_rad == 0.2
    assert profile.speed > 0.0
    assert profile.max_run_time > 0.0
    assert not profile.uses_legacy_bearing_compat()
    torpedoes = profile.build_torpedoes(np.random.default_rng(1))
    assert len(torpedoes) == 4
    assert all(np.isclose(torp.initial_heading_rad(), 0.2) for torp in torpedoes)
