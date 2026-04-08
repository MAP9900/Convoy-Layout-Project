"""Tests for attack profile schema and library sampling."""

import math
import numpy as np
import pytest

from convoy_sim.attack_profiles import (
    AttackProfile,
    AttackProfileLibrary,
    DEFAULT_ATTACK_PROFILE_LIBRARY,
    make_placeholder_profile_library,
)


def test_default_library_has_25_profiles() -> None:
    assert len(DEFAULT_ATTACK_PROFILE_LIBRARY.profiles) == 30
    assert DEFAULT_ATTACK_PROFILE_LIBRARY.profile_ids()[0] == "P01"
    assert DEFAULT_ATTACK_PROFILE_LIBRARY.profile_ids()[-1] == "P30"


def test_placeholder_library_roundtrip() -> None:
    lib = make_placeholder_profile_library(25)
    payload = lib.to_dict()
    restored = AttackProfileLibrary.from_dict(payload)
    assert len(restored.profiles) == 25
    assert restored.profile_ids() == lib.profile_ids()


def test_library_requires_unique_ids() -> None:
    p1 = AttackProfile(profile_id="P01", name="one")
    p2 = AttackProfile(profile_id="P01", name="dup")
    with pytest.raises(ValueError):
        AttackProfileLibrary(profiles=[p1, p2])


def test_sampling_respects_zero_weights() -> None:
    lib = AttackProfileLibrary(
        profiles=[
            AttackProfile(profile_id="P01", name="off", weight=0.0),
            AttackProfile(profile_id="P02", name="on", weight=1.0),
        ]
    )
    rng = np.random.default_rng(0)
    sampled = [lib.sample_profile(rng).profile_id for _ in range(20)]
    assert set(sampled) == {"P02"}


def test_sampling_is_deterministic_for_seed() -> None:
    lib = AttackProfileLibrary(
        profiles=[
            AttackProfile(profile_id="P01", name="p1", weight=0.6),
            AttackProfile(profile_id="P02", name="p2", weight=0.4),
        ]
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    rng_c = np.random.default_rng(43)
    seq_a = [lib.sample_profile(rng_a).profile_id for _ in range(20)]
    seq_b = [lib.sample_profile(rng_b).profile_id for _ in range(20)]
    seq_c = [lib.sample_profile(rng_c).profile_id for _ in range(20)]
    assert seq_a == seq_b
    assert seq_a != seq_c


def test_profile_validation_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        AttackProfile(profile_id=" ", name="bad_id")
    with pytest.raises(ValueError):
        AttackProfile(profile_id="P01", name="bad_n", n=0)
    with pytest.raises(ValueError):
        AttackProfile(profile_id="P02", name="bad_speed", speed=0.0)
    with pytest.raises(ValueError):
        AttackProfile(profile_id="P03", name="bad_run", max_run_time=0.0)
    with pytest.raises(ValueError):
        AttackProfile(profile_id="P04", name="bad_delay", launch_delay_s=-1.0)
    with pytest.raises(ValueError):
        AttackProfile(profile_id="P05", name="bad_spread", spread_rad=-0.1)
    with pytest.raises(ValueError):
        AttackProfile(
            profile_id="P06",
            name="bad_parallel_spacing",
            mode="parallel",
            lateral_spacing=-1.0,
        )
    with pytest.raises(ValueError):
        AttackProfile(
            profile_id="P07",
            name="bad_explicit_offsets_missing",
            mode="fan",
            spread_doctrine="explicit_divergent",
        )
    with pytest.raises(ValueError):
        AttackProfile(
            profile_id="P08",
            name="bad_explicit_offsets_len",
            mode="fan",
            n=3,
            spread_doctrine="explicit_divergent",
            per_torpedo_heading_offsets_rad=(0.0, 0.1),
        )
    with pytest.raises(ValueError):
        AttackProfile(
            profile_id="P09",
            name="bad_offsets_wrong_doctrine",
            mode="fan",
            spread_doctrine="longitudinal",
            per_torpedo_heading_offsets_rad=(0.0,),
        )


def test_profile_build_torpedoes_uses_sim_named_fields() -> None:
    profile = AttackProfile(
        profile_id="P01",
        name="fan_demo",
        mode="fan",
        u_pos=(-500.0, 100.0),
        n=3,
        speed=20.0,
        max_run_time=120.0,
        base_bearing_rad=0.2,
        spread_rad=0.1,
        launch_delay_s=5.0,
        salvo_interval_s=2.0,
    )
    rng = np.random.default_rng(0)
    torpedoes = profile.build_torpedoes(rng)
    assert len(torpedoes) == 3
    # Legacy-bearing compatibility: absent explicit sub heading, the profile's
    # requested bearing becomes the U-boat heading used for bow-origin launch.
    launch_t = profile.launch_delay_s
    launch_center = np.array(profile.u_pos, dtype=float) + np.array(
        [
            math.cos(profile.base_bearing_rad) * profile.u_boat_initial_speed_mps * launch_t,
            math.sin(profile.base_bearing_rad) * profile.u_boat_initial_speed_mps * launch_t,
        ]
    )
    bow_offset = np.array(
        [
            math.cos(profile.base_bearing_rad) * profile.sub_length_m * 0.5,
            math.sin(profile.base_bearing_rad) * profile.sub_length_m * 0.5,
        ]
    )
    assert np.allclose(torpedoes[0].launch_position, launch_center + bow_offset)
    assert torpedoes[0].launch_delay == 5.0
    assert torpedoes[1].launch_delay == 7.0
    assert torpedoes[2].launch_delay == 9.0
    headings = np.array([torp.heading_rad for torp in torpedoes], dtype=float)
    launch_headings = np.array([torp.initial_heading_rad() for torp in torpedoes], dtype=float)
    assert np.isclose(np.mean(headings), profile.base_bearing_rad, atol=1e-6)
    assert np.allclose(launch_headings, profile.base_bearing_rad)
    assert all(np.isclose(torp.gyro_turn_distance_m, profile.gyro_straight_run_m) for torp in torpedoes)


def test_longitudinal_doctrine_produces_zero_heading_offsets() -> None:
    profile = AttackProfile(
        profile_id="PLONG",
        name="longitudinal",
        mode="fan",
        n=3,
        base_bearing_rad=0.2,
        spread_rad=0.2,
        spread_doctrine="longitudinal",
        u_boat_initial_heading_rad=0.2,
    )
    torpedoes = profile.build_torpedoes(np.random.default_rng(0))
    headings = np.array([torp.heading_rad for torp in torpedoes], dtype=float)
    assert np.allclose(headings, profile.base_bearing_rad)
    assert profile.is_standard_convoy_doctrine() is False
    assert "Rare/nonstandard convoy doctrine" in profile.doctrine_note()


def test_uniform_divergent_doctrine_preserves_linear_spread_behavior() -> None:
    profile = AttackProfile(
        profile_id="PUNIF",
        name="uniform_divergent",
        mode="fan",
        n=4,
        base_bearing_rad=0.3,
        spread_rad=0.24,
        spread_doctrine="uniform_divergent",
        u_boat_initial_heading_rad=0.3,
    )
    expected_offsets = np.array([-0.12, -0.04, 0.04, 0.12], dtype=float)
    assert np.allclose(profile.fan_heading_offsets_rad(), expected_offsets)
    torpedoes = profile.build_torpedoes(np.random.default_rng(1))
    final_offsets = np.array([torp.heading_rad - profile.base_bearing_rad for torp in torpedoes], dtype=float)
    assert np.allclose(final_offsets, expected_offsets)
    assert profile.is_standard_convoy_doctrine() is True
    assert "Standard convoy doctrine" in profile.doctrine_note()


def test_explicit_divergent_doctrine_applies_non_uniform_offsets() -> None:
    profile = AttackProfile(
        profile_id="PEXP",
        name="explicit_divergent",
        mode="fan",
        n=4,
        base_bearing_rad=0.3,
        spread_doctrine="explicit_divergent",
        per_torpedo_heading_offsets_rad=(-0.15, -0.03, 0.01, 0.11),
        u_boat_initial_heading_rad=0.3,
    )
    torpedoes = profile.build_torpedoes(np.random.default_rng(2))
    final_offsets = np.array([torp.heading_rad - profile.base_bearing_rad for torp in torpedoes], dtype=float)
    assert np.allclose(final_offsets, np.array(profile.per_torpedo_heading_offsets_rad, dtype=float))
    assert profile.is_standard_convoy_doctrine() is True
    assert "Manual convoy doctrine" in profile.doctrine_note()


def test_default_library_p01_uses_profile_bearing_as_attack_intent() -> None:
    profile = next(p for p in DEFAULT_ATTACK_PROFILE_LIBRARY.profiles if p.profile_id == "P01")
    torpedoes = profile.build_torpedoes(np.random.default_rng(0))
    headings = np.array([torp.heading_rad for torp in torpedoes], dtype=float)
    launch_headings = np.array([torp.initial_heading_rad() for torp in torpedoes], dtype=float)
    mean_heading = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    expected = float(np.arctan2(np.sin(profile.base_bearing_rad), np.cos(profile.base_bearing_rad)))
    assert np.isclose(mean_heading, expected, atol=1e-3)
    assert np.allclose(launch_headings, profile.u_boat_initial_heading_rad)


def test_backward_compatibility_infers_doctrine_from_legacy_spread_rad() -> None:
    legacy_uniform = AttackProfile.from_dict(
        {
            "profile_id": "PLEG1",
            "name": "legacy_uniform",
            "mode": "fan",
            "n": 3,
            "base_bearing_rad": 0.2,
            "spread_rad": 0.1,
        }
    )
    legacy_longitudinal = AttackProfile.from_dict(
        {
            "profile_id": "PLEG2",
            "name": "legacy_longitudinal",
            "mode": "fan",
            "n": 3,
            "base_bearing_rad": 0.2,
            "spread_rad": 0.0,
        }
    )
    assert legacy_uniform.resolved_spread_doctrine() == "uniform_divergent"
    assert legacy_longitudinal.resolved_spread_doctrine() == "longitudinal"
    assert np.allclose(legacy_longitudinal.fan_heading_offsets_rad(), np.zeros(3))


def test_to_dict_omits_new_doctrine_fields_for_legacy_equivalent_profiles() -> None:
    profile = AttackProfile(
        profile_id="PLEG3",
        name="legacy_equivalent",
        mode="fan",
        n=3,
        base_bearing_rad=0.2,
        spread_rad=0.1,
    )
    payload = profile.to_dict()
    assert "spread_doctrine" not in payload
    assert "per_torpedo_heading_offsets_rad" not in payload


def test_default_library_profiles_no_longer_depend_on_legacy_bearing_compat() -> None:
    assert all(not profile.uses_legacy_bearing_compat() for profile in DEFAULT_ATTACK_PROFILE_LIBRARY.profiles)


@pytest.mark.parametrize(
    ("spread_doctrine", "spread_rad", "per_offsets"),
    [
        ("uniform_divergent", 0.1, ()),
        ("explicit_divergent", 0.0, (-0.05, 0.04)),
        ("longitudinal", 0.0, ()),
    ],
)
def test_profile_rejects_turning_u_boat_during_salvo_by_default(
    spread_doctrine: str,
    spread_rad: float,
    per_offsets: tuple[float, ...],
) -> None:
    profile = AttackProfile(
        profile_id="PTURN",
        name="turning_salvo",
        mode="fan",
        u_pos=(-1000.0, 0.0),
        n=2,
        speed=15.0,
        max_run_time=200.0,
        base_bearing_rad=0.3,
        spread_rad=spread_rad,
        spread_doctrine=spread_doctrine,
        per_torpedo_heading_offsets_rad=per_offsets,
        salvo_interval_s=5.0,
        u_boat_mode="moving",
        u_boat_initial_heading_rad=0.0,
        u_boat_initial_speed_mps=2.0,
        u_boat_launch_time_s=20.0,
        u_boat_turn_rate_limit_rad_s=0.02,
        u_boat_motion_legs=((100.0, 0.5, 2.0),),
    )
    with pytest.raises(
        ValueError,
        match="steady firing course|heading changes during the firing window",
    ):
        profile.build_torpedoes(np.random.default_rng(1))


def test_fan_spread_width_is_preserved_as_final_gyro_deflection() -> None:
    profile = AttackProfile(
        profile_id="PGYRO",
        name="gyro_spread",
        mode="fan",
        u_pos=(-1000.0, 0.0),
        n=4,
        speed=15.0,
        max_run_time=200.0,
        base_bearing_rad=0.3,
        spread_rad=0.24,
        u_boat_initial_heading_rad=0.3,
    )
    torpedoes = profile.build_torpedoes(np.random.default_rng(4))
    final_headings = np.array([torpedo.heading_rad for torpedo in torpedoes], dtype=float)
    launch_headings = np.array([torpedo.initial_heading_rad() for torpedo in torpedoes], dtype=float)
    width = float(final_headings.max() - final_headings.min())
    assert np.isclose(width, profile.spread_rad, atol=1e-9)
    assert np.allclose(launch_headings, profile.u_boat_initial_heading_rad)
