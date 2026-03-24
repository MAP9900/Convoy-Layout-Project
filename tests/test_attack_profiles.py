"""Tests for attack profile schema and library sampling."""

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
    assert np.allclose(torpedoes[0].launch_position, np.array([-500.0, 100.0]))
    assert torpedoes[0].launch_delay == 5.0
    assert torpedoes[1].launch_delay == 7.0
    assert torpedoes[2].launch_delay == 9.0
