from __future__ import annotations

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.attack_profiles import AttackProfileLibrary
from convoy_sim.entities import Ship, ShipClass, Torpedo
from convoy_sim.feasibility import Environment
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.noise import NoiseModel
from convoy_sim.realism import (
    AttackerObservationConfig,
    ShipMovementRealismConfig,
    UBoatLeg,
    UBoatMotionPlan,
    apply_ship_movement_realism,
    build_attacker_observation,
)
from convoy_sim.simulation import apply_noise_to_torpedoes
from convoy_sim.workflows import evaluate_layout_over_profiles


def _ships() -> list[Ship]:
    return [
        Ship(
            id="S1",
            position=np.array([0.0, 0.0]),
            speed=5.0,
            heading_rad=0.0,
            length=150.0,
            beam=20.0,
            ship_class=ShipClass.FREIGHTER,
        ),
        Ship(
            id="S2",
            position=np.array([1000.0, 0.0]),
            speed=5.0,
            heading_rad=0.0,
            length=150.0,
            beam=20.0,
            ship_class=ShipClass.ESCORT,
        ),
    ]


def test_uboat_motion_plan_deterministic_and_bounds() -> None:
    plan = UBoatMotionPlan(
        initial_position=np.array([-2000.0, 0.0]),
        initial_heading_rad=0.0,
        initial_speed_mps=2.0,
        mode="moving",
        legs=(
            UBoatLeg(duration_s=60.0, heading_rad=0.5, speed_mps=3.0),
            UBoatLeg(duration_s=60.0, heading_rad=1.0, speed_mps=4.0),
        ),
        launch_time_s=90.0,
        turn_rate_limit_rad_s=0.01,
        accel_limit_mps2=0.05,
    )
    pos_a, heading_a, speed_a = plan.state_at(90.0)
    pos_b, heading_b, speed_b = plan.state_at(90.0)
    assert np.allclose(pos_a, pos_b)
    assert heading_a == heading_b
    assert speed_a == speed_b
    assert heading_a <= 1.0
    assert speed_a <= 4.0


def test_attack_profile_moving_default_and_static_fallback() -> None:
    moving = AttackProfile(
        profile_id="PX1",
        name="moving",
        mode="fan",
        u_pos=(-2000.0, 0.0),
        n=1,
        speed=15.0,
        max_run_time=200.0,
        base_bearing_rad=0.0,
        spread_rad=0.0,
        u_boat_mode="moving",
        u_boat_initial_heading_rad=0.0,
        u_boat_initial_speed_mps=2.0,
        u_boat_launch_time_s=100.0,
    )
    static = AttackProfile(
        profile_id="PX2",
        name="static",
        mode="fan",
        u_pos=(-2000.0, 0.0),
        n=1,
        speed=15.0,
        max_run_time=200.0,
        base_bearing_rad=0.0,
        spread_rad=0.0,
        u_boat_mode="static",
        u_boat_initial_heading_rad=0.0,
        u_boat_initial_speed_mps=2.0,
        u_boat_launch_time_s=100.0,
    )
    rng = np.random.default_rng(7)
    torp_moving = moving.build_torpedoes(rng=rng, ships=_ships(), env=Environment("night", 3000.0, 4))[0]
    torp_static = static.build_torpedoes(rng=np.random.default_rng(7), ships=_ships(), env=Environment("night", 3000.0, 4))[0]
    assert torp_moving.launch_position[0] > torp_static.launch_position[0]
    assert np.isclose(torp_static.launch_position[0], -2000.0)
    assert np.isclose(torp_moving.launch_delay, 100.0)


def test_attack_profile_enforces_bow_fire_direction() -> None:
    profile = AttackProfile(
        profile_id="PX3",
        name="bow_fire",
        mode="fan",
        u_pos=(-2000.0, 0.0),
        n=3,
        speed=15.0,
        max_run_time=200.0,
        base_bearing_rad=2.5,  # should be ignored by bow-fire constraint
        spread_rad=0.1,
        u_boat_mode="moving",
        u_boat_initial_heading_rad=0.4,
        u_boat_initial_speed_mps=2.0,
        u_boat_launch_time_s=50.0,
    )
    plan = UBoatMotionPlan(
        initial_position=np.array(profile.u_pos, dtype=float),
        initial_heading_rad=profile.u_boat_initial_heading_rad,
        initial_speed_mps=profile.u_boat_initial_speed_mps,
        mode=profile.u_boat_mode,
        launch_time_s=profile.u_boat_launch_time_s,
    )
    _pos, launch_heading, _speed = plan.state_at(profile.u_boat_launch_time_s)
    torps = profile.build_torpedoes(np.random.default_rng(1), ships=_ships(), env=Environment("night", 3000.0, 4))
    headings = np.array([t.heading_rad for t in torps], dtype=float)
    assert np.isclose(np.mean(headings), launch_heading, atol=1e-6)


def test_partial_observation_is_reproducible() -> None:
    env = Environment(time_of_day="night", visibility_m=3500.0, sea_state=4)
    cfg = AttackerObservationConfig()
    obs_a = build_attacker_observation(
        ships=_ships(),
        u_boat_pos=np.array([-2000.0, 0.0]),
        env=env,
        rng=np.random.default_rng(123),
        cfg=cfg,
    )
    obs_b = build_attacker_observation(
        ships=_ships(),
        u_boat_pos=np.array([-2000.0, 0.0]),
        env=env,
        rng=np.random.default_rng(123),
        cfg=cfg,
    )
    assert obs_a == obs_b
    assert "estimated_bearing_rad" in obs_a
    assert "observation_quality" in obs_a


def test_ship_movement_overlay_is_bounded() -> None:
    ships = _ships()
    originals = [np.array(ship.position, dtype=float) for ship in ships]
    cfg = ShipMovementRealismConfig(
        position_jitter_std_m=20.0,
        heading_jitter_std_rad=0.05,
        deviation_offset_cap_m=15.0,
        enable_slot_swaps=False,
    )
    out = apply_ship_movement_realism(ships, rng=np.random.default_rng(3), cfg=cfg)
    for ship, original in zip(out, originals):
        displacement = float(np.linalg.norm(np.asarray(ship.position) - original))
        assert displacement <= 15.0001


def test_noise_model_applies_speed_and_dud() -> None:
    torpedo = Torpedo(
        id="T1",
        launch_position=np.array([0.0, 0.0]),
        speed=10.0,
        heading_rad=0.0,
        max_run_time=100.0,
    )
    model = NoiseModel(sigma_speed_mps=1.0, p_dud=1.0)
    adjusted = apply_noise_to_torpedoes([torpedo], model, np.random.default_rng(11))[0]
    assert adjusted.speed != 10.0
    assert adjusted.is_dud is True


def test_static_mode_profile_compatibility_in_workflow_eval() -> None:
    profile = AttackProfile(
        profile_id="PSTAT",
        name="static_compat",
        mode="fan",
        u_pos=(-1800.0, 0.0),
        n=2,
        speed=16.0,
        max_run_time=350.0,
        base_bearing_rad=0.0,
        spread_rad=0.1,
        u_boat_mode="static",
    )
    rows = evaluate_layout_over_profiles(
        model_name="compat",
        layout_fn=make_rectangular_convoy,
        layout_kwargs={
            "n_rows": 1,
            "n_cols": 2,
            "spacing_along": 450.0,
            "spacing_across": 300.0,
            "speed": 5.0,
            "heading_rad": 0.0,
            "length": 120.0,
            "beam": 18.0,
            "origin": np.array([0.0, 0.0]),
        },
        library=AttackProfileLibrary(profiles=[profile]),
        profile_ids=["PSTAT"],
        seeds=[7],
        n_trials_per_seed=2,
        t_max=200.0,
        env=Environment("night", 3000.0, 4),
    )
    assert len(rows) == 1
    assert rows[0].samples == 2
