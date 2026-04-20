"""Tests for RL compatibility wrappers."""

import numpy as np

from convoy_sim.attack_profiles import AttackProfile, AttackProfileLibrary
from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import LayoutAction, ThreatPrior, ThreatType
from convoy_sim.game import AttackerStrategy, DefenderStrategy
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.rl_layout_builder import RLLayoutBuilderConfig
from convoy_sim.rl_wrapper import (
    ActionSpaceMap,
    OBS_SCHEMA_VERSION,
    RLLayoutBuilderEpisode,
    RLEpisode,
    build_observation,
)


def test_action_space_map_roundtrip() -> None:
    mapping = ActionSpaceMap(names=["a", "b"])
    assert mapping.to_name(mapping.to_index("b")) == "b"


def test_build_observation_schema_version() -> None:
    obs = build_observation(
        time_step=0,
        threat=None,
        defender_action=None,
        attacker_action=None,
        layout_metrics=None,
        outcome=None,
    )
    assert obs["schema_version"] == OBS_SCHEMA_VERSION
    assert "attack_profile_id" in obs
    assert "attack_profile" in obs


def test_episode_step_returns_reward_and_done() -> None:
    defenders = [
        DefenderStrategy(
            name="rect",
            kind="layout_action",
            payload=LayoutAction(
                name="rect",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 1,
                    "n_cols": 1,
                    "spacing_along": 100.0,
                    "spacing_across": 100.0,
                    "speed": 0.0,
                    "heading_rad": 0.0,
                    "length": 40.0,
                    "beam": 10.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        )
    ]
    attackers = [
        AttackerStrategy(
            name="fan1",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=1,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )
    ]
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    env = RLEpisode(
        defenders=defenders,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 200.0},
        max_steps=1,
        rng=np.random.default_rng(0),
    )
    env.reset(seed=0)
    obs, reward, done, info = env.step(0, 0)
    assert "defender_loss" in info
    assert isinstance(reward, float)
    assert done


def test_episode_reward_sign_consistency() -> None:
    defenders = [
        DefenderStrategy(
            name="rect",
            kind="layout_action",
            payload=LayoutAction(
                name="rect",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 1,
                    "n_cols": 1,
                    "spacing_along": 100.0,
                    "spacing_across": 100.0,
                    "speed": 0.0,
                    "heading_rad": 0.0,
                    "length": 40.0,
                    "beam": 10.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        )
    ]
    attackers = [
        AttackerStrategy(
            name="fan1",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=1,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )
    ]
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    env_def = RLEpisode(
        defenders=defenders,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 200.0},
        max_steps=1,
        reward_perspective="defender",
        rng=np.random.default_rng(0),
    )
    env_att = RLEpisode(
        defenders=defenders,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 200.0},
        max_steps=1,
        reward_perspective="attacker",
        rng=np.random.default_rng(0),
    )
    env_def.reset(seed=0)
    env_att.reset(seed=0)
    _obs_d, reward_d, _done_d, info_d = env_def.step(0, 0)
    _obs_a, reward_a, _done_a, info_a = env_att.step(0, 0)
    assert np.isclose(reward_a, info_d["attacker_utility"])
    assert np.isclose(reward_d, -info_d["defender_loss"])


def test_episode_samples_attack_profile_on_reset_and_logs_payload() -> None:
    defenders = [
        DefenderStrategy(
            name="rect",
            kind="layout_action",
            payload=LayoutAction(
                name="rect",
                layout_fn=make_rectangular_convoy,
                layout_kwargs={
                    "n_rows": 1,
                    "n_cols": 1,
                    "spacing_along": 100.0,
                    "spacing_across": 100.0,
                    "speed": 0.0,
                    "heading_rad": 0.0,
                    "length": 40.0,
                    "beam": 10.0,
                    "origin": np.array([0.0, 0.0]),
                },
                complexity_cost=1.0,
            ),
        )
    ]
    attackers = [
        AttackerStrategy(
            name="fan1",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=1,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )
    ]
    profile_lib = AttackProfileLibrary(
        profiles=[
            AttackProfile(
                profile_id="P01",
                name="single_profile",
                mode="fan",
                u_pos=(-200.0, 0.0),
                n=1,
                speed=20.0,
                max_run_time=200.0,
                base_bearing_rad=0.0,
                spread_rad=0.0,
            )
        ]
    )
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    env = RLEpisode(
        defenders=defenders,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 200.0},
        attack_profile_library=profile_lib,
        max_steps=1,
        rng=np.random.default_rng(0),
    )
    obs0 = env.reset(seed=123)
    assert obs0["attack_profile_id"] == "P01"
    assert obs0["attack_profile"]["profile_id"] == "P01"

    obs, _reward, _done, info = env.step(0, 0)
    assert obs["attack_profile_id"] == "P01"
    assert obs["attack_profile"]["profile_id"] == "P01"
    assert info["attack_profile_id"] == "P01"
    assert info["attack_profile"]["profile_id"] == "P01"


def test_builder_episode_returns_terminal_reward_only_after_final_choice() -> None:
    attackers = [
        AttackerStrategy(
            name="fan1",
            kind="torpedo_sampler",
            payload=lambda rng: fan_spread(
                u_pos=np.array([-200.0, 0.0]),
                base_bearing_rad=0.0,
                n=1,
                spread_rad=0.0,
                speed=20.0,
                max_run_time=200.0,
            ),
        )
    ]
    builder_cfg = RLLayoutBuilderConfig.from_dict(
        {
            "enabled": True,
            "base_n_rows": 1,
            "base_n_cols": 2,
            "speed": 0.0,
            "heading_rad": 0.0,
            "length": 40.0,
            "beam": 10.0,
            "origin": [0.0, 0.0],
            "layout_families": ["rectangular", "staggered"],
            "row_patterns": {"uniform": [2]},
            "row_offset_policies": ["none"],
            "class_placement_policies": ["mixed_balanced"],
            "spacing_along_options": {"compact": 100.0, "loose": 150.0},
            "spacing_across_options": {"compact": 90.0, "loose": 120.0},
        }
    )
    prior = ThreatPrior(probs={ThreatType.ABEAM_FAN: 1.0})
    env = RLLayoutBuilderEpisode(
        builder_config=builder_cfg,
        attackers=attackers,
        prior=prior,
        env=None,
        constraints=None,
        dynamics=None,
        sim_params={"t_max": 200.0},
        rng=np.random.default_rng(0),
    )

    obs0 = env.reset(seed=0)
    assert obs0["builder_mode"] is True
    assert obs0["valid_defender_actions"] == ["family:rectangular", "family:staggered"]

    obs1, reward1, done1, _info1 = env.step(env.action_space.to_index("family:rectangular"), 0)
    assert reward1 == 0.0
    assert done1 is False
    assert obs1["builder_state"]["family"] == "rectangular"
    assert obs1["valid_defender_actions"] == ["pattern:uniform"]

    obs2, reward2, done2, _info2 = env.step(env.action_space.to_index("pattern:uniform"), 0)
    assert reward2 == 0.0
    assert done2 is False
    assert obs2["builder_state"]["row_pattern"] == "uniform"
    assert obs2["valid_defender_actions"] == ["offset:none"]

    obs3, reward3, done3, _info3 = env.step(env.action_space.to_index("offset:none"), 0)
    assert reward3 == 0.0
    assert done3 is False
    assert obs3["builder_state"]["row_offset_policy"] == "none"
    assert obs3["valid_defender_actions"] == ["placement:mixed_balanced"]

    obs4, reward4, done4, _info4 = env.step(env.action_space.to_index("placement:mixed_balanced"), 0)
    assert reward4 == 0.0
    assert done4 is False
    assert obs4["builder_state"]["class_placement_policy"] == "mixed_balanced"
    assert obs4["valid_defender_actions"] == ["along:compact", "along:loose"]

    obs5, reward5, done5, _info5 = env.step(env.action_space.to_index("along:compact"), 0)
    assert reward5 == 0.0
    assert done5 is False
    assert obs5["builder_state"]["spacing_along_bucket"] == "compact"
    assert obs5["valid_defender_actions"] == ["across:compact", "across:loose"]

    obs6, reward6, done6, info6 = env.step(env.action_space.to_index("across:loose"), 0)
    assert isinstance(reward6, float)
    assert done6 is True
    assert obs6["defender_action"] == "rect_uniform_none_mixed_balanced_compact_loose"
    assert info6["materialized_action"]["name"] == "rect_uniform_none_mixed_balanced_compact_loose"
