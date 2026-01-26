"""Tests for RL compatibility wrappers."""

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import LayoutAction, ThreatPrior, ThreatType
from convoy_sim.game import AttackerStrategy, DefenderStrategy
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.rl_wrapper import ActionSpaceMap, RLEpisode, build_observation, OBS_SCHEMA_VERSION


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
