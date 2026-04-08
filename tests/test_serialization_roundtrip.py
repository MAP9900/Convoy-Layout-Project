"""Serialization round-trip tests for scenarios, noise, and policies."""

from __future__ import annotations

import math

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import DefenderPolicy, LayoutAction, ThreatType
from convoy_sim.geometry import as_vec
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.noise import NoiseModel
from scenarios.convoy_profiles import get_convoy_layout_profile
from scenarios.scenario_base import Scenario


def _build_scenario() -> Scenario:
    profile = get_convoy_layout_profile("convoy_layout_1")

    def sampler(rng):
        return fan_spread(
            u_pos=as_vec(-2000.0, 0.0),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=math.radians(5.0),
            speed=15.0,
            max_run_time=500.0,
        )

    return Scenario(
        name="Serialization Roundtrip Smoke",
        layout_fn=profile.layout_fn,
        layout_kwargs=dict(profile.layout_kwargs),
        torpedo_sampler=sampler,
        n_trials=5,
        t_max=500.0,
        rng_seed=11,
        metadata={"profile_name": profile.name},
    )


def test_noise_model_roundtrip() -> None:
    noise = NoiseModel(sigma_heading_rad=0.1, sigma_launch_delay=0.2, p_dud=0.3)
    payload = noise.to_dict()
    restored = NoiseModel.from_dict(payload)
    assert restored == noise


def test_scenario_roundtrip() -> None:
    scenario = _build_scenario()
    payload = scenario.to_dict()
    restored = Scenario.from_dict(
        payload,
        layout_fn=scenario.layout_fn,
        torpedo_sampler=scenario.torpedo_sampler,
    )
    assert restored.name == scenario.name
    assert restored.layout_kwargs == scenario.layout_kwargs


def test_defender_policy_roundtrip() -> None:
    action = LayoutAction(
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
    )
    policy = DefenderPolicy(
        actions=[action],
        policy_table={ThreatType.ABEAM_FAN: {"rect": 1.0}},
    )
    payload = policy.to_dict()
    restored = DefenderPolicy.from_dict(
        payload,
        layout_fn_map={"make_rectangular_convoy": make_rectangular_convoy},
    )
    assert restored.actions[0].name == "rect"
    assert restored.policy_table[ThreatType.ABEAM_FAN]["rect"] == 1.0
