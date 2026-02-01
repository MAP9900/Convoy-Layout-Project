"""Serialization round-trip tests for scenarios, noise, policies, and plans."""

import numpy as np

from convoy_sim.attacker_tactics import AttackerPlan, PassSpec, SalvoSpec
from convoy_sim.defender_policy import DefenderPolicy, LayoutAction, ThreatType
from convoy_sim.feasibility import ApproachMode
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.noise import NoiseModel
from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_base import Scenario


def test_noise_model_roundtrip() -> None:
    noise = NoiseModel(sigma_heading_rad=0.1, sigma_launch_delay=0.2, p_dud=0.3)
    payload = noise.to_dict()
    restored = NoiseModel.from_dict(payload)
    assert restored == noise


def test_scenario_roundtrip() -> None:
    scenario = build_scenario_a(n_trials=5, rng_seed=11)
    payload = scenario.to_dict()
    restored = Scenario.from_dict(
        payload,
        layout_fn=scenario.layout_fn,
        torpedo_sampler=scenario.torpedo_sampler,
    )
    assert restored.name == scenario.name
    assert restored.layout_kwargs == scenario.layout_kwargs


def test_attacker_plan_roundtrip() -> None:
    plan = AttackerPlan(
        passes=[
            PassSpec(
                launch_time=0.0,
                u_boat_pos=np.array([-100.0, 0.0]),
                bearing_rad=0.0,
                approach_mode=ApproachMode.ABEAM,
                salvo=SalvoSpec(n_torpedoes=2, pattern="fan", spread_rad=0.1),
            )
        ],
        name="test",
    )
    payload = plan.to_dict()
    restored = AttackerPlan.from_dict(payload)
    assert restored.name == plan.name
    assert len(restored.passes) == 1


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
