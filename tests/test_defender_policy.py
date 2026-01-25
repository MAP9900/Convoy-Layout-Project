"""Tests for defender policy scaffolding."""

import math

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.defender_policy import (
    DefenderPolicy,
    LayoutAction,
    ThreatPrior,
    ThreatType,
    evaluate_defender_policy,
    make_deterministic_policy,
)
from convoy_sim.layouts import make_rectangular_convoy


def _rect_kwargs(spacing_across: float) -> dict[str, float]:
    return {
        "n_rows": 2,
        "n_cols": 2,
        "spacing_along": 200.0,
        "spacing_across": spacing_across,
        "speed": 5.0,
        "heading_rad": 0.0,
        "length": 120.0,
        "beam": 20.0,
        "origin": np.array([0.0, 0.0]),
    }


def test_threat_prior_normalization() -> None:
    prior = ThreatPrior(
        probs={
            ThreatType.ABEAM_FAN: 2.0,
            ThreatType.BOW_ON_FAN: 1.0,
        }
    ).normalized()
    assert math.isclose(sum(prior.probs.values()), 1.0)
    assert math.isclose(prior.probs[ThreatType.ABEAM_FAN], 2.0 / 3.0)
    assert math.isclose(prior.probs[ThreatType.BOW_ON_FAN], 1.0 / 3.0)


def test_threat_prior_sampling_respects_weights() -> None:
    rng = np.random.default_rng(123)
    prior = ThreatPrior(
        probs={
            ThreatType.ABEAM_FAN: 0.7,
            ThreatType.BOW_ON_FAN: 0.3,
        }
    )
    draws = [prior.sample(rng) for _ in range(5000)]
    freq = draws.count(ThreatType.ABEAM_FAN) / len(draws)
    assert abs(freq - 0.7) < 0.05


def test_deterministic_policy_always_picks_same_action() -> None:
    actions = [
        LayoutAction(
            name="tight_rect",
            layout_fn=make_rectangular_convoy,
            layout_kwargs=_rect_kwargs(200.0),
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="wide_rect",
            layout_fn=make_rectangular_convoy,
            layout_kwargs=_rect_kwargs(400.0),
            complexity_cost=2.0,
        ),
    ]
    policy = make_deterministic_policy(
        actions,
        threat_action_map={ThreatType.ABEAM_FAN: "tight_rect"},
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = policy.sample_action(ThreatType.ABEAM_FAN, rng)
        assert action.name == "tight_rect"


def test_evaluate_defender_policy_smoke() -> None:
    actions = [
        LayoutAction(
            name="tight_rect",
            layout_fn=make_rectangular_convoy,
            layout_kwargs=_rect_kwargs(200.0),
            complexity_cost=1.0,
        ),
        LayoutAction(
            name="wide_rect",
            layout_fn=make_rectangular_convoy,
            layout_kwargs=_rect_kwargs(400.0),
            complexity_cost=2.0,
        ),
    ]
    policy = DefenderPolicy(
        actions=actions,
        policy_table={
            ThreatType.ABEAM_FAN: {"tight_rect": 1.0},
            ThreatType.BOW_ON_FAN: {"wide_rect": 1.0},
        },
    )
    prior = ThreatPrior(
        probs={
            ThreatType.ABEAM_FAN: 0.5,
            ThreatType.BOW_ON_FAN: 0.5,
        }
    )

    def attacker_factory(threat: ThreatType):
        def sampler(_: np.random.Generator):
            bearing = 0.0 if threat == ThreatType.BOW_ON_FAN else math.pi / 2.0
            return fan_spread(
                u_pos=np.array([-2000.0, 0.0]),
                base_bearing_rad=bearing,
                n=2,
                spread_rad=math.radians(5.0),
                speed=25.0,
                max_run_time=800.0,
            )

        return sampler

    result = evaluate_defender_policy(
        prior=prior,
        policy=policy,
        attacker_factory=attacker_factory,
        n_trials=5,
        sim_kwargs={"t_max": 400.0},
        rng=np.random.default_rng(42),
    )

    summary = result["summary"]
    assert "expected_hits" in summary
    assert "expected_value_destroyed" in summary
    assert "action_frequencies" in summary
    assert len(result["trials"]) == 5
