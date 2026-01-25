"""Policy optimization helpers for defender layout selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .defender_policy import (
    DefenderPolicy,
    LayoutAction,
    ThreatPrior,
    ThreatType,
    evaluate_defender_policy,
)
from .entities import Torpedo
from .objectives import ObjectiveSpec


@dataclass(frozen=True)
class PolicyObjective:
    """Objective weights and hard budgets for policy optimization."""

    w_loss: float = 1.0
    w_footprint: float = 0.0
    w_complexity: float = 0.0
    footprint_budget: float | None = None
    complexity_budget: float | None = None


def score_policy_eval(eval_result: dict[str, Any], obj: PolicyObjective) -> float:
    """Return a scalar score (lower is better) for a policy evaluation."""

    summary = eval_result.get("summary", {})
    expected_loss = summary.get("expected_loss")
    if expected_loss is None:
        expected_loss = float(summary.get("expected_value_destroyed", 0.0))
    mean_footprint = float(summary.get("mean_footprint_area", 0.0))
    mean_complexity = float(summary.get("mean_complexity_cost", 0.0))

    if obj.footprint_budget is not None and mean_footprint > obj.footprint_budget:
        return math.inf
    if obj.complexity_budget is not None and mean_complexity > obj.complexity_budget:
        return math.inf

    return (
        obj.w_loss * float(expected_loss)
        + obj.w_footprint * mean_footprint
        + obj.w_complexity * mean_complexity
    )


def _evaluate_policy_for_threat(
    threat: ThreatType,
    policy: DefenderPolicy,
    attacker_factory: Callable[[ThreatType], Callable[[np.random.Generator], list[Torpedo]]],
    n_trials: int,
    sim_kwargs: dict[str, Any],
    objective_spec: ObjectiveSpec | None,
    rng_seed: int,
) -> dict[str, Any]:
    prior = ThreatPrior(probs={threat: 1.0})
    rng = np.random.default_rng(rng_seed)
    return evaluate_defender_policy(
        prior=prior,
        policy=policy,
        attacker_factory=attacker_factory,
        n_trials=n_trials,
        sim_kwargs=sim_kwargs,
        objective=objective_spec,
        rng=rng,
    )


def optimize_policy_deterministic(
    prior: ThreatPrior,
    actions: list[LayoutAction],
    threats: list[ThreatType],
    attacker_factory: Callable[[ThreatType], Callable[[np.random.Generator], list[Torpedo]]],
    n_trials: int,
    sim_kwargs: dict[str, Any],
    objective_spec: ObjectiveSpec | None,
    policy_obj: PolicyObjective,
    rng_seed: int = 0,
) -> tuple[DefenderPolicy, dict[str, Any]]:
    """Select the best deterministic action per threat under tradeoffs."""

    policy_table: dict[ThreatType, dict[str, float]] = {}
    per_threat: dict[str, Any] = {}

    for threat_idx, threat in enumerate(threats):
        best_score = math.inf
        best_action = actions[0]
        best_eval: dict[str, Any] | None = None
        for action_idx, action in enumerate(actions):
            policy = DefenderPolicy(
                actions=actions,
                policy_table={threat: {action.name: 1.0}},
            )
            eval_result = _evaluate_policy_for_threat(
                threat,
                policy,
                attacker_factory,
                n_trials,
                sim_kwargs,
                objective_spec,
                rng_seed + threat_idx * 100 + action_idx,
            )
            score = score_policy_eval(eval_result, policy_obj)
            if score < best_score:
                best_score = score
                best_action = action
                best_eval = eval_result
        policy_table[threat] = {best_action.name: 1.0}
        per_threat[threat.value] = {
            "best_action": best_action.name,
            "score": best_score,
            "summary": None if best_eval is None else best_eval.get("summary"),
        }

    policy = DefenderPolicy(actions=actions, policy_table=policy_table)
    final_eval = evaluate_defender_policy(
        prior=prior,
        policy=policy,
        attacker_factory=attacker_factory,
        n_trials=n_trials,
        sim_kwargs=sim_kwargs,
        objective=objective_spec,
        rng=np.random.default_rng(rng_seed + 999),
    )
    return policy, {"per_threat": per_threat, "final_eval": final_eval}


def optimize_policy_mixture_pairwise(
    prior: ThreatPrior,
    actions: list[LayoutAction],
    threats: list[ThreatType],
    attacker_factory: Callable[[ThreatType], Callable[[np.random.Generator], list[Torpedo]]],
    n_trials: int,
    sim_kwargs: dict[str, Any],
    objective_spec: ObjectiveSpec | None,
    policy_obj: PolicyObjective,
    rng_seed: int = 0,
    mix_grid: list[float] | None = None,
) -> tuple[DefenderPolicy, dict[str, Any]]:
    """Select up to two-action mixtures per threat over a coarse grid."""

    mix_grid = mix_grid or [0.0, 0.25, 0.5, 0.75, 1.0]
    policy_table: dict[ThreatType, dict[str, float]] = {}
    per_threat: dict[str, Any] = {}

    for threat_idx, threat in enumerate(threats):
        best_score = math.inf
        best_mix: dict[str, float] = {actions[0].name: 1.0}
        best_eval: dict[str, Any] | None = None
        for i, action_a in enumerate(actions):
            for j, action_b in enumerate(actions[i:], start=i):
                for mix_idx, p in enumerate(mix_grid):
                    weights = {
                        action_a.name: float(p),
                        action_b.name: float(1.0 - p),
                    }
                    if action_a.name == action_b.name:
                        weights = {action_a.name: 1.0}
                    policy = DefenderPolicy(actions=actions, policy_table={threat: weights})
                    eval_result = _evaluate_policy_for_threat(
                        threat,
                        policy,
                        attacker_factory,
                        n_trials,
                        sim_kwargs,
                        objective_spec,
                        rng_seed + threat_idx * 1000 + i * 100 + j * 10 + mix_idx,
                    )
                    score = score_policy_eval(eval_result, policy_obj)
                    if score < best_score:
                        best_score = score
                        best_mix = weights
                        best_eval = eval_result
        policy_table[threat] = best_mix
        per_threat[threat.value] = {
            "best_mix": dict(best_mix),
            "score": best_score,
            "summary": None if best_eval is None else best_eval.get("summary"),
        }

    policy = DefenderPolicy(actions=actions, policy_table=policy_table)
    final_eval = evaluate_defender_policy(
        prior=prior,
        policy=policy,
        attacker_factory=attacker_factory,
        n_trials=n_trials,
        sim_kwargs=sim_kwargs,
        objective=objective_spec,
        rng=np.random.default_rng(rng_seed + 1999),
    )
    return policy, {"per_threat": per_threat, "final_eval": final_eval}
