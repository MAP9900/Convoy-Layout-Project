"""Objective functions for defender/attacker optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from convoy_sim.entities import ShipClass


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective weights for scoring Monte Carlo outcomes."""

    w_total_value: float = 1.0
    w_total_hits: float = 0.0
    class_value_weights: dict[ShipClass, float] | None = None
    escort_loss_discount: float = 1.0
    mode: Literal["defender_minimize", "attacker_maximize"] = "defender_minimize"
    risk_alpha: float | None = None


def score_trial_result(trial: dict[str, Any], obj: ObjectiveSpec) -> float:
    """Return a scalar loss (lower is better) for a single trial."""

    value_by_class = trial.get("value_destroyed_by_class", {})
    class_weights = obj.class_value_weights or {}
    adjusted_value = 0.0
    for ship_class, value in value_by_class.items():
        weight = float(class_weights.get(ship_class, 1.0))
        if ship_class == ShipClass.ESCORT:
            weight *= obj.escort_loss_discount
        adjusted_value += float(value) * weight

    total_value = float(trial.get("total_value_destroyed", 0.0))
    total_hits = float(trial.get("n_hits", 0.0))
    loss = obj.w_total_value * max(total_value, adjusted_value) + obj.w_total_hits * total_hits
    return loss


def defender_loss_from_outcome(trial: dict[str, Any], obj: ObjectiveSpec | None) -> float:
    """Return defender loss (higher is worse); flips sign if obj is attacker-maximized."""

    if obj is None:
        return float(trial.get("total_value_destroyed", 0.0))
    loss = float(score_trial_result(trial, obj))
    if obj.mode == "attacker_maximize":
        return -loss
    return loss


def attacker_utility_from_outcome(trial: dict[str, Any], obj: ObjectiveSpec | None) -> float:
    """Return attacker utility (higher is better) using the same scalar convention."""

    if obj is None:
        return float(trial.get("total_value_destroyed", 0.0))
    score = float(score_trial_result(trial, obj))
    return score


def aggregate_objective(monte_carlo_result: dict[str, Any], obj: ObjectiveSpec) -> float:
    """Aggregate Monte Carlo outputs into a single scalar objective."""

    expected_value = float(monte_carlo_result.get("expected_value_destroyed", 0.0))
    expected_hits = float(monte_carlo_result.get("expected_hits", 0.0))
    loss = obj.w_total_value * expected_value + obj.w_total_hits * expected_hits
    if obj.risk_alpha is not None:
        label = int(round(obj.risk_alpha * 100))
        cvar_key = f"CVaR_{label}"
        if cvar_key in monte_carlo_result:
            loss += float(monte_carlo_result[cvar_key])
    return loss
