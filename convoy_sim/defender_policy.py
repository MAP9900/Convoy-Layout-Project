"""Defender policy modeling for layout selection under threat uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from .entities import Ship, ShipClass, Torpedo
from .feasibility import compute_convoy_reference
from .objectives import ObjectiveSpec, score_trial_result
from .trial_records import make_trial_record
from .simulation import simulate_attack_once_scored


class ThreatType(str, Enum):
    """Enumerated attacker threat families for policy conditioning."""

    ABEAM_FAN = "abeam_fan"
    BOW_ON_FAN = "bow_on_fan"
    STERN_CHASE = "stern_chase"
    PARALLEL_SPREAD = "parallel_spread"


@dataclass(frozen=True)
class ThreatPrior:
    """Prior distribution over threat types."""

    probs: dict[ThreatType, float]

    def normalized(self) -> "ThreatPrior":
        if any(value < 0.0 for value in self.probs.values()):
            raise ValueError("ThreatPrior.probs must be non-negative")
        total = float(sum(self.probs.values()))
        if total <= 0.0:
            raise ValueError("ThreatPrior.probs must sum to a positive value")
        normalized = {k: float(v) / total for k, v in self.probs.items()}
        return ThreatPrior(probs=normalized)

    def sample(self, rng: np.random.Generator) -> ThreatType:
        normalized = self.normalized()
        threats = list(normalized.probs.keys())
        weights = np.array([normalized.probs[t] for t in threats], dtype=float)
        idx = int(rng.choice(len(threats), p=weights))
        return threats[idx]


@dataclass(frozen=True)
class LayoutAction:
    """Defender layout choice with optional operational metadata."""

    name: str
    layout_fn: Callable[..., list[Ship]]
    layout_kwargs: dict[str, Any]
    complexity_cost: float
    footprint_limit: dict[str, float] | None = None


def compute_layout_metrics(ships: list[Ship]) -> dict[str, float]:
    """Compute simple geometric and composition metrics for a layout."""

    reference = compute_convoy_reference(ships)
    escort_count = sum(1 for ship in ships if ship.ship_class == ShipClass.ESCORT)
    total_value = float(sum(ship.value_weight for ship in ships))
    bbox_along = float(reference["bbox_along"])
    bbox_across = float(reference["bbox_across"])
    footprint_area = bbox_along * bbox_across
    return {
        "bbox_along": bbox_along,
        "bbox_across": bbox_across,
        "footprint_area": footprint_area,
        "ship_count": float(len(ships)),
        "escort_count": float(escort_count),
        "total_value": total_value,
    }


def _normalize_action_probs(
    action_probs: dict[str, float],
    actions: list[LayoutAction],
) -> dict[str, float]:
    names = {action.name for action in actions}
    filtered = {name: float(prob) for name, prob in action_probs.items() if name in names}
    if len(filtered) != len(action_probs):
        unknown = set(action_probs) - names
        raise ValueError(f"Unknown action names in policy table: {sorted(unknown)}")
    total = float(sum(filtered.values()))
    if total <= 0.0:
        raise ValueError("Policy action probabilities must sum to a positive value")
    return {name: prob / total for name, prob in filtered.items()}


@dataclass(frozen=True)
class DefenderPolicy:
    """Policy mapping threats to randomized layout choices."""

    actions: list[LayoutAction]
    policy_table: dict[ThreatType, dict[str, float]] = field(default_factory=dict)

    def action_distribution(self, threat: ThreatType) -> dict[str, float]:
        if not self.actions:
            raise ValueError("DefenderPolicy.actions must be non-empty")
        if threat not in self.policy_table:
            uniform = 1.0 / len(self.actions)
            return {action.name: uniform for action in self.actions}
        return _normalize_action_probs(self.policy_table[threat], self.actions)

    def sample_action(self, threat: ThreatType, rng: np.random.Generator) -> LayoutAction:
        distribution = self.action_distribution(threat)
        names = list(distribution.keys())
        weights = np.array([distribution[name] for name in names], dtype=float)
        idx = int(rng.choice(len(names), p=weights))
        chosen = names[idx]
        for action in self.actions:
            if action.name == chosen:
                return action
        raise ValueError(f"Unknown action name resolved: {chosen}")

    def expected_action(self, threat: ThreatType) -> dict[str, float]:
        return self.action_distribution(threat)


def make_uniform_policy(actions: list[LayoutAction], threats: list[ThreatType]) -> DefenderPolicy:
    """Return a policy that is uniform across actions for all threats."""

    if not actions:
        raise ValueError("actions must be non-empty")
    uniform = {action.name: 1.0 / len(actions) for action in actions}
    policy_table = {threat: dict(uniform) for threat in threats}
    return DefenderPolicy(actions=actions, policy_table=policy_table)


def make_deterministic_policy(
    actions: list[LayoutAction],
    threat_action_map: dict[ThreatType, str],
) -> DefenderPolicy:
    """Return a policy that selects a fixed action per threat."""

    policy_table: dict[ThreatType, dict[str, float]] = {}
    for threat, action_name in threat_action_map.items():
        policy_table[threat] = {action_name: 1.0}
    return DefenderPolicy(actions=actions, policy_table=policy_table)


def evaluate_defender_policy(
    prior: ThreatPrior,
    policy: DefenderPolicy,
    attacker_factory: Callable[[ThreatType], Callable[[np.random.Generator], list[Torpedo]]],
    n_trials: int,
    sim_kwargs: dict[str, Any],
    objective: ObjectiveSpec | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Evaluate a defender policy under a threat prior with nested randomness."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    hits = np.zeros(n_trials, dtype=float)
    values = np.zeros(n_trials, dtype=float)
    losses = np.zeros(n_trials, dtype=float)
    action_counts: dict[str, int] = {action.name: 0 for action in policy.actions}
    threat_counts: dict[ThreatType, int] = {threat: 0 for threat in prior.probs}
    action_by_threat: dict[ThreatType, dict[str, int]] = {
        threat: {action.name: 0 for action in policy.actions} for threat in prior.probs
    }
    footprint_values: list[float] = []
    complexity_values: list[float] = []
    trials: list[dict[str, Any]] = []

    for idx in range(n_trials):
        threat = prior.sample(generator)
        threat_counts[threat] = threat_counts.get(threat, 0) + 1
        action = policy.sample_action(threat, generator)
        action_counts[action.name] = action_counts.get(action.name, 0) + 1
        if threat not in action_by_threat:
            action_by_threat[threat] = {a.name: 0 for a in policy.actions}
        action_by_threat[threat][action.name] += 1

        ships = action.layout_fn(**action.layout_kwargs)
        torpedoes = attacker_factory(threat)(generator)
        metrics = compute_layout_metrics(ships)
        footprint_values.append(metrics["footprint_area"])
        complexity_values.append(float(action.complexity_cost))

        scored = simulate_attack_once_scored(
            ships=ships,
            torpedoes=torpedoes,
            t_max=float(sim_kwargs.get("t_max", 0.0)),
            max_hits_per_torpedo=sim_kwargs.get("max_hits_per_torpedo"),
        )
        hits[idx] = scored["n_hits"]
        values[idx] = scored["total_value_destroyed"]
        if objective is not None:
            losses[idx] = score_trial_result(scored, objective)

        outcome = {
            "n_hits": scored["n_hits"],
            "total_value_destroyed": scored["total_value_destroyed"],
        }
        if objective is not None:
            outcome["loss"] = losses[idx]
        trials.append(
            make_trial_record(
                trial_id=idx,
                seed=None,
                scenario=None,
                threat=threat.value,
                defender={"action": action.name, "complexity_cost": action.complexity_cost},
                attacker=None,
                layout_metrics=metrics,
                outcome=outcome,
            )
        )

    expected_hits = float(np.mean(hits))
    expected_value = float(np.mean(values))
    expected_loss = float(np.mean(losses)) if objective is not None else None
    action_freq = {name: count / n_trials for name, count in action_counts.items()}
    action_freq_by_threat: dict[str, dict[str, float]] = {}
    for threat, counts in action_by_threat.items():
        total = sum(counts.values())
        if total <= 0:
            action_freq_by_threat[threat.value] = {name: 0.0 for name in action_counts}
        else:
            action_freq_by_threat[threat.value] = {
                name: count / total for name, count in counts.items()
            }

    summary = {
        "expected_hits": expected_hits,
        "expected_value_destroyed": expected_value,
        "expected_loss": expected_loss,
        "action_frequencies": action_freq,
        "action_frequencies_by_threat": action_freq_by_threat,
        "mean_footprint_area": float(np.mean(footprint_values)) if footprint_values else 0.0,
        "mean_complexity_cost": float(np.mean(complexity_values)) if complexity_values else 0.0,
    }
    return {
        "summary": summary,
        "trials": trials,
    }
