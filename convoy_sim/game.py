"""Game-theoretic utilities for defender/attacker strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from convoy_sim.defender_policy import DefenderPolicy, LayoutAction, ThreatPrior, ThreatType, compute_layout_metrics
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics
from convoy_sim.entities import Ship, Torpedo
from convoy_sim.feasibility import AttackConstraints, Environment
from convoy_sim.objectives import ObjectiveSpec, defender_loss_from_outcome
from convoy_sim.realism import ShipMovementRealismConfig, apply_ship_movement_realism
from convoy_sim.simulation import apply_noise_to_torpedoes, simulate_attack_once_scored


@dataclass(frozen=True)
class DefenderStrategy:
    """Strategy wrapper for defender layout actions or policies."""

    name: str
    kind: Literal["layout_action", "policy"]
    payload: Any

    def sample_layout(
        self,
        threat: ThreatType | None,
        rng: np.random.Generator,
    ) -> tuple[list[Ship], dict[str, Any]]:
        def _build_ships(action: LayoutAction) -> list[Ship]:
            realism_cfg = ShipMovementRealismConfig.from_dict(action.layout_kwargs.get("ship_movement_realism"))
            layout_kwargs = {k: v for k, v in action.layout_kwargs.items() if k != "ship_movement_realism"}
            ships_local = action.layout_fn(**layout_kwargs)
            if not realism_cfg.inactive():
                ships_local = apply_ship_movement_realism(ships_local, rng=rng, cfg=realism_cfg)
            return ships_local

        if self.kind == "layout_action":
            action: LayoutAction = self.payload
            ships = _build_ships(action)
            metrics = compute_layout_metrics(ships)
            metrics["complexity_cost"] = float(action.complexity_cost)
            return ships, metrics
        if self.kind == "policy":
            policy: DefenderPolicy = self.payload
            if threat is None:
                raise ValueError("Threat must be provided for policy strategies")
            action = policy.sample_action(threat, rng)
            ships = _build_ships(action)
            metrics = compute_layout_metrics(ships)
            metrics["complexity_cost"] = float(action.complexity_cost)
            return ships, metrics
        raise ValueError(f"Unknown defender strategy kind: {self.kind}")


@dataclass(frozen=True)
class AttackerStrategy:
    """Strategy wrapper for torpedo samplers."""

    name: str
    kind: Literal["torpedo_sampler"]
    payload: Any

    def execute(
        self,
        ships_t0: list[Ship],
        constraints: AttackConstraints | None,
        env: Environment | None,
        dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
        sim_params: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if self.kind == "torpedo_sampler":
            sampler: Callable[[np.random.Generator], list[Torpedo]] = self.payload
            torpedoes = sampler(rng)
            noise_model = sim_params.get("noise_model")
            if noise_model and not noise_model.is_inactive():
                torpedoes = apply_noise_to_torpedoes(torpedoes, noise_model, rng)
            scored = simulate_attack_once_scored(
                ships=ships_t0,
                torpedoes=torpedoes,
                t_max=float(sim_params.get("t_max", 0.0)),
                max_hits_per_torpedo=sim_params.get("max_hits_per_torpedo"),
            )
            return scored
        raise ValueError(f"Unknown attacker strategy kind: {self.kind}")


def trial_loss_from_outcome(outcome: dict[str, Any], objective: ObjectiveSpec | None) -> float:
    """Return defender loss for a single trial outcome."""

    return defender_loss_from_outcome(outcome, objective)


def estimate_payoff_matrix(
    defenders: list[DefenderStrategy],
    attackers: list[AttackerStrategy],
    prior: ThreatPrior | None,
    env: Environment | None,
    constraints: AttackConstraints | None,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    sim_params: dict[str, Any],
    objective: ObjectiveSpec | None,
    n_trials: int,
    rng: np.random.Generator | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Estimate payoff matrix (mean loss, stderr) via Monte Carlo trials."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    d_count = len(defenders)
    a_count = len(attackers)
    mean_loss = np.zeros((d_count, a_count), dtype=float)
    stderr = np.zeros((d_count, a_count), dtype=float)
    raw_samples: list[list[list[float]]] | None = [] if include_raw else None

    for d_idx, defender in enumerate(defenders):
        row_samples: list[list[float]] | None = [] if include_raw else None
        for a_idx, attacker in enumerate(attackers):
            losses = np.zeros(n_trials, dtype=float)
            for t_idx in range(n_trials):
                threat = prior.sample(generator) if prior is not None else None
                ships, _layout_meta = defender.sample_layout(threat, generator)
                outcome = attacker.execute(
                    ships_t0=ships,
                    constraints=constraints,
                    env=env,
                    dynamics=dynamics,
                    sim_params=sim_params,
                    rng=generator,
                )
                loss = trial_loss_from_outcome(outcome, objective)
                losses[t_idx] = loss
            mean_loss[d_idx, a_idx] = float(np.mean(losses))
            stderr[d_idx, a_idx] = float(np.std(losses, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0
            if include_raw:
                row_samples.append(list(losses))
        if include_raw and raw_samples is not None:
            raw_samples.append(row_samples or [])

    payload = {
        "defender_names": [d.name for d in defenders],
        "attacker_names": [a.name for a in attackers],
        "matrix_mean_loss": mean_loss,
        "matrix_stderr": stderr,
    }
    if include_raw:
        payload["raw_loss_samples"] = raw_samples
    return payload


def defender_best_response(payoff_mean_loss: np.ndarray) -> int:
    """Return defender best response index (minimizes loss)."""

    return int(np.argmin(payoff_mean_loss))


def attacker_best_response(payoff_mean_loss: np.ndarray) -> int:
    """Return attacker best response index (maximizes defender loss)."""

    return int(np.argmax(payoff_mean_loss))


def expected_loss(p: np.ndarray, q: np.ndarray, m: np.ndarray) -> float:
    """Return expected loss for mixed strategies p (defender) and q (attacker)."""

    return float(np.dot(p, m @ q))


def best_response_value_defender(q: np.ndarray, m: np.ndarray) -> float:
    """Return best-response loss for defender against attacker mix."""

    return float(np.min(m @ q))


def best_response_value_attacker(p: np.ndarray, m: np.ndarray) -> float:
    """Return best-response loss for attacker against defender mix."""

    return float(np.max(p @ m))


def exploitability(p: np.ndarray, q: np.ndarray, m: np.ndarray) -> dict[str, float]:
    """Return exploitability metrics for mixed strategies."""

    exp_loss = expected_loss(p, q, m)
    def_br = best_response_value_defender(q, m)
    atk_br = best_response_value_attacker(p, m)
    defender_exploit = exp_loss - def_br
    attacker_exploit = atk_br - exp_loss
    return {
        "defender_exploitability": float(defender_exploit),
        "attacker_exploitability": float(attacker_exploit),
        "total": float(defender_exploit + attacker_exploit),
    }
