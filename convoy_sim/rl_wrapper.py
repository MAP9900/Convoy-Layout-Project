"""Thin RL compatibility wrappers for discrete strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .defender_policy import ThreatPrior
from .game import AttackerStrategy, DefenderStrategy
from .objectives import ObjectiveSpec, attacker_utility_from_outcome, defender_loss_from_outcome
from .trial_records import _to_serializable
from .feasibility import AttackConstraints, Environment
from .dynamics import ConvoyFormation, ConvoyKinematics


OBS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionSpaceMap:
    """Discrete mapping between strategy names and indices."""

    names: list[str]

    def to_index(self, name: str) -> int:
        if name not in self.names:
            raise ValueError(f"Unknown action name: {name}")
        return int(self.names.index(name))

    def to_name(self, index: int) -> str:
        if index < 0 or index >= len(self.names):
            raise ValueError("Action index out of range")
        return self.names[index]

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names)}


def build_observation(
    *,
    time_step: int,
    threat: Any | None,
    defender_action: str | None,
    attacker_action: str | None,
    layout_metrics: dict[str, Any] | None,
    outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a JSON-friendly observation record."""

    obs = {
        "schema_version": OBS_SCHEMA_VERSION,
        "time_step": int(time_step),
        "threat": threat,
        "defender_action": defender_action,
        "attacker_action": attacker_action,
        "layout_metrics": layout_metrics,
        "outcome": outcome,
    }
    return _to_serializable(obs)


@dataclass
class RLEpisode:
    """Minimal episode wrapper for discrete defender/attacker strategies."""

    defenders: list[DefenderStrategy]
    attackers: list[AttackerStrategy]
    prior: ThreatPrior | None
    env: Environment | None
    constraints: AttackConstraints | None
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None
    sim_params: dict[str, Any]
    objective: ObjectiveSpec | None = None
    max_steps: int = 1
    reward_perspective: Literal["defender", "attacker"] = "defender"
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._rng = self.rng or np.random.default_rng()
        self._step = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        return build_observation(
            time_step=self._step,
            threat=None,
            defender_action=None,
            attacker_action=None,
            layout_metrics=None,
            outcome=None,
        )

    def step(self, defender_action_idx: int, attacker_action_idx: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if defender_action_idx < 0 or defender_action_idx >= len(self.defenders):
            raise ValueError("Invalid defender action index")
        if attacker_action_idx < 0 or attacker_action_idx >= len(self.attackers):
            raise ValueError("Invalid attacker action index")

        threat = self.prior.sample(self._rng).value if self.prior is not None else None
        defender = self.defenders[defender_action_idx]
        attacker = self.attackers[attacker_action_idx]

        ships, layout_metrics = defender.sample_layout(threat, self._rng)
        outcome = attacker.execute(
            ships_t0=ships,
            constraints=self.constraints,
            env=self.env,
            dynamics=self.dynamics,
            sim_params=self.sim_params,
            rng=self._rng,
        )
        loss = defender_loss_from_outcome(outcome, self.objective)
        utility = attacker_utility_from_outcome(outcome, self.objective)
        reward = -loss if self.reward_perspective == "defender" else utility

        obs = build_observation(
            time_step=self._step,
            threat=threat,
            defender_action=defender.name,
            attacker_action=attacker.name,
            layout_metrics=layout_metrics,
            outcome=outcome,
        )
        info = {
            "defender_loss": float(loss),
            "attacker_utility": float(utility),
        }
        self._step += 1
        done = self._step >= self.max_steps
        return obs, float(reward), done, info
