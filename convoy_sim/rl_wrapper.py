"""Thin RL compatibility wrappers for discrete strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from convoy_sim.attack_profiles import AttackProfile, AttackProfileLibrary
from convoy_sim.defender_policy import ThreatPrior
from convoy_sim.game import AttackerStrategy, DefenderStrategy
from convoy_sim.objectives import ObjectiveSpec, attacker_utility_from_outcome, defender_loss_from_outcome
from convoy_sim.rl_layout_builder import RLLayoutBuilderConfig, RLLayoutBuilderState
from convoy_sim.trial_records import _to_serializable
from convoy_sim.feasibility import AttackConstraints, Environment
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics
from convoy_sim.simulation import apply_noise_to_torpedoes, simulate_attack_once_scored


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
    attack_profile_id: str | None = None,
    attack_profile: dict[str, Any] | None = None,
    builder_mode: bool = False,
    builder_state: dict[str, Any] | None = None,
    valid_defender_actions: list[str] | None = None,
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
        "attack_profile_id": attack_profile_id,
        "attack_profile": attack_profile,
        "builder_mode": bool(builder_mode),
        "builder_state": builder_state,
        "valid_defender_actions": valid_defender_actions,
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
    attack_profile_library: AttackProfileLibrary | None = None
    use_sampled_attack_profile_for_torpedo_sampler: bool = True
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._rng = self.rng or np.random.default_rng()
        self._step = 0
        self._sampled_attack_profile: AttackProfile | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._sampled_attack_profile = (
            self.attack_profile_library.sample_profile(self._rng) if self.attack_profile_library is not None else None
        )
        return build_observation(
            time_step=self._step,
            threat=None,
            defender_action=None,
            attacker_action=None,
            layout_metrics=None,
            outcome=None,
            attack_profile_id=(
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            attack_profile=(
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
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
        outcome = self._execute_attacker(attacker=attacker, ships=ships)
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
            attack_profile_id=(
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            attack_profile=(
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
        )
        info = {
            "defender_loss": float(loss),
            "attacker_utility": float(utility),
            "attack_profile_id": (
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            "attack_profile": (
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
        }
        self._step += 1
        done = self._step >= self.max_steps
        return obs, float(reward), done, info

    def _execute_attacker(self, *, attacker: AttackerStrategy, ships: list[Any]) -> dict[str, Any]:
        if (
            self.use_sampled_attack_profile_for_torpedo_sampler
            and self._sampled_attack_profile is not None
            and attacker.kind == "torpedo_sampler"
        ):
            torpedoes = self._sampled_attack_profile.build_torpedoes(
                self._rng,
                constraints=self.constraints,
                env=self.env,
                ships=ships,
            )
            noise_model = self.sim_params.get("noise_model")
            if noise_model and not noise_model.is_inactive():
                torpedoes = apply_noise_to_torpedoes(torpedoes, noise_model, self._rng)
            return simulate_attack_once_scored(
                ships=ships,
                torpedoes=torpedoes,
                t_max=float(self.sim_params.get("t_max", 0.0)),
                max_hits_per_torpedo=self.sim_params.get("max_hits_per_torpedo"),
            )

        return attacker.execute(
            ships_t0=ships,
            constraints=self.constraints,
            env=self.env,
            dynamics=self.dynamics,
            sim_params=self.sim_params,
            rng=self._rng,
        )


@dataclass
class RLLayoutBuilderEpisode:
    """Bounded multi-step layout-construction episode for RL experiments."""

    builder_config: RLLayoutBuilderConfig
    attackers: list[AttackerStrategy]
    prior: ThreatPrior | None
    env: Environment | None
    constraints: AttackConstraints | None
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None
    sim_params: dict[str, Any]
    objective: ObjectiveSpec | None = None
    reward_perspective: Literal["defender", "attacker"] = "defender"
    attack_profile_library: AttackProfileLibrary | None = None
    use_sampled_attack_profile_for_torpedo_sampler: bool = True
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if not self.builder_config.enabled:
            raise ValueError("RLLayoutBuilderEpisode requires builder_config.enabled=True")
        self._rng = self.rng or np.random.default_rng()
        self._step = 0
        self._sampled_attack_profile: AttackProfile | None = None
        self._sampled_threat: str | None = None
        self._builder_state = RLLayoutBuilderState()
        self._action_space = ActionSpaceMap(names=self.builder_config.action_space_names())

    @property
    def action_space(self) -> ActionSpaceMap:
        return self._action_space

    def valid_defender_action_names(self) -> list[str]:
        return self.builder_config.valid_action_names(self._builder_state)

    def valid_defender_action_indices(self) -> list[int]:
        return [self._action_space.to_index(name) for name in self.valid_defender_action_names()]

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._builder_state = RLLayoutBuilderState()
        self._sampled_threat = self.prior.sample(self._rng).value if self.prior is not None else None
        self._sampled_attack_profile = (
            self.attack_profile_library.sample_profile(self._rng) if self.attack_profile_library is not None else None
        )
        return build_observation(
            time_step=self._step,
            threat=None,
            defender_action=None,
            attacker_action=None,
            layout_metrics=None,
            outcome=None,
            attack_profile_id=(
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            attack_profile=(
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
            builder_mode=True,
            builder_state=self._builder_state.to_dict(),
            valid_defender_actions=self.valid_defender_action_names(),
        )

    def step(self, defender_action_idx: int, attacker_action_idx: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if attacker_action_idx < 0 or attacker_action_idx >= len(self.attackers):
            raise ValueError("Invalid attacker action index")
        action_name = self._action_space.to_name(defender_action_idx)
        if action_name not in self.valid_defender_action_names():
            raise ValueError(f"Invalid builder action for current step: {action_name}")

        attacker = self.attackers[attacker_action_idx]
        threat = self._sampled_threat
        self._builder_state = self.builder_config.apply_action(self._builder_state, action_name)
        self._step += 1

        if not self.builder_config.is_complete(self._builder_state):
            obs = build_observation(
                time_step=self._step,
                threat=threat,
                defender_action=action_name,
                attacker_action=attacker.name,
                layout_metrics=None,
                outcome=None,
                attack_profile_id=(
                    self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
                ),
                attack_profile=(
                    self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
                ),
                builder_mode=True,
                builder_state=self._builder_state.to_dict(),
                valid_defender_actions=self.valid_defender_action_names(),
            )
            info = {
                "attack_profile_id": (
                    self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
                ),
                "attack_profile": (
                    self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
                ),
                "builder_state": self._builder_state.to_dict(),
                "valid_defender_actions": self.valid_defender_action_names(),
            }
            return obs, 0.0, False, info

        layout_action = self.builder_config.materialize_layout_action(self._builder_state)
        ships = layout_action.layout_fn(**layout_action.layout_kwargs)
        from convoy_sim.defender_policy import compute_layout_metrics

        layout_metrics = compute_layout_metrics(ships)
        outcome = self._execute_attacker(attacker=attacker, ships=ships)
        loss = defender_loss_from_outcome(outcome, self.objective)
        utility = attacker_utility_from_outcome(outcome, self.objective)
        reward = -loss if self.reward_perspective == "defender" else utility

        obs = build_observation(
            time_step=self._step,
            threat=threat,
            defender_action=layout_action.name,
            attacker_action=attacker.name,
            layout_metrics=layout_metrics,
            outcome=outcome,
            attack_profile_id=(
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            attack_profile=(
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
            builder_mode=True,
            builder_state=self._builder_state.to_dict(),
            valid_defender_actions=[],
        )
        info = {
            "defender_loss": float(loss),
            "attacker_utility": float(utility),
            "attack_profile_id": (
                self._sampled_attack_profile.profile_id if self._sampled_attack_profile is not None else None
            ),
            "attack_profile": (
                self._sampled_attack_profile.to_dict() if self._sampled_attack_profile is not None else None
            ),
            "builder_state": self._builder_state.to_dict(),
            "materialized_action": layout_action.to_dict(),
        }
        return obs, float(reward), True, info

    def _execute_attacker(self, *, attacker: AttackerStrategy, ships: list[Any]) -> dict[str, Any]:
        if (
            self.use_sampled_attack_profile_for_torpedo_sampler
            and self._sampled_attack_profile is not None
            and attacker.kind == "torpedo_sampler"
        ):
            torpedoes = self._sampled_attack_profile.build_torpedoes(
                self._rng,
                constraints=self.constraints,
                env=self.env,
                ships=ships,
            )
            noise_model = self.sim_params.get("noise_model")
            if noise_model and not noise_model.is_inactive():
                torpedoes = apply_noise_to_torpedoes(torpedoes, noise_model, self._rng)
            return simulate_attack_once_scored(
                ships=ships,
                torpedoes=torpedoes,
                t_max=float(self.sim_params.get("t_max", 0.0)),
                max_hits_per_torpedo=self.sim_params.get("max_hits_per_torpedo"),
            )

        return attacker.execute(
            ships_t0=ships,
            constraints=self.constraints,
            env=self.env,
            dynamics=self.dynamics,
            sim_params=self.sim_params,
            rng=self._rng,
        )
