"""Gym-style environment stub for future RL integration.

This module defines a lightweight, dependency-free interface that mirrors a
subset of the OpenAI Gym API. The goal is to expose convoy/attacker parameters
as a vector observation and allow actions to propose parameter deltas. Future
phases can plug in full simulators, learned policies, or third-party RL
libraries without changing the public surface here.

Intended usage (later phases):
- Observation: concatenated vector of layout parameters, attack parameters, and
  noise settings, normalized by user-defined scales.
- Action: delta updates to either defender or attacker parameters.
- Reward: for defender, negative expected hits; for attacker, positive expected
  hits (based on a Monte Carlo estimate).
- Transition: apply actions to parameters, run a Monte Carlo estimate, return
  new observation and reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class EnvConfig:
    """Minimal configuration for the RL environment stub."""

    role: str = "defender"  # "defender" or "attacker"
    max_steps: int = 20
    seed: int | None = None


class ConvoyRLEnv:
    """Lightweight Gym-style environment stub (no external dependencies)."""

    def __init__(self, config: EnvConfig, initial_state: Dict[str, float]) -> None:
        self.config = config
        self.initial_state = dict(initial_state)
        self.state = dict(initial_state)
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Reset the environment state and return the initial observation."""

        self.state = dict(self.initial_state)
        self.step_count = 0
        return self._observe()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action and return (obs, reward, done, info).

        This stub currently applies action deltas directly to the parameter
        vector and uses a placeholder reward. A future implementation should
        run a Monte Carlo simulation to compute expected hits.
        """

        self._apply_action(action)
        reward = self._compute_reward_placeholder()
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        info = {"step": self.step_count}
        return self._observe(), reward, done, info

    def _observe(self) -> np.ndarray:
        """Return the current state as a flat vector."""

        return np.array(list(self.state.values()), dtype=float)

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action deltas to the state vector."""

        keys = list(self.state.keys())
        if action.shape[0] != len(keys):
            raise ValueError("Action dimension does not match state dimension")
        for key, delta in zip(keys, action):
            self.state[key] = float(self.state[key] + delta)

    def _compute_reward_placeholder(self) -> float:
        """Placeholder reward until full simulation integration is added."""

        expected_hits_estimate = float(self.rng.uniform(0.0, 1.0))
        if self.config.role == "defender":
            return -expected_hits_estimate
        return expected_hits_estimate
