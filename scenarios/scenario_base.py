"""Scenario definitions for running experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from convoy_sim import Ship, run_monte_carlo_attack
from convoy_sim.noise import NoiseModel


LayoutFn = Callable[..., list[Ship]]
TorpedoSamplerFactory = Callable[[np.random.Generator], list]


@dataclass
class Scenario:
    """Container for experiment configuration and sampler wiring."""

    name: str
    layout_fn: LayoutFn
    layout_kwargs: dict[str, Any]
    torpedo_sampler: Callable[[np.random.Generator], list]
    n_trials: int
    t_max: float
    rng_seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    noise_model: NoiseModel | None = None

    def run(self) -> dict[str, Any]:
        """Execute the configured Monte Carlo experiment."""

        rng = np.random.default_rng(self.rng_seed)
        result = run_monte_carlo_attack(
            layout_fn=self.layout_fn,
            layout_kwargs=self.layout_kwargs,
            torpedo_sampler=self.torpedo_sampler,
            n_trials=self.n_trials,
            t_max=self.t_max,
            rng=rng,
            noise_model=self.noise_model,
        )
        return {
            "scenario": self.name,
            "result": result,
            "metadata": self.metadata,
        }
