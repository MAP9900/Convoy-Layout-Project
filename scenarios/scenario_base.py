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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scenario description."""

        return {
            "name": self.name,
            "layout_fn": getattr(self.layout_fn, "__name__", str(self.layout_fn)),
            "layout_fn_module": getattr(self.layout_fn, "__module__", ""),
            "layout_kwargs": self.layout_kwargs,
            "torpedo_sampler": getattr(self.torpedo_sampler, "__name__", str(self.torpedo_sampler)),
            "torpedo_sampler_module": getattr(self.torpedo_sampler, "__module__", ""),
            "n_trials": int(self.n_trials),
            "t_max": float(self.t_max),
            "rng_seed": self.rng_seed,
            "metadata": self.metadata,
            "noise_model": None if self.noise_model is None else self.noise_model.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        layout_fn: LayoutFn,
        torpedo_sampler: Callable[[np.random.Generator], list],
    ) -> "Scenario":
        """Reconstruct a Scenario when callables are provided explicitly."""

        noise_payload = payload.get("noise_model")
        noise_model = NoiseModel.from_dict(noise_payload) if noise_payload else None
        return cls(
            name=payload["name"],
            layout_fn=layout_fn,
            layout_kwargs=payload["layout_kwargs"],
            torpedo_sampler=torpedo_sampler,
            n_trials=int(payload["n_trials"]),
            t_max=float(payload["t_max"]),
            rng_seed=payload.get("rng_seed"),
            metadata=payload.get("metadata", {}),
            noise_model=noise_model,
        )
