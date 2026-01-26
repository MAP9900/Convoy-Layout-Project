"""Reusable batch evaluation utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def run_batch(
    evaluator: Callable[[np.random.Generator, int], dict[str, Any]],
    n_trials: int,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
    max_evals: int | None = None,
) -> list[dict[str, Any]]:
    """Run evaluator across trials and return per-trial records."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if max_evals is not None and max_evals <= 0:
        raise ValueError("max_evals must be positive when provided")
    generator = rng or np.random.default_rng(rng_seed)
    limit = n_trials if max_evals is None else min(n_trials, int(max_evals))
    records: list[dict[str, Any]] = []
    for idx in range(limit):
        records.append(evaluator(generator, idx))
    return records
