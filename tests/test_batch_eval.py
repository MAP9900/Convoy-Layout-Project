"""Tests for batch evaluation utilities."""

import numpy as np

from convoy_sim.batch_eval import run_batch


def test_run_batch_honors_limit_and_seed() -> None:
    def evaluator(rng: np.random.Generator, idx: int):
        return {"idx": idx, "value": float(rng.uniform())}

    records_a = run_batch(evaluator, n_trials=5, rng_seed=123, max_evals=3)
    records_b = run_batch(evaluator, n_trials=5, rng_seed=123, max_evals=3)
    assert len(records_a) == 3
    assert records_a == records_b
