"""Tests for approximate Nash solvers."""

import numpy as np

from convoy_sim.nash import fictitious_play


def test_fictitious_play_matching_pennies() -> None:
    m = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = fictitious_play(m, n_iters=500, rng_seed=0)
    p = result["p"]
    q = result["q"]
    assert np.allclose(p.sum(), 1.0)
    assert np.allclose(q.sum(), 1.0)
    assert abs(p[0] - 0.5) < 0.2
    assert abs(q[0] - 0.5) < 0.2
    history = result["history"]
    if len(history) >= 2:
        assert history[-1]["total"] <= history[0]["total"] + 0.5
