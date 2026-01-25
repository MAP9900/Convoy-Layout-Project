"""Approximate Nash solvers for finite defender/attacker games."""

from __future__ import annotations

from typing import Any

import numpy as np

from .game import exploitability


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


def fictitious_play(
    m: np.ndarray,
    n_iters: int,
    rng_seed: int = 0,
    step: float | None = None,
    history_stride: int | None = None,
) -> dict[str, Any]:
    """Run fictitious play on loss matrix m (defender minimizes, attacker maximizes)."""

    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    m = np.asarray(m, dtype=float)
    if m.ndim != 2:
        raise ValueError("m must be a 2D array")
    d_count, a_count = m.shape
    rng = np.random.default_rng(rng_seed)

    p = np.ones(d_count, dtype=float) / d_count
    q = np.ones(a_count, dtype=float) / a_count
    counts_p = np.zeros(d_count, dtype=float)
    counts_q = np.zeros(a_count, dtype=float)

    stride = history_stride or max(1, n_iters // 50)
    history: list[dict[str, float]] = []

    for t in range(1, n_iters + 1):
        attacker_br = int(np.argmax(p @ m))
        defender_br = int(np.argmin(m @ q))
        if step is None:
            counts_p[defender_br] += 1.0
            counts_q[attacker_br] += 1.0
            p = counts_p / max(1.0, float(np.sum(counts_p)))
            q = counts_q / max(1.0, float(np.sum(counts_q)))
        else:
            alpha = float(step)
            if not 0.0 < alpha <= 1.0:
                raise ValueError("step must be in (0,1]")
            p = (1.0 - alpha) * p + alpha * _one_hot(defender_br, d_count)
            q = (1.0 - alpha) * q + alpha * _one_hot(attacker_br, a_count)

        if t % stride == 0 or t == n_iters:
            exp = exploitability(p, q, m)
            history.append({"iter": float(t), "total": exp["total"]})

    final_exp = exploitability(p, q, m)
    return {"p": p, "q": q, "history": history, "exploitability": final_exp}


def replicator_dynamics(
    m: np.ndarray,
    n_steps: int,
    eta: float = 0.5,
) -> dict[str, Any]:
    """Run simple replicator dynamics on the loss matrix m."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    m = np.asarray(m, dtype=float)
    if m.ndim != 2:
        raise ValueError("m must be a 2D array")
    d_count, a_count = m.shape

    p = np.ones(d_count, dtype=float) / d_count
    q = np.ones(a_count, dtype=float) / a_count
    history: list[dict[str, float]] = []

    for t in range(1, n_steps + 1):
        loss_by_def = m @ q
        avg_loss = float(np.dot(p, loss_by_def))
        p = p * np.exp(-eta * (loss_by_def - avg_loss))
        p_sum = float(np.sum(p))
        if p_sum <= 0.0:
            p = np.ones(d_count, dtype=float) / d_count
        else:
            p = p / p_sum

        payoff_by_att = p @ m
        avg_payoff = float(np.dot(q, payoff_by_att))
        q = q * np.exp(eta * (payoff_by_att - avg_payoff))
        q_sum = float(np.sum(q))
        if q_sum <= 0.0:
            q = np.ones(a_count, dtype=float) / a_count
        else:
            q = q / q_sum

        if t % max(1, n_steps // 50) == 0 or t == n_steps:
            exp = exploitability(p, q, m)
            history.append({"iter": float(t), "total": exp["total"]})

    final_exp = exploitability(p, q, m)
    return {"p": p, "q": q, "history": history, "exploitability": final_exp}
