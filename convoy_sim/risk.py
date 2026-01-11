"""Risk metrics for Monte Carlo hit distributions."""

from __future__ import annotations

import numpy as np


def empirical_var(x: np.ndarray, alpha: float) -> float:
    """Return the empirical Value-at-Risk at quantile ``alpha``.

    For hit counts, higher values are worse (attacker-favorable).
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    data = np.asarray(x, dtype=float)
    if data.size == 0:
        raise ValueError("x must be non-empty")
    return float(np.quantile(data, alpha))


def empirical_cvar(x: np.ndarray, alpha: float) -> float:
    """Return the mean of the worst ``alpha``-tail of the distribution."""

    var = empirical_var(x, alpha)
    data = np.asarray(x, dtype=float)
    tail = data[data >= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))
