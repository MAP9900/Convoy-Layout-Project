"""Tests for risk metrics."""

import numpy as np

from convoy_sim.risk import empirical_cvar, empirical_var


def test_empirical_var_cvar_known_array() -> None:
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert empirical_var(data, 0.8) == 3.2
    assert empirical_cvar(data, 0.8) == 4.0
