"""Standardized trial record helpers for downstream analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


TRIAL_SCHEMA_VERSION = 1


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def make_trial_record(
    *,
    trial_id: int,
    seed: int | None,
    scenario: str | None,
    threat: Any | None,
    defender: dict[str, Any] | None,
    attacker: dict[str, Any] | None,
    layout_metrics: dict[str, Any] | None,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    """Return a standardized, JSON-friendly trial record."""

    record = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "trial": int(trial_id),
        "seed": seed,
        "scenario": scenario,
        "threat": threat,
        "defender": defender,
        "attacker": attacker,
        "layout_metrics": layout_metrics,
        "outcome": outcome,
    }
    return _to_serializable(record)
