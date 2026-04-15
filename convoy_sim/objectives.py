"""Objective functions for defender/attacker optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from convoy_sim.entities import ShipClass


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective weights for scoring Monte Carlo outcomes."""

    preset_name: str | None = None
    w_total_value: float = 1.0
    w_total_hits: float = 0.0
    w_unique_ships_hit: float = 0.0
    w_repeat_hits: float = 0.0
    class_value_weights: dict[ShipClass, float] | None = None
    escort_loss_discount: float = 1.0
    mode: Literal["defender_minimize", "attacker_maximize"] = "defender_minimize"
    risk_alpha: float | None = None


OBJECTIVE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced_default": {
        "w_total_value": 1.0,
        "w_total_hits": 0.0,
        "w_unique_ships_hit": 1.0,
        "w_repeat_hits": 0.2,
        "class_value_weights": {
            ShipClass.FREIGHTER: 1.0,
            ShipClass.TANKER: 1.5,
            ShipClass.ESCORT: 0.5,
            ShipClass.DECOY: 0.2,
        },
        "escort_loss_discount": 0.75,
    },
    "protect_hulls": {
        "w_total_value": 1.0,
        "w_total_hits": 0.0,
        "w_unique_ships_hit": 1.5,
        "w_repeat_hits": 0.5,
        "class_value_weights": {
            ShipClass.FREIGHTER: 1.0,
            ShipClass.TANKER: 1.4,
            ShipClass.ESCORT: 0.6,
            ShipClass.DECOY: 0.2,
        },
        "escort_loss_discount": 0.75,
    },
    "protect_value": {
        "w_total_value": 1.5,
        "w_total_hits": 0.0,
        "w_unique_ships_hit": 0.8,
        "w_repeat_hits": 0.2,
        "class_value_weights": {
            ShipClass.FREIGHTER: 1.0,
            ShipClass.TANKER: 2.0,
            ShipClass.ESCORT: 0.4,
            ShipClass.DECOY: 0.2,
        },
        "escort_loss_discount": 0.75,
    },
    "accept_concentration": {
        "w_total_value": 1.0,
        "w_total_hits": 0.0,
        "w_unique_ships_hit": 1.2,
        "w_repeat_hits": 0.1,
        "class_value_weights": {
            ShipClass.FREIGHTER: 1.0,
            ShipClass.TANKER: 1.5,
            ShipClass.ESCORT: 0.5,
            ShipClass.DECOY: 0.2,
        },
        "escort_loss_discount": 0.75,
    },
}


def objective_preset_names() -> tuple[str, ...]:
    """Return the known named objective presets."""

    return tuple(sorted(OBJECTIVE_PRESETS))


def objective_preset_spec(name: str) -> ObjectiveSpec:
    """Return the ObjectiveSpec for a named preset."""

    try:
        raw = OBJECTIVE_PRESETS[str(name)]
    except KeyError as exc:
        valid = ", ".join(objective_preset_names())
        raise ValueError(f"Unknown objective preset '{name}'. Valid presets: {valid}") from exc
    return ObjectiveSpec(
        preset_name=str(name),
        w_total_value=float(raw["w_total_value"]),
        w_total_hits=float(raw.get("w_total_hits", 0.0)),
        w_unique_ships_hit=float(raw.get("w_unique_ships_hit", 0.0)),
        w_repeat_hits=float(raw.get("w_repeat_hits", 0.0)),
        class_value_weights=dict(raw.get("class_value_weights", {})) or None,
        escort_loss_discount=float(raw.get("escort_loss_discount", 1.0)),
        mode=str(raw.get("mode", "defender_minimize")),
        risk_alpha=(None if raw.get("risk_alpha") is None else float(raw["risk_alpha"])),
    )


def objective_from_config(payload: dict[str, Any] | None) -> ObjectiveSpec | None:
    """Parse an objective config block into an ObjectiveSpec."""

    cfg = dict(payload or {})
    if not cfg:
        return None
    preset_name = cfg.get("preset")
    base = objective_preset_spec(str(preset_name)) if preset_name else None
    raw_class_weights = dict(cfg.get("class_value_weights", {}))
    class_weights = dict(base.class_value_weights or {}) if base else {}
    class_weights.update({ShipClass(key): float(value) for key, value in raw_class_weights.items()})
    return ObjectiveSpec(
        preset_name=(str(preset_name) if preset_name else None),
        w_total_value=float(cfg.get("w_total_value", base.w_total_value if base else 1.0)),
        w_total_hits=float(cfg.get("w_total_hits", base.w_total_hits if base else 0.0)),
        w_unique_ships_hit=float(
            cfg.get("w_unique_ships_hit", base.w_unique_ships_hit if base else 0.0)
        ),
        w_repeat_hits=float(cfg.get("w_repeat_hits", base.w_repeat_hits if base else 0.0)),
        class_value_weights=class_weights or None,
        escort_loss_discount=float(
            cfg.get("escort_loss_discount", base.escort_loss_discount if base else 1.0)
        ),
        mode=str(cfg.get("mode", base.mode if base else "defender_minimize")),
        risk_alpha=(
            None
            if cfg.get("risk_alpha", base.risk_alpha if base else None) is None
            else float(cfg.get("risk_alpha", base.risk_alpha if base else None))
        ),
    )


def objective_to_dict(obj: ObjectiveSpec | None) -> dict[str, Any] | None:
    """Serialize an ObjectiveSpec for manifests and resolved config output."""

    if obj is None:
        return None
    return {
        "preset_name": obj.preset_name,
        "w_total_value": float(obj.w_total_value),
        "w_total_hits": float(obj.w_total_hits),
        "w_unique_ships_hit": float(obj.w_unique_ships_hit),
        "w_repeat_hits": float(obj.w_repeat_hits),
        "class_value_weights": (
            {ship_class.value: float(weight) for ship_class, weight in obj.class_value_weights.items()}
            if obj.class_value_weights
            else {}
        ),
        "escort_loss_discount": float(obj.escort_loss_discount),
        "mode": obj.mode,
        "risk_alpha": obj.risk_alpha,
    }


def weighted_value_destroyed_from_trial(trial: dict[str, Any], obj: ObjectiveSpec | None) -> float:
    """Return class-weighted destroyed value for a single trial."""

    if obj is None:
        return float(trial.get("total_value_destroyed", 0.0))
    value_by_class = trial.get("value_destroyed_by_class", {})
    class_weights = obj.class_value_weights or {}
    adjusted_value = 0.0
    for ship_class, value in value_by_class.items():
        ship_class_enum = ship_class if isinstance(ship_class, ShipClass) else ShipClass(ship_class)
        weight = float(class_weights.get(ship_class_enum, 1.0))
        if ship_class_enum == ShipClass.ESCORT:
            weight *= obj.escort_loss_discount
        adjusted_value += float(value) * weight
    total_value = float(trial.get("total_value_destroyed", 0.0))
    return max(total_value, adjusted_value)


def score_trial_result(trial: dict[str, Any], obj: ObjectiveSpec) -> float:
    """Return a scalar loss (lower is better) for a single trial."""

    adjusted_value = weighted_value_destroyed_from_trial(trial, obj)
    total_hits = float(trial.get("n_hits", 0.0))
    unique_ships_hit = float(trial.get("unique_ships_hit", 0.0))
    repeat_hits = float(trial.get("repeat_hits", max(0.0, total_hits - unique_ships_hit)))
    loss = (
        obj.w_total_value * adjusted_value
        + obj.w_total_hits * total_hits
        + obj.w_unique_ships_hit * unique_ships_hit
        + obj.w_repeat_hits * repeat_hits
    )
    return loss


def defender_loss_from_outcome(trial: dict[str, Any], obj: ObjectiveSpec | None) -> float:
    """Return defender loss (higher is worse); flips sign if obj is attacker-maximized."""

    if obj is None:
        return float(trial.get("total_value_destroyed", 0.0))
    loss = float(score_trial_result(trial, obj))
    if obj.mode == "attacker_maximize":
        return -loss
    return loss


def attacker_utility_from_outcome(trial: dict[str, Any], obj: ObjectiveSpec | None) -> float:
    """Return attacker utility (higher is better) using the same scalar convention."""

    if obj is None:
        return float(trial.get("total_value_destroyed", 0.0))
    score = float(score_trial_result(trial, obj))
    return score


def aggregate_objective(monte_carlo_result: dict[str, Any], obj: ObjectiveSpec) -> float:
    """Aggregate Monte Carlo outputs into a single scalar objective."""

    if "expected_loss" in monte_carlo_result:
        loss = float(monte_carlo_result["expected_loss"])
    else:
        expected_value = float(
            monte_carlo_result.get(
                "expected_weighted_value_destroyed",
                monte_carlo_result.get("expected_value_destroyed", 0.0),
            )
        )
        expected_hits = float(monte_carlo_result.get("expected_hits", 0.0))
        expected_unique = float(monte_carlo_result.get("expected_unique_ships_hit", 0.0))
        expected_repeat = float(monte_carlo_result.get("expected_repeat_hits", 0.0))
        loss = (
            obj.w_total_value * expected_value
            + obj.w_total_hits * expected_hits
            + obj.w_unique_ships_hit * expected_unique
            + obj.w_repeat_hits * expected_repeat
        )
    if obj.risk_alpha is not None:
        label = int(round(obj.risk_alpha * 100))
        cvar_key = f"CVaR_{label}_loss"
        if cvar_key in monte_carlo_result:
            loss += float(monte_carlo_result[cvar_key])
        else:
            legacy_cvar_key = f"CVaR_{label}"
            if legacy_cvar_key in monte_carlo_result:
                loss += float(monte_carlo_result[legacy_cvar_key])
    return loss
