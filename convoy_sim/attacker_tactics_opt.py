"""Coarse grid search utilities for attacker tactics plans."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np

from .attacker_tactics import AttackerPlan, PassSpec, SalvoSpec, execute_attacker_plan
from .dynamics import ConvoyFormation, ConvoyKinematics
from .entities import Ship
from .feasibility import ApproachMode, AttackConstraints, Environment
from .objectives import ObjectiveSpec
from .risk import empirical_cvar, empirical_var


@dataclass(frozen=True)
class PlanTemplate:
    """Compact plan parameterization for coarse grid search."""

    n_passes_options: list[int]
    launch_time_1: float
    u_boat_pos_1: np.ndarray
    bearing_rad_1: float
    approach_mode_1: ApproachMode
    pattern_1: str
    salvo_sizes_1: list[int]
    spread_options_1: list[float]
    asymmetry_options_1: list[float]
    edge_bias_options_1: list[float]
    launch_delay_options_2: list[float]
    u_boat_pos_2: np.ndarray
    bearing_rad_2: float
    approach_mode_2: ApproachMode
    pattern_2: str
    salvo_sizes_2: list[int]
    spread_options_2: list[float]
    asymmetry_options_2: list[float]
    edge_bias_options_2: list[float]
    abort_if_risk_above_options: list[float | None]


def _build_salvo_spec(
    pattern: str,
    size: int,
    spread: float,
    asymmetry: float,
    edge_bias: float,
) -> SalvoSpec:
    if pattern == "fan":
        return SalvoSpec(
            n_torpedoes=size,
            pattern="fan",
            spread_rad=spread,
            asymmetry=asymmetry,
            edge_bias=edge_bias,
        )
    return SalvoSpec(
        n_torpedoes=size,
        pattern="parallel",
        lateral_spacing=spread,
        asymmetry=asymmetry,
        edge_bias=edge_bias,
    )


def instantiate_plans(template: PlanTemplate) -> list[AttackerPlan]:
    """Instantiate all attacker plans from the template grid."""

    plans: list[AttackerPlan] = []
    for n_passes in template.n_passes_options:
        if n_passes not in (1, 2):
            raise ValueError("n_passes_options must contain only 1 or 2")
        pass1_grid = itertools.product(
            template.salvo_sizes_1,
            template.spread_options_1,
            template.asymmetry_options_1,
            template.edge_bias_options_1,
        )
        for salvo_size_1, spread_1, asym_1, edge_1 in pass1_grid:
            pass2_grid = itertools.product(
                template.launch_delay_options_2,
                template.salvo_sizes_2,
                template.spread_options_2,
                template.asymmetry_options_2,
                template.edge_bias_options_2,
            )
            pass1_base = PassSpec(
                launch_time=template.launch_time_1,
                u_boat_pos=template.u_boat_pos_1,
                bearing_rad=template.bearing_rad_1,
                approach_mode=template.approach_mode_1,
                salvo=_build_salvo_spec(
                    template.pattern_1,
                    int(salvo_size_1),
                    float(spread_1),
                    float(asym_1),
                    float(edge_1),
                ),
                abort_if_risk_above=None,
            )
            if n_passes == 1:
                for risk in template.abort_if_risk_above_options:
                    plans.append(
                        AttackerPlan(
                            passes=[PassSpec(
                                **{**pass1_base.__dict__, "abort_if_risk_above": risk}
                            )],
                            name="plan_p1",
                        )
                    )
                continue
            for delay, salvo_size_2, spread_2, asym_2, edge_2 in pass2_grid:
                pass2_base = PassSpec(
                    launch_time=template.launch_time_1 + float(delay),
                    u_boat_pos=template.u_boat_pos_2,
                    bearing_rad=template.bearing_rad_2,
                    approach_mode=template.approach_mode_2,
                    salvo=_build_salvo_spec(
                        template.pattern_2,
                        int(salvo_size_2),
                        float(spread_2),
                        float(asym_2),
                        float(edge_2),
                    ),
                    abort_if_risk_above=None,
                )
                for risk in template.abort_if_risk_above_options:
                    p1 = PassSpec(**{**pass1_base.__dict__, "abort_if_risk_above": risk})
                    p2 = PassSpec(**{**pass2_base.__dict__, "abort_if_risk_above": risk})
                    plans.append(AttackerPlan(passes=[p1, p2], name="plan_p2"))
    return plans


def _plan_to_dict(plan: AttackerPlan) -> dict[str, Any]:
    return {
        "name": plan.name,
        "passes": [
            {
                "launch_time": p.launch_time,
                "u_boat_pos": np.asarray(p.u_boat_pos, dtype=float).tolist(),
                "bearing_rad": p.bearing_rad,
                "approach_mode": p.approach_mode.value,
                "salvo": {
                    "n_torpedoes": p.salvo.n_torpedoes,
                    "pattern": p.salvo.pattern,
                    "spread_rad": p.salvo.spread_rad,
                    "lateral_spacing": p.salvo.lateral_spacing,
                    "asymmetry": p.salvo.asymmetry,
                    "edge_bias": p.salvo.edge_bias,
                },
                "abort_if_risk_above": p.abort_if_risk_above,
            }
            for p in plan.passes
        ],
    }


def evaluate_plan_monte_carlo(
    ships_t0: list[Ship],
    plan: AttackerPlan,
    constraints: AttackConstraints | None,
    env: Environment | None,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    torpedo_params: dict[str, Any],
    n_trials: int,
    t_max_global: float,
    objective: ObjectiveSpec | None,
    rng: np.random.Generator | None,
) -> dict[str, Any]:
    """Evaluate an attacker plan via repeated execution."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    generator = rng or np.random.default_rng()
    hits = np.zeros(n_trials, dtype=float)
    values = np.zeros(n_trials, dtype=float)
    losses = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        result = execute_attacker_plan(
            ships_t0=ships_t0,
            plan=plan,
            constraints=constraints,
            env=env,
            dynamics=dynamics,
            torpedo_params=torpedo_params,
            t_max_global=t_max_global,
            rng=generator,
            objective=objective,
        )
        totals = result["totals"]
        hits[idx] = totals["total_hits"]
        values[idx] = totals["total_value_destroyed"]
        if objective is not None:
            losses[idx] = totals.get("objective_loss", 0.0)
    payload = {
        "expected_hits": float(np.mean(hits)),
        "expected_value_destroyed": float(np.mean(values)),
        "var_hits": float(np.var(hits)),
        "hit_prob_at_least_one": float(np.mean(hits > 0)),
    }
    if objective is not None:
        payload["expected_loss"] = float(np.mean(losses))
    payload["VaR_90"] = empirical_var(values, 0.9)
    payload["CVaR_90"] = empirical_cvar(values, 0.9)
    return payload


def search_attacker_plans(
    ships_t0: list[Ship],
    template: PlanTemplate,
    constraints: AttackConstraints | None,
    env: Environment | None,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    torpedo_params: dict[str, Any],
    n_trials: int,
    t_max_global: float,
    objective: ObjectiveSpec | None,
    rng_seed: int = 0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Generate candidate plans, evaluate each, and return top results."""

    plans = instantiate_plans(template)
    results: list[dict[str, Any]] = []
    for idx, plan in enumerate(plans):
        eval_result = evaluate_plan_monte_carlo(
            ships_t0=ships_t0,
            plan=plan,
            constraints=constraints,
            env=env,
            dynamics=dynamics,
            torpedo_params=torpedo_params,
            n_trials=n_trials,
            t_max_global=t_max_global,
            objective=objective,
            rng=np.random.default_rng(rng_seed + idx),
        )
        utility = eval_result.get("expected_value_destroyed", 0.0)
        if objective is not None and objective.mode == "defender_minimize":
            utility = -float(eval_result.get("expected_loss", 0.0))
        results.append(
            {
                "plan": _plan_to_dict(plan),
                "metrics": eval_result,
                "utility": float(utility),
            }
        )
    results.sort(key=lambda r: r["utility"], reverse=True)
    return results[: max(1, int(top_k))]
