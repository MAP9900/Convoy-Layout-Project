"""Attacker tactics layer for multi-pass plans and shaped salvos."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from .dynamics import ConvoyFormation, ConvoyKinematics, ship_positions_at, validate_dt
from .entities import Ship, Torpedo
from .feasibility import (
    ApproachMode,
    AttackConstraints,
    AttackProposal,
    Environment,
    detection_risk_score,
    is_attack_feasible,
)
from .geometry import Vec2, as_vec, distance
from .noise import NoiseModel
from .objectives import ObjectiveSpec, score_trial_result
from .simulation import apply_noise_to_torpedoes, simulate_attack_once_scored


class TacticAction(str, Enum):
    """High-level tactical action for a pass."""

    DELAY = "delay"
    ABORT = "abort"
    COMMIT = "commit"


@dataclass(frozen=True)
class SalvoSpec:
    """Specification for a torpedo salvo pattern."""

    n_torpedoes: int
    pattern: Literal["fan", "parallel"]
    spread_rad: float | None = None
    lateral_spacing: float | None = None
    asymmetry: float = 0.0
    edge_bias: float = 0.0


@dataclass(frozen=True)
class PassSpec:
    """Single attack pass definition."""

    launch_time: float
    u_boat_pos: Vec2
    bearing_rad: float
    approach_mode: ApproachMode
    salvo: SalvoSpec
    allow_abort: bool = True
    abort_if_risk_above: float | None = None
    abort_if_infeasible: bool = True


@dataclass(frozen=True)
class AttackerPlan:
    """Multi-pass attacker plan with optional delay/abort logic."""

    passes: list[PassSpec]
    name: str = ""

    def max_time(self, run_time_buffer: float = 0.0) -> float:
        """Return max launch time plus optional run-time buffer."""

        if not self.passes:
            return 0.0
        latest = max(pass_spec.launch_time for pass_spec in self.passes)
        return float(latest) + float(run_time_buffer)


def fan_headings_from_salvo(
    base_bearing_rad: float,
    salvo: SalvoSpec,
) -> list[float]:
    """Return fan headings with asymmetry and edge bias applied.

    asymmetry shifts the centerline by +/- spread/2. edge_bias in [0,1] pulls
    angles toward the outer edges using a power transform on normalized indices.
    """

    if salvo.n_torpedoes <= 0:
        return []
    spread = float(salvo.spread_rad or 0.0)
    asym = float(np.clip(salvo.asymmetry, -1.0, 1.0))
    bias = float(np.clip(salvo.edge_bias, 0.0, 1.0))
    if salvo.n_torpedoes == 1 or spread == 0.0:
        return [float(base_bearing_rad) + asym * (spread * 0.5)]
    indices = np.linspace(-1.0, 1.0, salvo.n_torpedoes)
    exponent = 1.0 + 4.0 * bias
    shaped = np.sign(indices) * (np.abs(indices) ** exponent)
    center_shift = asym * (spread * 0.5)
    return [
        float(base_bearing_rad) + center_shift + (spread * 0.5) * float(x)
        for x in shaped
    ]


def parallel_offsets_from_salvo(salvo: SalvoSpec) -> list[float]:
    """Return lateral offsets for a parallel spread with shaping applied."""

    if salvo.n_torpedoes <= 0:
        return []
    spacing = float(salvo.lateral_spacing or 0.0)
    if salvo.n_torpedoes == 1 or spacing == 0.0:
        return [0.0]
    asym = float(np.clip(salvo.asymmetry, -1.0, 1.0))
    bias = float(np.clip(salvo.edge_bias, 0.0, 1.0))
    indices = np.linspace(-1.0, 1.0, salvo.n_torpedoes)
    exponent = 1.0 + 4.0 * bias
    shaped = np.sign(indices) * (np.abs(indices) ** exponent)
    max_offset = spacing * (salvo.n_torpedoes - 1) / 2.0
    center_shift = asym * max_offset
    return [center_shift + max_offset * float(x) for x in shaped]


def _build_torpedoes_for_pass(
    pass_spec: PassSpec,
    speed: float,
    max_run_time: float,
) -> list[Torpedo]:
    if pass_spec.salvo.n_torpedoes <= 0:
        return []
    if pass_spec.salvo.pattern == "fan":
        headings = fan_headings_from_salvo(pass_spec.bearing_rad, pass_spec.salvo)
        return [
            Torpedo(
                id=f"F{i+1:02d}",
                launch_position=as_vec(*pass_spec.u_boat_pos),
                speed=speed,
                heading_rad=heading,
                max_run_time=max_run_time,
                launch_delay=pass_spec.launch_time,
            )
            for i, heading in enumerate(headings)
        ]
    offsets = parallel_offsets_from_salvo(pass_spec.salvo)
    perp = as_vec(-float(np.sin(pass_spec.bearing_rad)), float(np.cos(pass_spec.bearing_rad)))
    torpedoes = []
    for i, offset in enumerate(offsets):
        launch_pos = as_vec(*pass_spec.u_boat_pos) + perp * float(offset)
        torpedoes.append(
            Torpedo(
                id=f"P{i+1:02d}",
                launch_position=launch_pos,
                speed=speed,
                heading_rad=pass_spec.bearing_rad,
                max_run_time=max_run_time,
                launch_delay=pass_spec.launch_time,
            )
        )
    return torpedoes


def _stationary_ships(ships: list[Ship]) -> list[Ship]:
    static = []
    for ship in ships:
        static.append(
            Ship(
                id=ship.id,
                position=ship.position.copy(),
                speed=0.0,
                heading_rad=ship.heading_rad,
                length=ship.length,
                beam=ship.beam,
                ship_class=ship.ship_class,
                value_weight=ship.value_weight,
                hit_radius=ship.hit_radius,
            )
        )
    return static


def _simulate_pass_dynamic(
    torpedoes: list[Torpedo],
    formation: ConvoyFormation,
    kin: ConvoyKinematics,
    t_max_global: float,
    dt: float,
    safety_margin: float,
    max_hits_per_torpedo: int | None,
) -> dict[str, Any]:
    window_start = min(torpedoes[0].launch_delay, t_max_global) if torpedoes else 0.0
    window_end = min(
        t_max_global,
        max(t.launch_delay + t.max_run_time for t in torpedoes) if torpedoes else 0.0,
    )
    hit_pairs: set[tuple[int, int]] = set()
    hit_ships: set[str] = set()
    hits = 0
    time = float(window_start)
    while time <= window_end + 1e-9:
        ship_positions = ship_positions_at(time, formation, kin, dt=dt)
        for torp_idx, torpedo in enumerate(torpedoes):
            if torpedo.is_dud:
                continue
            if time < torpedo.launch_delay:
                continue
            torp_pos = torpedo.position_at(time)
            if max_hits_per_torpedo == 1 and any(pair[0] == torp_idx for pair in hit_pairs):
                continue
            for ship_idx, ship in enumerate(formation.ships0):
                key = (torp_idx, ship_idx)
                if key in hit_pairs:
                    continue
                radius = ship.effective_hit_radius() + float(safety_margin)
                if distance(ship_positions[ship_idx], torp_pos) <= radius:
                    hit_pairs.add(key)
                    hits += 1
                    hit_ships.add(ship.id)
                    if max_hits_per_torpedo == 1:
                        break
        time += dt
    value_destroyed = float(sum(ship.value_weight for ship in formation.ships0 if ship.id in hit_ships))
    return {
        "n_hits": hits,
        "hit_ship_ids": list(hit_ships),
        "total_value_destroyed": value_destroyed,
    }


def execute_attacker_plan(
    ships_t0: list[Ship],
    plan: AttackerPlan,
    constraints: AttackConstraints | None,
    env: Environment | None,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    torpedo_params: dict[str, Any],
    t_max_global: float,
    rng: np.random.Generator | None = None,
    objective: ObjectiveSpec | None = None,
    noise_model: NoiseModel | None = None,
) -> dict[str, Any]:
    """Execute an attacker plan and return per-pass outcomes and totals."""

    if not plan.passes:
        return {
            "per_pass": [],
            "totals": {"total_hits": 0, "total_value_destroyed": 0.0, "unique_ships_hit": []},
        }
    if "speed" not in torpedo_params or "max_run_time" not in torpedo_params:
        raise ValueError("torpedo_params must include 'speed' and 'max_run_time'")
    speed = float(torpedo_params.get("speed"))
    max_run_time = float(torpedo_params.get("max_run_time"))
    dt = validate_dt(torpedo_params.get("dt", 1.0))
    safety_margin = float(torpedo_params.get("safety_margin", 0.0))
    max_hits_per_torpedo = torpedo_params.get("max_hits_per_torpedo")
    if dt <= 0.0:
        raise ValueError("torpedo_params['dt'] must be > 0")

    generator = rng or np.random.default_rng()
    per_pass: list[dict[str, Any]] = []
    unique_ship_ids: set[str] = set()
    total_hits = 0
    total_value = 0.0

    passes_sorted = sorted(plan.passes, key=lambda p: p.launch_time)
    for pass_spec in passes_sorted:
        proposal = AttackProposal(
            u_boat_pos=as_vec(*pass_spec.u_boat_pos),
            target_point=as_vec(*pass_spec.u_boat_pos),
            bearing_rad=pass_spec.bearing_rad,
            approach_mode=pass_spec.approach_mode,
            salvo_size=pass_spec.salvo.n_torpedoes,
            launch_time=pass_spec.launch_time,
            metadata={"plan": plan.name},
        )

        feasible = True
        feasibility_details: dict[str, Any] = {}
        if constraints is not None:
            if dynamics is not None:
                formation, kin = dynamics
                feasible, feasibility_details = is_attack_feasible(
                    ships_t0,
                    proposal,
                    constraints,
                    env=env,
                    formation=formation,
                    kin=kin,
                    dt=dt,
                )
            else:
                feasible, feasibility_details = is_attack_feasible(
                    ships_t0,
                    proposal,
                    constraints,
                    env=env,
                )

        risk_score = None
        if env is not None:
            risk_score = detection_risk_score(proposal, constraints.escort_zones if constraints else [], env)

        status = "committed"
        if not feasible and pass_spec.abort_if_infeasible and pass_spec.allow_abort:
            status = "aborted_infeasible"
        if (
            status == "committed"
            and pass_spec.abort_if_risk_above is not None
            and risk_score is not None
            and risk_score > pass_spec.abort_if_risk_above
            and pass_spec.allow_abort
        ):
            status = "aborted_risk"

        pass_record = {
            "launch_time": pass_spec.launch_time,
            "status": status,
            "feasible": feasible,
            "risk_score": risk_score,
            "n_torpedoes_fired": 0,
            "n_hits": 0,
            "total_value_destroyed": 0.0,
            "hit_ship_ids": [],
            "feasibility": feasibility_details,
        }

        if status != "committed":
            per_pass.append(pass_record)
            continue

        torpedoes = _build_torpedoes_for_pass(pass_spec, speed, max_run_time)
        if noise_model and not noise_model.is_inactive():
            torpedoes = apply_noise_to_torpedoes(torpedoes, noise_model, generator)
        pass_record["n_torpedoes_fired"] = len(torpedoes)

        if not torpedoes:
            scored = {"n_hits": 0, "total_value_destroyed": 0.0, "hit_ship_ids": []}
        elif dynamics is not None:
            formation, kin = dynamics
            scored = _simulate_pass_dynamic(
                torpedoes,
                formation,
                kin,
                t_max_global=t_max_global,
                dt=dt,
                safety_margin=safety_margin,
                max_hits_per_torpedo=max_hits_per_torpedo,
            )
        else:
            static_ships = _stationary_ships(ships_t0)
            scored = simulate_attack_once_scored(
                ships=static_ships,
                torpedoes=torpedoes,
                t_max=t_max_global,
                max_hits_per_torpedo=max_hits_per_torpedo,
            )

        pass_record["n_hits"] = scored["n_hits"]
        pass_record["total_value_destroyed"] = scored["total_value_destroyed"]
        pass_record["hit_ship_ids"] = scored.get("hit_ship_ids", [])
        per_pass.append(pass_record)

        total_hits += int(scored["n_hits"])
        total_value += float(scored["total_value_destroyed"])
        unique_ship_ids.update(pass_record["hit_ship_ids"])

    totals = {
        "total_hits": total_hits,
        "total_value_destroyed": total_value,
        "unique_ships_hit": sorted(unique_ship_ids),
    }
    if objective is not None:
        totals["objective_loss"] = score_trial_result(
            {
                "n_hits": total_hits,
                "total_value_destroyed": total_value,
                "value_destroyed_by_class": {},
            },
            objective,
        )
    return {"per_pass": per_pass, "totals": totals}
