"""Belief-limited candidate observation and selection helpers.

This module is the first POMDP bridge for VAE-derived attack candidates. It
does not train a policy and it does not generate new attack profiles. It builds
noisy attacker-facing observations for existing candidates, then ranks them with
a simple heuristic that intentionally avoids simulation outcome fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship
from convoy_sim.feasibility import Environment
from convoy_sim.realism import AttackerObservationConfig, build_attacker_observation


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass(frozen=True)
class CandidateSelectorConfig:
    """Heuristic belief-limited selector weights and shape parameters."""

    ideal_range_m: float = 2400.0
    range_width_m: float = 2300.0
    ideal_spread_rad: float = float(np.deg2rad(5.0))
    spread_width_rad: float = float(np.deg2rad(5.0))
    range_weight: float = 0.30
    bearing_alignment_weight: float = 0.24
    aspect_weight: float = 0.18
    formation_span_weight: float = 0.10
    contact_density_weight: float = 0.07
    spread_weight: float = 0.08
    contact_weight: float = 0.05
    inside_bonus_weight: float = 0.03
    uncertainty_penalty_weight: float = 0.35
    inside_uncertainty_penalty_weight: float = 0.35
    ideal_contact_density_per_km2: float = 8.0
    broadside_span_m: float = 4200.0


def _profile_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    if "profile" not in record:
        raise ValueError("candidate record is missing required key: profile")
    return dict(record["profile"])


def _candidate_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    profile = _profile_dict(record)
    return str(profile.get("profile_id", "")), str(profile.get("name", ""))


def _intent_value(record: Mapping[str, Any], key: str, default: Any = "") -> Any:
    intent = dict(record.get("intent", {}))
    audit = dict(record.get("audit", {}))
    return intent.get(key, audit.get(key, default))


def build_candidate_observation_row(
    record: Mapping[str, Any],
    *,
    ships: Sequence[Ship],
    rng: np.random.Generator,
    env: Environment | None = None,
    observation_cfg: AttackerObservationConfig | None = None,
    selector_cfg: CandidateSelectorConfig | None = None,
) -> dict[str, Any]:
    """Build one noisy observation row and heuristic score for a candidate.

    The score uses profile parameters, candidate intent metadata, and noisy
    attacker observation only. It deliberately does not use dynamic outcome
    labels, hit counts, value-loss metrics, or full-state evaluation scores.
    """

    profile = AttackProfile.from_dict(_profile_dict(record))
    resolved_obs_cfg = observation_cfg or AttackerObservationConfig()
    resolved_selector_cfg = selector_cfg or CandidateSelectorConfig()
    env = env or Environment(time_of_day="night", visibility_m=3500.0, sea_state=4)
    observation = build_attacker_observation(
        ships=list(ships),
        u_boat_pos=np.asarray(profile.u_pos, dtype=float),
        env=env,
        rng=rng,
        cfg=resolved_obs_cfg,
    )

    estimated_range_m = float(observation["estimated_range_m"])
    estimated_bearing_rad = float(observation["estimated_bearing_rad"])
    estimated_heading_rad = float(observation["estimated_convoy_heading_rad"])
    estimated_speed_mps = float(observation["estimated_convoy_speed_mps"])
    estimated_contact_count = int(observation["estimated_contact_count"])
    estimated_formation_width_m = float(observation["estimated_formation_width_m"])
    estimated_formation_depth_m = float(observation["estimated_formation_depth_m"])
    estimated_contact_density_per_km2 = float(observation["estimated_contact_density_per_km2"])
    estimated_aspect_rad = _wrap_angle_rad(estimated_heading_rad - (estimated_bearing_rad + np.pi))
    bearing_alignment_error_rad = abs(_wrap_angle_rad(float(profile.base_bearing_rad) - estimated_bearing_rad))
    attack_heading_delta_rad = abs(_wrap_angle_rad(estimated_bearing_rad - estimated_heading_rad))
    projected_target_span_m = float(
        estimated_formation_depth_m * abs(np.sin(attack_heading_delta_rad))
        + estimated_formation_width_m * abs(np.cos(attack_heading_delta_rad))
    )

    range_score = float(
        np.exp(-((estimated_range_m - resolved_selector_cfg.ideal_range_m) / resolved_selector_cfg.range_width_m) ** 2)
    )
    bearing_alignment_score = _clip01((1.0 + float(np.cos(bearing_alignment_error_rad))) / 2.0)
    aspect_score = _clip01(abs(float(np.sin(estimated_aspect_rad))))
    formation_span_score = _clip01(projected_target_span_m / max(float(resolved_selector_cfg.broadside_span_m), 1.0))
    contact_density_score = _clip01(
        estimated_contact_density_per_km2 / max(float(resolved_selector_cfg.ideal_contact_density_per_km2), 1.0)
    )
    spread_score = float(
        np.exp(-((float(profile.spread_rad) - resolved_selector_cfg.ideal_spread_rad) / resolved_selector_cfg.spread_width_rad) ** 2)
    )
    contact_score = _clip01(float(estimated_contact_count) / max(float(len(ships)), 1.0))
    inside_score = 1.0 if bool(_intent_value(record, "inside_convoy_envelope", False)) else 0.0
    obs_quality = dict(observation.get("observation_quality", {}))
    range_uncertainty = min(float(obs_quality.get("range_sigma_m", 0.0)) / max(estimated_range_m, 1.0), 1.0)
    width_uncertainty = min(
        float(obs_quality.get("formation_width_sigma_m", 0.0)) / max(estimated_formation_width_m, 1.0),
        1.0,
    )
    depth_uncertainty = min(
        float(obs_quality.get("formation_depth_sigma_m", 0.0)) / max(estimated_formation_depth_m, 1.0),
        1.0,
    )
    uncertainty_score = float(
        np.sqrt(
            float(obs_quality.get("bearing_sigma_rad", 0.0)) ** 2
            + float(obs_quality.get("heading_sigma_rad", 0.0)) ** 2
            + range_uncertainty**2
            + width_uncertainty**2
            + depth_uncertainty**2
        )
    )

    score = (
        resolved_selector_cfg.range_weight * range_score
        + resolved_selector_cfg.bearing_alignment_weight * bearing_alignment_score
        + resolved_selector_cfg.aspect_weight * aspect_score
        + resolved_selector_cfg.formation_span_weight * formation_span_score
        + resolved_selector_cfg.contact_density_weight * contact_density_score
        + resolved_selector_cfg.spread_weight * spread_score
        + resolved_selector_cfg.contact_weight * contact_score
        + resolved_selector_cfg.inside_bonus_weight * inside_score
        - resolved_selector_cfg.uncertainty_penalty_weight * uncertainty_score
        - resolved_selector_cfg.inside_uncertainty_penalty_weight * inside_score * uncertainty_score
    )

    profile_id, name = _candidate_identity(record)
    return {
        "profile_id": profile_id,
        "name": name,
        "belief_score": float(score),
        "u_pos_x": float(profile.u_pos[0]),
        "u_pos_y": float(profile.u_pos[1]),
        "estimated_range_m": estimated_range_m,
        "estimated_bearing_rad": estimated_bearing_rad,
        "estimated_convoy_heading_rad": estimated_heading_rad,
        "estimated_convoy_speed_mps": estimated_speed_mps,
        "estimated_contact_count": int(estimated_contact_count),
        "estimated_formation_width_m": estimated_formation_width_m,
        "estimated_formation_depth_m": estimated_formation_depth_m,
        "estimated_contact_density_per_km2": estimated_contact_density_per_km2,
        "estimated_aspect_rad": float(estimated_aspect_rad),
        "projected_target_span_m": projected_target_span_m,
        "bearing_alignment_error_rad": float(bearing_alignment_error_rad),
        "range_score": float(range_score),
        "bearing_alignment_score": float(bearing_alignment_score),
        "aspect_score": float(aspect_score),
        "formation_span_score": float(formation_span_score),
        "contact_density_score": float(contact_density_score),
        "spread_score": float(spread_score),
        "contact_score": float(contact_score),
        "inside_score": float(inside_score),
        "uncertainty_score": float(uncertainty_score),
        "inside_uncertainty_penalty": float(inside_score * uncertainty_score),
        "spawn_region": str(_intent_value(record, "spawn_region", "")),
        "approach_side": str(_intent_value(record, "approach_side", "")),
        "inside_convoy_envelope": bool(_intent_value(record, "inside_convoy_envelope", False)),
        "spread_rad": float(profile.spread_rad),
        "launch_delay_s": float(profile.launch_delay_s),
        "salvo_interval_s": float(profile.salvo_interval_s),
        "u_boat_initial_speed_mps": float(profile.u_boat_initial_speed_mps),
        "candidate_source_label": str(dict(record.get("generator_meta", {})).get("mode", "")),
        "selector_method": "belief_limited_heuristic_v1",
    }


def build_candidate_observation_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    ships: Sequence[Ship],
    seed: int = 1945,
    env: Environment | None = None,
    observation_cfg: AttackerObservationConfig | None = None,
    selector_cfg: CandidateSelectorConfig | None = None,
) -> list[dict[str, Any]]:
    """Build noisy observation rows for all candidates."""

    rng = np.random.default_rng(int(seed))
    return [
        build_candidate_observation_row(
            record,
            ships=ships,
            rng=rng,
            env=env,
            observation_cfg=observation_cfg,
            selector_cfg=selector_cfg,
        )
        for record in records
    ]


def rank_candidate_observation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rank rows by belief-limited heuristic score, highest first."""

    ranked = [dict(row) for row in rows]
    ranked.sort(
        key=lambda row: (
            -float(row["belief_score"]),
            float(row["estimated_range_m"]),
            str(row["profile_id"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["belief_rank"] = int(rank)
    if top_k is not None:
        return ranked[: int(top_k)]
    return ranked


def rank_candidate_records_by_belief(
    records: Sequence[Mapping[str, Any]],
    *,
    ships: Sequence[Ship],
    seed: int = 1945,
    top_k: int | None = None,
    env: Environment | None = None,
    observation_cfg: AttackerObservationConfig | None = None,
    selector_cfg: CandidateSelectorConfig | None = None,
) -> list[dict[str, Any]]:
    """Build observations and return belief-ranked candidate rows."""

    rows = build_candidate_observation_rows(
        records,
        ships=ships,
        seed=seed,
        env=env,
        observation_cfg=observation_cfg,
        selector_cfg=selector_cfg,
    )
    return rank_candidate_observation_rows(rows, top_k=top_k)


def write_belief_ranked_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write belief-ranked candidate rows to CSV."""

    if not rows:
        raise ValueError("rows must be non-empty")
    fieldnames = list(dict(rows[0]).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_belief_ranked_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write belief-ranked candidate rows to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": [dict(row) for row in rows]}, indent=2) + "\n", encoding="utf-8")


def write_belief_selected_candidate_pool(
    path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    observation_preset: str,
) -> None:
    """Write selected original candidate records in belief-rank order."""

    records_by_id = {str(record["profile"]["profile_id"]): dict(record) for record in records}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            profile_id = str(row["profile_id"])
            if profile_id not in records_by_id:
                raise ValueError(f"belief row references unknown profile_id: {profile_id}")
            record = dict(records_by_id[profile_id])
            record["selection_meta"] = {
                "method": "belief_limited_heuristic_v1",
                "observation_preset": str(observation_preset),
                "belief_rank": int(row["belief_rank"]),
                "belief_score": float(row["belief_score"]),
            }
            handle.write(json.dumps(record) + "\n")
