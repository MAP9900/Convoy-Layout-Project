"""Evaluate and rank JSONL attack candidate pools."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile, AttackProfileLibrary
from convoy_sim.feasibility import Environment
from convoy_sim.noise import NoiseModel
from convoy_sim.objectives import objective_from_config, objective_to_dict
from convoy_sim.workflows import (
    ProfileEvalRow,
    evaluate_layout_over_profiles,
    git_sha,
    resolve_run_dir,
    summarize_profile_rows,
    write_json,
    write_yaml,
)
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


DEFAULT_SHIP_MOVEMENT_REALISM: dict[str, Any] = {
    "position_jitter_std_m": 8.0,
    "heading_jitter_std_rad": 0.015,
    "deviation_offset_cap_m": 12.0,
    "enable_slot_swaps": False,
    "max_swap_fraction": 0.0,
    "freighter_scale": 1.0,
    "tanker_scale": 0.9,
    "escort_scale": 0.6,
    "decoy_scale": 1.1,
}


def load_candidate_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records with at least a ``profile`` payload."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc
            if "profile" not in record:
                raise ValueError(f"Line {line_number} in {path} is missing required key: profile")
            records.append(record)
    return records


def candidate_records_to_library(
    records: Sequence[Mapping[str, Any]],
    *,
    max_profiles: int | None = None,
) -> tuple[AttackProfileLibrary, list[dict[str, Any]]]:
    """Build an attack-profile library from candidate JSONL records."""

    selected = list(records[:max_profiles]) if max_profiles is not None else list(records)
    if not selected:
        raise ValueError("candidate records must be non-empty")
    profiles = [AttackProfile.from_dict(dict(record["profile"])) for record in selected]
    return AttackProfileLibrary(profiles=profiles), selected


def candidate_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact candidate-pool metadata for ranked CSV output."""

    profile = dict(record.get("profile", {}))
    audit = dict(record.get("audit", {}))
    intent = dict(record.get("intent", {}))
    outcome = dict(record.get("outcome", {}))
    return {
        "profile_id": str(profile.get("profile_id", "")),
        "spawn_region": str(intent.get("spawn_region", audit.get("spawn_region", ""))),
        "approach_side": str(intent.get("approach_side", audit.get("approach_side", ""))),
        "inside_convoy_envelope": bool(
            intent.get("inside_convoy_envelope", audit.get("inside_convoy_envelope", False))
        ),
        "source_label": str(audit.get("suggested_label", "")),
        "source_actual_outcome_label": str(
            audit.get("actual_outcome_label", outcome.get("actual_outcome_label", ""))
        ),
        "source_clearance_m": _float_or_blank(audit.get("clearance_m", intent.get("nearest_ship_clearance_m"))),
        "source_n_hits": _int_or_blank(audit.get("n_hits", outcome.get("n_hits"))),
        "source_unique_ships_hit": _int_or_blank(
            audit.get("unique_ships_hit", outcome.get("unique_ships_hit"))
        ),
        "source_closest_any_ship_distance_m": _float_or_blank(
            audit.get("closest_any_ship_distance_m", outcome.get("closest_any_ship_distance_m"))
        ),
        "source_closest_any_ship_id": str(
            audit.get("closest_any_ship_id", outcome.get("closest_any_ship_id", ""))
        ),
    }


def _float_or_blank(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    return parsed if np.isfinite(parsed) else ""


def _int_or_blank(value: Any) -> int | str:
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def _row_score(row: ProfileEvalRow, *, objective_enabled: bool) -> float:
    return float(row.expected_loss if objective_enabled else row.expected_hits)


def ranked_candidate_rows(
    rows: Sequence[ProfileEvalRow],
    *,
    records: Sequence[Mapping[str, Any]],
    objective_enabled: bool,
) -> list[dict[str, Any]]:
    """Return ranked attacker rows, highest score first."""

    metadata_by_id = {str(record["profile"]["profile_id"]): candidate_metadata(record) for record in records}
    ranked: list[dict[str, Any]] = []
    for row in rows:
        item = row.to_dict()
        item.update(metadata_by_id.get(str(row.profile_id), {}))
        item["attacker_score"] = _row_score(row, objective_enabled=objective_enabled)
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["attacker_score"]),
            -float(item["expected_hits"]),
            -float(item["expected_unique_ships_hit"]),
            str(item["profile_id"]),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = int(rank)
    return ranked


def write_ranked_candidates_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ranked candidate metrics and source metadata to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "model_name",
        "profile_id",
        "attacker_score",
        "samples",
        "expected_hits",
        "CVaR_90",
        "p_hit_ge_1",
        "value_lost",
        "expected_unique_ships_hit",
        "expected_repeat_hits",
        "expected_loss",
        "CVaR_90_loss",
        "spawn_region",
        "approach_side",
        "inside_convoy_envelope",
        "source_label",
        "source_actual_outcome_label",
        "source_clearance_m",
        "source_n_hits",
        "source_unique_ships_hit",
        "source_closest_any_ship_distance_m",
        "source_closest_any_ship_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _parse_seed_list(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("seed list must be non-empty")
    return [int(value) for value in values]


def _objective_config_from_preset(preset: str) -> dict[str, Any]:
    if str(preset).lower() in {"", "none", "null"}:
        return {}
    return {"preset": str(preset)}


def evaluate_candidate_pool(
    *,
    candidate_path: Path,
    project_root: Path,
    output_root: Path = Path("results/runs"),
    run_name: str | None = None,
    convoy_profile: str | None = None,
    max_profiles: int | None = None,
    top_k: int = 25,
    seeds: Sequence[int] = (1942, 1943, 1944),
    n_trials_per_seed: int = 10,
    t_max: float = 400.0,
    max_hits_per_torpedo: int | None = 1,
    objective_cfg: Mapping[str, Any] | None = None,
    noise_cfg: Mapping[str, Any] | None = None,
    environment_cfg: Mapping[str, Any] | None = None,
    ship_movement_realism: Mapping[str, Any] | None = DEFAULT_SHIP_MOVEMENT_REALISM,
) -> Path:
    """Evaluate a candidate pool and write ranked attacker-selection artifacts."""

    started_at = time.perf_counter()
    records_all = load_candidate_records(candidate_path)
    if convoy_profile is None:
        convoy_profile = str(records_all[0].get("generator_meta", {}).get("convoy_profile", "convoy_layout_1"))
    library, selected_records = candidate_records_to_library(records_all, max_profiles=max_profiles)
    profile_ids = [str(profile.profile_id) for profile in library.profiles]
    layout_profile = get_convoy_layout_profile(str(convoy_profile))
    layout_kwargs = dict(layout_profile.layout_kwargs)
    if ship_movement_realism:
        layout_kwargs["ship_movement_realism"] = dict(ship_movement_realism)
    objective = objective_from_config(dict(objective_cfg or {}))
    noise_model = NoiseModel.from_dict(dict(noise_cfg or {}))
    env_raw = dict(environment_cfg or {})
    env = Environment(
        time_of_day=str(env_raw.get("time_of_day", "night")),
        visibility_m=float(env_raw.get("visibility_m", 3500.0)),
        sea_state=int(env_raw.get("sea_state", 4)),
        detection_risk_scale=float(env_raw.get("detection_risk_scale", 1.0)),
    )

    output_dir = resolve_run_dir(project_root / output_root, "candidate_pool_eval", run_name)
    rows = evaluate_layout_over_profiles(
        model_name="vae_candidate_pool_full_state",
        layout_fn=layout_profile.layout_fn,
        layout_kwargs=layout_kwargs,
        library=library,
        profile_ids=profile_ids,
        seeds=[int(seed) for seed in seeds],
        n_trials_per_seed=int(n_trials_per_seed),
        t_max=float(t_max),
        noise_model=noise_model,
        env=env,
        max_hits_per_torpedo=max_hits_per_torpedo,
        objective=objective,
    )
    ranked = ranked_candidate_rows(rows, records=selected_records, objective_enabled=objective is not None)
    top_rows = ranked[: int(top_k)]
    pool_summary = summarize_profile_rows(list(rows))
    top_summary = summarize_profile_rows(
        [row for row in rows if str(row.profile_id) in {str(item["profile_id"]) for item in top_rows}]
    )

    write_ranked_candidates_csv(output_dir / "ranked_candidates.csv", ranked)
    write_json(output_dir / "top_candidates.json", {"candidates": top_rows})
    write_json(
        output_dir / "metrics_summary.json",
        {
            "candidate_pool": pool_summary,
            "top_k": top_summary,
            "best_candidate": top_rows[0] if top_rows else None,
            "selection": {
                "method": "full_state_attacker_rank",
                "score": "expected_loss" if objective is not None else "expected_hits",
                "higher_is_better_for_attacker": True,
                "top_k": int(top_k),
            },
            "timing": {
                "total_seconds": float(time.perf_counter() - started_at),
            },
        },
    )
    resolved = {
        "candidate_path": str(candidate_path),
        "convoy_profile": str(convoy_profile),
        "profile_count": int(len(profile_ids)),
        "seeds": [int(seed) for seed in seeds],
        "n_trials_per_seed": int(n_trials_per_seed),
        "t_max": float(t_max),
        "max_hits_per_torpedo": max_hits_per_torpedo,
        "objective": objective_to_dict(objective),
        "noise": noise_model.to_dict(),
        "environment": {
            "time_of_day": env.time_of_day,
            "visibility_m": env.visibility_m,
            "sea_state": env.sea_state,
            "detection_risk_scale": env.detection_risk_scale,
        },
        "ship_movement_realism": dict(ship_movement_realism or {}),
    }
    write_yaml(output_dir / "config_resolved.yaml", resolved)
    write_json(
        output_dir / "run_manifest.json",
        {
            "workflow": "candidate_pool_eval",
            "git_sha": git_sha(project_root),
            "candidate_path": str(candidate_path),
            "candidate_record_count": int(len(records_all)),
            "evaluated_profile_count": int(len(profile_ids)),
            "convoy_profile": str(convoy_profile),
            "seed_sets": {"eval": [int(seed) for seed in seeds]},
            "n_trials_per_seed": int(n_trials_per_seed),
            "t_max": float(t_max),
            "objective": objective_to_dict(objective),
            "artifacts": {
                "ranked_candidates_csv": "ranked_candidates.csv",
                "top_candidates_json": "top_candidates.json",
                "metrics_summary_json": "metrics_summary.json",
                "run_manifest_json": "run_manifest.json",
                "config_resolved_yaml": "config_resolved.yaml",
            },
        },
    )
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and rank an attack candidate JSONL pool.")
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", default="vae_candidate_pool")
    parser.add_argument("--convoy-profile", choices=list_convoy_layout_profiles(), default=None)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seeds", default="1942,1943,1944")
    parser.add_argument("--n-trials-per-seed", type=int, default=10)
    parser.add_argument("--t-max", type=float, default=400.0)
    parser.add_argument("--max-hits-per-torpedo", type=int, default=1)
    parser.add_argument("--objective-preset", default="balanced_default")
    parser.add_argument("--disable-ship-realism", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path.cwd()
    objective_cfg = _objective_config_from_preset(str(args.objective_preset))
    ship_realism = None if bool(args.disable_ship_realism) else DEFAULT_SHIP_MOVEMENT_REALISM
    run_dir = evaluate_candidate_pool(
        candidate_path=args.candidate_path,
        project_root=project_root,
        output_root=args.output_root,
        run_name=str(args.run_name),
        convoy_profile=args.convoy_profile,
        max_profiles=args.max_profiles,
        top_k=int(args.top_k),
        seeds=_parse_seed_list(str(args.seeds)),
        n_trials_per_seed=int(args.n_trials_per_seed),
        t_max=float(args.t_max),
        max_hits_per_torpedo=args.max_hits_per_torpedo,
        objective_cfg=objective_cfg,
        ship_movement_realism=ship_realism,
    )
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
