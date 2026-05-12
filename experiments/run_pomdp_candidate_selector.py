"""Run the first belief-limited selector over a VAE candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convoy_sim.feasibility import Environment
from convoy_sim.pomdp_candidate_selector import (
    build_candidate_observation_rows,
    rank_candidate_observation_rows,
    write_belief_ranked_csv,
    write_belief_ranked_json,
)
from convoy_sim.realism import AttackerObservationConfig
from convoy_sim.workflows import git_sha, resolve_run_dir, write_json
from experiments.evaluate_attack_candidate_pool import load_candidate_records
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank VAE candidates with a noisy-observation heuristic selector.")
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path("data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", default="pomdp_belief_selector")
    parser.add_argument("--convoy-profile", choices=list_convoy_layout_profiles(), default="convoy_layout_1")
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1945)
    parser.add_argument("--bearing-sigma-rad", type=float, default=0.04)
    parser.add_argument("--range-sigma-m", type=float, default=120.0)
    parser.add_argument("--heading-sigma-rad", type=float, default=0.06)
    parser.add_argument("--speed-sigma-mps", type=float, default=0.5)
    parser.add_argument("--contact-count-sigma", type=float, default=0.4)
    return parser.parse_args()


def run_belief_selector(
    *,
    candidate_path: Path,
    project_root: Path,
    output_root: Path = Path("results/runs"),
    run_name: str = "pomdp_belief_selector",
    convoy_profile: str = "convoy_layout_1",
    max_profiles: int | None = 100,
    top_k: int = 25,
    seed: int = 1945,
    observation_cfg: AttackerObservationConfig | None = None,
) -> Path:
    records_all = load_candidate_records(candidate_path)
    records = records_all[: int(max_profiles)] if max_profiles is not None else records_all
    if not records:
        raise ValueError("candidate pool is empty")
    ships = get_convoy_layout_profile(convoy_profile).build_ships()
    env = Environment(time_of_day="night", visibility_m=3500.0, sea_state=4)
    obs_cfg = observation_cfg or AttackerObservationConfig()
    rows = build_candidate_observation_rows(
        records,
        ships=ships,
        seed=int(seed),
        env=env,
        observation_cfg=obs_cfg,
    )
    ranked = rank_candidate_observation_rows(rows)
    top_rows = ranked[: int(top_k)]
    output_dir = resolve_run_dir(project_root / output_root, "pomdp_candidate_selector", run_name)
    write_belief_ranked_csv(output_dir / "belief_ranked_candidates.csv", ranked)
    write_belief_ranked_json(output_dir / "top_belief_candidates.json", top_rows)
    write_json(
        output_dir / "metrics_summary.json",
        {
            "candidate_pool": {
                "profiles": int(len(ranked)),
                "mean_belief_score": float(sum(float(row["belief_score"]) for row in ranked) / len(ranked)),
                "best_belief_score": float(ranked[0]["belief_score"]),
                "best_profile_id": str(ranked[0]["profile_id"]),
            },
            "top_k": {
                "profiles": int(len(top_rows)),
                "mean_belief_score": float(sum(float(row["belief_score"]) for row in top_rows) / len(top_rows)),
                "best_profile_id": str(top_rows[0]["profile_id"]) if top_rows else "",
            },
            "selection": {
                "method": "belief_limited_heuristic_v1",
                "score": "belief_score",
                "higher_is_better_for_attacker": True,
                "top_k": int(top_k),
            },
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "workflow": "pomdp_candidate_selector",
            "git_sha": git_sha(project_root),
            "candidate_path": str(candidate_path),
            "candidate_record_count": int(len(records_all)),
            "evaluated_profile_count": int(len(records)),
            "convoy_profile": str(convoy_profile),
            "seed": int(seed),
            "observation_config": {
                "bearing_sigma_rad": float(obs_cfg.bearing_sigma_rad),
                "range_sigma_m": float(obs_cfg.range_sigma_m),
                "heading_sigma_rad": float(obs_cfg.heading_sigma_rad),
                "speed_sigma_mps": float(obs_cfg.speed_sigma_mps),
                "contact_count_sigma": float(obs_cfg.contact_count_sigma),
            },
            "artifacts": {
                "belief_ranked_candidates_csv": "belief_ranked_candidates.csv",
                "top_belief_candidates_json": "top_belief_candidates.json",
                "metrics_summary_json": "metrics_summary.json",
                "run_manifest_json": "run_manifest.json",
            },
        },
    )
    return output_dir


def main() -> None:
    args = _parse_args()
    obs_cfg = AttackerObservationConfig(
        bearing_sigma_rad=float(args.bearing_sigma_rad),
        range_sigma_m=float(args.range_sigma_m),
        heading_sigma_rad=float(args.heading_sigma_rad),
        speed_sigma_mps=float(args.speed_sigma_mps),
        contact_count_sigma=float(args.contact_count_sigma),
    )
    run_dir = run_belief_selector(
        candidate_path=args.candidate_path,
        project_root=Path.cwd(),
        output_root=args.output_root,
        run_name=str(args.run_name),
        convoy_profile=str(args.convoy_profile),
        max_profiles=args.max_profiles,
        top_k=int(args.top_k),
        seed=int(args.seed),
        observation_cfg=obs_cfg,
    )
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
