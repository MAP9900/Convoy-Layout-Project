"""Generate adversarial candidate pools from a trained attack-profile VAE."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.profile_outcome_audit import OutcomeAuditConfig
from convoy_sim.target_zones import ConvoyEnvelope, ConvoyFrame, convoy_frame_and_envelope
from convoy_sim.vae import (
    AttackProfileVAE,
    AttackProfileVAEPreprocessor,
    build_latent_bank,
    load_attack_profile_dataset_jsonl,
)
from convoy_sim.vae_diagnostics import audit_decoded_vae_payloads, summarize_decoded_vae_audit
from convoy_sim.workflows import git_sha, write_json
from experiments.generate_attack_profile_scaffold import (
    HIT_THREAT_LABEL,
    INTENTIONAL_MISS_LABEL,
    MIN_SPAWN_CLEARANCE_M,
    NEAR_MISS_LABEL,
)
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


VAE_CANDIDATE_MODE = "vae_candidate_pool_v1"
VAE_CANDIDATE_SOURCE = "generate_vae_candidate_pool"
VAE_CANDIDATE_VERSION = "v1"
DEFAULT_ACCEPTED_OUTCOMES = (HIT_THREAT_LABEL,)


def _profile_to_dict(profile: AttackProfile) -> dict[str, Any]:
    payload = profile.to_dict()
    payload["spread_doctrine"] = str(profile.spread_doctrine)
    payload["per_torpedo_heading_offsets_rad"] = [
        float(value) for value in profile.per_torpedo_heading_offsets_rad
    ]
    return payload


def _inside_envelope(local: np.ndarray, envelope: ConvoyEnvelope) -> bool:
    return bool(
        envelope.min_x <= float(local[0]) <= envelope.max_x
        and envelope.min_y <= float(local[1]) <= envelope.max_y
    )


def _spawn_region(local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    if _inside_envelope(local, envelope):
        return "inside_convoy_envelope"
    distances = {
        "ahead_vae": float(local[0] - envelope.max_x),
        "astern_vae": float(envelope.min_x - local[0]),
        "port_vae": float(local[1] - envelope.max_y),
        "starboard_vae": float(envelope.min_y - local[1]),
    }
    return max(distances.items(), key=lambda item: item[1])[0]


def _approach_side(spawn_region: str) -> str:
    if spawn_region == "inside_convoy_envelope":
        return "inside"
    return str(spawn_region).replace("_vae", "")


def _candidate_label_from_actual(actual_outcome_label: str) -> str:
    if actual_outcome_label == HIT_THREAT_LABEL:
        return HIT_THREAT_LABEL
    if actual_outcome_label == NEAR_MISS_LABEL:
        return NEAR_MISS_LABEL
    if actual_outcome_label == "miss":
        return INTENTIONAL_MISS_LABEL
    return str(actual_outcome_label)


def _load_model_from_run(
    run_dir: Path,
    *,
    checkpoint_name: str,
    device: str,
) -> tuple[AttackProfileVAE, AttackProfileVAEPreprocessor, dict[str, Any]]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "checkpoints" / checkpoint_name, map_location=device)
    preprocessor_payload = checkpoint.get("preprocessor", manifest["preprocessor"])
    preprocessor = AttackProfileVAEPreprocessor.from_dict(preprocessor_payload)
    hyper = checkpoint.get("hyperparameters", manifest.get("hyperparameters", {}))
    model = AttackProfileVAE(
        latent_dim=int(hyper["latent_dim"]),
        hidden_dim=int(hyper["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, preprocessor, manifest


@torch.no_grad()
def sample_vae_payloads(
    *,
    model: AttackProfileVAE,
    preprocessor: AttackProfileVAEPreprocessor,
    sample_count: int,
    sampling_method: str,
    device: str,
    seed: int,
    train_records: Sequence[Mapping[str, Any]] | None = None,
    latent_noise_scale: float = 0.10,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Sample decoded VAE payloads using either prior or latent-bank sampling."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if start_index <= 0:
        raise ValueError("start_index must be positive")
    if latent_noise_scale < 0.0:
        raise ValueError("latent_noise_scale must be non-negative")
    torch.manual_seed(int(seed))
    sample_device = torch.device(device)
    if str(sampling_method) == "prior":
        decoded = model.sample(int(sample_count), device=sample_device).cpu().numpy()
    elif str(sampling_method) == "latent_bank":
        if train_records is None:
            raise ValueError("train_records is required for latent_bank sampling")
        train_features = preprocessor.transform_records(train_records)
        latent_bank = build_latent_bank(model, train_features, device=sample_device)
        decoded = model.sample_from_latent_bank(
            latent_bank,
            int(sample_count),
            noise_scale=float(latent_noise_scale),
            device=sample_device,
        ).cpu().numpy()
    else:
        raise ValueError("sampling_method must be 'prior' or 'latent_bank'")

    payloads: list[dict[str, Any]] = []
    for index in range(int(sample_count)):
        profile_index = int(start_index) + index
        payloads.append(
            preprocessor.decode_profile_fields(
                decoded[index],
                profile_id=f"VAE{profile_index:06d}",
                name=f"vae_candidate_{profile_index:06d}",
            )
        )
    return payloads


def _derived_intent(
    *,
    row: Mapping[str, Any],
    profile: AttackProfile,
    ships: Sequence[Any],
    frame: ConvoyFrame,
    envelope: ConvoyEnvelope,
) -> dict[str, Any]:
    u_pos = np.asarray(profile.u_pos, dtype=float)
    local = frame.world_to_local(u_pos)
    spawn_region = _spawn_region(local, envelope)
    actual_label = str(row["actual_outcome_label"])
    candidate_label = _candidate_label_from_actual(actual_label)
    hit_ship_ids = list(row.get("outcome", {}).get("hit_ship_ids", []))
    target_ship_ids = hit_ship_ids if candidate_label == HIT_THREAT_LABEL else []
    closest_ship_id = str(row.get("closest_any_ship_id", ""))
    ship_by_id = {str(getattr(ship, "id")): ship for ship in ships}
    target_ship = ship_by_id.get((target_ship_ids or [closest_ship_id])[0], None) if (target_ship_ids or closest_ship_id) else None
    target_point = None if target_ship is None else np.asarray(getattr(target_ship, "position"), dtype=float)
    target_local = None if target_point is None else frame.world_to_local(target_point)
    return {
        "target_zone_id": f"{profile.profile_id}_derived",
        "target_zone_kind": "vae_decoded_candidate",
        "target_ship_ids": target_ship_ids,
        "target_point": None if target_point is None else [float(target_point[0]), float(target_point[1])],
        "target_local": None if target_local is None else [float(target_local[0]), float(target_local[1])],
        "spawn_local": [float(local[0]), float(local[1])],
        "spawn_region": spawn_region,
        "approach_side": _approach_side(spawn_region),
        "approach_lane": f"{spawn_region}:vae_decoded",
        "inside_convoy_envelope": _inside_envelope(local, envelope),
        "nearest_ship_clearance_m": float(row["clearance_m"]),
        "intended_label": candidate_label,
        "actual_outcome_label": actual_label,
        "profile_first_outcome_label": True,
        "derived_from_vae_sample": True,
        "closest_any_ship_id": closest_ship_id,
    }


def _record_from_row(
    *,
    row: Mapping[str, Any],
    profile: AttackProfile,
    intent: Mapping[str, Any],
    seed: int,
    convoy_profile: str,
    run_dir: Path,
    checkpoint_name: str,
    sampling_method: str,
    latent_noise_scale: float,
) -> dict[str, Any]:
    outcome = dict(row["outcome"])
    candidate_label = str(intent["intended_label"])
    target_ship_ids = list(intent.get("target_ship_ids", []))
    hit_ship_ids = [str(value) for value in outcome.get("hit_ship_ids", [])]
    intended_target_hit = bool(set(target_ship_ids).intersection(hit_ship_ids))
    outcome["intended_label"] = candidate_label
    outcome["target_ship_ids"] = target_ship_ids
    outcome["intended_target_hit"] = intended_target_hit
    if intended_target_hit:
        outcome["closest_intended_target_distance_m"] = float(row["closest_any_ship_distance_m"])
    outcome["outcome_matches_intent"] = True
    outcome["passes_outcome_gate"] = True
    outcome["spawn_region"] = str(intent["spawn_region"])
    outcome["approach_side"] = str(intent["approach_side"])
    outcome["target_zone_kind"] = str(intent["target_zone_kind"])
    return {
        "profile": _profile_to_dict(profile),
        "audit": {
            "profile_id": str(row["profile_id"]),
            "name": str(row["name"]),
            "suggested_label": candidate_label,
            "actual_outcome_label": str(row["actual_outcome_label"]),
            "clearance_m": float(row["clearance_m"]),
            "clearance_ok": bool(row["clearance_ok"]),
            "any_ship_hit": bool(row["any_ship_hit"]),
            "n_hits": int(row["n_hits"]),
            "unique_ships_hit": int(row["unique_ships_hit"]),
            "closest_any_ship_distance_m": float(row["closest_any_ship_distance_m"]),
            "closest_any_ship_id": str(row["closest_any_ship_id"]),
            "centroid_static_label": str(row["centroid_static_label"]),
            "centroid_static_flags": list(row["centroid_static_flags"]),
            "centroid_static_bearing_error_deg": float(row["centroid_static_bearing_error_deg"]),
            "spawn_region": str(intent["spawn_region"]),
            "approach_side": str(intent["approach_side"]),
            "inside_convoy_envelope": bool(intent["inside_convoy_envelope"]),
        },
        "intent": dict(intent),
        "outcome": outcome,
        "generator_meta": {
            "mode": VAE_CANDIDATE_MODE,
            "seed": int(seed),
            "convoy_profile": str(convoy_profile),
            "source": VAE_CANDIDATE_SOURCE,
            "generator_version": VAE_CANDIDATE_VERSION,
            "vae_run_dir": str(run_dir),
            "checkpoint_name": str(checkpoint_name),
            "sampling_method": str(sampling_method),
            "latent_noise_scale": float(latent_noise_scale),
        },
    }


def build_vae_candidate_records(
    *,
    run_dir: Path,
    train_path: Path,
    checkpoint_name: str = "model_best.pt",
    sample_count: int = 1000,
    keep_count: int | None = None,
    seed: int = 1945,
    start_index: int = 1,
    convoy_profile: str = "convoy_layout_1",
    device: str = "cpu",
    sampling_method: str = "latent_bank",
    latent_noise_scale: float = 0.10,
    min_clearance_m: float = MIN_SPAWN_CLEARANCE_M,
    accepted_outcomes: Sequence[str] = DEFAULT_ACCEPTED_OUTCOMES,
    outcome_cfg: OutcomeAuditConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate filtered VAE candidate records for adversarial selection."""

    if keep_count is not None and keep_count <= 0:
        raise ValueError("keep_count must be positive when provided")
    model, preprocessor, manifest = _load_model_from_run(
        Path(run_dir),
        checkpoint_name=str(checkpoint_name),
        device=str(device),
    )
    train_records = load_attack_profile_dataset_jsonl(train_path) if str(sampling_method) == "latent_bank" else None
    payloads = sample_vae_payloads(
        model=model,
        preprocessor=preprocessor,
        sample_count=int(sample_count),
        sampling_method=str(sampling_method),
        device=str(device),
        seed=int(seed),
        train_records=train_records,
        latent_noise_scale=float(latent_noise_scale),
        start_index=int(start_index),
    )
    ships = get_convoy_layout_profile(convoy_profile).build_ships()
    rows = audit_decoded_vae_payloads(
        payloads,
        ships,
        min_clearance_m=float(min_clearance_m),
        rng_seed=int(seed),
        outcome_cfg=outcome_cfg or OutcomeAuditConfig(t_max_s=600.0, hit_dt_s=0.5),
    )
    row_by_id = {str(row["profile_id"]): row for row in rows}
    frame, envelope, _ = convoy_frame_and_envelope(ships)
    accepted_set = {str(label) for label in accepted_outcomes}
    records: list[dict[str, Any]] = []
    for payload in payloads:
        row = row_by_id[str(payload["profile_id"])]
        if not bool(row["clearance_ok"]):
            continue
        if str(row["actual_outcome_label"]) not in accepted_set:
            continue
        profile = AttackProfile(**payload)
        intent = _derived_intent(row=row, profile=profile, ships=ships, frame=frame, envelope=envelope)
        records.append(
            _record_from_row(
                row=row,
                profile=profile,
                intent=intent,
                seed=int(seed),
                convoy_profile=str(convoy_profile),
                run_dir=Path(run_dir),
                checkpoint_name=str(checkpoint_name),
                sampling_method=str(sampling_method),
                latent_noise_scale=float(latent_noise_scale),
            )
        )
        if keep_count is not None and len(records) >= int(keep_count):
            break

    summary = {
        "workflow": VAE_CANDIDATE_SOURCE,
        "mode": VAE_CANDIDATE_MODE,
        "generator_version": VAE_CANDIDATE_VERSION,
        "git_sha": git_sha(Path.cwd()),
        "vae_run_dir": str(run_dir),
        "checkpoint_name": str(checkpoint_name),
        "train_path": str(train_path),
        "convoy_profile": str(convoy_profile),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "keep_count": None if keep_count is None else int(keep_count),
        "accepted_count": int(len(records)),
        "filtered_out_count": int(len(payloads) - len(records)),
        "sampling_method": str(sampling_method),
        "latent_noise_scale": float(latent_noise_scale),
        "min_clearance_m": float(min_clearance_m),
        "accepted_outcomes": sorted(accepted_set),
        "pre_filter_audit": summarize_decoded_vae_audit(rows),
        "accepted_labels": dict(Counter(str(record["audit"]["suggested_label"]) for record in records)),
        "accepted_spawn_regions": dict(Counter(str(record["intent"]["spawn_region"]) for record in records)),
        "source_manifest": manifest,
    }
    return records, summary


def render_vae_candidate_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    """Render candidate records as newline-delimited JSON."""

    return "\n".join(json.dumps(_json_safe(dict(record)), allow_nan=False) for record in records) + (
        "\n" if records else ""
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a VAE-derived attack candidate JSONL pool.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="model_best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--keep-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1945)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--convoy-profile", choices=list_convoy_layout_profiles(), default="convoy_layout_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampling-method", choices=("prior", "latent_bank"), default="latent_bank")
    parser.add_argument("--latent-noise-scale", type=float, default=0.10)
    parser.add_argument("--min-clearance-m", type=float, default=MIN_SPAWN_CLEARANCE_M)
    parser.add_argument(
        "--accepted-outcome",
        action="append",
        dest="accepted_outcomes",
        default=None,
        help="Outcome label to keep. May be passed multiple times. Defaults to credible_hit_threat.",
    )
    parser.add_argument("--t-max-s", type=float, default=600.0)
    parser.add_argument("--hit-dt-s", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records, summary = build_vae_candidate_records(
        run_dir=args.run_dir,
        train_path=args.train_path,
        checkpoint_name=str(args.checkpoint_name),
        sample_count=int(args.sample_count),
        keep_count=args.keep_count,
        seed=int(args.seed),
        start_index=int(args.start_index),
        convoy_profile=str(args.convoy_profile),
        device=str(args.device),
        sampling_method=str(args.sampling_method),
        latent_noise_scale=float(args.latent_noise_scale),
        min_clearance_m=float(args.min_clearance_m),
        accepted_outcomes=args.accepted_outcomes or DEFAULT_ACCEPTED_OUTCOMES,
        outcome_cfg=OutcomeAuditConfig(t_max_s=float(args.t_max_s), hit_dt_s=float(args.hit_dt_s)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_vae_candidate_jsonl(records), encoding="utf-8")
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    write_json(summary_output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
