"""Build mixed curated/random attack-profile JSONL datasets for VAE training."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from convoy_sim.vae import load_attack_profile_dataset_jsonl


MIXED_DATASET_MODE = "mixed_curated_random_v1"
MIXED_DATASET_SOURCE = "build_mixed_attack_profile_dataset"
MIXED_DATASET_VERSION = "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 70/30 curated/random mixed VAE JSONL datasets.")
    parser.add_argument(
        "--curated-train",
        type=Path,
        default=Path("data/attack_profiles/synthetic/train_random_tactical_v4_45k.jsonl"),
        help="Curated v4 training JSONL.",
    )
    parser.add_argument(
        "--random-train",
        type=Path,
        default=Path("data/attack_profiles/synthetic/train_random_profile_v1_45k.jsonl"),
        help="Random-baseline training JSONL.",
    )
    parser.add_argument(
        "--curated-valid",
        type=Path,
        default=Path("data/attack_profiles/synthetic/valid_random_tactical_v4_5k.jsonl"),
        help="Curated v4 validation JSONL.",
    )
    parser.add_argument(
        "--random-valid",
        type=Path,
        default=Path("data/attack_profiles/synthetic/valid_random_profile_v1_5k.jsonl"),
        help="Random-baseline validation JSONL.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/attack_profiles/synthetic/train_mixed_curated70_random30_45k.jsonl"),
        help="Output mixed training JSONL.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=Path("data/attack_profiles/synthetic/valid_mixed_curated70_random30_5k.jsonl"),
        help="Output mixed validation JSONL.",
    )
    parser.add_argument("--train-count", type=int, default=45_000, help="Mixed training record count.")
    parser.add_argument("--valid-count", type=int, default=5_000, help="Mixed validation record count.")
    parser.add_argument(
        "--curated-fraction",
        type=float,
        default=0.70,
        help="Fraction of each mixed split sampled from curated v4 records.",
    )
    parser.add_argument("--seed", type=int, default=1945, help="Sampling/shuffle seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def _sample_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    source_dataset: str,
    split: str,
) -> list[dict[str, Any]]:
    if int(count) > len(records):
        raise ValueError(f"Requested {count} {source_dataset} {split} records, but only {len(records)} are available")
    indices = rng.choice(np.arange(len(records)), size=int(count), replace=False)
    sampled: list[dict[str, Any]] = []
    for index in indices.tolist():
        record = deepcopy(records[int(index)])
        record["_mixed_source_dataset"] = source_dataset
        record["_mixed_source_index"] = int(index)
        sampled.append(record)
    return sampled


def _rewrite_record(
    record: dict[str, Any],
    *,
    profile_id: str,
    name: str,
    split: str,
    mixed_index: int,
    seed: int,
    curated_fraction: float,
) -> dict[str, Any]:
    source_dataset = str(record.pop("_mixed_source_dataset"))
    source_index = int(record.pop("_mixed_source_index"))
    original_profile = dict(record["profile"])
    original_meta = dict(record["generator_meta"])
    rewritten = deepcopy(record)

    rewritten["profile"]["profile_id"] = profile_id
    rewritten["profile"]["name"] = name
    for key in ("audit", "outcome"):
        if key in rewritten and isinstance(rewritten[key], dict):
            rewritten[key]["profile_id"] = profile_id
            rewritten[key]["name"] = name

    convoy_profile = str(original_meta.get("convoy_profile", "convoy_layout_1"))
    rewritten["generator_meta"] = {
        "mode": MIXED_DATASET_MODE,
        "source": MIXED_DATASET_SOURCE,
        "generator_version": MIXED_DATASET_VERSION,
        "seed": int(seed),
        "convoy_profile": convoy_profile,
        "split": split,
        "curated_fraction": float(curated_fraction),
        "random_fraction": float(1.0 - float(curated_fraction)),
        "source_dataset": source_dataset,
    }
    rewritten["mixture_meta"] = {
        "mixed_index": int(mixed_index),
        "source_dataset": source_dataset,
        "source_index": int(source_index),
        "source_profile_id": str(original_profile.get("profile_id", "")),
        "source_name": str(original_profile.get("name", "")),
        "source_mode": str(original_meta.get("mode", "")),
        "source_generator": str(original_meta.get("source", "")),
        "source_generator_version": str(original_meta.get("generator_version", "")),
    }
    return rewritten


def _count_labels(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        label = (
            record.get("intent", {}).get("intended_label")
            or record.get("audit", {}).get("suggested_label")
            or record.get("outcome", {}).get("actual_outcome_label")
            or ""
        )
        counts[str(label)] += 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_summary(path: Path, summary: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mixed_split(
    *,
    curated_records: Sequence[dict[str, Any]],
    random_records: Sequence[dict[str, Any]],
    total_count: int,
    curated_fraction: float,
    split: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 <= float(curated_fraction) <= 1.0:
        raise ValueError("curated_fraction must be between 0 and 1")
    curated_count = int(round(int(total_count) * float(curated_fraction)))
    random_count = int(total_count) - int(curated_count)
    rng = np.random.default_rng(int(seed))
    selected = []
    selected.extend(
        _sample_records(
            curated_records,
            count=curated_count,
            rng=rng,
            source_dataset="curated_v4",
            split=split,
        )
    )
    selected.extend(
        _sample_records(
            random_records,
            count=random_count,
            rng=rng,
            source_dataset="random_profile_v1",
            split=split,
        )
    )
    rng.shuffle(selected)

    prefix = "MIXTRN" if split == "train" else "MIXVAL"
    rewritten = [
        _rewrite_record(
            record,
            profile_id=f"{prefix}{idx + 1:06d}",
            name=f"{split}_mixed_curated70_random30_{idx + 1:06d}",
            split=split,
            mixed_index=idx,
            seed=seed,
            curated_fraction=curated_fraction,
        )
        for idx, record in enumerate(selected)
    ]
    source_counts = Counter(str(record["mixture_meta"]["source_dataset"]) for record in rewritten)
    summary = {
        "split": split,
        "total_count": int(len(rewritten)),
        "curated_fraction": float(curated_fraction),
        "source_counts": dict(sorted(source_counts.items())),
        "labels": _count_labels(rewritten),
    }
    return rewritten, summary


def main() -> int:
    args = parse_args()
    curated_train = load_attack_profile_dataset_jsonl(args.curated_train)
    random_train = load_attack_profile_dataset_jsonl(args.random_train)
    curated_valid = load_attack_profile_dataset_jsonl(args.curated_valid)
    random_valid = load_attack_profile_dataset_jsonl(args.random_valid)

    train_records, train_summary = build_mixed_split(
        curated_records=curated_train,
        random_records=random_train,
        total_count=int(args.train_count),
        curated_fraction=float(args.curated_fraction),
        split="train",
        seed=int(args.seed),
    )
    valid_records, valid_summary = build_mixed_split(
        curated_records=curated_valid,
        random_records=random_valid,
        total_count=int(args.valid_count),
        curated_fraction=float(args.curated_fraction),
        split="valid",
        seed=int(args.seed) + 1,
    )

    summary = {
        "workflow": MIXED_DATASET_SOURCE,
        "mode": MIXED_DATASET_MODE,
        "generator_version": MIXED_DATASET_VERSION,
        "seed": int(args.seed),
        "curated_fraction": float(args.curated_fraction),
        "random_fraction": float(1.0 - float(args.curated_fraction)),
        "inputs": {
            "curated_train": str(args.curated_train),
            "random_train": str(args.random_train),
            "curated_valid": str(args.curated_valid),
            "random_valid": str(args.random_valid),
        },
        "outputs": {
            "train": str(args.train_output),
            "valid": str(args.valid_output),
        },
        "splits": {
            "train": train_summary,
            "valid": valid_summary,
        },
    }
    _write_jsonl(args.train_output, train_records, overwrite=bool(args.overwrite))
    _write_jsonl(args.valid_output, valid_records, overwrite=bool(args.overwrite))
    _write_summary(args.train_output.with_suffix(args.train_output.suffix + ".summary.json"), summary, overwrite=True)
    _write_summary(args.valid_output.with_suffix(args.valid_output.suffix + ".summary.json"), summary, overwrite=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
