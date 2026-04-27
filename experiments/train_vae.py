"""Train the first-pass VAE over synthetic attack-profile datasets.

This entrypoint is intentionally narrower than the canonical baseline/RL
workflows. It trains a small beta-VAE over the dataset-mode JSONL corpora and
emits the minimum artifacts needed to inspect learning behavior:

- resolved config snapshot
- training history CSV
- metrics summary JSON
- run manifest JSON
- best/latest checkpoints
- a small decoded sample file for manual inspection

Torch imports are kept local to the runtime path so this module can still be
imported in environments that do not have the ML dependencies installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from convoy_sim.workflows import ensure_dir, git_sha, resolve_run_dir, write_json, write_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v1 attack-profile VAE")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/attack_profiles/synthetic/train_random_v2.jsonl"),
        help="Path to the training JSONL corpus",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=Path("data/attack_profiles/synthetic/valid_random_v2.jsonl"),
        help="Path to the validation JSONL corpus",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/runs"),
        help="Root output directory; workflow artifacts go under output_root/vae",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Optional run suffix in the output directory name",
    )
    parser.add_argument("--seed", type=int, default=1945, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--beta", type=float, default=0.05, help="KL weight in the beta-VAE loss")
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Torch device selection",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="How many latent-prior samples to decode after training",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    if not stats_list:
        return {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "beta": 0.0}
    keys = stats_list[0].keys()
    return {key: float(np.mean([item[key] for item in stats_list], dtype=float)) for key in keys}


def _train_one_epoch(
    *,
    model: Any,
    data_loader: Any,
    optimizer: Any,
    device: str,
    beta: float,
) -> dict[str, float]:
    model.train()
    batch_stats: list[dict[str, float]] = []
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar = model(batch)
        loss, stats = model.loss_fn(recon, batch, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        batch_stats.append(stats)
    return _mean_stats(batch_stats)


def _eval_one_epoch(
    *,
    model: Any,
    data_loader: Any,
    device: str,
    beta: float,
) -> dict[str, float]:
    model.eval()
    batch_stats: list[dict[str, float]] = []
    with model.no_grad_ctx():
        for batch in data_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            _, stats = model.loss_fn(recon, batch, mu, logvar, beta=beta)
            batch_stats.append(stats)
    return _mean_stats(batch_stats)


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_recon_loss",
        "train_kl_loss",
        "valid_loss",
        "valid_recon_loss",
        "valid_kl_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def run_from_args(args: argparse.Namespace, *, project_root: Path | None = None) -> Path:
    try:
        import torch
        from torch.optim import Adam
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError(
            "train_vae.py requires torch; install the ML environment before running this script"
        ) from exc

    from convoy_sim.vae import AttackProfileVAE, build_vae_datasets, vae_loss

    project_root = project_root or Path(__file__).resolve().parents[1]
    output_root = project_root / str(args.output_root)
    run_dir = resolve_run_dir(output_root, "vae", args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    ensure_dir(checkpoint_dir)

    overall_start = time.perf_counter()
    _seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))

    data_start = time.perf_counter()
    train_ds, valid_ds, preprocessor = build_vae_datasets(
        train_path=project_root / args.train_path,
        valid_path=project_root / args.valid_path,
    )
    data_seconds = time.perf_counter() - data_start

    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(args.num_workers),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = AttackProfileVAE(
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    model.loss_fn = staticmethod(vae_loss)
    model.no_grad_ctx = torch.no_grad
    optimizer = Adam(model.parameters(), lr=float(args.learning_rate))

    history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    train_start = time.perf_counter()

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = _train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=float(args.beta),
        )
        valid_stats = _eval_one_epoch(
            model=model,
            data_loader=valid_loader,
            device=device,
            beta=float(args.beta),
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_stats["loss"]),
                "train_recon_loss": float(train_stats["recon_loss"]),
                "train_kl_loss": float(train_stats["kl_loss"]),
                "valid_loss": float(valid_stats["loss"]),
                "valid_recon_loss": float(valid_stats["recon_loss"]),
                "valid_kl_loss": float(valid_stats["kl_loss"]),
            }
        )
        if float(valid_stats["loss"]) < best_valid_loss:
            best_valid_loss = float(valid_stats["loss"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "preprocessor": preprocessor.to_dict(),
                    "hyperparameters": {
                        "latent_dim": int(args.latent_dim),
                        "hidden_dim": int(args.hidden_dim),
                        "learning_rate": float(args.learning_rate),
                        "batch_size": int(args.batch_size),
                        "beta": float(args.beta),
                    },
                },
                checkpoint_dir / "model_best.pt",
            )

    training_seconds = time.perf_counter() - train_start
    torch.save(
        {
            "epoch": int(args.epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "preprocessor": preprocessor.to_dict(),
            "hyperparameters": {
                "latent_dim": int(args.latent_dim),
                "hidden_dim": int(args.hidden_dim),
                "learning_rate": float(args.learning_rate),
                "batch_size": int(args.batch_size),
                "beta": float(args.beta),
            },
        },
        checkpoint_dir / "model_latest.pt",
    )

    sample_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        decoded = model.sample(int(args.sample_count), device=torch.device(device)).cpu().numpy()
    sample_payload = [
        preprocessor.decode_profile_fields(
            decoded[idx],
            profile_id=f"VAE_SAMPLE_{idx + 1:04d}",
            name=f"vae_sample_{idx + 1:04d}",
        )
        for idx in range(int(args.sample_count))
    ]
    sample_seconds = time.perf_counter() - sample_start

    _write_history_csv(run_dir / "training_history.csv", history)
    write_json(run_dir / "sampled_profiles.json", {"profiles": sample_payload})

    final_train = history[-1]
    best_row = history[best_epoch - 1]
    metrics_summary = {
        "dataset": {
            "train_samples": int(len(train_ds)),
            "valid_samples": int(len(valid_ds)),
            "input_dim": int(model.input_dim),
            "feature_names": list(preprocessor.feature_names),
        },
        "model": {
            "latent_dim": int(model.latent_dim),
            "hidden_dim": int(model.hidden_dim),
            "parameter_count": int(sum(int(p.numel()) for p in model.parameters())),
            "beta": float(args.beta),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "best_epoch": int(best_epoch),
            "final_train_loss": float(final_train["train_loss"]),
            "final_train_recon_loss": float(final_train["train_recon_loss"]),
            "final_train_kl_loss": float(final_train["train_kl_loss"]),
            "final_valid_loss": float(final_train["valid_loss"]),
            "final_valid_recon_loss": float(final_train["valid_recon_loss"]),
            "final_valid_kl_loss": float(final_train["valid_kl_loss"]),
            "best_valid_loss": float(best_row["valid_loss"]),
            "best_valid_recon_loss": float(best_row["valid_recon_loss"]),
            "best_valid_kl_loss": float(best_row["valid_kl_loss"]),
        },
        "samples": {
            "count": int(args.sample_count),
            "output_file": "sampled_profiles.json",
        },
        "timing": {
            "dataset_load_seconds": float(data_seconds),
            "training_seconds": float(training_seconds),
            "sample_decode_seconds": float(sample_seconds),
            "total_seconds": float(time.perf_counter() - overall_start),
        },
    }
    manifest = {
        "workflow": "vae_train",
        "git_sha": git_sha(project_root),
        "device": device,
        "seed": int(args.seed),
        "train_path": str(args.train_path),
        "valid_path": str(args.valid_path),
        "output_root": str(args.output_root),
        "run_name": str(args.run_name),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "beta": float(args.beta),
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "sample_count": int(args.sample_count),
        },
        "preprocessor": preprocessor.to_dict(),
        "artifacts": {
            "history_csv": "training_history.csv",
            "metrics_summary_json": "metrics_summary.json",
            "run_manifest_json": "run_manifest.json",
            "best_checkpoint": "checkpoints/model_best.pt",
            "latest_checkpoint": "checkpoints/model_latest.pt",
            "sampled_profiles_json": "sampled_profiles.json",
        },
    }
    resolved_cfg = {
        "dataset": {
            "train_path": str(args.train_path),
            "valid_path": str(args.valid_path),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "beta": float(args.beta),
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "device": device,
            "num_workers": int(args.num_workers),
            "sample_count": int(args.sample_count),
            "seed": int(args.seed),
        },
    }

    write_yaml(run_dir / "config_resolved.yaml", resolved_cfg)
    write_json(run_dir / "metrics_summary.json", metrics_summary)
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = run_from_args(args)
    print(f"VAE run completed: {run_dir}")


if __name__ == "__main__":
    main()
