from __future__ import annotations

from pathlib import Path

from experiments.train_vae import parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.train_path == Path("data/attack_profiles/synthetic/train_random_v2.jsonl")
    assert args.valid_path == Path("data/attack_profiles/synthetic/valid_random_v2.jsonl")
    assert args.epochs == 50
    assert args.batch_size == 128
    assert args.learning_rate == 1e-3
    assert args.beta == 0.05
    assert args.device == "auto"
    assert args.sample_count == 16


def test_parse_args_overrides() -> None:
    args = parse_args(
        [
            "--train-path",
            "tmp/train.jsonl",
            "--valid-path",
            "tmp/valid.jsonl",
            "--epochs",
            "5",
            "--batch-size",
            "64",
            "--learning-rate",
            "0.0005",
            "--beta",
            "0.1",
            "--latent-dim",
            "6",
            "--hidden-dim",
            "48",
            "--device",
            "cpu",
            "--sample-count",
            "8",
        ]
    )
    assert args.train_path == Path("tmp/train.jsonl")
    assert args.valid_path == Path("tmp/valid.jsonl")
    assert args.epochs == 5
    assert args.batch_size == 64
    assert args.learning_rate == 0.0005
    assert args.beta == 0.1
    assert args.latent_dim == 6
    assert args.hidden_dim == 48
    assert args.device == "cpu"
    assert args.sample_count == 8
