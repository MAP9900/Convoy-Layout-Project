"""VAE utilities for synthetic U-boat attack-profile generation.

This module implements the agreed v1 setup:

- feature extraction from dataset-mode JSONL records
- train-set normalization / inverse transform
- a small MLP VAE for continuous geometry/timing features
- reconstruction + KL loss

The model intentionally learns only the continuous plausible-attack manifold for
now. Fixed doctrine / realism fields are reattached after decode.

"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from convoy_sim.attack_profiles import AttackProfile

"""
VAE Model for Diverse U-Boat Attack Profile Generation

Input Schema (continous variables):
    u_pos_x
    u_pos_y
    sin(base_bearing_rad)
    cos(base_bearing_rad)
    spread_rad
    launch_delay_s
    salvo_interval_s
    u_boat_initial_speed_mps

Fixed / omitted from learning for now:
    n = 4
    spread_doctrine = uniform_divergent
    u_boat_mode = moving
    sub_length_m
    sub_beam_m
    launch_from
    max_bow_offset_deg
    gyro_straight_run_m

"""


FEATURE_NAMES: tuple[str, ...] = (
    "u_pos_x",
    "u_pos_y",
    "sin_base_bearing_rad",
    "cos_base_bearing_rad",
    "spread_rad",
    "launch_delay_s",
    "salvo_interval_s",
    "u_boat_initial_speed_mps",
)

FIXED_PROFILE_FIELDS: dict[str, Any] = {
    "mode": "fan",
    "n": 4,
    "speed": 15.4333,
    "max_run_time": 486.0,
    "spread_doctrine": "uniform_divergent",
    "per_torpedo_heading_offsets_rad": (),
    "u_boat_mode": "moving",
    "sub_length_m": 67.0,
    "sub_beam_m": 6.5,
    "launch_from": "bow",
    "max_bow_offset_deg": 15.0,
    "gyro_straight_run_m": 10.0,
}


def load_attack_profile_dataset_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load dataset-mode JSONL records produced by generate_attack_profile_scaffold."""

    dataset_path = Path(path)
    records: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            if not {"profile", "audit", "generator_meta"} <= set(row):
                raise ValueError(f"Line {line_number} of {dataset_path} is missing required keys")
            records.append(row)
    return records


def _profile_to_feature_vector(profile: dict[str, Any]) -> np.ndarray:
    """Convert one profile payload into the v1 continuous feature vector."""

    bearing = float(profile["base_bearing_rad"])
    vector = np.array(
        [
            float(profile["u_pos"][0]),
            float(profile["u_pos"][1]),
            float(np.sin(bearing)),
            float(np.cos(bearing)),
            float(profile["spread_rad"]),
            float(profile["launch_delay_s"]),
            float(profile["salvo_interval_s"]),
            float(profile["u_boat_initial_speed_mps"]),
        ],
        dtype=np.float32,)
    if vector.shape != (len(FEATURE_NAMES),):
        raise ValueError("Unexpected feature vector shape")
    return vector


@dataclass
class AttackProfileVAEPreprocessor:
    """Feature extractor, normalizer, and decoder helper for v1 VAE training."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, records: Sequence[dict[str, Any]]) -> "AttackProfileVAEPreprocessor":
        if not records:
            raise ValueError("records must be non-empty")
        features = np.vstack([_profile_to_feature_vector(dict(record["profile"])) for record in records]).astype(
            np.float32
        )
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(feature_names=FEATURE_NAMES, mean=mean.astype(np.float32), std=std)

    def transform_profile_dict(self, profile: dict[str, Any]) -> np.ndarray:
        raw = _profile_to_feature_vector(profile)
        return ((raw - self.mean) / self.std).astype(np.float32)

    def transform_records(self, records: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.vstack([self.transform_profile_dict(dict(record["profile"])) for record in records]).astype(
            np.float32
        )

    def inverse_transform(self, normalized: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(normalized, torch.Tensor):
            values = normalized.detach().cpu().numpy()
        else:
            values = np.asarray(normalized, dtype=np.float32)
        return (values * self.std) + self.mean

    def decode_profile_fields(
        self,
        normalized: np.ndarray | torch.Tensor,
        *,
        profile_id: str,
        name: str,
        weight: float = 1.0,) -> dict[str, Any]:
        raw = self.inverse_transform(normalized)
        if raw.shape != (len(self.feature_names),):
            raise ValueError("Expected a single v1 feature vector")
        sin_bearing = float(raw[2])
        cos_bearing = float(raw[3])
        base_bearing_rad = float(atan2(sin_bearing, cos_bearing))
        spread_rad = float(max(raw[4], 0.0))
        launch_delay_s = float(max(raw[5], 0.0))
        salvo_interval_s = float(max(raw[6], 0.0))
        u_boat_initial_speed_mps = float(np.clip(raw[7], 1.0, 2.0))
        return {
            "profile_id": profile_id,
            "name": name,
            "weight": float(weight),
            "u_pos": (float(raw[0]), float(raw[1])),
            "base_bearing_rad": base_bearing_rad,
            "spread_rad": spread_rad,
            "launch_delay_s": launch_delay_s,
            "salvo_interval_s": salvo_interval_s,
            "u_boat_initial_heading_rad": base_bearing_rad,
            "u_boat_initial_speed_mps": u_boat_initial_speed_mps,
            **FIXED_PROFILE_FIELDS,
        }

    def decode_attack_profile(
        self,
        normalized: np.ndarray | torch.Tensor,
        *,
        profile_id: str,
        name: str,
        weight: float = 1.0,
    ) -> AttackProfile:
        payload = self.decode_profile_fields(normalized, profile_id=profile_id, name=name, weight=weight)
        return AttackProfile(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackProfileVAEPreprocessor":
        return cls(
            feature_names=tuple(str(x) for x in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


class AttackProfileVAEDataset(Dataset[torch.Tensor]):
    """Torch dataset wrapper for normalized attack-profile feature vectors."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        preprocessor: AttackProfileVAEPreprocessor) -> None:
        if not records:
            raise ValueError("records must be non-empty")
        self._records = list(records)
        self._preprocessor = preprocessor
        self._features = torch.as_tensor(preprocessor.transform_records(self._records), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._features[index]

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def preprocessor(self) -> AttackProfileVAEPreprocessor:
        return self._preprocessor


class AttackProfileVAE(nn.Module):
    """Small MLP VAE for the v1 continuous attack-profile feature space."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        latent_dim: int = 4,
        hidden_dim: int = 32,) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_head = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples: int, *, device: torch.device | None = None) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        parameter = next(self.parameters())
        sample_device = device or parameter.device
        z = torch.randn(int(n_samples), self.latent_dim, device=sample_device)
        return self.decode(z)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return beta-VAE loss and scalar diagnostics."""

    if recon_x.shape != x.shape:
        raise ValueError("recon_x and x must have matching shape")
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_per_item = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_per_item)
    total_loss = recon_loss + (float(beta) * kl_loss)
    stats = {
        "loss": float(total_loss.detach().cpu().item()),
        "recon_loss": float(recon_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "beta": float(beta),
    }
    return total_loss, stats


def build_vae_datasets(
    *,
    train_path: str | Path,
    valid_path: str | Path,
) -> tuple[AttackProfileVAEDataset, AttackProfileVAEDataset, AttackProfileVAEPreprocessor]:
    """Load JSONL corpora, fit train-set preprocessing, and return Torch datasets."""

    train_records = load_attack_profile_dataset_jsonl(train_path)
    valid_records = load_attack_profile_dataset_jsonl(valid_path)
    preprocessor = AttackProfileVAEPreprocessor.fit(train_records)
    train_ds = AttackProfileVAEDataset(train_records, preprocessor=preprocessor)
    valid_ds = AttackProfileVAEDataset(valid_records, preprocessor=preprocessor)
    return train_ds, valid_ds, preprocessor
