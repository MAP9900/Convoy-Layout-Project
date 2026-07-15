"""Small VAE and preprocessing helpers for synthetic attack profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from convoy_sim.attack_profiles import AttackProfile


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
    """Load and lightly validate attack-profile JSONL records."""

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
    """Convert one explicit profile payload into the agreed v1 feature vector."""

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
        dtype=np.float32,
    )
    if vector.shape != (len(FEATURE_NAMES),):
        raise ValueError("Unexpected feature vector shape")
    return vector


def load_attack_profile_features_jsonl(path: str | Path) -> np.ndarray:
    """Read the eight VAE features without retaining full JSONL records."""

    feature_rows: list[np.ndarray] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if "profile" not in row:
                raise ValueError(f"Line {line_number} of {path} is missing profile")
            feature_rows.append(_profile_to_feature_vector(dict(row["profile"])))

    if not feature_rows:
        raise ValueError(f"Dataset must be non-empty: {path}")
    return np.vstack(feature_rows).astype(np.float32)


@dataclass
class AttackProfileVAEPreprocessor:
    """Normalize VAE features and decode them back into attack profiles."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_features(cls, features: np.ndarray) -> "AttackProfileVAEPreprocessor":
        """Fit normalization statistics from an unnormalized feature matrix."""

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)})")
        if features.shape[0] == 0:
            raise ValueError("features must be non-empty")
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(feature_names=FEATURE_NAMES, mean=mean.astype(np.float32), std=std)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize an existing raw feature matrix."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"features must have shape (n, {len(self.feature_names)})")
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, normalized: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map normalized vectors back into the raw physical feature space."""

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
        weight: float = 1.0,
    ) -> dict[str, Any]:
        """Decode one vector and clamp fields to physically valid ranges."""

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
        """Decode one normalized vector directly into an ``AttackProfile``."""

        payload = self.decode_profile_fields(normalized, profile_id=profile_id, name=name, weight=weight)
        return AttackProfile(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted preprocessing statistics for checkpoint metadata."""

        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackProfileVAEPreprocessor":
        """Rehydrate a saved preprocessor payload from checkpoint metadata."""

        return cls(
            feature_names=tuple(str(x) for x in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


class AttackProfileVAE(nn.Module):
    """Two-layer MLP beta-VAE for the eight continuous profile features."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        latent_dim: int = 4,
        hidden_dim: int = 32,
    ) -> None:
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
        """Encode a batch into latent mean and log-variance tensors."""

        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors using the standard VAE reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back into normalized feature space."""

        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full forward pass: encode, sample, decode."""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples: int, *, device: torch.device | None = None) -> torch.Tensor:
        """Sample decoded feature vectors from a standard normal latent prior."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        parameter = next(self.parameters())
        sample_device = device or parameter.device
        z = torch.randn(int(n_samples), self.latent_dim, device=sample_device)
        return self.decode(z)

    @torch.no_grad()
    def sample_from_latent_bank(
        self,
        latent_bank: torch.Tensor,
        n_samples: int,
        *,
        noise_scale: float = 0.10,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode samples drawn near encoded training-set latent vectors."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if latent_bank.ndim != 2 or int(latent_bank.shape[1]) != int(self.latent_dim):
            raise ValueError("latent_bank must have shape (n, latent_dim)")
        if int(latent_bank.shape[0]) <= 0:
            raise ValueError("latent_bank must be non-empty")
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")

        parameter = next(self.parameters())
        sample_device = device or parameter.device
        bank = latent_bank.to(sample_device)
        indices = torch.randint(
            low=0,
            high=int(bank.shape[0]),
            size=(int(n_samples),),
            device=torch.device("cpu"),
            generator=generator,
        ).to(sample_device)
        z = bank[indices]
        if float(noise_scale) > 0.0:
            noise = torch.randn(
                z.shape,
                device=sample_device,
                dtype=z.dtype,
                generator=generator,
            )
            z = z + (float(noise_scale) * noise)
        return self.decode(z)


@torch.no_grad()
def build_latent_bank(
    model: AttackProfileVAE,
    features: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 4096,
    device: torch.device | str | None = None,
    use_mu: bool = True,
) -> torch.Tensor:
    """Encode normalized features into a bank of latent vectors."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(features, torch.Tensor):
        feature_tensor = features.detach().to(dtype=torch.float32)
    else:
        feature_tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    if feature_tensor.ndim != 2 or int(feature_tensor.shape[1]) != int(model.input_dim):
        raise ValueError("features must have shape (n, input_dim)")
    if int(feature_tensor.shape[0]) <= 0:
        raise ValueError("features must be non-empty")

    parameter = next(model.parameters())
    encode_device = torch.device(device) if device is not None else parameter.device
    was_training = bool(model.training)
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, int(feature_tensor.shape[0]), int(batch_size)):
        batch = feature_tensor[start : start + int(batch_size)].to(encode_device)
        mu, logvar = model.encode(batch)
        z = mu if use_mu else model.reparameterize(mu, logvar)
        chunks.append(z.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return MSE plus beta-weighted KL loss and scalar diagnostics."""

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


def load_vae_dataset(
    path: str | Path,
    preprocessor: AttackProfileVAEPreprocessor | None = None,
) -> tuple[torch.Tensor, AttackProfileVAEPreprocessor]:
    """Load one JSONL corpus as a normalized feature tensor."""

    features = load_attack_profile_features_jsonl(path)
    preprocessor = preprocessor or AttackProfileVAEPreprocessor.fit_features(features)
    tensor = torch.as_tensor(preprocessor.transform_features(features), dtype=torch.float32)
    return tensor, preprocessor
