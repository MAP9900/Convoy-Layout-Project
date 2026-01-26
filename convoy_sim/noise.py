"""Optional noise toggles for Phase 2 realism experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseModel:
    """Container for optional torpedo aim/timing/dud parameters."""

    sigma_heading_rad: float = 0.0
    sigma_launch_delay: float = 0.0
    p_dud: float = 0.0

    def is_inactive(self) -> bool:
        return (
            self.sigma_heading_rad == 0.0
            and self.sigma_launch_delay == 0.0
            and self.p_dud == 0.0
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dict of noise parameters."""

        return {
            "sigma_heading_rad": float(self.sigma_heading_rad),
            "sigma_launch_delay": float(self.sigma_launch_delay),
            "p_dud": float(self.p_dud),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "NoiseModel":
        """Create a NoiseModel from a dict payload."""

        return cls(
            sigma_heading_rad=float(payload.get("sigma_heading_rad", 0.0)),
            sigma_launch_delay=float(payload.get("sigma_launch_delay", 0.0)),
            p_dud=float(payload.get("p_dud", 0.0)),
        )
