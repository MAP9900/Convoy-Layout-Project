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
