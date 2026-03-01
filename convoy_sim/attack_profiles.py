"""Attack profile schema and sampling helpers for RL scenario randomization.

Design goal: keep profile fields aligned with attacker sampler names so each profile
can be read and edited without translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.entities import Ship, Torpedo
from convoy_sim.feasibility import AttackConstraints, Environment


SpreadMode = Literal["fan", "parallel"]


def _vec2_tuple(value: Sequence[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("u_pos must be a 2D vector")
    return (float(arr[0]), float(arr[1]))


@dataclass(frozen=True)
class AttackProfile:
    """Flat attack profile with parameter names matching the sim samplers.

    Common params:
    - mode, u_pos, n, speed, max_run_time

    fan mode params:
    - base_bearing_rad, spread_rad

    parallel mode params:
    - bearing_rad, lateral_spacing

    Timing params:
    - launch_delay_s: applied to first torpedo
    - salvo_interval_s: added per torpedo index (0, 1, 2, ...)
    """

    profile_id: str
    name: str
    weight: float = 1.0

    mode: SpreadMode = "fan"
    u_pos: tuple[float, float] = (-2000.0, 0.0)
    n: int = 4
    speed: float = 25.0
    max_run_time: float = 800.0

    base_bearing_rad: float = 0.0
    spread_rad: float = 0.0

    bearing_rad: float = 0.0
    lateral_spacing: float = 0.0

    launch_delay_s: float = 0.0
    salvo_interval_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_pos", _vec2_tuple(self.u_pos))
        if self.mode not in ("fan", "parallel"):
            raise ValueError("mode must be 'fan' or 'parallel'")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        if self.max_run_time <= 0.0:
            raise ValueError("max_run_time must be positive")
        if self.salvo_interval_s < 0.0:
            raise ValueError("salvo_interval_s must be non-negative")

    def build_torpedoes(
        self,
        rng: np.random.Generator,
        *,
        ships: list[Ship] | None = None,
        proposal_cfg: dict[str, Any] | None = None,
        constraints: AttackConstraints | None = None,
        env: Environment | None = None,
    ) -> list[Torpedo]:
        """Instantiate torpedoes from this profile using sim-native samplers."""

        if self.mode == "fan":
            torpedoes = fan_spread(
                u_pos=np.asarray(self.u_pos, dtype=float),
                base_bearing_rad=float(self.base_bearing_rad),
                n=int(self.n),
                spread_rad=float(self.spread_rad),
                speed=float(self.speed),
                max_run_time=float(self.max_run_time),
                ships=ships,
                proposal_cfg=proposal_cfg,
                constraints=constraints,
                env=env,
                rng=rng,
            )
        else:
            torpedoes = parallel_spread(
                u_pos=np.asarray(self.u_pos, dtype=float),
                bearing_rad=float(self.bearing_rad),
                n=int(self.n),
                lateral_spacing=float(self.lateral_spacing),
                speed=float(self.speed),
                max_run_time=float(self.max_run_time),
                ships=ships,
                proposal_cfg=proposal_cfg,
                constraints=constraints,
                env=env,
                rng=rng,
            )

        for idx, torpedo in enumerate(torpedoes):
            torpedo.launch_delay = float(self.launch_delay_s + idx * self.salvo_interval_s)
        return torpedoes

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "weight": float(self.weight),
            "mode": self.mode,
            "u_pos": [float(self.u_pos[0]), float(self.u_pos[1])],
            "n": int(self.n),
            "speed": float(self.speed),
            "max_run_time": float(self.max_run_time),
            "base_bearing_rad": float(self.base_bearing_rad),
            "spread_rad": float(self.spread_rad),
            "bearing_rad": float(self.bearing_rad),
            "lateral_spacing": float(self.lateral_spacing),
            "launch_delay_s": float(self.launch_delay_s),
            "salvo_interval_s": float(self.salvo_interval_s),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackProfile":
        return cls(
            profile_id=str(payload["profile_id"]),
            name=str(payload["name"]),
            weight=float(payload.get("weight", 1.0)),
            mode=payload.get("mode", "fan"),
            u_pos=tuple(payload.get("u_pos", (-2000.0, 0.0))),
            n=int(payload.get("n", 4)),
            speed=float(payload.get("speed", 25.0)),
            max_run_time=float(payload.get("max_run_time", 800.0)),
            base_bearing_rad=float(payload.get("base_bearing_rad", 0.0)),
            spread_rad=float(payload.get("spread_rad", 0.0)),
            bearing_rad=float(payload.get("bearing_rad", 0.0)),
            lateral_spacing=float(payload.get("lateral_spacing", 0.0)),
            launch_delay_s=float(payload.get("launch_delay_s", 0.0)),
            salvo_interval_s=float(payload.get("salvo_interval_s", 0.0)),
        )


@dataclass(frozen=True)
class AttackProfileLibrary:
    """Collection of profiles with weighted random sampling."""

    profiles: list[AttackProfile]

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        ids = [profile.profile_id for profile in self.profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("profile_id values must be unique")
        if any(profile.weight < 0.0 for profile in self.profiles):
            raise ValueError("profile weights must be non-negative")
        if all(profile.weight == 0.0 for profile in self.profiles):
            raise ValueError("at least one profile weight must be positive")

    def profile_ids(self) -> list[str]:
        return [profile.profile_id for profile in self.profiles]

    def sample_profile(self, rng: np.random.Generator) -> AttackProfile:
        """Sample one profile (intended for episode reset)."""

        weights = np.array([profile.weight for profile in self.profiles], dtype=float)
        probs = weights / float(np.sum(weights))
        index = int(rng.choice(len(self.profiles), p=probs))
        return self.profiles[index]

    def to_dict(self) -> dict[str, Any]:
        return {"profiles": [profile.to_dict() for profile in self.profiles]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackProfileLibrary":
        return cls(
            profiles=[AttackProfile.from_dict(item) for item in payload.get("profiles", [])],
        )


def make_placeholder_profile(index: int, *, weight: float = 1.0) -> AttackProfile:
    """Create an editable template profile with sim-native parameter names."""

    if index <= 0:
        raise ValueError("index must be positive")
    profile_id = f"P{index:02d}"
    name = f"profile_{index:02d}"
    return AttackProfile(profile_id=profile_id, name=name, weight=float(weight))


def make_placeholder_profile_library(n_profiles: int = 25) -> AttackProfileLibrary:
    """Create a starter library with ``n_profiles`` placeholder profiles."""

    if n_profiles <= 0:
        raise ValueError("n_profiles must be positive")
    profiles = [make_placeholder_profile(index=i + 1) for i in range(n_profiles)]
    return AttackProfileLibrary(profiles=profiles)


def build_scaffolded_attack_profile_library() -> AttackProfileLibrary:
    """Return 25 explicit profile stubs for manual curation.

    Edit each block directly. Field names intentionally match `fan_spread` and
    `parallel_spread` arguments used by the simulator.
    """

    profiles = [
        AttackProfile(profile_id="P01", name="profile_01", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P01)
        AttackProfile(profile_id="P02", name="profile_02", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P02)
        AttackProfile(profile_id="P03", name="profile_03", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P03)
        AttackProfile(profile_id="P04", name="profile_04", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P04)
        AttackProfile(profile_id="P05", name="profile_05", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P05)
        AttackProfile(profile_id="P06", name="profile_06", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P06)
        AttackProfile(profile_id="P07", name="profile_07", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P07)
        AttackProfile(profile_id="P08", name="profile_08", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P08)
        AttackProfile(profile_id="P09", name="profile_09", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P09)
        AttackProfile(profile_id="P10", name="profile_10", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P10)
        AttackProfile(profile_id="P11", name="profile_11", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P11)
        AttackProfile(profile_id="P12", name="profile_12", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P12)
        AttackProfile(profile_id="P13", name="profile_13", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P13)
        AttackProfile(profile_id="P14", name="profile_14", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P14)
        AttackProfile(profile_id="P15", name="profile_15", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P15)
        AttackProfile(profile_id="P16", name="profile_16", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P16)
        AttackProfile(profile_id="P17", name="profile_17", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P17)
        AttackProfile(profile_id="P18", name="profile_18", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P18)
        AttackProfile(profile_id="P19", name="profile_19", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P19)
        AttackProfile(profile_id="P20", name="profile_20", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P20)
        AttackProfile(profile_id="P21", name="profile_21", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P21)
        AttackProfile(profile_id="P22", name="profile_22", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P22)
        AttackProfile(profile_id="P23", name="profile_23", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P23)
        AttackProfile(profile_id="P24", name="profile_24", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P24)
        AttackProfile(profile_id="P25", name="profile_25", mode="fan", u_pos=(-2000.0, 0.0), n=4, speed=25.0, max_run_time=800.0, base_bearing_rad=0.0, spread_rad=0.0, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.0, salvo_interval_s=0.0),  # TODO(P25)
    ]
    return AttackProfileLibrary(profiles=profiles)


DEFAULT_ATTACK_PROFILE_LIBRARY = build_scaffolded_attack_profile_library()
