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

_MIN_TORPEDO_COUNT = 1
_MAX_TORPEDO_COUNT = 32
_MIN_SPEED = 0.01
_MAX_SPEED = 200.0
_MIN_MAX_RUN_TIME = 0.01
_MAX_MAX_RUN_TIME = 7200.0
_MIN_BEARING_RAD = -2.0 * np.pi
_MAX_BEARING_RAD = 2.0 * np.pi
_MIN_SPREAD_RAD = 0.0
_MAX_SPREAD_RAD = np.pi
_MIN_LATERAL_SPACING = 0.0
_MAX_LATERAL_SPACING = 5000.0
_MIN_LAUNCH_DELAY = 0.0
_MAX_LAUNCH_DELAY = 7200.0
_MIN_SALVO_INTERVAL = 0.0
_MAX_SALVO_INTERVAL = 3600.0


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
        if not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not _MIN_TORPEDO_COUNT <= self.n <= _MAX_TORPEDO_COUNT:
            raise ValueError(f"n must be in [{_MIN_TORPEDO_COUNT}, {_MAX_TORPEDO_COUNT}]")
        if not np.isfinite(self.speed) or not _MIN_SPEED <= self.speed <= _MAX_SPEED:
            raise ValueError(f"speed must be finite and in [{_MIN_SPEED}, {_MAX_SPEED}]")
        if not np.isfinite(self.max_run_time) or not _MIN_MAX_RUN_TIME <= self.max_run_time <= _MAX_MAX_RUN_TIME:
            raise ValueError(
                f"max_run_time must be finite and in [{_MIN_MAX_RUN_TIME}, {_MAX_MAX_RUN_TIME}]"
            )
        if not np.isfinite(self.launch_delay_s) or not _MIN_LAUNCH_DELAY <= self.launch_delay_s <= _MAX_LAUNCH_DELAY:
            raise ValueError(
                f"launch_delay_s must be finite and in [{_MIN_LAUNCH_DELAY}, {_MAX_LAUNCH_DELAY}]"
            )
        if not np.isfinite(self.salvo_interval_s) or not _MIN_SALVO_INTERVAL <= self.salvo_interval_s <= _MAX_SALVO_INTERVAL:
            raise ValueError(
                f"salvo_interval_s must be finite and in [{_MIN_SALVO_INTERVAL}, {_MAX_SALVO_INTERVAL}]"
            )
        if self.mode == "fan":
            if not np.isfinite(self.base_bearing_rad) or not _MIN_BEARING_RAD <= self.base_bearing_rad <= _MAX_BEARING_RAD:
                raise ValueError(
                    f"base_bearing_rad must be finite and in [{_MIN_BEARING_RAD}, {_MAX_BEARING_RAD}]"
                )
            if not np.isfinite(self.spread_rad) or not _MIN_SPREAD_RAD <= self.spread_rad <= _MAX_SPREAD_RAD:
                raise ValueError(f"spread_rad must be finite and in [{_MIN_SPREAD_RAD}, {_MAX_SPREAD_RAD}]")
        else:
            if not np.isfinite(self.bearing_rad) or not _MIN_BEARING_RAD <= self.bearing_rad <= _MAX_BEARING_RAD:
                raise ValueError(
                    f"bearing_rad must be finite and in [{_MIN_BEARING_RAD}, {_MAX_BEARING_RAD}]"
                )
            if not np.isfinite(self.lateral_spacing) or not _MIN_LATERAL_SPACING <= self.lateral_spacing <= _MAX_LATERAL_SPACING:
                raise ValueError(
                    f"lateral_spacing must be finite and in [{_MIN_LATERAL_SPACING}, {_MAX_LATERAL_SPACING}]"
                )

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

    Torpedo Data: https://uboat.net/technical/torpedoes.htm

    DEFAULT TORPEDO: "T3a. Range was 7500m at 30 knots (preheated state - 4500m at 28 knots)"

    Parameter reference (for each AttackProfile):
    - profile_id: unique profile identifier (e.g., "P01").
    - name: Readable label.
    - weight: weighted sampling mass used by AttackProfileLibrary.sample_profile(...).

    - mode: spread model and required parameters. DEFAULT: fan
      - "fan": torpedoes launch from one point and fan by bearing offsets.
        Uses: base_bearing_rad + spread_rad.
      - "parallel": torpedoes launch from laterally offset positions with shared heading.
        Uses: bearing_rad + lateral_spacing.

    - u_pos: launch origin in world coordinates (meters), as (x, y).
    - n: number of torpedoes in the salvo. DEFAULT: 4 for attack against convoy (4 stern tubes)
    - speed: torpedo speed (m/s). DEFAULT: 30kts -> 15.4333 m/s
    - max_run_time: max torpedo run horizon (seconds).  DEFAULT: 7500/15.4333 -> 486 seconds

    fan mode fields:
    - base_bearing_rad: center launch bearing in radians. (Centered around base_bearing_rad) 
    - spread_rad: total fan width in radians.
        DEFAULT: 4° = 0.0698 rad, 5° = 0.0873 rad, 6° = 0.1047 rad

    parallel mode fields:
    - bearing_rad: shared launch bearing in radians.
    - lateral_spacing: spacing between adjacent parallel launch tracks (meters).

    timing fields:
    - launch_delay_s: delay before first torpedo launch (seconds). DEFAULT: 0.5-2.5 s
    - salvo_interval_s: delay increment between torpedoes (seconds). Typically 2-3
    """

    profiles = [
        AttackProfile(profile_id="P01", name="profile_01", mode="fan", u_pos=(2600.0, 3200.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=4.0400, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.5, salvo_interval_s=2.0),
        AttackProfile(profile_id="P02", name="profile_02", mode="fan", u_pos=(3000.0, 2600.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=3.8600, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.7, salvo_interval_s=2.0),
        AttackProfile(profile_id="P03", name="profile_03", mode="fan", u_pos=(2200.0, 3500.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=4.1500, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.9, salvo_interval_s=3.0),
        AttackProfile(profile_id="P04", name="profile_04", mode="fan", u_pos=(2800.0, 1800.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=3.7100, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.5, salvo_interval_s=2.0),
        AttackProfile(profile_id="P05", name="profile_05", mode="fan", u_pos=(2400.0, 2500.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=3.9500, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.5, salvo_interval_s=3.0),
        AttackProfile(profile_id="P06", name="profile_06", mode="fan", u_pos=(3200.0, 1400.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=3.5500, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.0, salvo_interval_s=3.0),
        AttackProfile(profile_id="P07", name="profile_07", mode="fan", u_pos=(2000.0, 3000.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=4.1200, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.0, salvo_interval_s=3.0),
        AttackProfile(profile_id="P08", name="profile_08", mode="fan", u_pos=(2600.0, -3200.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.2400, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.2, salvo_interval_s=2.0),
        AttackProfile(profile_id="P09", name="profile_09", mode="fan", u_pos=(3000.0, -2600.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.4300, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.3, salvo_interval_s=3.0),
        AttackProfile(profile_id="P10", name="profile_10", mode="fan", u_pos=(2200.0, -3500.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.1400, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.1, salvo_interval_s=2.0),
        AttackProfile(profile_id="P11", name="profile_11", mode="fan", u_pos=(2800.0, -1800.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.5700, spread_rad=0.10470, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.7, salvo_interval_s=2.0),
        AttackProfile(profile_id="P12", name="profile_12", mode="fan", u_pos=(2400.0, -2500.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.3400, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.8, salvo_interval_s=2.0),
        AttackProfile(profile_id="P13", name="profile_13", mode="fan", u_pos=(3200.0, -1400.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=2.7300, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.7, salvo_interval_s=3.0),
        # GEOMETRY_AUDIT_FLAGGED (implausible_geometry): P14, P15, P16, P17, P18, P19, P20, P21, P23, P24
        # Keep visible for manual correction; avoid using these as held-out eval profiles until corrected.
        AttackProfile(profile_id="P14", name="profile_14", mode="fan", u_pos=(-300.0, 700.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.1500, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.3, salvo_interval_s=2.0),
        AttackProfile(profile_id="P15", name="profile_15", mode="fan", u_pos=(100.0, -900.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=-0.0500, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.4, salvo_interval_s=3.0),
        AttackProfile(profile_id="P16", name="profile_16", mode="fan", u_pos=(400.0, 1300.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.2200, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.6, salvo_interval_s=0.0),
        AttackProfile(profile_id="P17", name="profile_17", mode="fan", u_pos=(-500.0, -1500.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=-0.1800, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.2, salvo_interval_s=2.0),
        AttackProfile(profile_id="P18", name="profile_18", mode="fan", u_pos=(250.0, 0.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.0000, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.0, salvo_interval_s=2.0),
        AttackProfile(profile_id="P19", name="profile_19", mode="fan", u_pos=(-150.0, 1900.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.2800, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.5, salvo_interval_s=2.0),
        AttackProfile(profile_id="P20", name="profile_20", mode="fan", u_pos=(350.0, -2100.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=-0.2400, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.9, salvo_interval_s=2.0),
        AttackProfile(profile_id="P21", name="profile_21", mode="fan", u_pos=(0.0, 1100.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.1000, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.4, salvo_interval_s=2.0),
        AttackProfile(profile_id="P22", name="profile_22", mode="fan", u_pos=(-2800.0, 0.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.0000, spread_rad=0.1047, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.9, salvo_interval_s=3.0),
        # GEOMETRY_AUDIT_FLAGGED (implausible_geometry): fan_not_covering_target (lower severity than near-range opposite-direction group)
        AttackProfile(profile_id="P23", name="profile_23", mode="fan", u_pos=(-3200.0, 900.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=-0.0600, spread_rad=0.0873, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=0.5, salvo_interval_s=3.0),
        AttackProfile(profile_id="P24", name="profile_24", mode="fan", u_pos=(-3000.0, -900.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=0.0600, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=1.9, salvo_interval_s=2.0),
        AttackProfile(profile_id="P25", name="profile_25", mode="fan", u_pos=(3600.0, 0.0), n=4, speed=15.4333, max_run_time=486.0, base_bearing_rad=3.1416, spread_rad=0.0698, bearing_rad=0.0, lateral_spacing=120.0, launch_delay_s=2.0, salvo_interval_s=3.0),
    ]
    return AttackProfileLibrary(profiles=profiles)


DEFAULT_ATTACK_PROFILE_LIBRARY = build_scaffolded_attack_profile_library()
