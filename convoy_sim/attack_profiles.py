"""Attack profile schema and sampling helpers for RL scenario randomization.

Design goal: keep profile fields aligned with attacker sampler names so each profile
can be read and edited without translation.
"""

#Imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Sequence
import numpy as np
from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.entities import Ship, Torpedo
from convoy_sim.feasibility import AttackConstraints, Environment
from convoy_sim.realism import AttackerObservationConfig, UBoatMotionPlan, build_attacker_observation


SpreadMode = Literal["fan", "parallel"]
SpreadDoctrine = Literal["longitudinal", "uniform_divergent", "explicit_divergent"]

_MIN_TORPEDO_COUNT = 1
_MAX_TORPEDO_COUNT = 10
_MIN_SPEED = 0.01
_MAX_SPEED = 50
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


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""

    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass(frozen=True)
class AttackProfile:
    """Flat attack profile with parameter names matching the sim samplers.

    Common params:
    - mode, u_pos, n, speed, max_run_time

    fan mode params:
    - base_bearing_rad
    - spread_doctrine
      - `uniform_divergent`: standard convoy doctrine; evenly spaced gyro offsets
      - `explicit_divergent`: authored per-torpedo gyro offsets
      - `longitudinal`: same final heading for every torpedo; rare/nonstandard for convoy attacks
    - spread_rad (total fan width for `uniform_divergent`)
    - per_torpedo_heading_offsets_rad (used by `explicit_divergent`)

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
    spread_doctrine: SpreadDoctrine = "uniform_divergent"
    per_torpedo_heading_offsets_rad: tuple[float, ...] = ()

    bearing_rad: float = 0.0
    lateral_spacing: float = 0.0

    launch_delay_s: float = 0.0
    salvo_interval_s: float = 0.0

    # V2 realism: moving U-boat is default, but firing remains a hold-course action
    # unless callers deliberately disable the steady-boat guardrails below.
    u_boat_mode: Literal["moving", "static"] = "moving"
    u_boat_initial_heading_rad: float = 0.0
    u_boat_initial_speed_mps: float = 2.0
    u_boat_launch_time_s: float = 0.0
    u_boat_turn_rate_limit_rad_s: float | None = None
    u_boat_accel_limit_mps2: float | None = None
    u_boat_motion_legs: tuple[tuple[float, float, float], ...] = ()
    sub_length_m: float = 67.0
    sub_beam_m: float = 6.5
    launch_from: Literal["bow", "center"] = "bow"
    max_bow_offset_deg: float = 15.0
    gyro_straight_run_m: float = 30.0
    require_stable_u_boat_during_salvo: bool = True
    max_u_boat_turn_rate_at_fire_rad_s: float = float(np.deg2rad(0.1))
    max_u_boat_heading_drift_during_salvo_deg: float = 0.25

    # Partial-observability noise (attacker-facing context only).
    obs_bearing_sigma_rad: float = 0.04
    obs_range_sigma_m: float = 120.0
    obs_heading_sigma_rad: float = 0.06
    obs_speed_sigma_mps: float = 0.5
    obs_contact_count_sigma: float = 0.4

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_pos", _vec2_tuple(self.u_pos))
        #Checks
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
        if self.u_boat_mode not in {"moving", "static"}:
            raise ValueError("u_boat_mode must be 'moving' or 'static'")
        if self.u_boat_launch_time_s < 0.0:
            raise ValueError("u_boat_launch_time_s must be >= 0")
        if self.sub_length_m <= 0.0:
            raise ValueError("sub_length_m must be > 0")
        if self.sub_beam_m <= 0.0:
            raise ValueError("sub_beam_m must be > 0")
        if self.launch_from not in {"bow", "center"}:
            raise ValueError("launch_from must be 'bow' or 'center'")
        if self.max_bow_offset_deg < 0.0:
            raise ValueError("max_bow_offset_deg must be >= 0")
        if self.gyro_straight_run_m < 0.0:
            raise ValueError("gyro_straight_run_m must be >= 0")
        if self.max_u_boat_turn_rate_at_fire_rad_s < 0.0:
            raise ValueError("max_u_boat_turn_rate_at_fire_rad_s must be >= 0")
        if self.max_u_boat_heading_drift_during_salvo_deg < 0.0:
            raise ValueError("max_u_boat_heading_drift_during_salvo_deg must be >= 0")
        if self.mode == "fan":
            if self.spread_doctrine not in {"longitudinal", "uniform_divergent", "explicit_divergent"}:
                raise ValueError("spread_doctrine must be 'longitudinal', 'uniform_divergent', or 'explicit_divergent'")
            if not np.isfinite(self.base_bearing_rad) or not _MIN_BEARING_RAD <= self.base_bearing_rad <= _MAX_BEARING_RAD:
                raise ValueError(
                    f"base_bearing_rad must be finite and in [{_MIN_BEARING_RAD}, {_MAX_BEARING_RAD}]"
                )
            if not np.isfinite(self.spread_rad) or not _MIN_SPREAD_RAD <= self.spread_rad <= _MAX_SPREAD_RAD:
                raise ValueError(f"spread_rad must be finite and in [{_MIN_SPREAD_RAD}, {_MAX_SPREAD_RAD}]")
            if any(not np.isfinite(float(offset)) for offset in self.per_torpedo_heading_offsets_rad):
                raise ValueError("per_torpedo_heading_offsets_rad entries must be finite")
            if self.spread_doctrine == "explicit_divergent":
                if not self.per_torpedo_heading_offsets_rad:
                    raise ValueError("explicit_divergent doctrine requires per_torpedo_heading_offsets_rad")
                if len(self.per_torpedo_heading_offsets_rad) != int(self.n):
                    raise ValueError("per_torpedo_heading_offsets_rad length must match n")
            elif self.per_torpedo_heading_offsets_rad:
                raise ValueError(
                    "per_torpedo_heading_offsets_rad is only valid with spread_doctrine='explicit_divergent'"
                )
        else:
            if not np.isfinite(self.bearing_rad) or not _MIN_BEARING_RAD <= self.bearing_rad <= _MAX_BEARING_RAD:
                raise ValueError(
                    f"bearing_rad must be finite and in [{_MIN_BEARING_RAD}, {_MAX_BEARING_RAD}]"
                )
            if not np.isfinite(self.lateral_spacing) or not _MIN_LATERAL_SPACING <= self.lateral_spacing <= _MAX_LATERAL_SPACING:
                raise ValueError(
                    f"lateral_spacing must be finite and in [{_MIN_LATERAL_SPACING}, {_MAX_LATERAL_SPACING}]"
                )
            if self.per_torpedo_heading_offsets_rad:
                raise ValueError("per_torpedo_heading_offsets_rad is only supported for fan mode")

    def requested_attack_bearing_rad(self) -> float:
        """Return the profile's intended shot axis in world coordinates."""

        if self.mode == "fan":
            return float(self.base_bearing_rad)
        return float(self.bearing_rad)

    def legacy_inferred_spread_doctrine(self) -> SpreadDoctrine:
        """Infer legacy doctrine semantics from historical `spread_rad` usage."""

        if self.mode != "fan":
            return "uniform_divergent"
        if float(self.spread_rad) == 0.0:
            return "longitudinal"
        return "uniform_divergent"

    def resolved_spread_doctrine(self) -> SpreadDoctrine:
        """Return the doctrine to apply for this profile."""

        if self.spread_doctrine == "uniform_divergent" and not self.per_torpedo_heading_offsets_rad:
            return self.legacy_inferred_spread_doctrine()
        return self.spread_doctrine

    def is_standard_convoy_doctrine(self) -> bool:
        """Return whether this doctrine is standard for convoy-attack authoring."""

        return self.resolved_spread_doctrine() in {"uniform_divergent", "explicit_divergent"}

    def doctrine_note(self) -> str:
        """Return a short authoring note for the resolved spread doctrine."""

        doctrine = self.resolved_spread_doctrine()
        if doctrine == "uniform_divergent":
            return "Standard convoy doctrine: evenly spaced gyro offsets with the submarine held steady."
        if doctrine == "explicit_divergent":
            return "Manual convoy doctrine: authored per-torpedo gyro offsets with the submarine held steady."
        return "Rare/nonstandard convoy doctrine: same final heading for every torpedo with staggered launch timing."

    def fan_heading_offsets_rad(self) -> list[float]:
        """Return final per-torpedo heading offsets for fan doctrine resolution."""

        n_torp = int(self.n)
        doctrine = self.resolved_spread_doctrine()
        if doctrine == "longitudinal":
            return [0.0 for _ in range(n_torp)]
        if doctrine == "explicit_divergent":
            return [float(offset) for offset in self.per_torpedo_heading_offsets_rad]
        requested_half_spread = float(self.spread_rad) * 0.5
        if n_torp == 1 or self.spread_rad == 0.0:
            return [0.0 for _ in range(n_torp)]
        return [
            float(-requested_half_spread + (float(self.spread_rad) * i / (n_torp - 1)))
            for i in range(n_torp)
        ]

    def uses_legacy_bearing_compat(self) -> bool:
        """Return whether the profile should auto-align the sub to legacy bearing intent.

        Older profiles encoded attack geometry primarily through `base_bearing_rad`
        / `bearing_rad` and often left the U-boat heading at the default `0.0`.
        Treat those profiles as legacy and derive a coherent heading from the
        intended attack axis unless explicit motion/heading data is present.
        """

        return (
            not self.u_boat_motion_legs
            and abs(_wrap_angle_rad(float(self.u_boat_initial_heading_rad))) <= 1e-12
        )

    def _enforce_firing_stability(self, motion_plan: UBoatMotionPlan, launch_times: Sequence[float]) -> None:
        """Reject salvo windows where the U-boat is still turning materially.

        By default every spread doctrine in the simulator assumes a steady firing
        platform during the salvo. Longitudinal doctrine changes only the final
        torpedo heading offsets; it does not imply a heading sweep or pivot turn.
        """

        if not self.require_stable_u_boat_during_salvo or not launch_times:
            return

        headings = [float(motion_plan.state_at(launch_t)[1]) for launch_t in launch_times]
        baseline = headings[0]
        max_drift_rad = float(np.deg2rad(self.max_u_boat_heading_drift_during_salvo_deg))
        for heading in headings[1:]:
            drift = abs(_wrap_angle_rad(heading - baseline))
            if drift > max_drift_rad + 1e-12:
                raise ValueError(
                    "U-boat heading changes during the firing window; all doctrines assume a steady firing course by default, so set a steadier motion plan or disable require_stable_u_boat_during_salvo"
                )

        probe_dt = 0.25
        for launch_t in launch_times:
            before_t = max(0.0, float(launch_t) - probe_dt)
            after_t = float(launch_t) + probe_dt
            heading_before = float(motion_plan.state_at(before_t)[1])
            heading_after = float(motion_plan.state_at(after_t)[1])
            turn_rate = abs(_wrap_angle_rad(heading_after - heading_before)) / max(after_t - before_t, 1e-9)
            if turn_rate > float(self.max_u_boat_turn_rate_at_fire_rad_s) + 1e-12:
                raise ValueError(
                    "U-boat is turning during torpedo launch; all doctrines assume a steady firing course by default, so set a steadier motion plan or disable require_stable_u_boat_during_salvo"
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

        requested_bearing_rad = self.requested_attack_bearing_rad()
        initial_heading_rad = float(self.u_boat_initial_heading_rad)
        if self.uses_legacy_bearing_compat():
            initial_heading_rad = requested_bearing_rad

        motion_plan = UBoatMotionPlan(
            initial_position=np.asarray(self.u_pos, dtype=float),
            initial_heading_rad=initial_heading_rad,
            initial_speed_mps=float(self.u_boat_initial_speed_mps),
            mode=str(self.u_boat_mode),
            legs=(),
            launch_time_s=float(self.u_boat_launch_time_s),
            turn_rate_limit_rad_s=self.u_boat_turn_rate_limit_rad_s,
            accel_limit_mps2=self.u_boat_accel_limit_mps2,
        )
        if self.u_boat_motion_legs:
            motion_payload = {
                "mode": self.u_boat_mode,
                "initial_position": list(self.u_pos),
                "initial_heading_rad": initial_heading_rad,
                "initial_speed_mps": self.u_boat_initial_speed_mps,
                "launch_time_s": self.u_boat_launch_time_s,
                "turn_rate_limit_rad_s": self.u_boat_turn_rate_limit_rad_s,
                "accel_limit_mps2": self.u_boat_accel_limit_mps2,
                "legs": [
                    {"duration_s": float(a), "heading_rad": float(b), "speed_mps": float(c)}
                    for (a, b, c) in self.u_boat_motion_legs
                ],
            }
            motion_plan = UBoatMotionPlan.from_dict(motion_payload, fallback_u_pos=self.u_pos)

        launch_pos, launch_heading_rad, _launch_speed = motion_plan.state_at(float(self.u_boat_launch_time_s))

        observed_context = None
        if ships is not None:
            obs_cfg = AttackerObservationConfig(
                bearing_sigma_rad=float(self.obs_bearing_sigma_rad),
                range_sigma_m=float(self.obs_range_sigma_m),
                heading_sigma_rad=float(self.obs_heading_sigma_rad),
                speed_sigma_mps=float(self.obs_speed_sigma_mps),
                contact_count_sigma=float(self.obs_contact_count_sigma),
            )
            observed_context = build_attacker_observation(
                ships=ships,
                u_boat_pos=np.asarray(launch_pos, dtype=float),
                env=env,
                rng=rng,
                cfg=obs_cfg,
            )

        n_torp = int(self.n)
        if n_torp <= 0:
            return []
        max_bow_offset_rad = float(np.deg2rad(self.max_bow_offset_deg))
        launch_times = [
            float(self.u_boat_launch_time_s + self.launch_delay_s + idx * self.salvo_interval_s)
            for idx in range(n_torp)
        ]
        self._enforce_firing_stability(motion_plan, launch_times)
        if self.mode == "fan":
            rel_offsets = self.fan_heading_offsets_rad()
        else:
            rel_offsets = [0.0 for _ in range(n_torp)]

        proposal_cfg_local = _proposal_with_observation(proposal_cfg, observed_context)
        if constraints is not None:
            # Run feasibility proposal once for compatibility, but launch geometry remains
            # constrained to bow-direction realism below.
            if self.mode == "fan":
                _ = fan_spread(
                    u_pos=np.asarray(launch_pos, dtype=float),
                    base_bearing_rad=float(requested_bearing_rad),
                    n=n_torp,
                    spread_rad=float(self.spread_rad),
                    speed=float(self.speed),
                    max_run_time=float(self.max_run_time),
                    ships=ships,
                    proposal_cfg=proposal_cfg_local,
                    constraints=constraints,
                    env=env,
                    rng=rng,
                )
            else:
                _ = parallel_spread(
                    u_pos=np.asarray(launch_pos, dtype=float),
                    bearing_rad=float(requested_bearing_rad),
                    n=n_torp,
                    lateral_spacing=float(self.lateral_spacing),
                    speed=float(self.speed),
                    max_run_time=float(self.max_run_time),
                    ships=ships,
                    proposal_cfg=proposal_cfg_local,
                    constraints=constraints,
                    env=env,
                    rng=rng,
                )

        torpedoes: list[Torpedo] = []
        for idx in range(n_torp):
            launch_t = launch_times[idx]
            center_pos, sub_heading, _sub_speed = motion_plan.state_at(launch_t)
            center_pos = np.asarray(center_pos, dtype=float)
            sub_heading = float(sub_heading)
            centerline_rel = _wrap_angle_rad(float(requested_bearing_rad) - sub_heading)

            if self.launch_from == "bow":
                bow_offset = np.asarray([np.cos(sub_heading), np.sin(sub_heading)], dtype=float) * float(self.sub_length_m * 0.5)
                launch_origin = center_pos + bow_offset
            else:
                launch_origin = center_pos

            if self.mode == "fan":
                if abs(centerline_rel) > max_bow_offset_rad + 1e-12:
                    raise ValueError(
                        "requested fan centerline exceeds bow tube arc limit; adjust sub heading/motion or widen max_bow_offset_deg"
                    )
                # Doctrine only changes the final gyro-selected course. The torpedo
                # still exits on the submarine's bow heading at this launch time.
                torp_heading = _wrap_angle_rad(requested_bearing_rad + rel_offsets[idx])
                torp_launch = launch_origin
                torp_id = f"F{idx + 1:02d}"
            else:
                if abs(centerline_rel) > max_bow_offset_rad + 1e-12:
                    raise ValueError(
                        "requested parallel bearing exceeds bow tube arc limit; adjust sub heading/motion or widen max_bow_offset_deg"
                    )
                torp_heading = _wrap_angle_rad(sub_heading + centerline_rel)
                # Keep parallel spacing by offsetting launch points perpendicular to heading.
                perp = np.asarray([-np.sin(sub_heading), np.cos(sub_heading)], dtype=float)
                center = (n_torp - 1) / 2.0
                torp_launch = launch_origin + perp * ((idx - center) * float(self.lateral_spacing))
                torp_id = f"P{idx + 1:02d}"

            torpedoes.append(
                Torpedo(
                    id=torp_id,
                    launch_position=np.asarray(torp_launch, dtype=float),
                    speed=float(self.speed),
                    heading_rad=float(torp_heading),
                    max_run_time=float(self.max_run_time),
                    launch_delay=float(launch_t),
                    launch_heading_rad=float(sub_heading),
                    gyro_turn_distance_m=float(self.gyro_straight_run_m),
                )
            )
        return torpedoes

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
            "u_boat_mode": self.u_boat_mode,
            "u_boat_initial_heading_rad": float(self.u_boat_initial_heading_rad),
            "u_boat_initial_speed_mps": float(self.u_boat_initial_speed_mps),
            "u_boat_launch_time_s": float(self.u_boat_launch_time_s),
            "u_boat_turn_rate_limit_rad_s": (
                None if self.u_boat_turn_rate_limit_rad_s is None else float(self.u_boat_turn_rate_limit_rad_s)
            ),
            "u_boat_accel_limit_mps2": (
                None if self.u_boat_accel_limit_mps2 is None else float(self.u_boat_accel_limit_mps2)
            ),
            "u_boat_motion_legs": [
                [float(a), float(b), float(c)] for (a, b, c) in self.u_boat_motion_legs
            ],
            "sub_length_m": float(self.sub_length_m),
            "sub_beam_m": float(self.sub_beam_m),
            "launch_from": self.launch_from,
            "max_bow_offset_deg": float(self.max_bow_offset_deg),
            "gyro_straight_run_m": float(self.gyro_straight_run_m),
            "require_stable_u_boat_during_salvo": bool(self.require_stable_u_boat_during_salvo),
            "max_u_boat_turn_rate_at_fire_rad_s": float(self.max_u_boat_turn_rate_at_fire_rad_s),
            "max_u_boat_heading_drift_during_salvo_deg": float(self.max_u_boat_heading_drift_during_salvo_deg),
            "obs_bearing_sigma_rad": float(self.obs_bearing_sigma_rad),
            "obs_range_sigma_m": float(self.obs_range_sigma_m),
            "obs_heading_sigma_rad": float(self.obs_heading_sigma_rad),
            "obs_speed_sigma_mps": float(self.obs_speed_sigma_mps),
            "obs_contact_count_sigma": float(self.obs_contact_count_sigma),
        }
        if self.mode == "fan":
            legacy_doctrine = self.legacy_inferred_spread_doctrine()
            if self.spread_doctrine != legacy_doctrine or self.per_torpedo_heading_offsets_rad:
                payload["spread_doctrine"] = self.spread_doctrine
            if self.per_torpedo_heading_offsets_rad:
                payload["per_torpedo_heading_offsets_rad"] = [
                    float(offset) for offset in self.per_torpedo_heading_offsets_rad
                ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackProfile":
        spread_rad = float(payload.get("spread_rad", 0.0))
        inferred_doctrine: SpreadDoctrine = "longitudinal" if spread_rad == 0.0 else "uniform_divergent"
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
            spread_rad=spread_rad,
            spread_doctrine=str(payload.get("spread_doctrine", inferred_doctrine)),
            per_torpedo_heading_offsets_rad=tuple(
                float(item) for item in payload.get("per_torpedo_heading_offsets_rad", [])
            ),
            bearing_rad=float(payload.get("bearing_rad", 0.0)),
            lateral_spacing=float(payload.get("lateral_spacing", 0.0)),
            launch_delay_s=float(payload.get("launch_delay_s", 0.0)),
            salvo_interval_s=float(payload.get("salvo_interval_s", 0.0)),
            u_boat_mode=str(payload.get("u_boat_mode", "moving")),
            u_boat_initial_heading_rad=float(payload.get("u_boat_initial_heading_rad", 0.0)),
            u_boat_initial_speed_mps=float(payload.get("u_boat_initial_speed_mps", 2.0)),
            u_boat_launch_time_s=float(payload.get("u_boat_launch_time_s", 0.0)),
            u_boat_turn_rate_limit_rad_s=(
                None
                if payload.get("u_boat_turn_rate_limit_rad_s") is None
                else float(payload.get("u_boat_turn_rate_limit_rad_s"))
            ),
            u_boat_accel_limit_mps2=(
                None
                if payload.get("u_boat_accel_limit_mps2") is None
                else float(payload.get("u_boat_accel_limit_mps2"))
            ),
            u_boat_motion_legs=tuple(
                (
                    float(item[0]),
                    float(item[1]),
                    float(item[2]),
                )
                for item in payload.get("u_boat_motion_legs", [])
            ),
            sub_length_m=float(payload.get("sub_length_m", 67.0)),
            sub_beam_m=float(payload.get("sub_beam_m", 6.5)),
            launch_from=str(payload.get("launch_from", "bow")),
            max_bow_offset_deg=float(payload.get("max_bow_offset_deg", 15.0)),
            gyro_straight_run_m=float(payload.get("gyro_straight_run_m", 30.0)),
            require_stable_u_boat_during_salvo=bool(payload.get("require_stable_u_boat_during_salvo", True)),
            max_u_boat_turn_rate_at_fire_rad_s=float(payload.get("max_u_boat_turn_rate_at_fire_rad_s", np.deg2rad(0.1))),
            max_u_boat_heading_drift_during_salvo_deg=float(payload.get("max_u_boat_heading_drift_during_salvo_deg", 0.25)),
            obs_bearing_sigma_rad=float(payload.get("obs_bearing_sigma_rad", 0.04)),
            obs_range_sigma_m=float(payload.get("obs_range_sigma_m", 120.0)),
            obs_heading_sigma_rad=float(payload.get("obs_heading_sigma_rad", 0.06)),
            obs_speed_sigma_mps=float(payload.get("obs_speed_sigma_mps", 0.5)),
            obs_contact_count_sigma=float(payload.get("obs_contact_count_sigma", 0.4)),
        )


def _proposal_with_observation(
    proposal_cfg: dict[str, Any] | None,
    observed_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if proposal_cfg is None and observed_context is None:
        return None
    cfg = dict(proposal_cfg or {})
    if observed_context is not None:
        metadata = dict(cfg.get("metadata", {}))
        metadata["attacker_observation"] = observed_context
        cfg["metadata"] = metadata
    return cfg


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
            profiles=[AttackProfile.from_dict(item) for item in payload.get("profiles", [])],)


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
    """Return 30 scaffolded convoy-attack profiles for manual curation.

    Torpedo Data: https://uboat.net/technical/torpedoes.htm

    DEFAULT TORPEDO: "T3a. Range was 7500m at 30 knots (preheated state - 4500m at 28 knots)"

    Parameter reference (for each AttackProfile):
    - profile_id: unique profile identifier (e.g., "P01").
    - name: Readable label.
    - weight: weighted sampling mass used by AttackProfileLibrary.sample_profile(...).

    - mode: spread model and required parameters. DEFAULT: fan
      - "fan": torpedoes share one launch point and fan by post-launch gyro deflection.
        Uses: base_bearing_rad + spread_doctrine.
        Standard convoy authoring should use `spread_doctrine="uniform_divergent"`.
      - "parallel": torpedoes launch from laterally offset positions with shared heading.
        Uses: bearing_rad + lateral_spacing.

    Common geometry / torpedo fields:
    - u_pos: launch origin in world coordinates (meters), as (x, y).
    - n: number of torpedoes in the salvo (count). DEFAULT: 4 for attack against convoy.
    - speed: torpedo speed (m/s). DEFAULT: 30 kts -> 15.4333 m/s.
    - max_run_time: max torpedo run endurance after launch (seconds). DEFAULT: 7500/15.4333 -> 486 s.

    fan mode fields:
    - base_bearing_rad: center attack bearing (radians).
    - spread_doctrine: fan doctrine label.
    - spread_rad: total fan width in radians for `uniform_divergent`.
        DEFAULT: 4 deg = 0.0698 rad, 5 deg = 0.0873 rad, 6 deg = 0.1047 rad.
    - per_torpedo_heading_offsets_rad: explicit per-shot gyro offsets (radians) for `explicit_divergent`.
    - `longitudinal` remains supported but is best treated as a rare/nonstandard convoy doctrine.

    parallel mode fields:
    - bearing_rad: shared launch bearing (radians).
    - lateral_spacing: spacing between adjacent parallel launch tracks (meters).

    Timing fields:
    - launch_delay_s: delay before first torpedo launch (seconds). DEFAULT: 0.5-2.5 s.
    - salvo_interval_s: delay increment between torpedoes (seconds). Typically 2-3 s.

    U-boat motion / launch realism fields:
    - u_boat_mode: firing platform mode (`moving` or `static`).
    - u_boat_initial_heading_rad: initial U-boat heading at motion-plan start (radians).
    - u_boat_initial_speed_mps: initial U-boat speed at motion-plan start (m/s).
    - u_boat_launch_time_s: reference time when the firing solution begins (seconds).
    - u_boat_turn_rate_limit_rad_s: optional U-boat turn-rate bound (rad/s).
    - u_boat_accel_limit_mps2: optional U-boat acceleration bound (m/s^2).
    - u_boat_motion_legs: optional list of `(duration_s, heading_rad, speed_mps)` tuples.

    Tube / hull realism fields:
    - sub_length_m: submarine length (meters).
    - sub_beam_m: submarine beam (meters).
    - launch_from: launch origin on submarine (`bow` or `center`).
    - max_bow_offset_deg: max bow tube centerline offset allowed at fire time (degrees).
    - gyro_straight_run_m: straight tube-exit run before the torpedo takes its gyro angle (meters).

    Firing stability guardrail fields:
    - require_stable_u_boat_during_salvo: reject turning salvos when True.
    - max_u_boat_turn_rate_at_fire_rad_s: max allowed turn rate during firing (rad/s).
    - max_u_boat_heading_drift_during_salvo_deg: max allowed heading drift across the salvo (degrees).

    Observation-noise fields:
    - obs_bearing_sigma_rad: bearing observation sigma (radians).
    - obs_range_sigma_m: range observation sigma (meters).
    - obs_heading_sigma_rad: target-heading observation sigma (radians).
    - obs_speed_sigma_mps: target-speed observation sigma (m/s).
    - obs_contact_count_sigma: contact-count observation sigma (count units).
    """

    def _scaffolded_fan_profile(
        *,
        profile_id: str,
        name: str,
        u_pos: tuple[float, float],
        base_bearing_rad: float,
        spread_rad: float,
        launch_delay_s: float,
        salvo_interval_s: float,
        u_boat_initial_speed_mps: float,
    ) -> AttackProfile:
        return AttackProfile(
            profile_id=profile_id,
            name=name,
            mode="fan",
            u_pos=u_pos,
            n=4,
            speed=15.4333,
            max_run_time=486.0,
            base_bearing_rad=base_bearing_rad,
            spread_doctrine="uniform_divergent",
            spread_rad=spread_rad,
            bearing_rad=0.0,
            lateral_spacing=120.0,
            launch_delay_s=launch_delay_s,
            salvo_interval_s=salvo_interval_s,
            u_boat_mode="moving",
            u_boat_initial_heading_rad=base_bearing_rad,
            u_boat_initial_speed_mps=u_boat_initial_speed_mps,
            sub_length_m=67.0,
            sub_beam_m=6.5,
            launch_from="bow",
            max_bow_offset_deg=15.0,
            gyro_straight_run_m=10.0,
        )

    profiles = [
        _scaffolded_fan_profile(profile_id="P01", name="profile_01", u_pos=(2600.0, 3200.0), base_bearing_rad=4.0400, spread_rad=0.0698, launch_delay_s=0.5, salvo_interval_s=2.0, u_boat_initial_speed_mps=2.0),
        _scaffolded_fan_profile(profile_id="P02", name="profile_02", u_pos=(3000.0, 2600.0), base_bearing_rad=3.8600, spread_rad=0.1047, launch_delay_s=0.7, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.5),
        _scaffolded_fan_profile(profile_id="P03", name="profile_03", u_pos=(2200.0, 3500.0), base_bearing_rad=4.1500, spread_rad=0.0873, launch_delay_s=0.9, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.6),
        _scaffolded_fan_profile(profile_id="P04", name="profile_04", u_pos=(2800.0, 1800.0), base_bearing_rad=3.7100, spread_rad=0.0873, launch_delay_s=1.5, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.9),
        _scaffolded_fan_profile(profile_id="P05", name="profile_05", u_pos=(2400.0, 2500.0), base_bearing_rad=3.9500, spread_rad=0.1047, launch_delay_s=0.5, salvo_interval_s=3.0, u_boat_initial_speed_mps=2.0),
        _scaffolded_fan_profile(profile_id="P06", name="profile_06", u_pos=(3200.0, 1400.0), base_bearing_rad=3.5500, spread_rad=0.0698, launch_delay_s=1.0, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.1),
        _scaffolded_fan_profile(profile_id="P07", name="profile_07", u_pos=(2000.0, 3000.0), base_bearing_rad=4.1200, spread_rad=0.0873, launch_delay_s=2.0, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.3),
        _scaffolded_fan_profile(profile_id="P08", name="profile_08", u_pos=(2600.0, -3200.0), base_bearing_rad=2.2400, spread_rad=0.0873, launch_delay_s=2.2, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.7),
        _scaffolded_fan_profile(profile_id="P09", name="profile_09", u_pos=(3000.0, -2600.0), base_bearing_rad=2.4300, spread_rad=0.1047, launch_delay_s=2.3, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.5),
        _scaffolded_fan_profile(profile_id="P10", name="profile_10", u_pos=(2200.0, -3500.0), base_bearing_rad=2.1400, spread_rad=0.0873, launch_delay_s=1.1, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.8),
        _scaffolded_fan_profile(profile_id="P11", name="profile_11", u_pos=(2800.0, -1800.0), base_bearing_rad=2.5700, spread_rad=0.10470, launch_delay_s=1.7, salvo_interval_s=2.0, u_boat_initial_speed_mps=2.0),
        _scaffolded_fan_profile(profile_id="P12", name="profile_12", u_pos=(2400.0, -2500.0), base_bearing_rad=2.3400, spread_rad=0.1047, launch_delay_s=0.8, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.0),
        _scaffolded_fan_profile(profile_id="P13", name="profile_13", u_pos=(3200.0, -1400.0), base_bearing_rad=2.7300, spread_rad=0.0873, launch_delay_s=1.7, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.4),
        _scaffolded_fan_profile(profile_id="P14", name="profile_14", u_pos=(-300.0, 700.0), base_bearing_rad=5.1173, spread_rad=0.0873, launch_delay_s=1.3, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.9),
        _scaffolded_fan_profile(profile_id="P15", name="profile_15", u_pos=(100.0, -900.0), base_bearing_rad=1.6815, spread_rad=0.0873, launch_delay_s=2.4, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.8),
        _scaffolded_fan_profile(profile_id="P16", name="profile_16", u_pos=(400.0, 900.0), base_bearing_rad=4.2942, spread_rad=0.0698, launch_delay_s=0.6, salvo_interval_s=0.0, u_boat_initial_speed_mps=1.7),
        _scaffolded_fan_profile(profile_id="P17", name="profile_17", u_pos=(-500.0, -1900.0), base_bearing_rad=1.309, spread_rad=0.1047, launch_delay_s=1.2, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.0),
        _scaffolded_fan_profile(profile_id="P18", name="profile_18", u_pos=(250.0, -2500.0), base_bearing_rad=1.6705, spread_rad=0.0873, launch_delay_s=1.0, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.7),
        _scaffolded_fan_profile(profile_id="P19", name="profile_19", u_pos=(-150.0, 1900.0), base_bearing_rad=4.88692, spread_rad=0.0873, launch_delay_s=2.5, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.4),
        _scaffolded_fan_profile(profile_id="P20", name="profile_20", u_pos=(350.0, -2100.0), base_bearing_rad=1.7360, spread_rad=0.0698, launch_delay_s=1.9, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.8),
        _scaffolded_fan_profile(profile_id="P21", name="profile_21", u_pos=(0.0, 1100.0), base_bearing_rad=4.7124, spread_rad=0.1047, launch_delay_s=1.4, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.8),
        _scaffolded_fan_profile(profile_id="P22", name="profile_22", u_pos=(-2800.0, 200.0), base_bearing_rad=6.19592, spread_rad=0.1047, launch_delay_s=0.9, salvo_interval_s=3.0, u_boat_initial_speed_mps=2.0),
        _scaffolded_fan_profile(profile_id="P23", name="profile_23", u_pos=(-3200.0, 900.0), base_bearing_rad=6.0090, spread_rad=0.0873, launch_delay_s=0.5, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.6),
        _scaffolded_fan_profile(profile_id="P24", name="profile_24", u_pos=(-3000.0, -900.0), base_bearing_rad=0.261799, spread_rad=0.08738, launch_delay_s=1.9, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.2),
        _scaffolded_fan_profile(profile_id="P25", name="profile_25", u_pos=(3600.0, -1100.0), base_bearing_rad=2.8450, spread_rad=0.0698, launch_delay_s=2.0, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.8),
        # Intentional near-miss variants (plausible geometry with moderate bearing offset). (Hits Can Still Occur!)
        _scaffolded_fan_profile(profile_id="P26", name="profile_26", u_pos=(350.0, -2100.0), base_bearing_rad=1.8931, spread_rad=0.2443, launch_delay_s=1.9, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.6),
        _scaffolded_fan_profile(profile_id="P27", name="profile_27", u_pos=(0.0, 1100.0), base_bearing_rad=4.8695, spread_rad=0.2443, launch_delay_s=1.4, salvo_interval_s=2.0, u_boat_initial_speed_mps=2.0),
        _scaffolded_fan_profile(profile_id="P28", name="profile_28", u_pos=(-2800.0, 200.0), base_bearing_rad=0.0858, spread_rad=0.2443, launch_delay_s=0.9, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.9),
        _scaffolded_fan_profile(profile_id="P29", name="profile_29", u_pos=(-3200.0, 900.0), base_bearing_rad=6.1661, spread_rad=0.2443, launch_delay_s=0.5, salvo_interval_s=3.0, u_boat_initial_speed_mps=1.0),
        _scaffolded_fan_profile(profile_id="P30", name="profile_30", u_pos=(-3000.0, -900.0), base_bearing_rad=0.4485, spread_rad=0.2443, launch_delay_s=1.9, salvo_interval_s=2.0, u_boat_initial_speed_mps=1.2),

    ]
    return AttackProfileLibrary(profiles=profiles)


DEFAULT_ATTACK_PROFILE_LIBRARY = build_scaffolded_attack_profile_library()


