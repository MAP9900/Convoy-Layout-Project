"""Deterministic attacker-side fire-control baseline helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from convoy_sim.attack_profiles import AttackProfile


SpeedSettingName = Literal["fast", "medium", "long_range"]
DoctrineMode = Literal["night_surface", "day_submerged"]


def _wrap_angle_rad(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass(frozen=True)
class G7aSpeedSetting:
    """Coarse G7a speed/range pairing used by the baseline doctrine."""

    name: SpeedSettingName
    speed_kts: float
    speed_mps: float
    nominal_range_m: float

    @property
    def max_run_time_s(self) -> float:
        return float(self.nominal_range_m / self.speed_mps)


@dataclass(frozen=True)
class FireControlLiteConfig:
    """Deterministic coarse fire-control solution config."""

    doctrine_mode: DoctrineMode = "night_surface"
    desired_salvo_size: int = 4
    base_spread_rad: float = float(np.deg2rad(4.0))
    max_spread_rad: float = float(np.deg2rad(16.0))
    max_intercept_lead_rad: float = float(np.deg2rad(12.0))
    uncertainty_spread_gain: float = 1.6
    range_spread_gain_rad: float = float(np.deg2rad(4.0))
    max_bow_centerline_offset_deg: float = 15.0
    night_surface_speed_multiplier: float = 1.0
    day_submerged_speed_multiplier: float = 0.85
    fast_setting_max_range_m: float = 1600.0
    medium_setting_max_range_m: float = 3200.0

    def speed_multiplier(self) -> float:
        if self.doctrine_mode == "day_submerged":
            return float(self.day_submerged_speed_multiplier)
        return float(self.night_surface_speed_multiplier)


@dataclass(frozen=True)
class FireControlLiteSolution:
    """Resolved attacker-side firing solution."""

    centerline_bearing_rad: float
    spread_rad: float
    salvo_size: int
    target_point_est_m: np.ndarray
    selected_speed_setting: SpeedSettingName
    torpedo_speed_mps: float
    torpedo_max_run_time_s: float
    estimated_range_m: float
    estimated_convoy_speed_mps: float
    estimated_convoy_heading_rad: float
    estimated_target_aspect_rad: float
    lead_angle_rad: float
    bow_offset_rad: float
    uncertainty_score: float
    doctrine_mode: DoctrineMode

    def as_metadata(self) -> dict[str, Any]:
        return {
            "centerline_bearing_rad": float(self.centerline_bearing_rad),
            "spread_rad": float(self.spread_rad),
            "salvo_size": int(self.salvo_size),
            "target_point_est_m": [float(self.target_point_est_m[0]), float(self.target_point_est_m[1])],
            "selected_speed_setting": self.selected_speed_setting,
            "torpedo_speed_mps": float(self.torpedo_speed_mps),
            "torpedo_max_run_time_s": float(self.torpedo_max_run_time_s),
            "estimated_range_m": float(self.estimated_range_m),
            "estimated_convoy_speed_mps": float(self.estimated_convoy_speed_mps),
            "estimated_convoy_heading_rad": float(self.estimated_convoy_heading_rad),
            "estimated_target_aspect_rad": float(self.estimated_target_aspect_rad),
            "lead_angle_rad": float(self.lead_angle_rad),
            "bow_offset_rad": float(self.bow_offset_rad),
            "uncertainty_score": float(self.uncertainty_score),
            "doctrine_mode": self.doctrine_mode,
        }


G7A_SPEED_SETTINGS: dict[SpeedSettingName, G7aSpeedSetting] = {
    "fast": G7aSpeedSetting(
        name="fast",
        speed_kts=44.0,
        speed_mps=float(44.0 * 0.514444),
        nominal_range_m=1800.0,
    ),
    "medium": G7aSpeedSetting(
        name="medium",
        speed_kts=40.0,
        speed_mps=float(40.0 * 0.514444),
        nominal_range_m=3200.0,
    ),
    "long_range": G7aSpeedSetting(
        name="long_range",
        speed_kts=30.0,
        speed_mps=float(30.0 * 0.514444),
        nominal_range_m=7500.0,
    ),
}


def choose_g7a_speed_setting(
    estimated_range_m: float,
    *,
    cfg: FireControlLiteConfig,
) -> G7aSpeedSetting:
    """Return a deterministic G7a speed/range setting from estimated range."""

    if estimated_range_m <= float(cfg.fast_setting_max_range_m):
        return G7A_SPEED_SETTINGS["fast"]
    if estimated_range_m <= float(cfg.medium_setting_max_range_m):
        return G7A_SPEED_SETTINGS["medium"]
    return G7A_SPEED_SETTINGS["long_range"]


def _require_observation_value(attacker_observation: dict[str, Any], key: str) -> float:
    if key not in attacker_observation:
        raise ValueError(f"attacker_observation missing required key: {key}")
    return float(attacker_observation[key])


def solve_fire_control_lite(
    *,
    u_boat_position: np.ndarray,
    u_boat_heading_rad: float,
    attacker_observation: dict[str, Any],
    cfg: FireControlLiteConfig | None = None,
) -> FireControlLiteSolution:
    """Resolve a deterministic coarse firing solution from attacker observation."""

    resolved_cfg = cfg or FireControlLiteConfig()
    u_pos = np.asarray(u_boat_position, dtype=float)
    if u_pos.shape != (2,):
        raise ValueError("u_boat_position must have shape (2,)")

    est_bearing = _require_observation_value(attacker_observation, "estimated_bearing_rad")
    est_range = max(1.0, _require_observation_value(attacker_observation, "estimated_range_m"))
    est_heading = _require_observation_value(attacker_observation, "estimated_convoy_heading_rad")
    est_speed = max(0.0, _require_observation_value(attacker_observation, "estimated_convoy_speed_mps"))
    obs_quality = attacker_observation.get("observation_quality", {})

    setting = choose_g7a_speed_setting(est_range, cfg=resolved_cfg)
    torpedo_speed = float(setting.speed_mps)
    torpedo_max_run_time = float(setting.max_run_time_s)

    line_of_sight = np.asarray([math.cos(est_bearing), math.sin(est_bearing)], dtype=float)
    estimated_target_point = u_pos + line_of_sight * est_range

    convoy_velocity = np.asarray([math.cos(est_heading), math.sin(est_heading)], dtype=float) * est_speed
    lateral_axis = np.asarray([-line_of_sight[1], line_of_sight[0]], dtype=float)
    lateral_speed = float(np.dot(convoy_velocity, lateral_axis))
    lead_ratio = np.clip(lateral_speed / max(torpedo_speed, 1e-9), -0.95, 0.95)
    lead_angle_rad = float(np.arcsin(lead_ratio))
    lead_angle_rad = float(np.clip(lead_angle_rad, -resolved_cfg.max_intercept_lead_rad, resolved_cfg.max_intercept_lead_rad))

    range_sigma_m = float(obs_quality.get("range_sigma_m", 0.0))
    bearing_sigma_rad = float(obs_quality.get("bearing_sigma_rad", 0.0))
    heading_sigma_rad = float(obs_quality.get("heading_sigma_rad", 0.0))
    normalized_range_unc = min(range_sigma_m / est_range, 1.0)
    uncertainty_score = float(
        math.sqrt(bearing_sigma_rad ** 2 + heading_sigma_rad ** 2 + normalized_range_unc ** 2)
    )

    spread_rad = float(
        resolved_cfg.base_spread_rad
        + resolved_cfg.uncertainty_spread_gain * uncertainty_score
        + resolved_cfg.range_spread_gain_rad * min(est_range / 4000.0, 1.0)
    )
    spread_rad = float(np.clip(spread_rad, resolved_cfg.base_spread_rad, resolved_cfg.max_spread_rad))

    target_aspect_rad = _wrap_angle_rad(est_heading - (est_bearing + math.pi))
    centerline_bearing_rad = _wrap_angle_rad(est_bearing + lead_angle_rad)
    max_bow_offset_rad = float(np.deg2rad(resolved_cfg.max_bow_centerline_offset_deg))
    bow_offset_rad = float(
        np.clip(
            _wrap_angle_rad(centerline_bearing_rad - float(u_boat_heading_rad)),
            -max_bow_offset_rad,
            max_bow_offset_rad,
        )
    )
    centerline_bearing_rad = _wrap_angle_rad(float(u_boat_heading_rad) + bow_offset_rad)

    return FireControlLiteSolution(
        centerline_bearing_rad=float(centerline_bearing_rad),
        spread_rad=spread_rad,
        salvo_size=max(1, int(resolved_cfg.desired_salvo_size)),
        target_point_est_m=estimated_target_point,
        selected_speed_setting=setting.name,
        torpedo_speed_mps=torpedo_speed,
        torpedo_max_run_time_s=torpedo_max_run_time,
        estimated_range_m=est_range,
        estimated_convoy_speed_mps=est_speed,
        estimated_convoy_heading_rad=est_heading,
        estimated_target_aspect_rad=float(target_aspect_rad),
        lead_angle_rad=float(lead_angle_rad),
        bow_offset_rad=float(bow_offset_rad),
        uncertainty_score=uncertainty_score,
        doctrine_mode=resolved_cfg.doctrine_mode,
    )


def build_attack_profile_from_fire_control(
    *,
    profile_id: str,
    name: str,
    u_boat_position: np.ndarray,
    u_boat_heading_rad: float,
    attacker_observation: dict[str, Any],
    cfg: FireControlLiteConfig | None = None,
    weight: float = 1.0,
    u_boat_mode: Literal["moving", "static"] = "moving",
    u_boat_initial_speed_mps: float = 2.0,
    u_boat_launch_time_s: float = 0.0,
    u_boat_motion_legs: tuple[tuple[float, float, float], ...] = (),
    launch_delay_s: float = 0.5,
    salvo_interval_s: float = 2.0,
    sub_length_m: float = 67.0,
    sub_beam_m: float = 6.5,
    launch_from: Literal["bow", "center"] = "bow",
    max_bow_offset_deg: float = 15.0,
    gyro_straight_run_m: float = 30.0,
    require_stable_u_boat_during_salvo: bool = True,
) -> AttackProfile:
    """Build a normal `AttackProfile` from the fire-control-lite solution."""

    solution = solve_fire_control_lite(
        u_boat_position=np.asarray(u_boat_position, dtype=float),
        u_boat_heading_rad=float(u_boat_heading_rad),
        attacker_observation=attacker_observation,
        cfg=cfg,
    )
    return AttackProfile(
        profile_id=profile_id,
        name=name,
        weight=weight,
        mode="fan",
        u_pos=(float(u_boat_position[0]), float(u_boat_position[1])),
        n=int(solution.salvo_size),
        speed=float(solution.torpedo_speed_mps),
        max_run_time=float(solution.torpedo_max_run_time_s),
        base_bearing_rad=float(solution.centerline_bearing_rad),
        spread_rad=float(solution.spread_rad),
        launch_delay_s=float(launch_delay_s),
        salvo_interval_s=float(salvo_interval_s),
        u_boat_mode=u_boat_mode,
        u_boat_initial_heading_rad=float(u_boat_heading_rad),
        u_boat_initial_speed_mps=float(u_boat_initial_speed_mps),
        u_boat_launch_time_s=float(u_boat_launch_time_s),
        u_boat_motion_legs=u_boat_motion_legs,
        sub_length_m=float(sub_length_m),
        sub_beam_m=float(sub_beam_m),
        launch_from=launch_from,
        max_bow_offset_deg=float(max_bow_offset_deg),
        gyro_straight_run_m=float(gyro_straight_run_m),
        require_stable_u_boat_during_salvo=bool(require_stable_u_boat_during_salvo),
    )
