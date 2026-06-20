"""POMDP fire-control rebuild helpers for VAE candidate locations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship
from convoy_sim.feasibility import Environment
from convoy_sim.fire_control import FireControlLiteConfig, build_attack_profile_from_fire_control, solve_fire_control_lite
from convoy_sim.realism import AttackerObservationConfig, build_attacker_observation, get_attacker_observation_config


def _profile_dict(record: Mapping[str, Any]) -> dict[str, Any]:
    if "profile" not in record:
        raise ValueError("candidate record is missing required key: profile")
    return dict(record["profile"])


def _resolve_observation_config(
    observation_preset: str,
    observation_cfg: AttackerObservationConfig | None,
) -> AttackerObservationConfig:
    return observation_cfg or get_attacker_observation_config(observation_preset)


def build_fire_control_rebuilt_record(
    record: Mapping[str, Any],
    *,
    ships: Sequence[Ship],
    rng: np.random.Generator,
    env: Environment | None = None,
    observation_preset: str = "good_contact",
    observation_cfg: AttackerObservationConfig | None = None,
    fire_control_cfg: FireControlLiteConfig | None = None,
    profile_id_prefix: str = "POMDP_FC",
    sequence_idx: int = 1,
    observation_seed: int | None = None,
) -> dict[str, Any]:
    """Rebuild one candidate profile from noisy observation and fire-control-lite.

    The source candidate contributes the U-boat location and basic motion/timing
    fields. Bearing, spread, torpedo speed, and max run time are rebuilt from
    the attacker-facing observation rather than copied from the source profile.
    """

    source_profile = AttackProfile.from_dict(_profile_dict(record))
    resolved_env = env or Environment(time_of_day="night", visibility_m=3500.0, sea_state=4)
    resolved_obs_cfg = _resolve_observation_config(observation_preset, observation_cfg)
    u_pos = np.asarray(source_profile.u_pos, dtype=float)
    observation = build_attacker_observation(
        ships=list(ships),
        u_boat_pos=u_pos,
        env=resolved_env,
        rng=rng,
        cfg=resolved_obs_cfg,
    )
    u_boat_heading_rad = float(observation["estimated_bearing_rad"])
    solution = solve_fire_control_lite(
        u_boat_position=u_pos,
        u_boat_heading_rad=u_boat_heading_rad,
        attacker_observation=observation,
        cfg=fire_control_cfg,
    )
    profile_id = f"{profile_id_prefix}_{sequence_idx:04d}"
    rebuilt_profile = build_attack_profile_from_fire_control(
        profile_id=profile_id,
        name=f"{profile_id.lower()}_from_{source_profile.profile_id}",
        u_boat_position=u_pos,
        u_boat_heading_rad=u_boat_heading_rad,
        attacker_observation=observation,
        cfg=fire_control_cfg,
        weight=float(source_profile.weight),
        u_boat_mode=source_profile.u_boat_mode,
        u_boat_initial_speed_mps=float(source_profile.u_boat_initial_speed_mps),
        u_boat_launch_time_s=float(source_profile.u_boat_launch_time_s),
        u_boat_motion_legs=(),
        launch_delay_s=float(source_profile.launch_delay_s),
        salvo_interval_s=float(source_profile.salvo_interval_s),
        sub_length_m=float(source_profile.sub_length_m),
        sub_beam_m=float(source_profile.sub_beam_m),
        launch_from=source_profile.launch_from,
        max_bow_offset_deg=float(source_profile.max_bow_offset_deg),
        gyro_straight_run_m=float(source_profile.gyro_straight_run_m),
        require_stable_u_boat_during_salvo=bool(source_profile.require_stable_u_boat_during_salvo),
    )
    intent = dict(record.get("intent", {}))
    intent.update(
        {
            "source_profile_id": str(source_profile.profile_id),
            "source_profile_name": str(source_profile.name),
            "source_u_pos": [float(u_pos[0]), float(u_pos[1])],
            "source_base_bearing_rad": float(source_profile.base_bearing_rad),
            "rebuilt_profile_id": profile_id,
            "rebuild_method": "pomdp_fire_control_lite_v1",
        }
    )
    selection_meta = dict(record.get("selection_meta", {}))
    selection_meta.update(
        {
            "method": "pomdp_fire_control_lite_v1",
            "observation_preset": str(observation_preset),
        }
    )
    if observation_seed is not None:
        selection_meta["observation_seed"] = int(observation_seed)

    generator_meta = dict(record.get("generator_meta", {}))
    generator_meta.update(
        {
            "mode": "pomdp_fire_control_lite_v1",
            "source_profile_id": str(source_profile.profile_id),
        }
    )
    return {
        "profile": rebuilt_profile.to_dict(),
        "intent": intent,
        "selection_meta": selection_meta,
        "generator_meta": generator_meta,
        "fire_control_meta": {
            "observation_preset": str(observation_preset),
            "observation_config": resolved_obs_cfg.to_dict(),
            "attacker_observation": observation,
            "solution": solution.as_metadata(),
            "u_boat_heading_rad": float(u_boat_heading_rad),
        },
    }


def rebuild_records_with_fire_control(
    records: Sequence[Mapping[str, Any]],
    *,
    ships: Sequence[Ship],
    seed: int = 1945,
    env: Environment | None = None,
    observation_preset: str = "good_contact",
    observation_cfg: AttackerObservationConfig | None = None,
    fire_control_cfg: FireControlLiteConfig | None = None,
    profile_id_prefix: str = "POMDP_FC",
) -> list[dict[str, Any]]:
    """Rebuild a sequence of selected records with fire-control-lite."""

    rng = np.random.default_rng(int(seed))
    return [
        build_fire_control_rebuilt_record(
            record,
            ships=ships,
            rng=rng,
            env=env,
            observation_preset=observation_preset,
            observation_cfg=observation_cfg,
            fire_control_cfg=fire_control_cfg,
            profile_id_prefix=profile_id_prefix,
            sequence_idx=idx,
            observation_seed=int(seed),
        )
        for idx, record in enumerate(records, start=1)
    ]


def write_fire_control_candidate_pool(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write rebuilt fire-control candidate records as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record)) + "\n")
