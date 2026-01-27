"""Render frames (and optional MP4) for a dynamic attack demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan, ZigZagPlan
from convoy_sim.entities import Ship, ShipClass
from convoy_sim.geometry import as_vec
from convoy_sim.viz_attack import save_attack_animation_mp4, save_attack_frames


def _make_convoy() -> list[Ship]:
    ships = []
    positions = [as_vec(0.0, 0.0), as_vec(-200.0, 80.0), as_vec(-200.0, -80.0)]
    for idx, pos in enumerate(positions, start=1):
        ships.append(
            Ship(
                id=f"S{idx}",
                position=pos,
                speed=5.0,
                heading_rad=0.0,
                length=80.0,
                beam=12.0,
                ship_class=ShipClass.FREIGHTER,
            )
        )
    return ships


def main() -> None:
    ships = _make_convoy()
    formation = ConvoyFormation(
        ships0=ships,
        convoy_origin0=as_vec(0.0, 0.0),
        convoy_heading0=0.0,
    )
    route = RoutePlan(
        legs=[
            RouteLeg(duration_s=60.0, heading_rad=0.0),
            RouteLeg(duration_s=60.0, heading_rad=np.pi / 2.0),
        ]
    )
    zigzag = ZigZagPlan(enabled=True, amplitude_rad=0.15, period_s=30.0, phase_s=0.0)
    kin = ConvoyKinematics(route=route, zigzag=zigzag)

    torpedoes = fan_spread(
        u_pos=as_vec(-1500.0, 0.0),
        base_bearing_rad=0.0,
        n=3,
        spread_rad=np.radians(10.0),
        speed=20.0,
        max_run_time=200.0,
    )
    for idx, torpedo in enumerate(torpedoes):
        torpedo.launch_delay = float(idx * 10.0)

    out_dir = Path("results/frames/demo_attack")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_attack_frames(
        str(out_dir),
        ships_t0=ships,
        torpedoes=torpedoes,
        t_start=0.0,
        t_end=120.0,
        fps=5,
        dynamics=(formation, kin),
        color_by="class",
        show_trails=True,
        trail_length_s=40.0,
    )

    mp4_path = Path("results/frames/demo_attack.mp4")
    try:
        save_attack_animation_mp4(
            str(mp4_path),
            ships_t0=ships,
            torpedoes=torpedoes,
            t_start=0.0,
            t_end=120.0,
            fps=5,
            dynamics=(formation, kin),
            color_by="class",
            show_trails=True,
            trail_length_s=40.0,
        )
    except ImportError as exc:
        print(f"Skipping MP4 export: {exc}")

    print(f"Wrote frames to {out_dir}")


if __name__ == "__main__":
    main()
