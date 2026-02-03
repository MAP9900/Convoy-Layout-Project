"""Render frames (and optional MP4) for a dynamic attack demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan, ZigZagPlan
from convoy_sim.entities import Ship
from convoy_sim.geometry import as_vec
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.viz_attack import save_attack_animation_mp4, save_attack_frames


def _make_convoy() -> list[Ship]:
    return make_rectangular_convoy(
        n_rows=3,
        n_cols=4,
        spacing_along=600.0,
        spacing_across=350.0,
        speed=5.0,
        heading_rad=0.0,
        length=150.0,
        beam=20.0,
        origin=as_vec(0.0, 0.0),
    )


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
        u_pos=as_vec(0, -1250), # Submarine Position 
        base_bearing_rad=np.radians(80), # Launch Radian 0.0 = due east np.radians(180) = due west
        n=4, #Number of torpedos (Typically 4 was max slavo size)
        spread_rad=np.radians(10.0), #Torpedo Spread (ex 10 at launch degree goes -5, 0, 5)
        speed=20.0, # Torpedo Speed
        max_run_time=200.0, 
    )
    for idx, torpedo in enumerate(torpedoes):
        torpedo.launch_delay = float(idx * 10.0) # Torpedo Lauch Delay

    out_dir = Path("results/frames/demo_attack")
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = np.array([ship.position for ship in ships], dtype=float)
    xmin = float(np.min(positions[:, 0]))
    xmax = float(np.max(positions[:, 0]))
    ymin = float(np.min(positions[:, 1]))
    ymax = float(np.max(positions[:, 1]))
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    pad = max(2000.0, span * 1.5)
    view_bounds = (
        center_x - pad,
        center_x + pad,
        center_y - pad,
        center_y + pad,
    )

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
        show_footprint=False,
        ship_marker="ship",
        rotate_by_heading=True,
        use_hull_dimensions=True,
        trail_color="red",
        legend_bbox_to_anchor=(0.5, -0.24),
        view_bounds=view_bounds,
        hide_spines=True,
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
            show_footprint=False,
            ship_marker="ship",
            rotate_by_heading=True,
            use_hull_dimensions=True,
            trail_color="red",
            legend_bbox_to_anchor=(0.5, -0.24),
            view_bounds=view_bounds,
            hide_spines=True,
        )
    except ImportError as exc:
        print(f"Skipping MP4 export: {exc}")

    print(f"Wrote frames to {out_dir}")


if __name__ == "__main__":
    main()
