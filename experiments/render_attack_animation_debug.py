"""Render debug frames with heading arrows for the attack demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from convoy_sim.attackers import fan_spread
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, RouteLeg, RoutePlan, ZigZagPlan, ship_positions_at
from convoy_sim.entities import Ship
from convoy_sim.geometry import as_vec
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.simulation import HitSlowdownSpec, init_dynamic_hit_state
from convoy_sim.viz_attack import render_attack_frame


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
            RouteLeg(duration_s=120.0, heading_rad=0.0),
        ]
    )
    zigzag = ZigZagPlan(enabled=True, amplitude_rad=0.12, period_s=60.0, phase_s=0.0, waveform="sine")
    kin = ConvoyKinematics(route=route, zigzag=zigzag)

    torpedoes = fan_spread(
        u_pos=as_vec(0, -1250),
        base_bearing_rad=np.radians(80),
        n=4,
        spread_rad=np.radians(10.0),
        speed=20.0,
        max_run_time=200.0,
    )
    for idx, torpedo in enumerate(torpedoes):
        torpedo.launch_delay = float(idx * 10.0)

    apply_hit_slowdown = False
    hit_slowdown = HitSlowdownSpec(
        enabled=apply_hit_slowdown,
        decay_rate=0.02,
        min_factor=0.4,
    )
    hit_state = init_dynamic_hit_state(0.0)

    out_dir = Path("results/frames/demo_attack_debug")
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

    t_start = 0.0
    t_end = 120.0
    fps = 5
    dt = 1.0 / float(fps)
    times = np.arange(float(t_start), float(t_end) + 1e-9, dt)

    for idx, t in enumerate(times):
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="lightgrey")
        render_attack_frame(
            ships_t0=ships,
            torpedoes=torpedoes,
            t_global=float(t),
            t_max=float(t_end),
            dynamics=(formation, kin),
            ax=ax,
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
            apply_hit_slowdown=apply_hit_slowdown,
            hit_slowdown=hit_slowdown,
            hit_state=hit_state,
            hit_dt=dt,
            figure_facecolor="lightgrey",
        )

        heading = kin.convoy_heading_at(float(t), formation.convoy_heading0)
        arrow_len = 150
        ship_positions = ship_positions_at(
            float(t),
            formation,
            kin,
            dt=1.0,
            motion="independent",
        )
        for pos in ship_positions:
            dx = arrow_len * float(np.cos(heading))
            dy = arrow_len * float(np.sin(heading))
            ax.plot(
                [float(pos[0]), float(pos[0]) + dx],
                [float(pos[1]), float(pos[1]) + dy],
                color="#2f2f2f",
                linewidth=3.0,
                alpha=0.7,
                zorder=10,
            )
            ax.scatter(
                float(pos[0]) + dx,
                float(pos[1]) + dy,
                s=60.0,
                marker=">",
                color="#2f2f2f",
                alpha=0.8,
                zorder=11,
            )

        fig.tight_layout()
        path = out_dir / f"frame_{idx:04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

    mp4_path = Path("results/frames/demo_attack_debug.mp4")
    try:
        from matplotlib import animation  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 6), facecolor="lightgrey")

        def _update(frame_idx: int):
            ax.clear()
            t = float(times[frame_idx])
            render_attack_frame(
                ships_t0=ships,
                torpedoes=torpedoes,
                t_global=t,
                t_max=float(t_end),
                dynamics=(formation, kin),
                ax=ax,
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
                apply_hit_slowdown=apply_hit_slowdown,
                hit_slowdown=hit_slowdown,
                hit_state=hit_state,
                hit_dt=dt,
                figure_facecolor="lightgrey",
            )
            heading = kin.convoy_heading_at(float(t), formation.convoy_heading0)
            ship_positions = ship_positions_at(
                float(t),
                formation,
                kin,
                dt=1.0,
                motion="independent",
            )
            for pos in ship_positions:
                dx = arrow_len * float(np.cos(heading))
                dy = arrow_len * float(np.sin(heading))
                ax.plot(
                    [float(pos[0]), float(pos[0]) + dx],
                    [float(pos[1]), float(pos[1]) + dy],
                    color="#2f2f2f",
                    linewidth=3.0,
                    alpha=0.7,
                    zorder=10,
                )
                ax.scatter(
                    float(pos[0]) + dx,
                    float(pos[1]) + dy,
                    s=60.0,
                    marker=">",
                    color="#2f2f2f",
                    alpha=0.8,
                    zorder=11,
                )
            return ax

        anim = animation.FuncAnimation(fig, _update, frames=len(times), interval=1000 / fps)
        if not animation.writers.is_available("ffmpeg"):
            raise ImportError("ffmpeg is required for MP4 output")
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(str(mp4_path), writer=writer)
        plt.close(fig)
    except ImportError as exc:
        print(f"Skipping MP4 export: {exc}")

    print(f"Wrote debug frames to {out_dir}")
    print(f"Wrote debug mp4 to {mp4_path}")


if __name__ == "__main__":
    main()
