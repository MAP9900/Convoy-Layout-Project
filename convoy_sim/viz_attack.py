"""Optional visualization and debug helpers for static attacks."""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

import numpy as np
from pathlib import Path

from convoy_sim.entities import Ship, ShipClass, Torpedo, torpedo_hit_time
from convoy_sim.geometry import as_vec, distance, min_distance_over_interval
from convoy_sim.realism import UBoatMotionPlan
from convoy_sim.viz import plot_convoy_planview
from convoy_sim.dynamics import ConvoyFormation, ConvoyKinematics, ship_positions_at
from convoy_sim.simulation import (
    DynamicHitState,
    HitSlowdownSpec,
    advance_dynamic_hit_state,
    init_dynamic_hit_state,
)


def torpedo_segment(
    torpedo: Torpedo,
    t0: float = 0.0,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the start/end points of the torpedo ray segment."""

    start_t = float(max(0.0, t0))
    end_limit = torpedo.end_time_s()
    end_t = float(end_limit if t1 is None else min(t1, end_limit))
    p_start = torpedo.position_at(start_t)
    p_end = torpedo.position_at(end_t)
    return p_start, p_end


def torpedo_path_points(
    torpedo: Torpedo,
    t0: float = 0.0,
    t1: float | None = None,
) -> np.ndarray:
    """Return polyline points for the torpedo path over the requested window."""

    start_t = float(max(0.0, t0))
    end_limit = torpedo.end_time_s()
    end_t = float(end_limit if t1 is None else min(t1, end_limit))
    if end_t < start_t:
        return np.empty((0, 2), dtype=float)
    sample_times = [start_t, end_t]
    turn_t = torpedo.gyro_turn_time_s()
    if torpedo.uses_gyro_turn() and start_t < turn_t < end_t:
        sample_times.insert(1, float(turn_t))
    points = [np.asarray(torpedo.position_at(t), dtype=float) for t in dict.fromkeys(sample_times)]
    return np.asarray(points, dtype=float)


def torpedo_position_at_global_time(torpedo: Torpedo, t_global: float) -> np.ndarray:
    """Return torpedo position at global time, respecting launch_delay."""

    return torpedo.position_at(float(t_global))


def sample_u_boat_track(
    motion_plan: UBoatMotionPlan,
    t_start: float,
    t_end: float,
    *,
    n_points: int = 200,
) -> np.ndarray:
    """Sample a U-boat motion plan into polyline points."""

    start = float(t_start)
    end = float(t_end)
    if end < start:
        raise ValueError("t_end must be >= t_start")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    times = np.linspace(start, end, int(n_points), dtype=float)
    return np.asarray([motion_plan.position_at(float(t)) for t in times], dtype=float)


def torpedo_heading_table_rows(torpedoes: list[Torpedo]) -> list[dict[str, float | str]]:
    """Return compact launch/final-heading rows for reporting or notebook display."""

    rows: list[dict[str, float | str]] = []
    for idx, torpedo in enumerate(torpedoes, start=1):
        launch_heading_deg = float(np.degrees(torpedo.initial_heading_rad()))
        final_heading_deg = float(np.degrees(torpedo.heading_rad))
        gyro_offset_deg = float(
            np.degrees(
                np.arctan2(
                    np.sin(float(torpedo.heading_rad) - float(torpedo.initial_heading_rad())),
                    np.cos(float(torpedo.heading_rad) - float(torpedo.initial_heading_rad())),
                )
            )
        )
        rows.append(
            {
                "shot": idx,
                "torpedo_id": torpedo.id,
                "launch_time_s": float(torpedo.launch_delay),
                "launch_x_m": float(torpedo.launch_position[0]),
                "launch_y_m": float(torpedo.launch_position[1]),
                "launch_heading_deg": launch_heading_deg,
                "final_heading_deg": final_heading_deg,
                "gyro_offset_deg": gyro_offset_deg,
            }
        )
    return rows


def format_torpedo_heading_table(torpedoes: list[Torpedo]) -> str:
    """Return a monospace table for launch positions and final headings."""

    rows = torpedo_heading_table_rows(torpedoes)
    if not rows:
        return "No torpedoes."
    header = (
        f"{'shot':>4} {'id':>5} {'t_launch_s':>10} {'launch_x_m':>11} "
        f"{'launch_y_m':>11} {'launch_deg':>11} {'final_deg':>10} {'gyro_deg':>9}"
    )
    body = [
        (
            f"{int(row['shot']):>4} {str(row['torpedo_id']):>5} {float(row['launch_time_s']):>10.1f} "
            f"{float(row['launch_x_m']):>11.1f} {float(row['launch_y_m']):>11.1f} "
            f"{float(row['launch_heading_deg']):>11.1f} {float(row['final_heading_deg']):>10.1f} "
            f"{float(row['gyro_offset_deg']):>9.1f}"
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _draw_heading_stub(
    ax: Any,
    position: np.ndarray,
    heading_rad: float,
    *,
    length_m: float = 120.0,
    color: str = "#111111",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    zorder: int = 7,
) -> None:
    """Draw a short heading line for the U-boat orientation."""

    direction = np.asarray([np.cos(float(heading_rad)), np.sin(float(heading_rad))], dtype=float)
    start = np.asarray(position, dtype=float)
    end = start + direction * float(length_m)
    ax.plot(
        [float(start[0]), float(end[0])],
        [float(start[1]), float(end[1])],
        color=color,
        linewidth=float(linewidth),
        alpha=float(alpha),
        solid_capstyle="round",
        zorder=int(zorder),
    )


def plot_torpedo_doctrine_snapshot(
    torpedoes: list[Torpedo],
    *,
    snapshot_time_s: float,
    ax: Any | None = None,
    title: str | None = None,
    u_boat_track: np.ndarray | None = None,
    u_boat_position: np.ndarray | None = None,
    u_boat_heading_rad: float | None = None,
    centerline_bearing_rad: float | None = None,
    view_bounds: tuple[float, float, float, float] | None = None,
    figure_facecolor: str | None = None,
    axes_facecolor: str = "#06768d",
    grid_color: str = "lightgrey",
    torpedo_color: str = "#b00020",
    path_color: str = "#6c757d",
    centerline_color: str = "#8d99ae",
    launch_point_color: str = "#111111",
    launch_point_marker: str = ".",
    launch_point_size: float = 20.0,
    torpedo_linewidth: float = 1,
    show_launch_points: bool = True,
    show_centerline: bool = True,
    show_u_boat_path: bool = True,
    show_u_boat_heading: bool = False,
    show_full_torpedo_run: bool = False,
    u_boat_marker: str = ".",
    u_boat_size: float = 75,
) -> Any:
    """Render a submarine-centric attack snapshot for doctrine comparison."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for doctrine plotting") from exc

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 8), facecolor=figure_facecolor)
    elif figure_facecolor is not None:
        ax.figure.set_facecolor(figure_facecolor)
    ax.set_facecolor(axes_facecolor)

    snapshot = float(snapshot_time_s)
    if show_u_boat_path and u_boat_track is not None and len(u_boat_track) >= 2:
        track = np.asarray(u_boat_track, dtype=float)
        ax.plot(
            track[:, 0],
            track[:, 1],
            linestyle="--",
            linewidth=1.4,
            color=path_color,
            alpha=0.9,
            zorder=1,
        )

    if show_centerline and centerline_bearing_rad is not None:
        reference_origin = None
        if u_boat_position is not None:
            reference_origin = np.asarray(u_boat_position, dtype=float)
        elif torpedoes:
            reference_origin = np.asarray(torpedoes[0].launch_position, dtype=float)
        if reference_origin is not None:
            direction = np.asarray(
                [np.cos(float(centerline_bearing_rad)), np.sin(float(centerline_bearing_rad))],
                dtype=float,
            )
            line_length = 1_200.0
            start = reference_origin
            end = reference_origin + direction * line_length
            ax.plot(
                [float(start[0]), float(end[0])],
                [float(start[1]), float(end[1])],
                linestyle=(0, (4, 4)),
                linewidth=1.0,
                color=centerline_color,
                alpha=0.8,
                zorder=2,
            )

    for torpedo in torpedoes:
        path_end = torpedo.end_time_s() if show_full_torpedo_run else snapshot
        path = torpedo_path_points(torpedo, t0=0.0, t1=path_end)
        if path.size == 0:
            continue
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=torpedo_color,
            linewidth=float(torpedo_linewidth),
            alpha=0.95,
            zorder=3,
        )
        if show_launch_points:
            ax.scatter(
                float(torpedo.launch_position[0]),
                float(torpedo.launch_position[1]),
                s=float(launch_point_size),
                c=launch_point_color,
                marker=launch_point_marker,
                edgecolors="none",
                linewidths=0.0,
                zorder=5,
            )

    if u_boat_position is not None:
        pos = np.asarray(u_boat_position, dtype=float)
        ax.scatter(
            float(pos[0]),
            float(pos[1]),
            s=float(u_boat_size),
            c="#111111",
            marker=u_boat_marker,
            edgecolors="none",
            linewidths=0.0,
            zorder=6,
        )
        if show_u_boat_heading and u_boat_heading_rad is not None:
            _draw_heading_stub(ax, pos, float(u_boat_heading_rad))

    if title is not None:
        ax.set_title(title)
    if view_bounds is not None:
        xmin, xmax, ymin, ymax = view_bounds
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.75, linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def get_ship_positions_dynamic(
    ships_t0: list[Ship],
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    t_global: float,
    *,
    speed_factors: dict[str, float] | None = None,
    hit_time_by_ship: dict[str, float] | None = None,
    hit_decay_rate: float | None = None,
    hit_min_factor: float = 0.3,
) -> list[np.ndarray]:
    """Return ship positions at global time for static or dynamic motion."""

    if dynamics is None:
        return [ship.position.copy() for ship in ships_t0]
    formation, kin = dynamics
    return ship_positions_at(
        t_global,
        formation,
        kin,
        dt=1.0,
        motion="independent",
        speed_factors=speed_factors,
        hit_time_by_ship=hit_time_by_ship,
        hit_decay_rate=hit_decay_rate,
        hit_min_factor=hit_min_factor,
    )


def min_miss_distance_ship_torpedo(
    ship: Ship,
    torpedo: Torpedo,
    t_max: float,
    dt: float | None = None,
) -> float:
    """Return the minimum distance between a static ship and torpedo track."""

    window = min(float(t_max), float(torpedo.end_time_s()))
    if window <= 0.0:
        return float("inf")
    if dt is not None:
        times = np.arange(0.0, window + 1e-9, float(dt))
        positions = np.array([torpedo.position_at(t) for t in times], dtype=float)
        dists = np.linalg.norm(positions - ship.position, axis=1)
        return float(np.min(dists))
    min_dist = float("inf")
    for start_t, end_t, torp_start, torp_velocity in torpedo.motion_segments(window):
        duration = float(end_t - start_t)
        if duration <= 0.0:
            continue
        ship_start = ship.position_at(start_t)
        dist = min_distance_over_interval(
            ship_start,
            ship.velocity_vec(),
            torp_start,
            torp_velocity,
            0.0,
            duration,
        )
        min_dist = min(min_dist, float(dist))
    return min_dist


def _earliest_static_hit_time(ship: Ship, torpedo: Torpedo, t_max: float) -> float | None:
    """Return earliest hit time for a stationary ship, or None."""

    return torpedo_hit_time(ship, torpedo, t_max)


def attack_debug_metrics(
    ships: list[Ship],
    torpedoes: list[Torpedo],
    t_max: float,
    margin: float = 0.0,
) -> dict[str, Any]:
    """Return debug metrics for a static attack realization."""

    per_torpedo = []
    per_ship = {
        ship.id: {
            "nearby_torpedoes": 0,
            "was_hit": False,}
        for ship in ships}
    for torpedo in torpedoes:
        closest_ship = None
        closest_dist = float("inf")
        hit_ship_id = None
        for ship in ships:
            d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max)
            if d_min < closest_dist:
                closest_dist = d_min
                closest_ship = ship
            if d_min <= ship.effective_hit_radius() + float(margin):
                per_ship[ship.id]["nearby_torpedoes"] += 1
            if d_min <= ship.effective_hit_radius():
                hit_ship_id = ship.id
                per_ship[ship.id]["was_hit"] = True
        per_torpedo.append(
            {
                "id": torpedo.id,
                "hit_ship_id": hit_ship_id,
                "min_miss_distance": float(closest_dist),
                "closest_ship_id": None if closest_ship is None else closest_ship.id,}
        )
    return {
        "torpedoes": per_torpedo,
        "ships": per_ship,
    }


def plot_attack_planview(
    ships: list[Ship],
    torpedoes: list[Torpedo],
    t_max: float,
    ax: Any | None = None,
    title: str | None = None,
    color_by: Literal["class", "value"] = "class",
    show_footprint: bool = True,
    show_miss_annotations: bool = False,
    miss_k: int = 5,
    ray_alpha: float = 0.6,
    miss_color: str = "gray",
    ship_marker: Literal["circle", "ship"] = "circle",
    rotate_by_heading: bool = False,
    use_hull_dimensions: bool = False,
) -> Any:
    """Plot ships plus torpedo rays in plan view."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for attack plotting") from exc

    ax = plot_convoy_planview(
        ships,
        ax=ax,
        title=title,
        color_by=color_by,
        show_footprint=show_footprint,
        ship_marker=ship_marker,
        rotate_by_heading=rotate_by_heading,
        use_hull_dimensions=use_hull_dimensions,
    )
    hit_info: dict[str, dict[str, Any]] = {}
    hit_ship_ids: set[str] = set()
    for torpedo in torpedoes:
        hit_time = None
        hit_ship_id = None
        min_dist = float("inf")
        closest_ship_id = None
        for ship in ships:
            candidate = _earliest_static_hit_time(ship, torpedo, t_max)
            if candidate is not None:
                if hit_time is None or candidate < hit_time:
                    hit_time = candidate
                    hit_ship_id = ship.id
            d_min = min_miss_distance_ship_torpedo(ship, torpedo, t_max)
            if d_min < min_dist:
                min_dist = d_min
                closest_ship_id = ship.id
        if hit_ship_id is not None:
            hit_ship_ids.add(hit_ship_id)
        hit_info[torpedo.id] = {
            "hit_ship_id": hit_ship_id,
            "hit_time": hit_time,
            "min_miss_distance": float(min_dist),
            "closest_ship_id": closest_ship_id,
        }

    for torpedo in torpedoes:
        info = hit_info.get(torpedo.id, {})
        hit_time = info.get("hit_time")
        t_end = float(t_max) if hit_time is None else float(hit_time)
        path = torpedo_path_points(torpedo, t0=0.0, t1=t_end)
        if path.size == 0:
            continue
        hit = hit_time is not None
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="red" if hit else miss_color,
            linewidth=1.5 if hit else 1.0,
            alpha=float(ray_alpha),
            linestyle="-" if hit else "--",
        )

    for ship in ships:
        if ship.id in hit_ship_ids:
            ax.scatter(
                ship.position[0],
                ship.position[1],
                s=300,
                facecolors="none",
                edgecolors="red",
                linewidths=0.75,
                zorder=5,
            )

    if show_miss_annotations and torpedoes:
        misses = [
            (info["min_miss_distance"], torp_id)
            for torp_id, info in hit_info.items()
            if info.get("hit_ship_id") is None
        ]
        misses.sort(key=lambda x: x[0])
        for dist, torp_id in misses[: max(0, int(miss_k))]:
            torpedo = next(t for t in torpedoes if t.id == torp_id)
            path = torpedo_path_points(torpedo, t0=0.0, t1=t_max)
            if path.size > 0:
                p_last = path[-1]
                ax.annotate(f"{dist:.1f}m", (p_last[0], p_last[1]), fontsize=8, color="gray")

    return ax


def render_attack_frame(
    ships_t0: list[Ship],
    torpedoes: list[Torpedo],
    t_global: float,
    t_max: float,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None = None,
    ax: Any | None = None,
    color_by: Literal["class", "value"] = "class",
    show_trails: bool = True,
    trail_length_s: float = 60.0,
    show_footprint: bool = True,
    ship_marker: Literal["circle", "ship"] = "circle",
    rotate_by_heading: bool = False,
    use_hull_dimensions: bool = False,
    trail_color: str = "gray",
    trail_linewidth: float = 1.0,
    trail_alpha: float = 0.7,
    trail_antialiased: bool = True,
    clip_trails_at_hits: bool = True,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    view_bounds: tuple[float, float, float, float] | None = None,
    hide_spines: bool = True,
    hit_state: DynamicHitState | None = None,
    hit_dt: float | None = None,
    apply_hit_slowdown: bool = False,
    hit_slowdown: HitSlowdownSpec | None = None,
    figure_facecolor: str | None = None,
    show_u_boat: bool = False,
    u_boat_position: np.ndarray | None = None,
    u_boat_position_fn: Callable[[float], np.ndarray] | None = None,
    u_boat_marker: str = "o",
    u_boat_color: str = "#111111",
    u_boat_size: float = 45.0,
    u_boat_label: str | None = "U-boat",
) -> Any:
    """Render a single time frame of a static/dynamic attack."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for attack frame rendering") from exc

    speed_factors = None
    hit_time_by_ship = None
    hit_decay_rate = None
    hit_min_factor = 0.3
    if apply_hit_slowdown and hit_state is not None and hit_slowdown and hit_slowdown.enabled:
        hit_time_by_ship = hit_state.hit_time_by_ship
        hit_decay_rate = float(hit_slowdown.decay_rate)
        hit_min_factor = float(hit_slowdown.min_factor)
    ship_positions = get_ship_positions_dynamic(
        ships_t0,
        dynamics,
        t_global,
        speed_factors=speed_factors,
        hit_time_by_ship=hit_time_by_ship,
        hit_decay_rate=hit_decay_rate,
        hit_min_factor=hit_min_factor,
    )
    heading_override = None
    if dynamics is not None:
        formation, kin = dynamics
        heading_override = kin.convoy_heading_at(t_global, formation.convoy_heading0)
    ships = []
    for ship, pos in zip(ships_t0, ship_positions):
        ships.append(
            Ship(
                id=ship.id,
                position=pos,
                speed=0.0,
                heading_rad=ship.heading_rad if heading_override is None else heading_override,
                length=ship.length,
                beam=ship.beam,
                ship_class=ship.ship_class,
                value_weight=ship.value_weight,
                hit_radius=ship.hit_radius,
            )
        )

    ax = plot_convoy_planview(
        ships,
        ax=ax,
        title=f"Convoy Attack Visual \nt={t_global:.1f}s",
        color_by=color_by,
        show_footprint=show_footprint,
        ship_marker=ship_marker,
        rotate_by_heading=rotate_by_heading,
        use_hull_dimensions=use_hull_dimensions,
    )
    if dynamics is not None and hit_state is not None and hit_dt is not None:
        formation, kin = dynamics
        advance_dynamic_hit_state(
            formation,
            kin,
            torpedoes,
            t_global,
            hit_dt,
            hit_state,
            max_hits_per_torpedo=1,
            hit_slowdown=hit_slowdown,
        )
    elif dynamics is None:
        hit_colors = ["#ffd0d0", "#ff9b9b", "#ff6666", "#d7191c"]
        for ship in ships:
            for torpedo in torpedoes:
                if t_global < torpedo.launch_delay:
                    continue
                torp_pos = torpedo.position_at(float(t_global))
                if distance(ship.position, torp_pos) <= ship.effective_hit_radius():
                    ax.scatter(
                        ship.position[0],
                        ship.position[1],
                        s=140.0,
                        facecolors="none",
                        edgecolors=hit_colors[0],
                        linewidths=2.0,
                        zorder=5,
                    )
                    break

    for torpedo in torpedoes:
        if t_global < torpedo.launch_delay:
            continue
        t_end = min(float(t_max), float(t_global))
        t_start = max(0.0, t_end - float(trail_length_s)) if show_trails else t_end
        if clip_trails_at_hits and hit_state is not None and torpedo.id in hit_state.torpedo_hit_times:
            t_end = min(t_end, float(hit_state.torpedo_hit_times[torpedo.id]))
            if t_end <= t_start:
                continue
        path = torpedo_path_points(torpedo, t0=t_start, t1=t_end)
        if path.size == 0:
            continue
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=trail_color,
            linewidth=float(trail_linewidth),
            alpha=float(trail_alpha),
            antialiased=bool(trail_antialiased),
        )

    if show_u_boat and torpedoes:
        if u_boat_position_fn is not None:
            u_pos = np.asarray(u_boat_position_fn(float(t_global)), dtype=float)
            if u_pos.shape != (2,):
                raise ValueError("u_boat_position_fn(t) must return shape (2,)")
        elif u_boat_position is None:
            inferred = np.array([torpedo.launch_position for torpedo in torpedoes], dtype=float)
            u_pos = np.mean(inferred, axis=0)
        else:
            u_pos = np.asarray(u_boat_position, dtype=float)
            if u_pos.shape != (2,):
                raise ValueError("u_boat_position must have shape (2,)")
        ax.scatter(
            float(u_pos[0]),
            float(u_pos[1]),
            s=float(u_boat_size),
            marker=u_boat_marker,
            c=u_boat_color,
            edgecolors="none",
            linewidths=0.0,
            zorder=8,
            label=u_boat_label,
        )

    hit_colors = ["#ffd0d0", "#ff9b9b", "#ff6666", "#d7191c"]
    for ship in ships:
        if hit_state is not None and ship.id in hit_state.hit_counts:
            count = int(hit_state.hit_counts.get(ship.id, 0))
            color_idx = min(max(count, 1), len(hit_colors)) - 1
            ax.scatter(
                ship.position[0],
                ship.position[1],
                s=140.0,
                facecolors="none",
                edgecolors=hit_colors[color_idx],
                linewidths=2.0,
                zorder=5,
            )
    if view_bounds is not None:
        xmin, xmax, ymin, ymax = view_bounds
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)
    if legend_bbox_to_anchor is not None:
        handles: list[Any] = []
        labels: list[str] = []
        if color_by == "class":
            from matplotlib.patches import Patch  # type: ignore

            class_colors = {
                ShipClass.FREIGHTER: "#0a0a0a",
                ShipClass.TANKER: "#38160d",
                ShipClass.ESCORT: "#001845",
                ShipClass.DECOY: "#ffc600",
            }
            for ship_class, color in class_colors.items():
                handles.append(Patch(facecolor=color, edgecolor="none"))
                labels.append(ship_class.value)
        if show_u_boat and torpedoes and u_boat_label:
            from matplotlib.lines import Line2D  # type: ignore

            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=u_boat_marker,
                    color="none",
                    markerfacecolor=u_boat_color,
                    markeredgecolor=u_boat_color,
                    markersize=max(4.0, float(u_boat_size) ** 0.5),
                    linewidth=0.0,
                )
            )
            labels.append(u_boat_label)
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=max(1, len(labels)),
                frameon=False,
                title="Ship class" if color_by == "class" else None,
            )
    ax.set_aspect("equal", adjustable="box")
    return ax


def save_attack_frames(
    out_dir: str,
    ships_t0: list[Ship],
    torpedoes: list[Torpedo],
    t_start: float,
    t_end: float,
    fps: int,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    **kwargs,
) -> list[str]:
    """Render and save PNG frames for an attack sequence."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for attack frame rendering") from exc

    if fps <= 0:
        raise ValueError("fps must be positive")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / float(fps)
    times = np.arange(float(t_start), float(t_end) + 1e-9, dt)
    paths: list[str] = []
    hit_state = init_dynamic_hit_state(t_start)
    for idx, t in enumerate(times):
        fig, ax = plt.subplots(
            figsize=(6, 6),
            facecolor=kwargs.get("figure_facecolor"),
        )
        render_attack_frame(
            ships_t0,
            torpedoes,
            t_global=float(t),
            t_max=float(t_end),
            dynamics=dynamics,
            ax=ax,
            hit_state=hit_state,
            hit_dt=dt,
            **kwargs,
        )
        fig.tight_layout()
        path = out_path / f"frame_{idx:04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_attack_animation_mp4(
    out_path: str,
    ships_t0: list[Ship],
    torpedoes: list[Torpedo],
    t_start: float,
    t_end: float,
    fps: int,
    dynamics: tuple[ConvoyFormation, ConvoyKinematics] | None,
    **kwargs,
) -> str:
    """Save an MP4 animation if matplotlib animation is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import animation  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib animation support is required for MP4 output") from exc
    if not animation.writers.is_available("ffmpeg"):
        raise ImportError("ffmpeg is required for MP4 output; use save_attack_frames instead")

    if fps <= 0:
        raise ValueError("fps must be positive")
    dt = 1.0 / float(fps)
    times = np.arange(float(t_start), float(t_end) + 1e-9, dt)
    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor=kwargs.get("figure_facecolor"),
    )
    hit_state = init_dynamic_hit_state(t_start)

    def _update(frame_idx: int):
        ax.clear()
        t = float(times[frame_idx])
        render_attack_frame(
            ships_t0,
            torpedoes,
            t_global=t,
            t_max=float(t_end),
            dynamics=dynamics,
            ax=ax,
            hit_state=hit_state,
            hit_dt=dt,
            **kwargs,
        )
        return ax

    anim = animation.FuncAnimation(fig, _update, frames=len(times), interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def save_attack_debug_json(path: str, metrics: dict[str, Any]) -> str:
    """Serialize debug metrics to JSON and return path."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return path
