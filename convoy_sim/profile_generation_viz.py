"""Notebook-friendly plotting helpers for attack-profile generation QA."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship, ShipClass


SHIP_CLASS_COLORS = {
    ShipClass.FREIGHTER: "#0a0a0a",
    ShipClass.TANKER: "#38160d",
    ShipClass.ESCORT: "#001845",
    ShipClass.DECOY: "#ffc600",
}
ACCEPT_COLOR = "#1b9e77"
REJECT_COLOR = "#d95f02"
SEA_COLOR = "#06768d"
GRID_COLOR = "lightgrey"
U_BOAT_COLOR = "#111111"
U_BOAT_EDGE = "#f5f5f5"
FIXED_X_LIMITS = (-6000.0, 6000.0)


def style_ax(ax: Any, title: str) -> None:
    """Apply the shared sea-background plan-view style."""

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_facecolor(SEA_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")


def ship_polygon_xy(length: float, beam: float) -> np.ndarray:
    """Return a simple rectangle-and-triangle ship polygon centered at origin."""

    bow_len = 0.3 * float(length)
    bow_base = 0.5 * float(length) - bow_len
    stern = -0.5 * float(length)
    half_beam = 0.5 * float(beam)
    return np.array(
        [
            [-half_beam, stern],
            [-half_beam, bow_base],
            [0.0, 0.5 * float(length)],
            [half_beam, bow_base],
            [half_beam, stern],
            [-half_beam, stern],
        ],
        dtype=float,
    )


def rotate_translate_xy(points: np.ndarray, heading_rad: float, origin_xy: Sequence[float]) -> np.ndarray:
    """Rotate local points by ship heading and translate to world coordinates."""

    heading = float(heading_rad) - 0.5 * np.pi
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return (rot @ points.T).T + np.asarray(origin_xy, dtype=float)


def simple_ship_polygons(ax: Any, ships: Sequence[Ship]) -> None:
    """Draw scaled solid-black ship hull polygons on ``ax``."""

    from matplotlib.patches import Polygon

    for ship in ships:
        base = ship_polygon_xy(float(ship.length), float(ship.beam))
        poly = rotate_translate_xy(base, float(ship.heading_rad), ship.position)
        patch = Polygon(poly, closed=True, facecolor="#111111", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=3)
        ax.add_patch(patch)


def draw_u_boat(ax: Any, u_pos: Sequence[float], *, size: float = 24.0) -> None:
    """Draw a U-boat spawn marker."""

    ax.scatter(
        float(u_pos[0]),
        float(u_pos[1]),
        s=float(size),
        c=U_BOAT_COLOR,
        edgecolors=U_BOAT_EDGE,
        linewidths=0.6,
        zorder=5,
    )


def combined_limits(
    ships: Sequence[Ship],
    extra_points: np.ndarray | Sequence[Sequence[float]] | None = None,
    pad: float = 350.0,
) -> tuple[float, float, float, float]:
    """Return padded x/y limits covering ships and optional points."""

    ship_xy = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
    points = [ship_xy]
    if extra_points is not None and len(extra_points):
        points.append(np.asarray(extra_points, dtype=float))
    all_xy = np.vstack(points)
    xmin = float(np.min(all_xy[:, 0]) - pad)
    xmax = float(np.max(all_xy[:, 0]) + pad)
    ymin = float(np.min(all_xy[:, 1]) - pad)
    ymax = float(np.max(all_xy[:, 1]) + pad)
    return xmin, xmax, ymin, ymax


def apply_limits(
    ax: Any,
    limits: tuple[float, float, float, float],
    *,
    fixed_x_limits: tuple[float, float] | None = None,
) -> None:
    """Apply x/y limits, optionally overriding x with fixed bounds."""

    xmin, xmax, ymin, ymax = limits
    if fixed_x_limits is not None:
        xmin, xmax = fixed_x_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def plot_spawn_comparison(
    ax: Any,
    ships: Sequence[Ship],
    sample_df: Any,
    *,
    title: str,
    accepted_only: bool = False,
    limits: tuple[float, float, float, float] | None = None,) -> None:
    """Plot decoded/generated spawn points with accepted/rejected coloring."""
    style_ax(ax, title)
    simple_ship_polygons(ax, ships)
    plot_df = sample_df.loc[sample_df["passes_gate"]].copy() if accepted_only else sample_df.copy()
    if len(plot_df):
        plot_xy = np.vstack(plot_df["u_pos"].to_list())
        accepted = plot_df["passes_gate"].to_numpy(dtype=bool)
        rejected = ~accepted
        if accepted.any():
            acc_xy = plot_xy[accepted]
            ax.scatter(
                acc_xy[:, 0],
                acc_xy[:, 1],
                c=ACCEPT_COLOR,
                s=4,
                alpha=0.4,
                zorder=2,
                label="accepted",)
        if rejected.any():
            rej_xy = plot_xy[rejected]
            ax.scatter(
                rej_xy[:, 0],
                rej_xy[:, 1],
                c=REJECT_COLOR,
                s=4,
                alpha=0.4,
                zorder=2,
                label="rejected",)
        ax.legend(frameon=False)
        apply_limits(ax, limits or combined_limits(ships, extra_points=plot_xy))
    else:
        apply_limits(ax, limits or combined_limits(ships))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    


def torpedo_path_xy(torpedo: Any, n_points: int = 120) -> np.ndarray:
    """Return sampled torpedo path points for plotting."""

    t_end = float(torpedo.launch_delay + torpedo.max_run_time)
    ts = np.linspace(0.0, t_end, int(n_points))
    return np.array([torpedo.position_at(float(t)) for t in ts], dtype=float)


def plot_attack_example(
    ax: Any,
    ships: Sequence[Ship],
    profile: AttackProfile,
    *,
    title: str,
    accepted: bool,
    seed: int = 1945,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    """Plot one attack profile with torpedo paths and U-boat spawn marker."""

    style_ax(ax, title)
    simple_ship_polygons(ax, ships)
    line_color = ACCEPT_COLOR if accepted else REJECT_COLOR
    rng = np.random.default_rng(int(seed))
    torpedoes = profile.build_torpedoes(rng, ships=ships)
    all_points = [np.asarray([profile.u_pos], dtype=float)]
    for torpedo in torpedoes:
        xy = torpedo_path_xy(torpedo)
        all_points.append(xy)
        ax.plot(xy[:, 0], xy[:, 1], color=line_color, linewidth=1.15, alpha=0.75, zorder=1)
    draw_u_boat(ax, profile.u_pos)
    apply_limits(ax, limits or combined_limits(ships, extra_points=np.vstack(all_points)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
