"""Matplotlib-based plan-view visualization helpers (optional dependency)."""

from __future__ import annotations
from typing import Any, Literal
import numpy as np
from .entities import Ship, ShipClass


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for visualization helpers") from exc
    return plt


def ship_color(
    ship: Ship,
    color_by: Literal["class", "value"] = "class",
    cmap: str = "YlGnBu",
    *,
    value_range: tuple[float, float] | None = None,
) -> Any:
    """Return a matplotlib-friendly color for a ship."""

    if color_by == "class":
        class_colors = {
            ShipClass.FREIGHTER: "#1f77b4",
            ShipClass.TANKER: "#ff7f0e",
            ShipClass.ESCORT: "#2ca02c",
            ShipClass.DECOY: "#d62728",
        }
        return class_colors.get(ship.ship_class, "#7f7f7f")
    plt = _require_matplotlib()
    vmin, vmax = value_range if value_range is not None else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (float(ship.value_weight) - vmin) / (vmax - vmin)
    norm = float(np.clip(norm, 0.0, 1.0))
    return plt.get_cmap(cmap)(norm)


def compute_footprint_polygon(ships: list[Ship], padding: float = 0.0) -> np.ndarray:
    """Return a closed bounding-box polygon around ship centers."""

    if not ships:
        return np.zeros((0, 2), dtype=float)
    positions = np.array([ship.position for ship in ships], dtype=float)
    xmin, ymin = np.min(positions, axis=0) - float(padding)
    xmax, ymax = np.max(positions, axis=0) + float(padding)
    return np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ],
        dtype=float,
    )


def plot_footprint(ax: Any, poly: np.ndarray, **kwargs) -> None:
    """Plot a footprint outline on the given axes."""

    if poly.size == 0:
        return
    ax.plot(poly[:, 0], poly[:, 1], **kwargs)


def layout_summary(ships: list[Ship]) -> dict[str, Any]:
    """Return a compact summary of ship counts, value, and bbox extents."""

    counts: dict[str, int] = {
        ShipClass.FREIGHTER.value: 0,
        ShipClass.TANKER.value: 0,
        ShipClass.ESCORT.value: 0,
        ShipClass.DECOY.value: 0,
    }
    for ship in ships:
        counts[ship.ship_class.value] = counts.get(ship.ship_class.value, 0) + 1
    total_value = float(sum(ship.value_weight for ship in ships))
    if not ships:
        bbox = {"xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0}
    else:
        positions = np.array([ship.position for ship in ships], dtype=float)
        xmin, ymin = np.min(positions, axis=0)
        xmax, ymax = np.max(positions, axis=0)
        bbox = {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
        }
    return {
        "counts_by_class": counts,
        "total_value": total_value,
        "bbox": bbox,
        "bbox_along": float(bbox["xmax"] - bbox["xmin"]),
        "bbox_across": float(bbox["ymax"] - bbox["ymin"]),
    }


def add_summary_text(
    ax: Any,
    summary: dict[str, Any],
    loc: str = "upper right",
    title: str = "",
) -> None:
    """Render a summary box on the axes."""

    plt = _require_matplotlib()
    loc_map = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }
    if loc not in loc_map:
        raise ValueError(f"Unknown loc: {loc}")
    x, y, ha, va = loc_map[loc]
    counts = summary.get("counts_by_class", {})
    lines = []
    if title:
        lines.append(str(title))
    lines.extend(
        [
            f"freighter: {counts.get(ShipClass.FREIGHTER.value, 0)}",
            f"tanker: {counts.get(ShipClass.TANKER.value, 0)}",
            f"escort: {counts.get(ShipClass.ESCORT.value, 0)}",
            f"decoy: {counts.get(ShipClass.DECOY.value, 0)}",
            f"value: {summary.get('total_value', 0.0):.1f}",
            f"bbox: {summary.get('bbox_along', 0.0):.1f} x {summary.get('bbox_across', 0.0):.1f}",
        ]
    )
    text = "\n".join(lines)
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )


def plot_convoy_planview(
    ships: list[Ship],
    ax: Any | None = None,
    title: str | None = None,
    color_by: Literal["class", "value"] = "class",
    show_labels: bool = False,
    show_footprint: bool = True,
    footprint_padding: float = 0.0,
    ship_marker_size: float = 40.0,
    alpha: float = 0.9,
    axes_facecolor: str = "#06768d",
    value_cmap: str = "YlGnBu",
) -> Any:
    """Plot convoy ship centers in plan view."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(axes_facecolor)
    positions = np.array([ship.position for ship in ships], dtype=float)
    colors: list[Any] = []
    if color_by == "value":
        values = np.array([ship.value_weight for ship in ships], dtype=float)
        vmin = float(np.min(values)) if len(values) else 0.0
        vmax = float(np.max(values)) if len(values) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=values,
            cmap=value_cmap,
            vmin=vmin,
            vmax=vmax,
            s=float(ship_marker_size),
            alpha=float(alpha),
        )
    else:
        colors = [ship_color(ship, color_by=color_by) for ship in ships]
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=colors,
            s=float(ship_marker_size),
            alpha=float(alpha),
        )

    if show_labels:
        for ship in ships:
            ax.annotate(ship.id, (ship.position[0], ship.position[1]), fontsize=8)

    if show_footprint:
        poly = compute_footprint_polygon(ships, padding=footprint_padding)
        plot_footprint(ax, poly, color="black", linewidth=1.0, linestyle="--")

    if title:
        ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")

    if color_by == "class":
        handles = []
        labels = []
        for ship_class, color in {
            ShipClass.FREIGHTER: "#1f77b4",
            ShipClass.TANKER: "#ff7f0e",
            ShipClass.ESCORT: "#2ca02c",
            ShipClass.DECOY: "#d62728",
        }.items():
            handles.append(ax.scatter([], [], c=color, s=ship_marker_size))
            labels.append(ship_class.value)
        ax.legend(
            handles,
            labels,
            title="Ship class",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=len(labels),
            frameon=False,
        )
    else:
        plt.colorbar(scatter, ax=ax, label="Value weight")
    return ax


def plot_layout_overlay(
    ships_a: list[Ship],
    ships_b: list[Ship],
    labels: tuple[str, str] = ("A", "B"),
    ax: Any | None = None,
    color_by: Literal["class", "value"] = "class",
    show_footprints: bool = True,
    footprint_padding: float = 0.0,
    alpha_a: float = 0.6,
    alpha_b: float = 0.6,
) -> Any:
    """Overlay two layouts with distinct markers and optional footprints."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    values = np.array([s.value_weight for s in ships_a + ships_b], dtype=float)
    vmin = float(np.min(values)) if len(values) else 0.0
    vmax = float(np.max(values)) if len(values) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    colors_a = [
        ship_color(ship, color_by=color_by, value_range=(vmin, vmax)) for ship in ships_a
    ]
    colors_b = [
        ship_color(ship, color_by=color_by, value_range=(vmin, vmax)) for ship in ships_b
    ]
    pos_a = np.array([ship.position for ship in ships_a], dtype=float)
    pos_b = np.array([ship.position for ship in ships_b], dtype=float)
    ax.scatter(pos_a[:, 0], pos_a[:, 1], c=colors_a, marker="o", alpha=float(alpha_a), label=labels[0])
    ax.scatter(pos_b[:, 0], pos_b[:, 1], c=colors_b, marker="^", alpha=float(alpha_b), label=labels[1])

    if show_footprints:
        poly_a = compute_footprint_polygon(ships_a, padding=footprint_padding)
        poly_b = compute_footprint_polygon(ships_b, padding=footprint_padding)
        plot_footprint(ax, poly_a, color="black", linewidth=1.0, linestyle="--")
        plot_footprint(ax, poly_b, color="gray", linewidth=1.0, linestyle=":")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(title="Layout")
    return ax


def plot_layout_comparison_grid(
    layouts: list[tuple[str, list[Ship]]],
    ncols: int = 2,
    color_by: Literal["class", "value"] = "class",
    show_footprint: bool = True,
) -> Any:
    """Plot multiple layouts in a grid of subplots and return the figure."""

    plt = _require_matplotlib()
    n_layouts = len(layouts)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_layouts / ncols)) if n_layouts else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, (name, ships) in zip(axes_list, layouts):
        plot_convoy_planview(
            ships,
            ax=ax,
            title=name,
            color_by=color_by,
            show_footprint=show_footprint,
        )
    for ax in axes_list[len(layouts):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_planview_png(ships: list[Ship], path: str, **plot_kwargs) -> str:
    """Save a plan-view figure to disk and return the path."""

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="lightgrey")
    plot_convoy_planview(ships, ax=ax, **plot_kwargs)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
