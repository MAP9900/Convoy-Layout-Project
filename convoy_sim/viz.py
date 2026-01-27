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
    cmap: str = "viridis",
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
) -> Any:
    """Plot convoy ship centers in plan view."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    positions = np.array([ship.position for ship in ships], dtype=float)
    colors = []
    if color_by == "value":
        values = np.array([ship.value_weight for ship in ships], dtype=float)
        vmin = float(np.min(values)) if len(values) else 0.0
        vmax = float(np.max(values)) if len(values) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        colors = [
            ship_color(ship, color_by="value", cmap="viridis", value_range=(vmin, vmax))
            for ship in ships
        ]
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
        ax.legend(handles, labels, title="Ship class")
    else:
        plt.colorbar(scatter, ax=ax, label="Value weight")
    return ax


def save_planview_png(ships: list[Ship], path: str, **plot_kwargs) -> str:
    """Save a plan-view figure to disk and return the path."""

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_convoy_planview(ships, ax=ax, **plot_kwargs)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
