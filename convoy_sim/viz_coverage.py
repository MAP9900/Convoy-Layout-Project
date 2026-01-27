"""Optional matplotlib helpers for coverage heatmaps."""

from __future__ import annotations

from typing import Any

import numpy as np

from .entities import Ship
from .viz import compute_footprint_polygon, plot_footprint


def plot_coverage_heatmap(
    grid_density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ships: list[Ship] | None = None,
    ax: Any | None = None,
    title: str | None = None,
) -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for coverage plots") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.pcolormesh(x_edges, y_edges, grid_density, shading="auto", cmap="inferno")
    plt.colorbar(mesh, ax=ax, label="Coverage density")

    if ships:
        positions = np.array([ship.position for ship in ships], dtype=float)
        ax.scatter(positions[:, 0], positions[:, 1], c="white", s=20.0, edgecolors="black")
        poly = compute_footprint_polygon(ships, padding=0.0)
        plot_footprint(ax, poly, color="white", linewidth=1.0, linestyle="--")

    if title:
        ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    return ax
