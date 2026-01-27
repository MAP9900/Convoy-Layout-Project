"""Pure-numpy coverage estimators for torpedo paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from .entities import Torpedo


def _grid_edges(bounds: tuple[float, float, float, float], grid_n: int) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    x_edges = np.linspace(float(xmin), float(xmax), int(grid_n) + 1)
    y_edges = np.linspace(float(ymin), float(ymax), int(grid_n) + 1)
    return x_edges, y_edges


def accumulate_torpedo_coverage(
    torpedoes_list: list[list[Torpedo]],
    t_max: float,
    bounds: tuple[float, float, float, float],
    grid_n: int = 200,
    dt: float = 2.0,
) -> dict[str, Any]:
    """Accumulate torpedo path coverage counts on a 2D grid."""

    if grid_n <= 0:
        raise ValueError("grid_n must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    window = float(t_max)
    x_edges, y_edges = _grid_edges(bounds, grid_n)
    counts = np.zeros((grid_n, grid_n), dtype=float)

    for torpedoes in torpedoes_list:
        for torpedo in torpedoes:
            t_end = min(window, float(torpedo.max_run_time))
            times = np.arange(0.0, t_end + 1e-9, float(dt))
            positions = np.array([torpedo.position_at(t) for t in times], dtype=float)
            x_idx = np.searchsorted(x_edges, positions[:, 0], side="right") - 1
            y_idx = np.searchsorted(y_edges, positions[:, 1], side="right") - 1
            mask = (x_idx >= 0) & (x_idx < grid_n) & (y_idx >= 0) & (y_idx < grid_n)
            for xi, yi in zip(x_idx[mask], y_idx[mask]):
                counts[int(yi), int(xi)] += 1.0

    total = float(np.sum(counts))
    density = counts / total if total > 0.0 else counts
    return {
        "grid_counts": counts,
        "grid_density": density,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def accumulate_hit_probability_map(
    torpedoes_list: list[list[Torpedo]],
    t_max: float,
    bounds: tuple[float, float, float, float],
    grid_n: int = 200,
    dt: float = 2.0,
    radius: float = 50.0,
) -> dict[str, Any]:
    """Approximate probability of any torpedo passing within radius."""

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    data = accumulate_torpedo_coverage(
        torpedoes_list=torpedoes_list,
        t_max=t_max,
        bounds=bounds,
        grid_n=grid_n,
        dt=dt,
    )
    counts = data["grid_counts"]
    if radius > 0.0:
        pixels = max(0, int(round(radius / (x_edges[1] - x_edges[0]))))
    else:
        pixels = 0
    hits = (counts > 0).astype(float)
    if pixels > 0:
        padded = np.pad(hits, pixels, mode="constant", constant_values=0.0)
        expanded = np.zeros_like(hits)
        for dy in range(-pixels, pixels + 1):
            for dx in range(-pixels, pixels + 1):
                if dx * dx + dy * dy > pixels * pixels:
                    continue
                expanded = np.maximum(
                    expanded,
                    padded[pixels + dy : pixels + dy + hits.shape[0], pixels + dx : pixels + dx + hits.shape[1]],
                )
        hits = expanded
    data["grid_hit_prob"] = hits
    data["radius"] = float(radius)
    return data
