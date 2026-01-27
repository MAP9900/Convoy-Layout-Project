"""Optional visualization and debug helpers for static attacks."""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

from .entities import Ship, Torpedo
from .geometry import as_vec, closest_approach_time, distance
from .viz import plot_convoy_planview


def torpedo_segment(
    torpedo: Torpedo,
    t0: float = 0.0,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the start/end points of the torpedo ray segment."""

    start_t = float(max(0.0, t0))
    end_t = float(torpedo.max_run_time if t1 is None else min(t1, torpedo.max_run_time))
    p_start = torpedo.position_at(start_t)
    p_end = torpedo.position_at(end_t)
    return p_start, p_end


def min_miss_distance_ship_torpedo(
    ship: Ship,
    torpedo: Torpedo,
    t_max: float,
    dt: float | None = None,
) -> float:
    """Return the minimum distance between a static ship and torpedo track."""

    window = min(float(t_max), float(torpedo.max_run_time))
    if window <= 0.0:
        return float("inf")
    if dt is not None:
        times = np.arange(0.0, window + 1e-9, float(dt))
        positions = np.array([torpedo.position_at(t) for t in times], dtype=float)
        dists = np.linalg.norm(positions - ship.position, axis=1)
        return float(np.min(dists))
    t_star = closest_approach_time(
        ship.position,
        as_vec(0.0, 0.0),
        torpedo.launch_position,
        torpedo.velocity_vec(),
    )
    t_star = float(np.clip(t_star, 0.0, window))
    torp_pos = torpedo.position_at(t_star)
    return distance(ship.position, torp_pos)


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
    )
    metrics = attack_debug_metrics(ships, torpedoes, t_max)
    hit_ship_ids = {t["hit_ship_id"] for t in metrics["torpedoes"] if t["hit_ship_id"]}

    for torpedo in torpedoes:
        p0, p1 = torpedo_segment(torpedo, t0=0.0, t1=t_max)
        hit = any(t["id"] == torpedo.id and t["hit_ship_id"] for t in metrics["torpedoes"])
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="red" if hit else "gray",
            linewidth=1.5 if hit else 1.0,
            alpha=float(ray_alpha),
            linestyle="-" if hit else "--",
        )

    for ship in ships:
        if ship.id in hit_ship_ids:
            ax.scatter(
                ship.position[0],
                ship.position[1],
                s=120.0,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
            )

    if show_miss_annotations and torpedoes:
        misses = [
            (t["min_miss_distance"], t)
            for t in metrics["torpedoes"]
            if t["hit_ship_id"] is None
        ]
        misses.sort(key=lambda x: x[0])
        for dist, torp in misses[: max(0, int(miss_k))]:
            torpedo = next(t for t in torpedoes if t.id == torp["id"])
            p0, p1 = torpedo_segment(torpedo, t0=0.0, t1=t_max)
            ax.annotate(f"{dist:.1f}m", (p1[0], p1[1]), fontsize=8, color="gray")

    return ax


def save_attack_debug_json(path: str, metrics: dict[str, Any]) -> str:
    """Serialize debug metrics to JSON and return path."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return path
