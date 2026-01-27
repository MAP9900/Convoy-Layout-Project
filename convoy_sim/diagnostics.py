"""Diagnostics helpers for comparing layouts and attack outcomes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .entities import Ship, ShipClass, Torpedo
from .geometry import distance
from .viz import add_summary_text, layout_summary, plot_convoy_planview
from .viz_attack import plot_attack_planview
from .viz_coverage import plot_coverage_heatmap


def _counts_by_class(ships: list[Ship]) -> dict[str, int]:
    counts = {c.value: 0 for c in ShipClass}
    for ship in ships:
        counts[ship.ship_class.value] = counts.get(ship.ship_class.value, 0) + 1
    return counts


def _value_by_class(ships: list[Ship]) -> dict[str, float]:
    values = {c.value: 0.0 for c in ShipClass}
    for ship in ships:
        values[ship.ship_class.value] = values.get(ship.ship_class.value, 0.0) + float(ship.value_weight)
    return values


def _nearest_neighbor_stats(ships: list[Ship]) -> dict[str, float]:
    if len(ships) < 2:
        return {"min": 0.0, "mean": 0.0}
    positions = [ship.position for ship in ships]
    distances = []
    for i, pos in enumerate(positions):
        min_d = float("inf")
        for j, other in enumerate(positions):
            if i == j:
                continue
            min_d = min(min_d, distance(pos, other))
        distances.append(min_d)
    return {"min": float(np.min(distances)), "mean": float(np.mean(distances))}


def compare_layout_metrics(ships_a: list[Ship], ships_b: list[Ship]) -> dict[str, Any]:
    """Compare layout metrics between two ship sets."""

    summary_a = layout_summary(ships_a)
    summary_b = layout_summary(ships_b)
    nn_a = _nearest_neighbor_stats(ships_a)
    nn_b = _nearest_neighbor_stats(ships_b)

    return {
        "before": {
            "summary": summary_a,
            "counts_by_class": _counts_by_class(ships_a),
            "value_by_class": _value_by_class(ships_a),
            "nn_distance": nn_a,
        },
        "after": {
            "summary": summary_b,
            "counts_by_class": _counts_by_class(ships_b),
            "value_by_class": _value_by_class(ships_b),
            "nn_distance": nn_b,
        },
        "delta": {
            "bbox_area": summary_b["bbox_along"] * summary_b["bbox_across"]
            - summary_a["bbox_along"] * summary_a["bbox_across"],
            "nn_min": nn_b["min"] - nn_a["min"],
            "nn_mean": nn_b["mean"] - nn_a["mean"],
            "total_value": summary_b["total_value"] - summary_a["total_value"],
        },
    }


def compare_attack_outcomes(samples_a: dict[str, Any], samples_b: dict[str, Any]) -> dict[str, Any]:
    """Compare Monte Carlo outputs between two runs."""

    return {
        "delta_expected_hits": float(samples_b.get("expected_hits", 0.0) - samples_a.get("expected_hits", 0.0)),
        "delta_expected_value": float(
            samples_b.get("expected_value_destroyed", 0.0) - samples_a.get("expected_value_destroyed", 0.0)
        ),
        "delta_p_hit_ge_1": float(
            samples_b.get("hit_prob_at_least_one", 0.0) - samples_a.get("hit_prob_at_least_one", 0.0)
        ),
        "delta_VaR_90": float(samples_b.get("VaR_90", 0.0) - samples_a.get("VaR_90", 0.0)),
        "delta_CVaR_90": float(samples_b.get("CVaR_90", 0.0) - samples_a.get("CVaR_90", 0.0)),
    }


def lane_vulnerability_proxy(
    ships: list[Ship],
    headings: np.ndarray,
    n_rays: int = 200,
) -> dict[str, Any]:
    """Approximate lane vulnerability by counting circle intersections per ray."""

    if not ships:
        return {"headings": headings.tolist(), "max_hits": [], "mean_hits": [], "lane_score": []}
    positions = np.array([ship.position for ship in ships], dtype=float)
    radii = np.array([ship.effective_hit_radius() for ship in ships], dtype=float)
    xmin, ymin = np.min(positions, axis=0)
    xmax, ymax = np.max(positions, axis=0)
    bbox = (xmin, xmax, ymin, ymax)

    max_hits = []
    mean_hits = []
    lane_score = []
    for heading in headings:
        direction = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        centers = np.linspace(-1.0, 1.0, n_rays)
        hits_per_ray = []
        for offset in centers:
            origin = np.array(
                [
                    (bbox[0] + bbox[1]) / 2.0,
                    (bbox[2] + bbox[3]) / 2.0,
                ]
            )
            origin = origin + perp * offset * max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 0.6
            count = 0
            for pos, radius in zip(positions, radii):
                rel = pos - origin
                proj = np.dot(rel, perp)
                if abs(proj) <= radius:
                    count += 1
            hits_per_ray.append(count)
        max_hits.append(float(np.max(hits_per_ray)))
        mean_hits.append(float(np.mean(hits_per_ray)))
        lane_score.append(float(np.max(hits_per_ray) - np.mean(hits_per_ray)))
    return {
        "headings": headings.tolist(),
        "max_hits": max_hits,
        "mean_hits": mean_hits,
        "lane_score": lane_score,
    }


def plot_before_after_layout(
    ships_before: list[Ship],
    ships_after: list[Ship],
    names: tuple[str, str] = ("Before", "After"),
    color_by: Literal["class", "value"] = "class",
    out_path: str | None = None,
) -> Any:
    """Plot side-by-side plan views with summary text."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for diagnostics plotting") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_convoy_planview(ships_before, ax=axes[0], title=names[0], color_by=color_by, show_footprint=True)
    plot_convoy_planview(ships_after, ax=axes[1], title=names[1], color_by=color_by, show_footprint=True)
    add_summary_text(axes[0], layout_summary(ships_before), loc="upper right", title=names[0])
    add_summary_text(axes[1], layout_summary(ships_after), loc="upper right", title=names[1])
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_before_after_attack_overlay(
    ships_before: list[Ship],
    torps_before: list[Torpedo],
    ships_after: list[Ship],
    torps_after: list[Torpedo],
    t_max: float,
    out_path: str | None = None,
    color_by: Literal["class", "value"] = "class",
) -> Any:
    """Plot a before/after attack overlay in two panels."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for diagnostics plotting") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_attack_planview(ships_before, torps_before, t_max=t_max, ax=axes[0], title="Before", color_by=color_by)
    plot_attack_planview(ships_after, torps_after, t_max=t_max, ax=axes[1], title="After", color_by=color_by)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_coverage_comparison(
    cov_before: dict[str, Any],
    cov_after: dict[str, Any],
    ships_overlay: list[Ship] | None = None,
    out_path: str | None = None,
) -> Any:
    """Plot side-by-side coverage heatmaps and a difference panel."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for diagnostics plotting") from exc

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_coverage_heatmap(
        cov_before["grid_density"],
        cov_before["x_edges"],
        cov_before["y_edges"],
        ships=ships_overlay,
        ax=axes[0],
        title="Before",
    )
    plot_coverage_heatmap(
        cov_after["grid_density"],
        cov_after["x_edges"],
        cov_after["y_edges"],
        ships=ships_overlay,
        ax=axes[1],
        title="After",
    )
    diff = cov_after["grid_density"] - cov_before["grid_density"]
    plot_coverage_heatmap(
        diff,
        cov_before["x_edges"],
        cov_before["y_edges"],
        ships=ships_overlay,
        ax=axes[2],
        title="After - Before",
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def render_diagnostics_report(
    ships_before: list[Ship],
    ships_after: list[Ship],
    mc_before: dict[str, Any] | None,
    mc_after: dict[str, Any] | None,
    lane_before: dict[str, Any] | None,
    lane_after: dict[str, Any] | None,
    out_json_path: str,
) -> str:
    """Write a JSON summary report with high-level explanations."""

    layout_cmp = compare_layout_metrics(ships_before, ships_after)
    attack_cmp = None
    if mc_before and mc_after:
        attack_cmp = compare_attack_outcomes(mc_before, mc_after)

    explanations = []
    delta_nn = layout_cmp["delta"]["nn_min"]
    if delta_nn > 0.0:
        explanations.append("Increased minimum ship spacing reduces clustering.")
    if lane_before and lane_after:
        before_score = np.max(lane_before.get("lane_score", [0.0]))
        after_score = np.max(lane_after.get("lane_score", [0.0]))
        if after_score < before_score:
            explanations.append("Reduced peak lane score suggests fewer aligned corridors.")
    if attack_cmp and attack_cmp.get("delta_expected_value", 0.0) < 0.0:
        explanations.append("Lower expected value destroyed indicates improved survivability.")

    payload = {
        "layout_comparison": layout_cmp,
        "attack_comparison": attack_cmp,
        "lane_before": lane_before,
        "lane_after": lane_after,
        "explanations": explanations[:3],
    }
    with open(out_json_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)
    return out_json_path
