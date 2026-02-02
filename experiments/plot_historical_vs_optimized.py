"""Plot overlay and comparison views for historical vs optimized layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D  # type: ignore

from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.viz import (
    layout_summary,
    plot_layout_comparison_grid,
    plot_layout_overlay,
    save_planview_png,
)
from scenarios.scenario_a import build_scenario_a


def _load_optimized_params(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("params", {})


def main() -> None:
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario_a(n_trials=1, rng_seed=0)
    base_kwargs = dict(scenario.layout_kwargs)
    historical = make_rectangular_convoy(**base_kwargs)

    optimized_params = _load_optimized_params(Path("results/defender_best.json"))
    if optimized_params:
        optimized_kwargs = dict(base_kwargs)
        optimized_kwargs.update(optimized_params)
        optimized = make_rectangular_convoy(**optimized_kwargs)
        optimized_label = "Optimized (rect)"
    else:
        optimized_kwargs = dict(base_kwargs)
        optimized_kwargs["spacing_across"] = float(base_kwargs["spacing_across"]) * 1.4
        optimized_kwargs["spacing_along"] = float(base_kwargs["spacing_along"]) * 1.2
        optimized = make_staggered_convoy(**optimized_kwargs)
        optimized_label = "Optimized (staggered)"

    overlay_path = out_dir / "historical_vs_optimized_overlay.png"
    overlay_ax = plot_layout_overlay(
        historical,
        optimized,
        labels=("Historical", optimized_label),
        color_by="class",
        show_footprints=True,
        footprint_padding=0.0,
    )
    overlay_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor="0.2",
            markeredgecolor="0.2",
            label="Historical (circle)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markersize=7,
            markerfacecolor="0.2",
            markeredgecolor="0.2",
            label="Optimized (triangle)",
        ),
    ]
    overlay_ax.legend(
        handles=overlay_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
    )
    def _summary_lines(summary: dict, title: str) -> str:
        counts = summary.get("counts_by_class", {})
        lines = [
            title,
            f"freighter: {counts.get('freighter', 0)}",
            f"tanker: {counts.get('tanker', 0)}",
            f"escort: {counts.get('escort', 0)}",
            f"decoy: {counts.get('decoy', 0)}",
            f"value: {summary.get('total_value', 0.0):.1f}",
            f"bbox: {summary.get('bbox_along', 0.0):.1f} x {summary.get('bbox_across', 0.0):.1f}",
        ]
        return "\n".join(lines)

    overlay_ax.text(
        1.02,
        0.98,
        _summary_lines(layout_summary(historical), "Historical"),
        transform=overlay_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )
    overlay_ax.text(
        1.02,
        0.45,
        _summary_lines(layout_summary(optimized), "Optimized"),
        transform=overlay_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )
    overlay_ax.figure.tight_layout(rect=(0.0, 0.08, 0.78, 0.92))
    overlay_ax.figure.subplots_adjust(bottom=0.28, right=0.74)
    overlay_ax.figure.savefig(overlay_path, dpi=150)

    grid_fig = plot_layout_comparison_grid(
        [
            ("Historical", historical),
            (optimized_label, optimized),
        ],
        ncols=2,
        color_by="class",
        show_footprint=False,
        ship_marker="ship",
    )
    for ax in grid_fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    grid_path = out_dir / "historical_vs_optimized_grid.png"
    grid_fig.savefig(grid_path, dpi=150)

    print(f"Wrote {overlay_path}")
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
