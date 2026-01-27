"""Plot overlay and comparison views for historical vs optimized layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.viz import (
    add_summary_text,
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
    add_summary_text(overlay_ax, layout_summary(historical), loc="upper left", title="Historical")
    add_summary_text(overlay_ax, layout_summary(optimized), loc="upper right", title="Optimized")
    overlay_ax.figure.tight_layout()
    overlay_ax.figure.savefig(overlay_path, dpi=150)

    grid_fig = plot_layout_comparison_grid(
        [
            ("Historical", historical),
            (optimized_label, optimized),
        ],
        ncols=2,
        color_by="class",
        show_footprint=True,
    )
    grid_path = out_dir / "historical_vs_optimized_grid.png"
    grid_fig.savefig(grid_path, dpi=150)

    print(f"Wrote {overlay_path}")
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
