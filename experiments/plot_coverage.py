"""Render torpedo coverage heatmaps from sampled attacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.coverage import accumulate_torpedo_coverage
from convoy_sim.layouts import make_rectangular_convoy
from convoy_sim.viz_coverage import plot_coverage_heatmap


def main() -> None:
    ships = make_rectangular_convoy(
        n_rows=3,
        n_cols=4,
        spacing_along=600.0,
        spacing_across=350.0,
        speed=5.0,
        heading_rad=0.0,
        length=140.0,
        beam=18.0,
        origin=np.array([0.0, 0.0]),
    )
    torpedoes_list = []
    for _ in range(200):
        torpedoes_list.append(
            fan_spread(
                u_pos=np.array([-2000.0, 0.0]),
                base_bearing_rad=0.0,
                n=4,
                spread_rad=np.radians(15.0),
                speed=25.0,
                max_run_time=800.0,
            )
        )
    bounds = (-2500.0, 500.0, -1500.0, 1500.0)
    data = accumulate_torpedo_coverage(
        torpedoes_list=torpedoes_list,
        t_max=400.0,
        bounds=bounds,
        grid_n=150,
        dt=5.0,
    )

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "coverage_heatmap.png"
    ax = plot_coverage_heatmap(
        data["grid_density"],
        data["x_edges"],
        data["y_edges"],
        ships=ships,
        title="Torpedo Coverage Heatmap",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
