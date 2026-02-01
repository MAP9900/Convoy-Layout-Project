"""Render basic convoy layout plan views to PNG."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.viz import save_planview_png


def main() -> None:
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rect = make_rectangular_convoy(
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
    save_planview_png(
        rect,
        str(out_dir / "rect_class.png"),
        title="Rectangular (class)",
        color_by="class",
        show_labels=False,
    )
    save_planview_png(
        rect,
        str(out_dir / "rect_value.png"),
        title="Rectangular (value)",
        color_by="value",
        show_labels=False,
    )

    staggered = make_staggered_convoy(
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
    save_planview_png(
        staggered,
        str(out_dir / "staggered_class.png"),
        title="Staggered (class)",
        color_by="class",
        show_labels=False,
    )
    save_planview_png(
        staggered,
        str(out_dir / "staggered_value.png"),
        title="Staggered (value)",
        color_by="value",
        show_labels=False,
    )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
