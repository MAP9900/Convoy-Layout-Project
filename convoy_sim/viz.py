"""Matplotlib-based plan-view visualization helpers (optional dependency)."""

from __future__ import annotations
from typing import Any, Literal
import numpy as np
from convoy_sim.entities import Ship, ShipClass


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("matplotlib is required for visualization helpers") from exc
    return plt


def _ship_marker_polygon(length: float, beam: float) -> np.ndarray:
    """Return a simple ship-like polygon centered at the origin pointing +y."""

    length = float(length)
    beam = float(beam)
    if length <= 0.0 or beam <= 0.0:
        return np.zeros((0, 2), dtype=float)
    bow_len = 0.3 * length
    bow_base = 0.5 * length - bow_len
    stern = -0.5 * length
    half_beam = 0.5 * beam
    return np.array(
        [
            [-half_beam, stern],
            [-half_beam, bow_base],
            [0.0, 0.5 * length],
            [half_beam, bow_base],
            [half_beam, stern],
            [-half_beam, stern],
        ],
        dtype=float,
    )


def _add_ship_polygons(
    ax: Any,
    ships: list[Ship],
    colors: list[Any],
    *,
    alpha: float,
    ship_marker_size: float,
    rotate_by_heading: bool,
    cmap: str | None = None,
    norm: Any | None = None,
    use_hull_dimensions: bool = False,
) -> Any:
    """Add ship-shaped polygons to the axes and return the collection."""

    from matplotlib.collections import PatchCollection  # type: ignore
    from matplotlib.patches import Polygon  # type: ignore

    patches = []
    for ship in ships:
        heading = (ship.heading_rad - 0.5 * np.pi) if rotate_by_heading else 0.0
        if use_hull_dimensions:
            length = float(ship.length)
            beam = float(ship.beam)
        else:
            length = float(ship_marker_size)
            beam = float(ship_marker_size) * 0.35
        base_poly = _ship_marker_polygon(length, beam)
        if base_poly.size == 0:
            continue
        cos_a = float(np.cos(heading))
        sin_a = float(np.sin(heading))
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
        rotated = (rotation @ base_poly.T).T
        translated = rotated + ship.position
        patches.append(Polygon(translated, closed=True))

    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        alpha=float(alpha),
        linewidths=0.0,
    )
    if cmap is not None and norm is not None:
        collection.set_array(np.asarray(colors, dtype=float))
    else:
        collection.set_facecolor(colors)
    ax.add_collection(collection)
    return collection


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
            ShipClass.FREIGHTER: "#0a0a0a",
            ShipClass.TANKER: "#38160d",
            ShipClass.ESCORT: "#001845",
            ShipClass.DECOY: "#ffc600",
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
    alpha: float = 1.0,
    axes_facecolor: str = "#06768d",
    value_cmap: str = "YlGnBu",
    grid_color: str = "lightgrey",
    ship_marker: Literal["circle", "ship"] = "circle",
    rotate_by_heading: bool = False,
    use_hull_dimensions: bool = False,
    highlight_ids: set[str] | None = None,
    highlight_color: str = "red",) -> Any:
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
        colors = list(values)
        mappable = plt.cm.ScalarMappable(
            cmap=value_cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        mappable.set_array(np.asarray(colors, dtype=float))
    else:
        colors = [ship_color(ship, color_by=color_by) for ship in ships]
        if highlight_ids:
            colors = [
                (highlight_color if ship.id in highlight_ids else color)
                for ship, color in zip(ships, colors)
            ]

    if ship_marker == "ship":
        if color_by == "value":
            scatter = _add_ship_polygons(
                ax,
                ships,
                colors,
                alpha=alpha,
                ship_marker_size=ship_marker_size,
                rotate_by_heading=rotate_by_heading,
                cmap=value_cmap,
                norm=mappable.norm,
                use_hull_dimensions=use_hull_dimensions,
            )
        else:
            scatter = _add_ship_polygons(
                ax,
                ships,
                colors,
                alpha=alpha,
                ship_marker_size=ship_marker_size,
                rotate_by_heading=rotate_by_heading,
                use_hull_dimensions=use_hull_dimensions,
            )
        if highlight_ids and color_by == "value":
            highlight_ships = [ship for ship in ships if ship.id in highlight_ids]
            if highlight_ships:
                highlight_colors = [highlight_color] * len(highlight_ships)
                overlay = _add_ship_polygons(
                    ax,
                    highlight_ships,
                    highlight_colors,
                    alpha=1.0,
                    ship_marker_size=ship_marker_size,
                    rotate_by_heading=rotate_by_heading,
                    use_hull_dimensions=use_hull_dimensions,
                )
                overlay.set_zorder(4)
        if len(positions):
            if use_hull_dimensions:
                max_length = max(float(ship.length) for ship in ships)
                pad = max(1.0, max_length * 0.6)
            else:
                pad = max(1.0, ship_marker_size * 0.6)
            ax.set_xlim(float(np.min(positions[:, 0]) - pad), float(np.max(positions[:, 0]) + pad))
            ax.set_ylim(float(np.min(positions[:, 1]) - pad), float(np.max(positions[:, 1]) + pad))
    else:
        if color_by == "value":
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=colors,
                cmap=value_cmap,
                vmin=vmin,
                vmax=vmax,
                s=float(ship_marker_size),
                alpha=float(alpha),
            )
            if highlight_ids:
                highlight_positions = np.array(
                    [ship.position for ship in ships if ship.id in highlight_ids],
                    dtype=float,
                )
                if len(highlight_positions):
                    ax.scatter(
                        highlight_positions[:, 0],
                        highlight_positions[:, 1],
                        c=highlight_color,
                        s=float(ship_marker_size),
                        alpha=1.0,
                        zorder=4,
                    )
        else:
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
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.75, linestyle='--')

    if color_by == "class":
        handles = []
        labels = []
        for ship_class, color in {
            ShipClass.FREIGHTER: "#0a0a0a",
            ShipClass.TANKER: "#38160d",
            ShipClass.ESCORT: "#001845",
            ShipClass.DECOY: "#ffc600",
        }.items():
            handles.append(ax.scatter([], [], c=color, s=ship_marker_size))
            labels.append(ship_class.value)
        if ship_marker == "ship":
            from matplotlib.patches import Patch  # type: ignore

            handles = [Patch(facecolor=color, edgecolor="none") for color in {
                ShipClass.FREIGHTER: "#0a0a0a",
                ShipClass.TANKER: "#38160d",
                ShipClass.ESCORT: "#001845",
                ShipClass.DECOY: "#ffc600",
            }.values()]
        ax.legend(
            handles,
            labels,
            title="Ship class",
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=len(labels),
            frameon=False,
        )
    else:
        if ship_marker == "ship":
            plt.colorbar(mappable, ax=ax, label="Value weight")
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
    alpha_a: float = 1.0,
    alpha_b: float = 1.0,
    ship_marker_size: float = 40.0,
    ship_marker: Literal["circle", "ship"] = "circle",
    rotate_by_heading: bool = False,
    use_hull_dimensions: bool = False,
) -> Any:
    """Overlay two layouts with distinct markers and optional footprints."""

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
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
    if ship_marker == "ship":
        _add_ship_polygons(
            ax,
            ships_a,
            colors_a,
            alpha=alpha_a,
            ship_marker_size=ship_marker_size,
            rotate_by_heading=rotate_by_heading,
            use_hull_dimensions=use_hull_dimensions,
        )
        _add_ship_polygons(
            ax,
            ships_b,
            colors_b,
            alpha=alpha_b,
            ship_marker_size=ship_marker_size,
            rotate_by_heading=rotate_by_heading,
            use_hull_dimensions=use_hull_dimensions,
        )
        from matplotlib.patches import Patch  # type: ignore

        handles = [
            Patch(facecolor="0.2", edgecolor="none", alpha=float(alpha_a)),
            Patch(facecolor="0.6", edgecolor="none", alpha=float(alpha_b)),
        ]
        ax.legend(handles, list(labels), title="Layout")
    else:
        ax.scatter(
            pos_a[:, 0],
            pos_a[:, 1],
            c=colors_a,
            marker="o",
            alpha=float(alpha_a),
            label=labels[0],
        )
        ax.scatter(
            pos_b[:, 0],
            pos_b[:, 1],
            c=colors_b,
            marker="^",
            alpha=float(alpha_b),
            label=labels[1],
        )

    if show_footprints:
        poly_a = compute_footprint_polygon(ships_a, padding=footprint_padding)
        poly_b = compute_footprint_polygon(ships_b, padding=footprint_padding)
        plot_footprint(ax, poly_a, color="black", linewidth=1.0, linestyle="--")
        plot_footprint(ax, poly_b, color="gray", linewidth=1.0, linestyle=":")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    return ax


def plot_layout_comparison_grid(
    layouts: list[tuple[str, list[Ship]]],
    ncols: int = 2,
    color_by: Literal["class", "value"] = "class",
    show_footprint: bool = True,
    ship_marker: Literal["circle", "ship"] = "circle",
    rotate_by_heading: bool = False,
) -> Any:
    """Plot multiple layouts in a grid of subplots and return the figure."""

    plt = _require_matplotlib()
    n_layouts = len(layouts)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_layouts / ncols)) if n_layouts else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows), facecolor="lightgrey")
    axes_list = np.atleast_1d(axes).ravel()
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    for ax, (name, ships) in zip(axes_list, layouts):
        plot_convoy_planview(
            ships,
            ax=ax,
            title=name,
            color_by=color_by,
            show_footprint=show_footprint,
            ship_marker=ship_marker,
            rotate_by_heading=rotate_by_heading,
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
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.32)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
