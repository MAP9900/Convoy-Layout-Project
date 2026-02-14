"""Render a single static attack with torpedo rays and debug metrics."""

from __future__ import annotations
import argparse

from pathlib import Path

import json

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from convoy_sim.attackers import fan_spread
from convoy_sim.viz_attack import plot_attack_planview, save_attack_debug_json, attack_debug_metrics
from scenarios.scenario_a import build_scenario_a
from scenarios.scenario_rl import build_scenario_rl


SCENARIOS = {
    "scenario_a": build_scenario_a,
    "scenario_rl": build_scenario_rl,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a static attack plan-view figure")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS.keys(),
        default="scenario_a",
        help="Scenario selector (default keeps current small setup)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario](n_trials=1, rng_seed=0)
    ships = scenario.layout_fn(**scenario.layout_kwargs)
    torpedoes = fan_spread(
        u_pos=np.array([-2000.0, 0.0]), #Submarine position
        base_bearing_rad=0.0, #Launch direction (radians)
        n=4, #Number of torpedos
        spread_rad=np.radians(15.0), #Torpedo Spread
        speed=25.0, #Torpedo Speed
        max_run_time=800.0, #Torpedo distance (TODO add units)
    )
    t_max = float(scenario.t_max)

    fig_dir = Path("results/figures")
    debug_dir = Path("results/debug")
    fig_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="lightgrey")
    ax = plot_attack_planview(
        ships,
        torpedoes,
        t_max=t_max,
        ax=ax,
        title="Static Attack Example",
        color_by="class",
        show_footprint=False,
        show_miss_annotations=True,
        miss_k=3,
        ship_marker="ship",
        use_hull_dimensions=True,
    )
    launch_pos = None
    if torpedoes:
        launch_pos = torpedoes[0].launch_position
        ax.scatter(
            launch_pos[0],
            launch_pos[1],
            s=70.0,
            marker="o",
            facecolor="white",
            edgecolor="black",
            zorder=4,
        )
    ax.set_axisbelow(True)
    ax.grid(True, color="lightgrey", linewidth=0.8, alpha=1.0)
    positions = np.array([ship.position for ship in ships], dtype=float)
    if len(positions):
        pad = max(200.0, float(np.ptp(positions[:, 0])) * 0.6, float(np.ptp(positions[:, 1])) * 0.6)
        ax.set_xlim(float(np.min(positions[:, 0]) - pad), float(np.max(positions[:, 0]) + pad))
        ax.set_ylim(float(np.min(positions[:, 1]) - pad), float(np.max(positions[:, 1]) + pad))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor="#0a0a0a",
            markeredgecolor="#0a0a0a",
            label="Freighter",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor="#38160d",
            markeredgecolor="#38160d",
            label="Tanker",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor="#001845",
            markeredgecolor="#001845",
            label="Escort",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor="#ffc600",
            markeredgecolor="#ffc600",
            label="Decoy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Submarine",
        )
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    positions = np.array([ship.position for ship in ships], dtype=float)
    if len(positions):
        xmin = float(np.min(positions[:, 0]))
        xmax = float(np.max(positions[:, 0]))
        ymin = float(np.min(positions[:, 1]))
        ymax = float(np.max(positions[:, 1]))
        if launch_pos is not None:
            xmin = min(xmin, float(launch_pos[0]))
            xmax = max(xmax, float(launch_pos[0]))
            ymin = min(ymin, float(launch_pos[1]))
            ymax = max(ymax, float(launch_pos[1]))
        pad = max(200.0, (xmax - xmin) * 0.25, (ymax - ymin) * 0.25)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_anchor("C")
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.18, right=0.95, bottom=0.28, top=0.92)
    fig_path = fig_dir / "attack_once.png"
    fig.savefig(fig_path, dpi=150)

    metrics = attack_debug_metrics(ships, torpedoes, t_max=t_max)
    debug_path = debug_dir / "attack_once.json"
    save_attack_debug_json(str(debug_path), metrics)

    print(f"Wrote {fig_path}")
    print(f"Wrote {debug_path}")


if __name__ == "__main__":
    main()
