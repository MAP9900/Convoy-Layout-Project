"""Render a single static attack with torpedo rays and debug metrics."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.viz_attack import plot_attack_planview, save_attack_debug_json, attack_debug_metrics
from scenarios.scenario_a import build_scenario_a


def main() -> None:
    scenario = build_scenario_a(n_trials=1, rng_seed=0)
    ships = scenario.layout_fn(**scenario.layout_kwargs)
    torpedoes = fan_spread(
        u_pos=np.array([-2000.0, 0.0]),
        base_bearing_rad=0.0,
        n=4,
        spread_rad=np.radians(15.0),
        speed=25.0,
        max_run_time=800.0,
    )
    t_max = float(scenario.t_max)

    fig_dir = Path("results/figures")
    debug_dir = Path("results/debug")
    fig_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ax = plot_attack_planview(
        ships,
        torpedoes,
        t_max=t_max,
        title="Attack Once",
        color_by="class",
        show_footprint=True,
        show_miss_annotations=True,
        miss_k=3,
    )
    fig = ax.figure
    fig.tight_layout()
    fig_path = fig_dir / "attack_once.png"
    fig.savefig(fig_path, dpi=150)

    metrics = attack_debug_metrics(ships, torpedoes, t_max=t_max)
    debug_path = debug_dir / "attack_once.json"
    save_attack_debug_json(str(debug_path), metrics)

    print(f"Wrote {fig_path}")
    print(f"Wrote {debug_path}")


if __name__ == "__main__":
    main()
