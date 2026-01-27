"""Run before/after diagnostics for layouts and attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.attackers import fan_spread
from convoy_sim.coverage import accumulate_torpedo_coverage
from convoy_sim.diagnostics import (
    compare_layout_metrics,
    lane_vulnerability_proxy,
    plot_before_after_attack_overlay,
    plot_before_after_layout,
    plot_coverage_comparison,
    render_diagnostics_report,
)
from convoy_sim.layouts import make_rectangular_convoy, make_staggered_convoy
from convoy_sim.simulation import run_monte_carlo_attack
from scenarios.scenario_a import build_scenario_a


def _load_layout_kwargs(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("params")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run before/after diagnostics")
    parser.add_argument("--before", type=str, default="scenario_a", help="Scenario name or JSON path")
    parser.add_argument("--after", type=str, default="results/defender_best.json", help="Scenario name or JSON path")
    parser.add_argument("--trials", type=int, default=50, help="Monte Carlo trials per layout")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    return parser.parse_args()


def _build_layout_from_spec(spec: str, base_kwargs: dict) -> list:
    if spec == "scenario_a":
        return make_rectangular_convoy(**base_kwargs)
    path = Path(spec)
    if path.exists():
        overrides = _load_layout_kwargs(path) or {}
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        return make_rectangular_convoy(**kwargs)
    return make_staggered_convoy(**base_kwargs)


def main() -> None:
    args = parse_args()
    scenario = build_scenario_a(n_trials=args.trials, rng_seed=args.seed)
    base_kwargs = dict(scenario.layout_kwargs)

    ships_before = _build_layout_from_spec(args.before, base_kwargs)
    ships_after = _build_layout_from_spec(args.after, base_kwargs)

    def sampler(_: np.random.Generator):
        return fan_spread(
            u_pos=np.array([-2000.0, 0.0]),
            base_bearing_rad=0.0,
            n=4,
            spread_rad=np.radians(15.0),
            speed=25.0,
            max_run_time=800.0,
        )

    rng = np.random.default_rng(args.seed)
    mc_before = run_monte_carlo_attack(
        layout_fn=lambda: ships_before,
        layout_kwargs={},
        torpedo_sampler=sampler,
        n_trials=args.trials,
        t_max=scenario.t_max,
        rng=rng,
    )
    rng = np.random.default_rng(args.seed)
    mc_after = run_monte_carlo_attack(
        layout_fn=lambda: ships_after,
        layout_kwargs={},
        torpedo_sampler=sampler,
        n_trials=args.trials,
        t_max=scenario.t_max,
        rng=rng,
    )

    headings = np.linspace(-np.pi, np.pi, 9)
    lane_before = lane_vulnerability_proxy(ships_before, headings=headings, n_rays=50)
    lane_after = lane_vulnerability_proxy(ships_after, headings=headings, n_rays=50)

    torpedo_samples = [sampler(rng) for _ in range(100)]
    bounds = (-2500.0, 500.0, -1500.0, 1500.0)
    cov_before = accumulate_torpedo_coverage(torpedo_samples, t_max=400.0, bounds=bounds, grid_n=120, dt=5.0)
    cov_after = cov_before

    fig_dir = Path("results/figures")
    diag_dir = Path("results/diag")
    fig_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    plot_before_after_layout(
        ships_before,
        ships_after,
        out_path=str(fig_dir / "diag_layout_before_after.png"),
    )
    plot_before_after_attack_overlay(
        ships_before,
        sampler(rng),
        ships_after,
        sampler(rng),
        t_max=scenario.t_max,
        out_path=str(fig_dir / "diag_attack_overlay.png"),
    )
    plot_coverage_comparison(
        cov_before,
        cov_after,
        ships_overlay=ships_after,
        out_path=str(fig_dir / "diag_coverage_compare.png"),
    )

    render_diagnostics_report(
        ships_before,
        ships_after,
        mc_before,
        mc_after,
        lane_before,
        lane_after,
        out_json_path=str(diag_dir / "diagnostics_summary.json"),
    )

    print(f"Wrote figures to {fig_dir}")
    print(f"Wrote diagnostics summary to {diag_dir}")


if __name__ == "__main__":
    main()
