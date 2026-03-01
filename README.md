# Optimal Convoy Layout Project

A research‑focused simulator for WWII‑style convoys and straight‑running torpedoes on a 2D plane (meters). The current phases emphasize fast, interpretable geometry, baseline Monte Carlo evaluation, and coarse optimization loops for defender/attacker layouts.

## What It Does

- **Geometry + kinematics**: straight‑line ship and torpedo motion with analytical closest‑approach checks.
- **Temporal dynamics (opt‑in)**: convoy‑level route legs + zig‑zag plans with time‑aware attack windows.
- **Convoy layouts**: rectangular, staggered, hex/triangular, and jittered formations.
- **Monte Carlo**: estimate expected hits, variance, and tail‑risk metrics (VaR/CVaR).
- **Value-based scoring**: optional value-destroyed metrics for heterogeneous ship classes.
- **Optimization**: brute‑force defender/attacker search and alternating best‑response loop.
- **Feasibility constraints**: optional range, escort zones, approach cones, and detection risk.
- **Defender policies (B1)**: randomized layout choice conditioned on threat priors.
- **Policy optimization**: coarse deterministic/mixture search over policy tables.
- **Attacker tactics (B2)**: multi-pass plans with delay/abort/commit and shaped salvos.
- **Attacker tactics search**: coarse grid search over multi-pass plan templates.
- **Game layer (B3)**: payoff matrix estimation, best responses, and exploitability.
- **Approximate Nash (B4)**: fictitious play/replicator solvers and double-oracle loop hooks.
- **Visualization (E1)**: optional matplotlib plan-view layout plots.
- **RL wrapper**: minimal episode-style interface + observation schema for ML experiments.
- **Scenarios & experiments**: scripted runs that save JSON/CSV results under `results/`.

## Project Map and Review

- Full repository map, dependency surface, workflow paths, and review findings:
  - `PROJECT_CODE_REVIEW.md`
- Script-by-script operator reference:
  - `docs/SCRIPTS.md`

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

If you only need core simulation (no ML/test tooling):

```bash
pip install -r requirements.txt
```

## Quick Start

Run a baseline scenario:

```bash
python -m experiments.run_scenario scenario_a --n_trials 1000 --seed 123
```

Run a constrained scenario (A1):

```bash
python -m experiments.run_scenario scenario_a1 --n_trials 1000 --seed 123
```

Run defender and attacker searches:

```bash
python -m experiments.optimize_defender --trials 200 --seed 0
python -m experiments.optimize_attacker --defense-json results/defender_best.json --trials 200 --seed 0 --mode fan
```

Run the min‑max loop:

```bash
python -m experiments.run_minmax --rounds 5 --trials 100 --seed 0
```

Run the defender policy demo (B1):

```bash
python -m experiments.run_policy scenario_b1 --trials 200 --seed 0
```

Optimize a policy table (deterministic + mixture):

```bash
python -m experiments.optimize_policy --trials 200 --seed 0
```

Run the multi-pass attacker plan demo (B2):

```bash
python -m experiments.run_attacker_plan scenario_b2 --trials 50 --seed 0
```

Optimize attacker tactics plans (grid search):

```bash
python -m experiments.optimize_attacker_tactics --trials 50 --seed 0 --top-k 10
```

Estimate a game payoff matrix (B3):

```bash
python -m experiments.estimate_game --trials 50 --seed 0
```

Solve approximate Nash from a saved matrix (B4):

```bash
python -m experiments.solve_nash_from_matrix results/game_matrix.json --iters 500 --seed 0
```

Render example plan-view layout PNGs (E1):

```bash
python -m experiments.plot_layout
```

Plot historical vs optimized overlay/comparison (E1):

```bash
python -m experiments.plot_historical_vs_optimized
```

Plot a single attack with torpedo rays (E2):

```bash
python -m experiments.plot_attack_once
```

Render temporal attack frames (E2):

```bash
python -m experiments.render_attack_animation
```

Render temporal attack frames with heading arrows (debug helper):

```bash
python -m experiments.render_attack_animation_debug
```

Run before/after diagnostics report (E3):

```bash
python -m experiments.run_diagnostics_before_after
```

## Surrogate Dataset + Training (Phase 7)

Generate a dataset:

```bash
python experiments/make_dataset.py --samples 200 --trials 50 --seed 0
```

Train baseline surrogates:

```bash
pip install scikit-learn joblib
python experiments/train_surrogate.py --data results/datasets/dataset.csv --target expected_hits
```

## Project Structure (High Level)

- `convoy_sim/`: core simulation, geometry, layout generation, and optimization utilities.
- `scenarios/`: scenario definitions (inputs only).
- `experiments/`: executable scripts that run scenarios and save results.
- `results/`: output data (gitignored).
- `tests/`: pytest suite.
- `docs/`: high‑level script and workflow reference.
- `docs/Visuals.md`: visualization guide (what can be rendered and how).

## Testing

```bash
python -m pytest
```

## Notes

- All units are **meters** and **seconds**.
- The simulation is deterministic unless optional noise is enabled.
- Time‑aware motion and launch windows are opt‑in via `convoy_sim.dynamics` and `run_monte_carlo_attack_dynamic`.
