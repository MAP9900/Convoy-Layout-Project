# Script Reference

Source-of-truth index for runnable scripts and scenario definitions.
All scripts write under `results/` unless overridden.

Related docs:
- Repo-wide map and review findings: `PROJECT_CODE_REVIEW.md`
- Visualization details: `docs/Visuals.md`

## Scenario Definitions

- `scenarios/scenario_base.py`
  - Scenario dataclass and `run()` helper for Monte Carlo execution.
- `scenarios/scenario_a.py`
  - Baseline Scenario A (rectangular layout + fan spread).
- `scenarios/scenario_a1_constraints.py`
  - Scenario A1 with feasibility constraints and optional value scoring.
- `scenarios/scenario_b1_policy_demo.py`
  - Scenario B1 policy demo with threat priors and randomized layout selection.
- `scenarios/scenario_b2_multisalvo_demo.py`
  - Scenario B2 multi-pass attacker plan demo with abort logic.
- `scenarios/scenario_rl.py`
  - RL convoy layout scaffold scenario using profile `rl_large`.
- `scenarios/convoy_profiles.py`
  - Convoy profile registry (`small_demo`, `rl_large`).

## Workflow Paths

Baseline simulation/optimization:
1. `python -m experiments.run_scenario scenario_a --trials 1000 --seed 123`
2. `python -m experiments.optimize_defender --trials 200 --seed 0`
3. `python -m experiments.optimize_attacker --defense-json results/defender_best.json --trials 200 --seed 0`
4. `python -m experiments.run_minmax --rounds 5 --trials 100 --seed 0`

Policy/tactics/game:
1. `python -m experiments.run_policy scenario_b1 --trials 200 --seed 0`
2. `python -m experiments.optimize_policy --trials 200 --seed 0`
3. `python -m experiments.run_attacker_plan scenario_b2 --trials 50 --seed 0`
4. `python -m experiments.optimize_attacker_tactics --trials 50 --seed 0 --top-k 10`
5. `python -m experiments.estimate_game --trials 50 --seed 0`
6. `python -m experiments.solve_nash_from_matrix results/game_matrix.json --iters 500 --seed 0`

Data/ML:
1. `python experiments/make_dataset.py --samples 200 --trials 50 --seed 0`
2. `python experiments/train_surrogate.py --data results/datasets/dataset.csv --target expected_hits`

Visual/diagnostics:
1. `python -m experiments.plot_layout`
2. `python -m experiments.plot_historical_vs_optimized`
3. `python -m experiments.plot_attack_once`
4. `python -m experiments.render_attack_animation`
5. `python -m experiments.render_attack_animation_debug`
6. `python -m experiments.run_diagnostics_before_after`
7. `python -m experiments.render_attack_profile_previews --convoy-profile convoy_layout_1 --run-mode verify --workers 8`
8. Open `docs/notebooks/attack_profile_tests.ipynb` to export first/middle/last frames for each attack profile (notebook now calls the same script)
9. `python -m experiments.audit_attack_profiles --convoy-profile convoy_layout_1` for fast profile plausibility triage

Notebook runtime modes (`docs/notebooks/attack_profile_tests.ipynb`):
- Fast mode:
  - `RUN_MODE='fast'`
  - Uses shorter horizon/coarser hit-step/profile cap for quick iteration.
  - Uses fewer workers and shorter trail length.
- Verify mode:
  - `RUN_MODE='verify'`
  - Uses full horizon/finer hit-step/full profile set for final checks.
  - Uses process parallelism (`PARALLEL_WORKERS`) with deterministic per-profile seeds.

## Experiment Scripts

### `experiments/run_scenario.py`
- Purpose: Run a named scenario and save JSON results.
- Usage:
  - `python -m experiments.run_scenario scenario_a --trials 1000 --seed 123`
- Outputs:
  - `results/scenario_a.json`
  - `results/scenario_a1.json` (for A1)

### `experiments/sensitivity_oat.py`
- Purpose: One-at-a-time sensitivity sweeps.
- Usage:
  - `python experiments/sensitivity_oat.py --n-trials 200 --seed 0`
- Outputs: `results/sensitivity.csv`

### `experiments/optimize_defender.py`
- Purpose: Brute-force defender layout search vs fixed attacker.
- Usage:
  - `python -m experiments.optimize_defender --trials 200 --seed 0`
- Outputs:
  - `results/defender_opt.csv`
  - `results/defender_best.json`

### `experiments/optimize_attacker.py`
- Purpose: Brute-force attacker search vs selected defense.
- Usage:
  - `python -m experiments.optimize_attacker --defense-json results/defender_best.json --trials 200 --seed 0 --mode fan`
- Outputs:
  - `results/attacker_opt.csv`
  - `results/attacker_best.json`

### `experiments/run_minmax.py`
- Purpose: Alternating defender/attacker best-response loop.
- Usage:
  - `python -m experiments.run_minmax --rounds 5 --trials 100 --seed 0`
- Outputs: `results/minmax_history.json`

### `experiments/run_policy.py`
- Purpose: Evaluate defender policy over threat priors.
- Usage:
  - `python -m experiments.run_policy scenario_b1 --trials 200 --seed 0`
- Outputs: `results/scenario_b1.json`

### `experiments/optimize_policy.py`
- Purpose: Optimize policy tables (deterministic + mixture search).
- Usage:
  - `python -m experiments.optimize_policy --trials 200 --seed 0`
- Outputs: `results/policy_opt.json`

### `experiments/run_attacker_plan.py`
- Purpose: Execute multi-pass attacker plans.
- Usage:
  - `python -m experiments.run_attacker_plan scenario_b2 --trials 50 --seed 0`
- Outputs: `results/scenario_b2.json`

### `experiments/optimize_attacker_tactics.py`
- Purpose: Grid search over attacker multi-pass plan templates.
- Usage:
  - `python -m experiments.optimize_attacker_tactics --trials 50 --seed 0 --top-k 10`
- Outputs: `results/attacker_tactics_opt.json`

### `experiments/estimate_game.py`
- Purpose: Estimate payoff matrices and exploitability.
- Usage:
  - `python -m experiments.estimate_game --trials 50 --seed 0`
- Outputs: `results/game_matrix.json`

### `experiments/solve_nash_from_matrix.py`
- Purpose: Approximate Nash solution from saved matrix.
- Usage:
  - `python -m experiments.solve_nash_from_matrix results/game_matrix.json --iters 500 --seed 0`
- Outputs: `results/nash_solution.json`

### `experiments/plot_layout.py`
- Purpose: Render plan-view layout PNGs.
- Usage:
  - `python -m experiments.plot_layout`
- Outputs: `results/figures/*.png`

### `experiments/plot_historical_vs_optimized.py`
- Purpose: Overlay historical vs optimized layouts.
- Usage:
  - `python -m experiments.plot_historical_vs_optimized`
- Outputs: `results/figures/historical_vs_optimized_*.png`

### `experiments/plot_attack_once.py`
- Purpose: Plot one attack with torpedo rays and debug JSON.
- Usage:
  - `python -m experiments.plot_attack_once`
- Outputs:
  - `results/figures/attack_once.png`
  - `results/debug/attack_once.json`

### `experiments/render_attack_animation.py`
- Purpose: Render time-indexed attack frames and optional MP4.
- Usage:
  - `python -m experiments.render_attack_animation`
- Outputs:
  - `results/frames/demo_attack/frame_*.png`
  - `results/frames/demo_attack.mp4` (optional)

### `experiments/render_attack_animation_debug.py`
- Purpose: Debug variant of animation rendering with heading arrows.
- Usage:
  - `python -m experiments.render_attack_animation_debug`
- Outputs:
  - `results/frames/demo_attack_debug/frame_*.png`
  - `results/frames/demo_attack_debug.mp4` (optional)

### `experiments/run_diagnostics_before_after.py`
- Purpose: Generate before/after diagnostics plot + summary.
- Usage:
  - `python -m experiments.run_diagnostics_before_after`
- Outputs:
  - `results/figures/diag_attack_overlay.png`
  - `results/diag/diagnostics_summary.json`

### `experiments/audit_attack_profiles.py`
- Purpose: Fast geometry plausibility audit for attack profiles against a selected convoy layout.
- Usage:
  - `python -m experiments.audit_attack_profiles --convoy-profile convoy_layout_1`
- Outputs:
  - `results/diag/attack_profile_geometry_audit.csv`
  - `results/diag/attack_profile_geometry_audit.json`

### `experiments/render_attack_profile_previews.py`
- Purpose: Render first/middle/last preview frames per attack profile with optional process parallelism.
- Usage:
  - `python -m experiments.render_attack_profile_previews --convoy-profile convoy_layout_1 --run-mode verify --workers 8`
  - `python -m experiments.render_attack_profile_previews --convoy-profile convoy_layout_1 --run-mode fast --workers 4 --select-profile-ids P15,P16,P19,P21`
- Outputs:
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_0001.png`
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_<middle>.png`
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_<last>.png`
  - `results/diag/attack_profile_geometry_audit.csv`
  - `results/diag/attack_profile_geometry_audit.json`
- Runtime knobs:
  - `--workers` parallel profiles (recommended for verify runs).
  - `--trail-length-s`, `--trail-linewidth`, `--trail-alpha`, `--trail-antialiased` for trail render cost/quality.
  - `--hit-dt` controls hit-state stepping fidelity/speed.
  - `--select-profile-ids` and `--profile-limit` reduce scope for tuning loops.

### `experiments/robustness_report.py`
- Purpose: Compare baseline vs optimized defense across noise settings.
- Usage:
  - `python experiments/robustness_report.py --defense-json results/defender_best.json --trials 200 --seed 0`
- Outputs: `results/robustness_report.csv`

### `experiments/make_dataset.py`
- Purpose: Generate surrogate dataset from random parameter samples.
- Usage:
  - `python experiments/make_dataset.py --samples 200 --trials 50 --seed 0`
- Outputs:
  - `results/datasets/dataset.csv`
  - `results/datasets/dataset.npz`

### `experiments/train_surrogate.py`
- Purpose: Train baseline regressors on dataset targets.
- Usage:
  - `python experiments/train_surrogate.py --data results/datasets/dataset.csv --target expected_hits`
- Outputs:
  - `results/surrogate_report_<target>.json`
  - `results/*.joblib`

## Core Utilities (`convoy_sim/`)

- `simulation.py`: core Monte Carlo execution.
- `dynamics.py`: time-aware route legs and convoy motion.
- `attackers.py`: fan/parallel torpedo spread samplers.
- `attack_profiles.py`: attacker profile schema and profile-to-torpedo builder.
- `attack_proposals.py`: attack proposal helpers and aimpoint selection.
- `feasibility.py`: range/cone/risk feasibility checks.
- `risk.py`: VaR/CVaR metrics.
- `objectives.py`: objective weighting/scoring.
- `layouts.py`: formation generation and helpers.
- `layout_roles.py`: role assignment helpers.
- `entities.py`: domain entities/dataclasses.
- `geometry.py`: geometric primitives and computations.
- `ship_catalog.py`: ship class/catalog metadata.
- `defender_opt.py`: defender search utilities.
- `attacker_opt.py`: attacker search utilities.
- `minmax_loop.py`: alternating best-response coordinator.
- `defender_policy.py`: policy representation/evaluation.
- `defender_policy_opt.py`: policy search/objectives.
- `attacker_tactics.py`: multi-pass tactics execution.
- `attacker_tactics_opt.py`: tactics template search.
- `game.py`: payoff matrix utilities/exploitability helpers.
- `nash.py`: fictitious-play/replicator solvers.
- `double_oracle.py`: double-oracle loop support.
- `datasets.py`: surrogate dataset assembly.
- `batch_eval.py`: batch evaluation utilities.
- `trial_records.py`: standardized trial record schema.
- `rl_wrapper.py`: RL episode wrapper and action mapping.
- `rl_env.py`: RL env data structures/helpers.
- `viz.py`: plan-view plotting helpers.
- `viz_attack.py`: static/temporal attack plotting helpers.
- `diagnostics.py`: comparative diagnostics generation.

## Notes

- Some scripts are designed for `python -m ...`; a few are plain-file invocation (`python experiments/...py`).
- Matplotlib is optional for plotting scripts only.
- `scikit-learn` and `joblib` are only needed for surrogate training.
