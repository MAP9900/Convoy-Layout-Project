# Script Reference

High‑level overview of runnable scripts and scenario definitions. All scripts write outputs under `results/` unless overridden.

## Scenario Definitions

- `scenarios/scenario_base.py`
  - Scenario dataclass and `run()` helper that executes Monte Carlo with a fixed layout and torpedo sampler.
- `scenarios/scenario_a.py`
  - Baseline Scenario A (rectangular layout + fan spread).
- `scenarios/scenario_a1_constraints.py`
  - Scenario A1 with feasibility constraints and optional value scoring.

## Experiment Scripts

### `experiments/run_scenario.py`
- Purpose: Run a named scenario and save JSON results.
- Usage:
  - `python -m experiments.run_scenario scenario_a --n_trials 1000 --seed 123`
- Key args:
  - `scenario`: scenario name (e.g., `scenario_a`)
  - `--trials`: override number of trials
  - `--seed`: RNG seed
  - `--output`: output directory
- Outputs: `results/scenario_a.json`
  - Scenario A1 writes `results/scenario_a1.json`

### `experiments/sensitivity_oat.py`
- Purpose: One‑at‑a‑time sensitivity sweep over layout/attack/noise parameters.
- Usage:
  - `python experiments/sensitivity_oat.py --n-trials 200 --seed 0`
- Key args:
  - `--scenario`: scenario name (default `scenario_a`)
  - `--n-trials`: trials per point
  - `--seed`: RNG seed
  - `--output`: CSV path
- Outputs: `results/sensitivity.csv`

### `experiments/optimize_defender.py`
- Purpose: Brute‑force defender layout search vs fixed attacker.
- Usage:
  - `python -m experiments.optimize_defender --trials 200 --seed 0`
- Key args:
  - `--trials`: trials per candidate
  - `--seed`: RNG seed
  - `--output`: output directory
- Outputs: `results/defender_opt.csv`, `results/defender_best.json`

### `experiments/optimize_attacker.py`
- Purpose: Brute‑force attacker search vs best defense JSON.
- Usage:
  - `python -m experiments.optimize_attacker --defense-json results/defender_best.json --trials 200 --seed 0 --mode fan`
- Key args:
  - `--defense-json`: path to `defender_best.json`
  - `--trials`: trials per candidate
  - `--seed`: RNG seed
  - `--mode`: `fan` or `parallel`
  - `--output`: output directory
- Outputs: `results/attacker_opt.csv`, `results/attacker_best.json`

### `experiments/run_minmax.py`
- Purpose: Alternating defender/attacker best‑response loop.
- Usage:
  - `python -m experiments.run_minmax --rounds 5 --trials 100 --seed 0`
- Key args:
  - `--rounds`: number of alternating rounds
  - `--trials`: trials per candidate
  - `--seed`: RNG seed
  - `--output`: output JSON path
- Outputs: `results/minmax_history.json`

### `experiments/robustness_report.py`
- Purpose: Compare baseline vs optimized defense across noise settings.
- Usage:
  - `python experiments/robustness_report.py --defense-json results/defender_best.json --trials 200 --seed 0`
- Key args:
  - `--defense-json`: optimized defense JSON path
  - `--trials`: trials per run
  - `--seed`: RNG seed
  - `--output`: CSV path
- Outputs: `results/robustness_report.csv`

### `experiments/make_dataset.py`
- Purpose: Generate surrogate modeling dataset from random parameter samples.
- Usage:
  - `python experiments/make_dataset.py --samples 200 --trials 50 --seed 0`
- Key args:
  - `--samples`: number of samples
  - `--trials`: trials per sample
  - `--seed`: RNG seed
  - `--output`: output directory
- Outputs: `results/datasets/dataset.csv`, `results/datasets/dataset.npz`

### `experiments/train_surrogate.py`
- Purpose: Train baseline regressors (RandomForest/GradientBoosting) on a dataset.
- Usage:
  - `PYTHONPATH=. python experiments/train_surrogate.py --data results/datasets/dataset.csv --target expected_hits`
- Key args:
  - `--data`: dataset CSV path
  - `--target`: target column name
  - `--seed`: RNG seed
  - `--test-size`: test split fraction
  - `--output`: output directory
- Outputs:
  - `results/surrogate_report_<target>.json`
  - `results/*.joblib`

## Utilities (Non‑Executable)

- `convoy_sim/attackers.py`: Deterministic fan/parallel torpedo spread samplers.
- `convoy_sim/attack_proposals.py`: Attack proposal generation and value-biased aimpoints.
- `convoy_sim/defender_opt.py`: Defender layout search utilities.
- `convoy_sim/attacker_opt.py`: Attacker parameter search utilities.
- `convoy_sim/minmax_loop.py`: Min‑max loop coordinator.
- `convoy_sim/datasets.py`: Dataset generator for surrogate modeling.
- `convoy_sim/risk.py`: VaR/CVaR metrics for Monte Carlo outputs.
- `convoy_sim/noise.py`: Noise toggles for heading/timing/duds.
- `convoy_sim/feasibility.py`: Attack feasibility checks and detection risk scoring.
