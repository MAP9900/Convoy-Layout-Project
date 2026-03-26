# Convoy RL-First Research Platform

RL-first convoy-defense simulator with shared baseline and RL artifact contracts for reproducible comparisons.

## Canonical Entrypoints

- Baseline suite (config-first):
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- RL train/eval (config-first):
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`

Both runners emit the same core artifacts per run:
- `config_resolved.yaml`
- `metrics_summary.json`
- `per_profile_metrics.csv`
- `run_manifest.json`

RL runs also emit:
- `checkpoints/policy_latest.json`

Default output root:
- `results/runs/baseline/<timestamp>_*`
- `results/runs/rl/<timestamp>_*`

## Repository Layout

- `convoy_sim/`: simulation engine, RL wrapper, risk/objective utilities, workflow helpers.
- `scenarios/`: scenario/profile definitions.
- `experiments/`: canonical entrypoints + retained core utilities.
- `configs/`: baseline and RL canonical configs.
- `tests/`: pytest regression/smoke coverage.
- `docs/`: workflow docs and planning references.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Core Validation

```bash
python -m pytest
```

## Reorg Record

For the keep/archive/remove map and migration rationale, see:
- `docs/REORG_PHASE2_4_AUDIT.md`
