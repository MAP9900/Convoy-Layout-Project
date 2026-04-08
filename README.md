# Convoy Layout Project

Convoy-defense simulation and optimization platform for studying WWII-style torpedo attacks against merchant convoys.

The repository provides:
- deterministic geometry and kinematic simulation primitives
- canonical baseline and RL workflows with shared artifact contracts
- realism layers (moving U-boat, partial observability, torpedo imperfections, ship station-keeping overlays)
- diagnostics and visualization tooling for manual verification

## Current Protocol

- Active simulation track: `Protocol V2-Realism`
- Primary technical reference: `docs/SIM_FEATURES.md`

If you are new to the repo, start here:
1. `docs/SIM_FEATURES.md`
2. `docs/PROTOCOL_V2_REALISM.md`
3. this README quick-start section

## Canonical Workflows

Generate configs (optional, if you want to re-split profiles/seeds):

```bash
python -m experiments.generate_run_config \
  --template configs/templates/baseline.template.toml \
  --output configs/baseline/default.toml \
  --convoy-profile convoy_layout_1 \
  --split-seed 1945 \
  --n-total 30 \
  --n-train 20

python -m experiments.generate_run_config \
  --template configs/templates/rl.template.toml \
  --output configs/rl/default.toml \
  --convoy-profile convoy_layout_1 \
  --split-seed 1945 \
  --n-total 30 \
  --n-train 20
```

Run canonical experiments:

```bash
python -m experiments.run_baseline_suite --config configs/baseline/default.toml
python -m experiments.run_rl_train --config configs/rl/default.toml
```

## Run Artifacts

Both baseline and RL runs emit:
- `config_resolved.yaml`
- `metrics_summary.json`
- `per_profile_metrics.csv`
- `run_manifest.json`

RL runs also emit:
- `checkpoints/policy_latest.json`

Default output roots:
- `results/runs/baseline/<timestamp>_*`
- `results/runs/rl/<timestamp>_*`

## Manual Verification Paths

Notebook-first verification:
- `notebooks/attack_manual_verification.ipynb` (historical realism checks, MP4/frames, hit summaries)
- `notebooks/attack_profile_tests.ipynb` (profile-oriented geometry checks)
- `notebooks/torpedo_firing_doctrine_comparison.ipynb` (large, submarine-centric doctrine comparison plots for static vs moving U-boat cases; saves PNGs by default to `notebooks/results/torpedo_firing_doctrine_comparison/`)

Script-based diagnostics:
- `python -m experiments.render_attack_profile_previews --help`
- `python -m experiments.audit_attack_profiles --help`
- `python -m experiments.render_attack_animation --help`

Supporting visual reference:
- `docs/Visuals.md`

## Repository Layout

- `convoy_sim/`: core simulation, realism, scoring, and wrappers
- `experiments/`: canonical run entrypoints and diagnostics scripts
- `scenarios/`: convoy layout profile registry
- `configs/`: canonical and template TOML configs
- `tests/`: regression and smoke test coverage
- `docs/`: protocol, feature reference, logs, and planning docs
- `notebooks/`: manual verification notebooks

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Recommended Validation

Fast targeted checks:

```bash
pytest -q tests/test_realism_v2.py tests/test_attack_profiles.py tests/test_canonical_entrypoints.py
```

Full suite:

```bash
pytest -q
```
