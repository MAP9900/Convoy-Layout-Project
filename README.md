# Convoy Layout Project

Convoy-defense simulation and optimization project for studying WWII-style torpedo attacks against merchant convoys.

The project combines:

- a deterministic 2D convoy/torpedo simulation
- synthetic U-boat attack-profile generation
- VAE-based attack-candidate generation
- POMDP-style partial-observation attacker modeling
- reinforcement-learning layout optimization
- notebook and script workflows for reproducible experiment reruns

Current results are being regenerated after repository cleanup. Generated data, old runs, and intermediate notebook outputs are intentionally excluded from the clean repo presentation unless promoted as final artifacts.

## Project Map

Start here:

1. `docs/SIMULATION_PHYSICS.md` - simulator behavior, assumptions, and current modeling limits
2. `docs/VAE.md` - attack-profile VAE design and candidate-pool workflow
3. `docs/POMDP.md` - partial-observation attacker selection and fire-control rebuilds
4. `docs/REINFORCEMENT_LEARNING.md` - RL layout-optimization design and validation plan
5. `docs/REPRODUCING.md` - commands and rerun order
6. `docs/PROJECT_MAP.md` - codebase map for future development
7. `docs/SCRIPTS.md` - runnable script entrypoints

## Repository Layout

- `convoy_sim/`: core simulation, realism, scoring, VAE, RL, and POMDP support code
- `experiments/`: command-line entrypoints for generation, audits, training, evaluation, and visuals
- `notebooks/`: notebook-first analysis and final rerun workflows
- `scenarios/`: convoy profile definitions
- `configs/`: baseline/RL config templates and defaults
- `tests/`: regression and smoke tests
- `docs/`: method docs, runbook, and codebase map

## Quick Start

Most commands assume the local `Python-DS` conda environment is already active:

```text
(Python-DS) ... Convoy Layout Project %
```

Install/update dependencies as needed:

```bash
pip install -r requirements-dev.txt
```

Run validation:

```bash
python -m pytest -q
python -m ruff check convoy_sim experiments scenarios tests --config pyproject.toml
```

Run the canonical baseline/RL entrypoints:

```bash
python -m experiments.run_baseline_suite --config configs/baseline/default.toml
python -m experiments.run_rl_train --config configs/rl/default.toml
```

For full data regeneration, VAE candidate pools, notebook order, and `Python-ML` notes, use `docs/REPRODUCING.md`.

## Outputs

Generated outputs are generally ignored:

- `results/runs/`
- `results/diag/`
- `results/figures/`
- `results/frames/`
- `results/notebook-results/`
- generated attack-profile datasets

Final rerun artifacts should be promoted deliberately into a small curated area such as:

```text
results/final/
  figures/
  metrics/
  manifests/
```

## Development Note

This is a personal research and portfolio project. AI (Codex GPT 5.4 & 5.5) were used to assist with code developement. Project direction, experiment framing, validation, and final interpretations were maintained by me, the author.
