# Project Code Review

Date: 2026-02-28
Reviewer: Codex
Scope: Full repository structure, dependency surface, script/workflow paths, and documentation consistency.

## Executive Summary

The codebase is well-organized around a clean split (`convoy_sim` core package, `scenarios` inputs, `experiments` entry points, `tests` validation). The largest source of confusion is not code layout itself, but documentation drift and missing dependency pinning.

Most important outcomes from this review:
- The project needs a canonical dependency manifest (`requirements*.txt` or `pyproject.toml`) so new environments can run tests/scripts reliably.
- A few docs reference scripts/modules that no longer exist.
- Script workflow paths can be made explicit so it is obvious what to run first and what artifacts are consumed downstream.

## Findings (High to Low)

1. High: No installable/testable dependency manifest is present.
- Evidence: `python -m pytest -q` fails with `No module named pytest` in this environment.
- Impact: New contributors cannot reliably bootstrap or run CI locally.
- Recommendation: add pinned dependency files (see "Dependency Map and Proposed Locking").
- Status: addressed in this pass by adding `requirements.txt`, `requirements-ml.txt`, and `requirements-dev.txt`.

2. Medium: Documentation drift previously referenced non-existent coverage tooling.
- Evidence at review start:
  - README referenced `python -m experiments.plot_coverage`.
  - `docs/SCRIPTS.md` listed `convoy_sim/coverage.py` and `convoy_sim/viz_coverage.py`.
- Impact: Contributors can spend time chasing missing modules.
- Status: addressed in this pass by updating README and `docs/SCRIPTS.md`.

3. Medium: Absolute import style is mixed between package-qualified and local-module imports.
- Evidence at review start: relative-local and package-qualified styles were mixed across modules.
- Impact: execution context/PYTHONPATH sensitivity in some scripts and tooling.
- Recommendation: standardize on package-qualified imports where possible for portability.
- Status: addressed in this pass by normalizing `convoy_sim/` and `scenarios/` imports to package-qualified form.

4. Low: Workflow sequencing is implicit across scripts.
- Evidence: several scripts consume artifacts created by earlier scripts (e.g., `optimize_attacker` expects `results/defender_best.json`) but this is distributed across docs.
- Impact: repeated onboarding friction.
- Recommendation: maintain a single workflow map (included below).

## Repository Map

Top-level:
- `convoy_sim/`: simulation engine and optimization/policy/game utilities.
- `scenarios/`: named scenario definitions and profiles.
- `experiments/`: runnable entry points.
- `tests/`: pytest suite for unit/smoke regression.
- `docs/`: operator/reference docs.
- `results/`: generated outputs/artifacts.

Python file counts:
- `convoy_sim`: 33
- `experiments`: 20
- `scenarios`: 7
- `tests`: 44
- Total: 104

## Dependency Map and Proposed Locking

Observed external dependencies from imports:
- Required runtime (core): `numpy`
- Optional visuals: `matplotlib`
- Optional ML/surrogate: `scikit-learn`, `joblib`
- Dev/test: `pytest`

Standard library usage is extensive (`argparse`, `dataclasses`, `json`, `math`, `pathlib`, `typing`, etc.) and does not need pinning.

Proposed files:
- `requirements.txt`
  - `numpy`
  - `matplotlib`
- `requirements-ml.txt`
  - `-r requirements.txt`
  - `scikit-learn`
  - `joblib`
- `requirements-dev.txt`
  - `-r requirements-ml.txt`
  - `pytest`

## Workflow Paths (What Depends on What)

Primary baseline path:
1. `python -m experiments.run_scenario scenario_a`
2. optional tuning: `python -m experiments.optimize_defender`
3. consumes step 2 artifact: `python -m experiments.optimize_attacker --defense-json results/defender_best.json`
4. iterate best responses: `python -m experiments.run_minmax`

Policy/tactics/game path:
1. `python -m experiments.run_policy scenario_b1`
2. `python -m experiments.optimize_policy`
3. `python -m experiments.run_attacker_plan scenario_b2`
4. `python -m experiments.optimize_attacker_tactics`
5. `python -m experiments.estimate_game`
6. consumes step 5 artifact: `python -m experiments.solve_nash_from_matrix results/game_matrix.json`

Data/ML path:
1. `python experiments/make_dataset.py`
2. consumes dataset CSV: `python experiments/train_surrogate.py --data ...`

Visual/diagnostic path:
1. `python -m experiments.plot_layout`
2. `python -m experiments.plot_historical_vs_optimized`
3. `python -m experiments.plot_attack_once`
4. `python -m experiments.render_attack_animation`
5. `python -m experiments.render_attack_animation_debug`
6. `python -m experiments.run_diagnostics_before_after`

## Script Inventory (Current)

`experiments/`
- Analysis/simulation:
  - `run_scenario.py`
  - `sensitivity_oat.py`
  - `run_minmax.py`
  - `robustness_report.py`
- Optimization:
  - `optimize_defender.py`
  - `optimize_attacker.py`
  - `optimize_policy.py`
  - `optimize_attacker_tactics.py`
- Policy/game:
  - `run_policy.py`
  - `run_attacker_plan.py`
  - `estimate_game.py`
  - `solve_nash_from_matrix.py`
- Data/ML:
  - `make_dataset.py`
  - `train_surrogate.py`
- Visualization:
  - `plot_layout.py`
  - `plot_historical_vs_optimized.py`
  - `plot_attack_once.py`
  - `render_attack_animation.py`
  - `render_attack_animation_debug.py`
  - `run_diagnostics_before_after.py`

## Testability Status

What was executed:
- `python -m compileall -q convoy_sim experiments scenarios tests` (pass)
- `python -m pytest -q` (fail: missing `pytest`)

Interpretation:
- Source parses cleanly.
- Functional correctness not re-validated in this environment due to missing test dependency.

## Maintainability Recommendations

1. Add dependency manifests and a bootstrap section in README.
2. Keep `docs/SCRIPTS.md` as the source-of-truth script index and link to it from README.
3. Standardize imports to package-qualified paths over time.
4. Add an internal "workflow quick map" (now present in this file + `docs/SCRIPTS.md`) and keep downstream artifact contracts explicit.
