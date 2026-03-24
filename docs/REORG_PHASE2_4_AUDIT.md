# Phase 2–4 Audit and Migration Map

Date: 2026-03-24
Scope: Combined execution of PROJECT_GUIDE Phase 2, 3, and 4 (with Phase 2 safety assumptions already satisfied externally).

## Safety Assumptions Applied

- External full backup exists (provided by user).
- No new git backup branch/tag created in this run.
- Structural edits preserve reproducibility and deterministic replay contracts.

## Repo Audit Summary

Current repo shape (high level):
- Core engine: `convoy_sim/`
- Scenario definitions: `scenarios/`
- Operational scripts: `experiments/`
- Tests: `tests/`
- Documentation: `docs/`, `README.md`, `PROJECT_GUIDE.md`
- Tracked historical outputs: `results/`

Primary finding:
- Core simulation and RL wrapper are already present and test-covered.
- Entrypoints are fragmented across many scripts, with no single canonical baseline suite runner and no canonical RL train/eval runner emitting a unified artifact contract.
- Several scripts are useful but off critical path for V1 RL-first direction (policy/game-theory side quests, surrogate training, and legacy orchestration demos).

## Keep / Archive / Remove Map

### Keep (Core)

- `convoy_sim/*.py` (except no module removals in this phase)
- `scenarios/scenario_base.py`
- `scenarios/scenario_a.py`
- `scenarios/scenario_a1_constraints.py`
- `scenarios/scenario_rl.py`
- `scenarios/convoy_profiles.py`
- `experiments/run_baseline_suite.py` (new canonical baseline runner)
- `experiments/run_rl_train.py` (new canonical RL runner)
- `experiments/run_scenario.py` (legacy-compatible scenario runner retained)
- `experiments/optimize_defender.py` (baseline primitive retained)
- `experiments/optimize_attacker.py` (baseline primitive retained)
- `experiments/sensitivity_oat.py` (retained; test dependency)
- `experiments/audit_attack_profiles.py`
- `experiments/plot_layout.py`
- `experiments/plot_historical_vs_optimized.py`
- `experiments/plot_attack_once.py`
- `experiments/render_attack_animation.py`
- `experiments/render_attack_profile_previews.py`
- `experiments/run_diagnostics_before_after.py`

### Remove (Legacy workflow scripts)

Deleted:
- `run_policy.py`
- `optimize_policy.py`
- `run_attacker_plan.py`
- `optimize_attacker_tactics.py`
- `estimate_game.py`
- `solve_nash_from_matrix.py`
- `run_minmax.py`
- `make_dataset.py`
- `train_surrogate.py`
- `robustness_report.py`
- `render_attack_animation_debug.py`

Rationale:
- These were off critical path for the V1 RL-first baseline-vs-RL workflow.

### Remove (Additional cleanup)

- Remove generated cache directories from tracked workflow consideration:
  - `convoy_sim/__pycache__/`
  - `experiments/__pycache__/`
  - `scenarios/__pycache__/`
  - `tests/__pycache__/`

Notes:
- Additional non-canonical Python modules and scenarios were removed in follow-up cleanup.
- Historical result artifacts under `results/` are preserved.

## Canonical Baseline and RL Entrypoints (Phase 4 Targets)

- Baseline: `python -m experiments.run_baseline_suite --config <path>`
- RL: `python -m experiments.run_rl_train --config <path>`

Shared artifact contract per run:
- `config_resolved.yaml`
- `metrics_summary.json`
- `per_profile_metrics.csv`
- `run_manifest.json`

Additional RL artifact:
- `checkpoints/policy_latest.json`

## Migration Map (Old -> New Primary Usage)

- Baseline comparisons:
  - old: `run_scenario.py` + `optimize_defender.py` + ad hoc scripts
  - new: `run_baseline_suite.py`

- RL train/eval:
  - old: no canonical runner
  - new: `run_rl_train.py`

- Legacy analytical workflows:
  - old: top-level `experiments/*.py`
  - new: removed from active repo

## Validation Protocol During Reorg

Validation checkpoints executed after each major step:
1. Phase 2 map creation
2. Structural reorg (archive moves)
3. Canonical entrypoint implementation
4. Documentation refresh
5. Final regression pass
