# Script Reference

Canonical script index after Phase 2–4 reorg.

## Canonical Workflows

### Baseline (primary)
- Command:
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- Outputs per run:
  - `config_resolved.yaml`
  - `metrics_summary.json`
  - `per_profile_metrics.csv`
  - `run_manifest.json`

### RL (primary)
- Command:
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`
- Outputs per run:
  - `config_resolved.yaml`
  - `metrics_summary.json`
  - `per_profile_metrics.csv`
  - `run_manifest.json`
  - `checkpoints/policy_latest.json`

## Retained Core Utility Scripts

- `python -m experiments.audit_attack_profiles --convoy-profile convoy_layout_1`
- `python -m experiments.plot_layout`
- `python -m experiments.plot_attack_once`
- `python -m experiments.render_attack_animation`
- `python -m experiments.render_attack_profile_previews --convoy-profile convoy_layout_1 --run-mode verify --workers 8`

## Scenario Definitions

- `scenarios/scenario_base.py`
- `scenarios/scenario_a.py`
- `scenarios/scenario_a1_constraints.py`
- `scenarios/scenario_rl.py`
- `scenarios/convoy_profiles.py`

## Notes

- Canonical keep/archive/remove map: `docs/REORG_PHASE2_4_AUDIT.md`
- Visual rendering guide: `docs/Visuals.md`
