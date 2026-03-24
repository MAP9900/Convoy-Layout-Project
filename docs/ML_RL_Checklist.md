# ML/RL Checklist

Status date: 2026-03-24

## Reorg + Canonical Workflow

- [x] Phase 2 audit and keep/archive/remove map completed.
- [x] Phase 3 RL-first structural reorg completed.
- [x] Phase 4 canonical baseline and RL entrypoints completed.
- [x] Shared artifact schema active for both workflows.

## Canonical Entry Commands

- Baseline:
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- RL:
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Required Artifacts Per Run

- [x] `config_resolved.yaml`
- [x] `metrics_summary.json`
- [x] `per_profile_metrics.csv`
- [x] `run_manifest.json`
- [x] RL checkpoint: `checkpoints/policy_latest.json`

## Verification Contract

- [x] Sim and visual hit logic remain shared.
- [x] Fixed-seed deterministic replay supported.
- [x] RL reset-time profile sampling covered by tests.
- [x] Canonical entrypoint smoke tests added.

## Outstanding Decisions

- [ ] Official benchmark split freeze (train/eval profiles).
- [ ] Official benchmark seed freeze.
- [ ] RL promotion thresholds (mean + CVaR guardrail criteria).
- [ ] Runtime policy for fast vs verify tiers in release process.

## Implementation Next Steps

- [ ] Upgrade learner in `experiments/run_rl_train.py` from tabular bandit to full RL algorithm.
- [ ] Add run-to-run comparator utility for baseline vs RL manifests.
- [ ] Add CI workflow for canonical runner smoke tests and selected core regression tests.

## References

- `PROJECT_GUIDE.md`
- `docs/REORG_PHASE2_4_AUDIT.md`
- `docs/RL_Road_Map.md`
- `docs/SCRIPTS.md`
