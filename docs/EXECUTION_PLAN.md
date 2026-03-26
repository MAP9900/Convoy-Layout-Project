# Execution Plan

Status date: 2026-03-26

## Current State

- Phase 2–4 repo refocus complete.
- Canonical baseline workflow is active.
- Canonical RL workflow is active.
- Shared artifact schema is active for baseline and RL runs.

## Canonical Commands

- Baseline:
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- RL:
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Required Artifacts Per Run

- `config_resolved.yaml`
- `metrics_summary.json`
- `per_profile_metrics.csv`
- `run_manifest.json`
- RL only: `checkpoints/policy_latest.json`

## Verification Contract

- Sim and visual hit logic remain shared.
- Fixed-seed deterministic replay is required.
- Profile math checks are required before training.
- Canonical entrypoint smoke tests are required.

## RL Road Map

### M1 (done)
- Defender-only RL wrapper path is operational.
- Reset-time attack-profile sampling is wired and test-covered.

### M2 (done)
- Canonical RL entrypoint and config-first execution are established.
- Shared baseline-vs-RL artifact contract is established.

### M3 (next)
- Replace tabular bandit baseline with a production RL learner.
- Expand observation/reward structure while preserving artifact outputs.

### M4 (next)
- Lock acceptance protocol across at least 3 seeds.
- Add per-profile regression thresholds for promotion gating.

### M5 (later)
- Extend to attacker RL and alternating training (V2).

## Action Board

### Now
- [ ] Finalize official benchmark train/eval profile split.
- [ ] Freeze official benchmark train/eval seed sets.
- [ ] Define RL promotion thresholds using `expected_hits` + `CVaR_90` guardrail.

### Next
- [ ] Replace tabular learner in `experiments/run_rl_train.py`.
- [ ] Add run-to-run comparator utility for baseline vs RL manifests.
- [ ] Add CI smoke checks for canonical baseline and RL entrypoints.

### Later
- [ ] V2: attacker RL + alternating training protocol.
- [ ] V3: multi-agent robustness and domain randomization matrix.

## References

- `PROJECT_GUIDE.md`
- `docs/SCRIPTS.md`
- `docs/Visuals.md`
- `docs/REORG_PHASE2_4_AUDIT.md`
