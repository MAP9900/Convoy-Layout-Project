# RL Road Map

## Current State (2026-03-24)

- Phase 2–4 reorg complete:
  - Keep/archive/remove map documented in `docs/REORG_PHASE2_4_AUDIT.md`.
  - Canonical baseline runner: `experiments/run_baseline_suite.py`.
  - Canonical RL runner: `experiments/run_rl_train.py`.
  - Shared artifact schema enforced across baseline/RL runs.

## Canonical RL Workflow

1. Load config (`configs/rl/*.toml`).
2. Lock deterministic seeds/splits.
3. Build defender action set + opponent source (`AttackProfileLibrary`).
4. Train policy.
5. Evaluate on fixed held-out profile split.
6. Write canonical run artifacts.

Run command:
- `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Milestone Progress

### M1 (done)
- Defender-only RL wrapper path operational (`convoy_sim/rl_wrapper.py`).
- Reset-time attack-profile sampling wired and test-covered.

### M2 (done)
- Canonical RL entrypoint and config-first execution established.
- Shared baseline-vs-RL artifact contract established.

### M3 (next)
- Upgrade learner from tabular bandit baseline to full RL algorithm.
- Add richer observation/reward structure while preserving artifact schema.

### M4 (next)
- Lock acceptance protocol across >=3 seeds with fixed eval split.
- Add per-profile regression thresholds for promotion gating.

### M5 (later)
- Extend to attacker RL and alternating training (V2).

## Evaluation Contract (locked)

- Fixed train/eval profile splits.
- Fixed train/eval seed sets.
- Shared metrics:
  - `expected_hits` (primary)
  - `CVaR_90` (guardrail)
  - `p_hit_ge_1`
  - `value_lost` (when enabled)

## Practical Guidance

- Train mostly with fast budgets; evaluate with locked verify split.
- Do not change profile splits or seed sets inside ad hoc scripts once an experiment series starts.
- Keep sim and visual logic aligned before accepting policy improvements.
