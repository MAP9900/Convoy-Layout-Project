# Optimization Log

Tracks how each test was produced, including optimization method, objective, search space, and reproducibility details.

## Test 1 - Baseline vs RL (2026-03-31)

### Run References

- Baseline run dir: `results/runs/baseline/20260331_005947_baseline_test1`
- RL run dir: `results/runs/rl/20260331_005922_rl_test1`
- Results summary: see `docs/RESULTS_LOG.md` (Test 1)

### Baseline Optimization (Heuristic Grid Search)

- Entrypoint:
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- What is fixed:
  - Attack profile train/eval split from `[splits]`
  - Seeds from `[splits.train_seeds]` and `[splits.eval_seeds]`
  - Simulation settings from `[simulation]` (`t_max`, `n_trials_per_seed`, `max_hits_per_torpedo`)
  - Static layout defaults from `[baseline.static_layout]`
- What is optimized:
  - Candidate values listed in `[baseline.heuristic_search.grid]`
  - In Test 1, optimized knobs were `spacing_along` and `spacing_across`
- Behind-the-scenes steps:
  1. Evaluate the static layout on eval profiles (`static_baseline`).
  2. Enumerate candidate parameter combinations from the grid (bounded by `max_candidates`).
  3. For each candidate, run Monte Carlo over **train** profiles/seeds.
  4. Compute candidate score using train aggregate `expected_hits`.
  5. Select candidate with lowest score.
  6. Re-evaluate the selected candidate on **eval** profiles/seeds (`heuristic_baseline`).
- Objective used for selection:
  - Minimize `expected_hits` on the train split.
- Notes:
  - This is derivative-free search (no gradient updates); it picks the best discrete grid point.

### RL Optimization (Current Tabular One-Step Policy Selection)

- Entrypoint:
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`
- What is fixed:
  - Attack profile train/eval split from `[splits]`
  - Seeds from `[splits.train_seeds]` and `[splits.eval_seeds]`
  - Training hyperparameters from `[training]` (`episodes`, `epsilon`, decay, `alpha`)
- What is optimized:
  - Choice among predefined layout actions in `[[rl.actions]]`
- Behind-the-scenes steps:
  1. Build discrete action set from `[[rl.actions]]` (each action is a complete layout).
  2. Initialize tabular `q_values` for each action.
  3. For each episode (`max_steps=1`):
     - Choose action with epsilon-greedy policy.
     - Simulate one defender-perspective outcome.
     - Update chosen action Q-value with incremental update (step size `alpha`).
  4. Select action with highest learned Q-value.
  5. Evaluate only that selected action on eval profiles/seeds.
- Objective used for learning/selection:
  - Maximize defender reward (equivalent to minimizing defender loss / hit impact proxy in current setup).
- Notes:
  - Current RL is a lightweight tabular contextual bandit-style selector, not a deep RL policy learner.

### Reproducibility Inputs For Test 1

- Config sources:
  - `configs/baseline/default.toml`
  - `configs/rl/default.toml`
- Split generation method:
  - Deterministic random partition via `experiments.generate_run_config`
- Comparability status:
  - Train/eval profile splits matched across baseline and RL.
  - Train/eval seeds matched across baseline and RL.
  - Test 1 is an apples-to-apples comparison.

### Retrospective Methodology Note (added 2026-04-08)

- Although the run is reproducible and comparable, it should be interpreted cautiously as an RL benchmark.
- Later review showed the canonical RL action space needed overhaul before RL-vs-heuristic comparisons could say much about learning quality.

## V2-Realism Test 1 - Baseline vs RL (2026-04-01)

### Run References

- Baseline run dir: `results/runs/baseline/20260401_210338_baseline_test1`
- RL run dir: `results/runs/rl/20260401_210544_rl_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 1)

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active; `V1` frozen as historical reference.
- `u_boat_mode` default: `moving` (static retained for compatibility checks only).
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).
- Explicitly excluded in this phase: depth-fuze proxy.

### Config Stamp (from run manifests)

- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`

### Compatibility Check

- Static U-boat compatibility verified in test coverage:
  - `tests/test_realism_v2.py::test_static_mode_profile_compatibility_in_workflow_eval`

### Retrospective Methodology Note (added 2026-04-08)

- This run pair remained useful for validating the V2 realism workflow, but the RL action space was still too narrow to treat the result as a strong optimization benchmark.

## V2-Realism Test 2 - Baseline vs RL (2026-04-08)

### Run References

- Baseline run dir: `results/runs/baseline/20260408_145827_baseline_test1`
- RL run dir: `results/runs/rl/20260408_150010_rl_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 2)

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active.
- `u_boat_mode` default: `moving`.
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).

### Config Stamp (from run manifests)

- Git SHA: `b2dedb11d8a2b6acaf134c49ecafde5a5f713ceb`
- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`
- RL episodes: `300`
- RL selected action: `rect_standard`

### Comparability Status

- Train/eval profile splits matched across baseline and RL.
- Train/eval seeds matched across baseline and RL.
- Both runs use the same realism stamp and git SHA.
- V2-Realism Test 2 is a valid apples-to-apples comparison.

### Retrospective Methodology Note

- This run pair is reproducible and comparable, but not yet a strong RL capability test.
- Post-run review found that the canonical `[[rl.actions]]` set in `configs/rl/default.toml` was effectively degenerate, so the learner had almost no meaningful convoy-layout freedom.

## V2-Realism Test 3 - Baseline vs RL After Phase 1 Action-Space Fix (2026-04-08)

### Run References

- Baseline run dir: `results/runs/baseline/20260408_153149_baseline_test1`
- RL run dir: `results/runs/rl/20260408_153137_rl_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 3)

### What Changed In This Test

- This was the first benchmark after Phase 1 of the RL overhaul.
- `configs/rl/default.toml` was updated so the canonical `[[rl.actions]]` menu became genuinely non-degenerate:
  - rectangular vs staggered layouts
  - compact / standard / loose spacing variants
- Baseline config remained unchanged.

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active.
- `u_boat_mode` default: `moving`.
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).

### Config Stamp (from run manifests)

- Git SHA: `bb7ba4e5ee3b33aac6167ab1b437c61f31f5069e`
- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`
- RL episodes: `300`
- RL selected action: `staggered_loose`

### Comparability Status

- Train/eval profile splits matched across baseline and RL.
- Train/eval seeds matched across baseline and RL.
- Both runs use the same realism stamp and git SHA.
- V2-Realism Test 3 is a valid apples-to-apples comparison.

### Methodology Interpretation

- This is the first RL run in the current protocol where the canonical action set actually provided meaningful geometric freedom.
- The result therefore says more than earlier RL tests:
  - Phase 1 action-space repair worked
  - the current one-step tabular learner still performs poorly even when given nontrivial choices
- This points the next work toward:
  - reward redesign
  - richer state/threat modeling
  - eventual learner upgrade

### Follow-Up Diagnostic: Direct RL Action Audit

- Audit run dir: `results/runs/rl_action_audit/20260408_154017_rl_test1_action_audit`
- Entrypoint:
  - `python -m experiments.audit_rl_actions --config configs/rl/default.toml`
- Why this was run:
  - determine whether the new Phase 1 action menu itself was weak, or whether the learner selected the wrong action

Key findings:
- Best train action: `staggered_loose`
  - train `expected_hits = 2.7375`
  - eval `expected_hits = 2.8883333333333336`
- Best eval action: `rect_standard`
  - train `expected_hits = 2.7904166666666663`
  - eval `expected_hits = 2.624166666666667`

Interpretation:
- The canonical RL action menu is no longer degenerate.
- There is at least one materially better action on eval (`rect_standard`) than the one RL selected in Test 3 (`staggered_loose`).
- This shifts the main bottleneck away from action degeneracy and toward:
  - train/eval objective mismatch
  - weak one-step tabular selection logic
  - reward/risk alignment issues

## V2-Realism Test 4 - RL After Phase 1.5 Train-Time Selection Fix (2026-04-08)

### Run References

- RL run dir: `results/runs/rl/20260408_155214_rl_test1`
- Comparison baseline reference: `results/runs/baseline/20260408_153149_baseline_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 4)

### What Changed In This Test

- This test implemented Phase 1.5 of the RL overhaul.
- `experiments/run_rl_train.py` now keeps the existing tabular training loop but changes final action selection:
  - raw Q-values are still tracked
  - final action is selected from direct train-split action summaries
  - selection score uses `expected_hits + risk_cvar_weight * CVaR_90`
  - near-tied actions use a complexity tie-break
- Canonical config now includes:
  - `[rl.selection]`
  - `risk_cvar_weight = 0.05`
  - `complexity_tiebreak_tolerance = 0.1`

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active.
- `u_boat_mode` default: `moving`.
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).

### Config Stamp (from run manifest)

- Git SHA: `a4e3317eb32e8e3942e76f23fd113e0322c9ef80`
- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`
- RL episodes: `300`
- Selected action: `rect_standard`
- Raw Q-value winner: `staggered_loose`

### Comparability Status

- RL run uses the same canonical train/eval profile split and seed sets as the recent V2 runs.
- Realism stamp remains aligned with recent V2 benchmarks.
- This is a valid follow-up comparison to Test 3 because the key methodological change was isolated to train-time action selection.

### Methodology Interpretation

- Phase 1.5 worked as intended:
  - the selector no longer followed the overfit/raw-Q preference for `staggered_loose`
  - it selected `rect_standard`, which the direct action audit had already identified as the best eval action
- This restored RL performance to the static baseline level.
- The remaining gap to heuristic baseline suggests the next work should focus on:
  - richer environment/action design
  - reward/objective redesign
  - only then learner replacement

## V2-Realism Test 5 - RL After Phase 2 Minimal Builder Slice (2026-04-08)

### Run References

- RL run dir: `results/runs/rl/20260408_175732_rl_test1`
- Comparison baseline reference: `results/runs/baseline/20260408_153149_baseline_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 5)

### What Changed In This Test

- This test introduced the minimal Phase 2 builder path.
- Canonical RL no longer had to choose only from a flat list of whole-layout actions.
- Instead, the policy constructed a layout in 3 bounded steps:
  1. layout family
  2. along-spacing bucket
  3. across-spacing bucket
- `configs/rl/default.toml` now enables `[rl.builder]`.
- The flat `[[rl.actions]]` path still remains available as a fallback, but this run used builder mode.

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active.
- `u_boat_mode` default: `moving`.
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).

### Config Stamp (from run manifest)

- Git SHA: `52c356bcb56f17085eee64634a549c9dad605366`
- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`
- RL episodes: `300`
- Training mode: `builder`
- Selected action: `rect_compact_standard`
- Raw Q-value builder trace: `family:rectangular -> along:compact -> across:loose`
- Final selected action by Q-value reconstruction: `rect_compact_loose`

### Comparability Status

- RL run uses the same canonical train/eval profile split and seed sets as the recent V2 runs.
- Realism stamp remains aligned with recent V2 benchmarks.
- This is a valid methodology follow-up to Test 4 because the primary change was the builder-mode environment.

### Methodology Interpretation

- The Phase 2 minimal slice worked mechanically:
  - builder-mode training ran end to end
  - builder metadata is written to manifests/checkpoints
  - the policy now composes layouts instead of only choosing from a fixed menu
- But optimization quality regressed:
  - train-time ranking favored `rect_compact_loose` and `rect_compact_standard`
  - the risk-aware selector chose the slightly simpler `rect_compact_standard`
  - eval `expected_hits = 2.8916666666666666`, worse than the prior flat-menu `rect_standard` result (`2.624166666666667`)
- That suggests the next bottleneck is not merely action compositionality. The builder now needs:
  - better reward shaping
  - explicit constraint/footprint guardrails
  - possibly broader threat conditioning
- In other words: Phase 2 minimal slice was necessary infrastructure, but not yet a performance win.

### Follow-Up Diagnostic: Direct Builder Action Audit

- Audit run dir: `results/runs/rl_action_audit/20260408_180301_rl_test1_action_audit`
- Entrypoint:
  - `python -m experiments.audit_rl_actions --config configs/rl/default.toml`
- Why this was run:
  - determine whether the new minimal builder space itself was weak, or whether RL/selection still picked the wrong composed layout

Key findings:
- Best train action: `rect_compact_loose`
  - train `expected_hits = 2.5704166666666666`
  - eval `expected_hits = 2.4316666666666666`
- Best eval action: `rect_compact_loose`
  - eval `expected_hits = 2.4316666666666666`
  - `CVaR_90 = 3.0257354395162213`

Interpretation:
- The minimal builder search space is already strong enough to contain a heuristic-level layout.
- The main failure in Test 5 was not the builder menu.
- It was the final selection rule:
  - raw builder Q trace reconstructed `rect_compact_loose`
  - train-split selector/tie-break switched to `rect_compact_standard`
  - that switch caused the eval regression
- This moves the immediate next task away from reward redesign and toward:
  - fixing builder-mode selection/tie-break behavior
  - possibly reducing or disabling the complexity tie-break when the lower-primary-score option is still clearly preferable

## V2-Realism Test 6 - RL After Phase 2.1 Builder Selection Fix (2026-04-08)

### Run References

- RL run dir: `results/runs/rl/20260408_182159_rl_test1`
- Comparison baseline reference: `results/runs/baseline/20260408_153149_baseline_test1`
- Results summary: see `docs/RESULTS_LOG.md` (V2-Realism Test 6)

### What Changed In This Test

- This test implemented Phase 2.1 of the RL overhaul.
- `experiments/run_rl_train.py` now applies a builder-aware effective tie-break tolerance:
  - flat action mode keeps the configured broad tolerance
  - builder mode shrinks the effective complexity tie-break band sharply
  - complexity can no longer override the best primary score in builder mode unless the scores are genuinely near-identical
- Added regression coverage for the previous failure mode between:
  - `rect_compact_loose`
  - `rect_compact_standard`

### Protocol/Realism Stamp

- Protocol track: `V2-Realism` active.
- `u_boat_mode` default: `moving`.
- Torpedo realism enabled: heading noise, launch-delay noise, speed variance, dud probability.
- Included movement realism: bounded station-keeping jitter + class-dependent cohesion + bounded per-ship deviation overlay.
- Included attacker input realism: partial observability (noisy bearing/range/course/speed/contact estimate + environment context).

### Config Stamp (from run manifest)

- Git SHA: `510092af5321235c4e2cfa9143c07eb05575c913`
- Noise: `sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`
- Environment: `time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`
- Ship movement realism enabled: `true`
- Train seeds: `[1939, 1940, 1941]`
- Eval seeds: `[1942, 1943, 1944]`
- RL episodes: `300`
- Training mode: `builder`
- Selected action: `rect_compact_loose`
- Raw Q-value builder trace: `family:rectangular -> along:compact -> across:loose`
- Final selected action by Q-value reconstruction: `rect_compact_loose`
- Effective builder tie-break tolerance: `0.002724603972523337`

### Comparability Status

- RL run uses the same canonical train/eval profile split and seed sets as the recent V2 runs.
- Realism stamp remains aligned with recent V2 benchmarks.
- This is a valid follow-up to Tests 5 and the builder audit because the primary change was isolated to builder-mode final selection behavior.

### Methodology Interpretation

- Phase 2.1 worked as intended:
  - the selector no longer discarded the audited winner
  - RL now recovers `rect_compact_loose` directly
  - eval `expected_hits = 2.4316666666666666`, matching the heuristic baseline
- This materially changes the RL diagnosis:
  - the minimal builder search space is now validated
  - selector logic is no longer the main bottleneck
  - the next work should move to richer objectives and harder generalization, not more selector repair
