# Results Log

## Test 1 - Baseline vs RL (2026-03-31)

- Baseline run:
  - `results/runs/baseline/20260331_005947_baseline_test1`
  - `static_baseline.expected_hits = 2.7`
  - `heuristic_baseline.expected_hits = 2.2` (winner)
- RL run:
  - `results/runs/rl/20260331_005922_rl_test1`
  - `evaluation.expected_hits = 2.7`
  - `training.selected_action = staggered_mid`

### Conclusion

- RL matched static baseline (`2.7`) but underperformed heuristic baseline (`2.2`).
- Relative gap vs heuristic baseline: `+0.5` expected hits (~18.5% worse than 2.7->2.2 benchmark).

### Validity Notes

- Train/eval profile splits matched across runs.
- Train/eval seeds matched across runs:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- This Test 1 result is a valid 1 to 1 comparison.

### Retrospective Interpretation Note (added 2026-04-08)

- Treat this RL comparison primarily as a pipeline/integration result, not strong evidence about RL optimization quality.
- The canonical RL setup at this stage used a very narrow predefined action set, and later review showed the action space needed overhaul before RL-vs-heuristic conclusions could be taken seriously.

## V2-Realism Test 1 - Baseline vs RL (2026-04-01)

- Baseline run:
  - `results/runs/baseline/20260401_210338_baseline_test1`
  - `static_baseline.expected_hits = 2.5225`
  - `heuristic_baseline.expected_hits = 2.495833333333333` (winner)
- RL run:
  - `results/runs/rl/20260401_210544_rl_test1`
  - `evaluation.expected_hits = 2.5225`
  - `training.selected_action = staggered_mid`

### Conclusion

- Under V2 realism defaults, heuristic baseline remained best in this run.
- RL evaluation matched static baseline and trailed heuristic baseline.

### Validity Notes

- Both runs use moving U-boat as default (`u_boat_mode=moving`).
- Both manifests stamp identical profile splits and seed sets.
- Both manifests include realism config stamps (noise/environment/ship-movement realism enabled).

### Retrospective Interpretation Note (added 2026-04-08)

- Treat this RL comparison primarily as a pipeline/integration result, not strong evidence about RL optimization quality.
- The canonical RL action space was still too narrow, so this test should not be treated as a decisive benchmark of RL capability.

## V2-Realism Test 2 - Baseline vs RL (2026-04-08)

- Baseline run:
  - `results/runs/baseline/20260408_145827_baseline_test1`
  - `static_baseline.expected_hits = 2.624166666666667`
  - `heuristic_baseline.expected_hits = 2.4316666666666666` (winner)
- RL run:
  - `results/runs/rl/20260408_150010_rl_test1`
  - `evaluation.expected_hits = 2.624166666666667`
  - `training.selected_action = rect_standard`

### Conclusion

- Under the current V2 realism setup, heuristic baseline again remained best.
- RL evaluation matched static baseline exactly and trailed heuristic baseline by `0.1925` expected hits.
- Relative gap vs heuristic baseline: about `7.9%` worse than the heuristic result.

### Validity Notes

- Both runs use the same git SHA: `b2dedb11d8a2b6acaf134c49ecafde5a5f713ceb`.
- Train/eval profile splits matched across both runs.
- Train/eval seeds matched across both runs:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- Both manifests stamp the same realism config family:
  - `u_boat_mode_default = moving`
  - noise enabled (`sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`)
  - environment enabled (`time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`)
  - ship movement realism enabled

### Retrospective Interpretation Note

- This run pair is best read as a workflow/debug benchmark, not a strong RL-vs-heuristic study.
- Review after the run showed the canonical RL action set was effectively degenerate in `configs/rl/default.toml`: the listed actions did not provide meaningful geometric freedom, so RL was not actually searching a rich convoy-layout space.

## V2-Realism Test 3 - Baseline vs RL After Phase 1 Action-Space Fix (2026-04-08)

- Baseline run:
  - `results/runs/baseline/20260408_153149_baseline_test1`
  - `static_baseline.expected_hits = 2.624166666666667`
  - `heuristic_baseline.expected_hits = 2.4316666666666666` (winner)
- RL run:
  - `results/runs/rl/20260408_153137_rl_test1`
  - `evaluation.expected_hits = 2.8883333333333336`
  - `training.selected_action = staggered_loose`

### Conclusion

- This is the first benchmark after fixing the degenerate canonical RL action set.
- RL now selected a materially different layout (`staggered_loose`), which confirms the action menu is no longer collapsing to one geometry.
- Performance worsened relative to both baselines:
  - RL trailed static baseline by `0.26416666666666694` expected hits
  - RL trailed heuristic baseline by `0.456666666666667` expected hits
- This suggests Phase 1 succeeded as an action-space fix, but the current one-step tabular learner and reward/training setup are still not good enough.

### Validity Notes

- Both runs use the same git SHA: `bb7ba4e5ee3b33aac6167ab1b437c61f31f5069e`.
- Train/eval profile splits matched across both runs.
- Train/eval seeds matched across both runs:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- Both manifests stamp the same realism config family:
  - `u_boat_mode_default = moving`
  - noise enabled (`sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`)
  - environment enabled (`time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`)
  - ship movement realism enabled
- The baseline config remained unchanged; the RL-side action menu was expanded in Phase 1.

### Interpretation Note

- Unlike earlier RL tests, this run does carry meaningful action-space information because the canonical RL menu now contains genuinely different layouts.
- However, it is still not a final RL capability benchmark because the environment remains a one-step selector and the learner/reward design have not yet been upgraded.

## V2-Realism Test 4 - RL After Phase 1.5 Train-Time Selection Fix (2026-04-08)

- Reference baseline for comparison:
  - `results/runs/baseline/20260408_153149_baseline_test1`
  - `static_baseline.expected_hits = 2.624166666666667`
  - `heuristic_baseline.expected_hits = 2.4316666666666666` (winner)
- RL run:
  - `results/runs/rl/20260408_155214_rl_test1`
  - `evaluation.expected_hits = 2.624166666666667`
  - `training.selected_action = rect_standard`
  - `training.selected_action_by_q_value = staggered_loose`

### Conclusion

- Phase 1.5 corrected the final action-selection failure exposed by the direct action audit.
- The new train-split risk-aware selector overrode the raw Q-value choice and selected `rect_standard`, which was the best eval action in the audit.
- RL evaluation returned to the static baseline level and avoided the worse `staggered_loose` outcome from Test 3.
- RL still trails the heuristic baseline by `0.1925` expected hits, so the selector fix helped, but it did not close the optimization gap.

### Validity Notes

- RL run git SHA: `a4e3317eb32e8e3942e76f23fd113e0322c9ef80`
- Baseline comparison reference git SHA: `bb7ba4e5ee3b33aac6167ab1b437c61f31f5069e`
- Train/eval profile splits matched the existing canonical V2 split:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- RL manifest stamps the same realism config family:
  - `u_boat_mode_default = moving`
  - noise enabled (`sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`)
  - environment enabled (`time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`)
  - ship movement realism enabled

### Interpretation Note

- This result shows the immediate next bottleneck is no longer final action selection.
- The remaining gap to heuristic baseline now points more toward:
  - coarse one-step action space
  - limited state/threat modeling
  - reward/objective limitations

## V2-Realism Test 5 - RL After Phase 2 Minimal Builder Slice (2026-04-08)

- Reference baseline for comparison:
  - `results/runs/baseline/20260408_153149_baseline_test1`
  - `static_baseline.expected_hits = 2.624166666666667`
  - `heuristic_baseline.expected_hits = 2.4316666666666666` (winner)
- RL run:
  - `results/runs/rl/20260408_175732_rl_test1`
  - `evaluation.expected_hits = 2.8916666666666666`
  - `training.selected_action = rect_compact_standard`
  - `training.selected_action_by_q_value = rect_compact_loose`
  - `training.mode = builder`

### Conclusion

- The minimal multi-step builder is functioning, but this first builder-mode benchmark performed worse than the prior flat-menu Phase 1.5 run.
- RL trailed static baseline by `0.2675` expected hits and trailed heuristic baseline by `0.46` expected hits.
- The selected builder layout (`rect_compact_standard`) was only marginally preferred over `rect_compact_loose` on the train objective, and both builder-selected rectangular compact variants were materially worse on eval than the earlier `rect_standard` flat action.

### Validity Notes

- RL run git SHA: `52c356bcb56f17085eee64634a549c9dad605366`
- Comparison baseline reference git SHA: `bb7ba4e5ee3b33aac6167ab1b437c61f31f5069e`
- RL run kept the same canonical V2 split and seeds:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- RL manifest stamps the same realism family as recent V2 runs:
  - `u_boat_mode_default = moving`
  - noise enabled (`sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`)
  - environment enabled (`time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`)
  - ship movement realism enabled
- This run used builder mode:
  - family choice
  - along-spacing bucket
  - across-spacing bucket

### Interpretation Note

- Phase 2 minimal builder integration succeeded technically, but it did not improve optimization quality yet.
- The next bottleneck now appears to be objective/training quality rather than missing compositional control alone:
  - train-time selection is still driven by aggregate `expected_hits + risk * CVaR_90`
  - the builder search space currently over-favors compact rectangular variants on the train split
  - no explicit feasibility/footprint guardrails or richer reward terms are shaping the builder policy yet

### Follow-Up Diagnostic Note (added 2026-04-08)

- Direct builder audit later changed the interpretation of Test 5.
- Audit run `results/runs/rl_action_audit/20260408_180301_rl_test1_action_audit` showed:
  - best train action = `rect_compact_loose`
  - best eval action = `rect_compact_loose`
  - eval `expected_hits = 2.4316666666666666`
- That means the minimal builder space already contains a layout matching the heuristic baseline.
- The immediate failure in Test 5 was therefore not the builder search space itself; it was the final selector/tie-break choosing `rect_compact_standard` over the better `rect_compact_loose`.

## V2-Realism Test 6 - RL After Phase 2.1 Builder Selection Fix (2026-04-08)

- Reference baseline for comparison:
  - `results/runs/baseline/20260408_153149_baseline_test1`
  - `static_baseline.expected_hits = 2.624166666666667`
  - `heuristic_baseline.expected_hits = 2.4316666666666666` (winner)
- RL run:
  - `results/runs/rl/20260408_182159_rl_test1`
  - `evaluation.expected_hits = 2.4316666666666666`
  - `training.selected_action = rect_compact_loose`
  - `training.selected_action_by_q_value = rect_compact_loose`
  - `training.mode = builder`

### Conclusion

- Phase 2.1 fixed the builder-mode selector failure.
- RL now selects the audited winner `rect_compact_loose`.
- This RL run:
  - beats static baseline by `0.1925` expected hits
  - matches heuristic baseline exactly on `expected_hits`
  - improves materially over the prior builder-mode run from Test 5

### Validity Notes

- RL run git SHA: `510092af5321235c4e2cfa9143c07eb05575c913`
- Comparison baseline reference git SHA: `bb7ba4e5ee3b33aac6167ab1b437c61f31f5069e`
- RL run kept the same canonical V2 split and seeds:
  - Train seeds: `[1939, 1940, 1941]`
  - Eval seeds: `[1942, 1943, 1944]`
- RL manifest stamps the same realism family as recent V2 runs:
  - `u_boat_mode_default = moving`
  - noise enabled (`sigma_heading_rad=0.01`, `sigma_launch_delay=0.05`, `sigma_speed_mps=0.25`, `p_dud=0.02`)
  - environment enabled (`time_of_day=night`, `visibility_m=3500`, `sea_state=4`, `detection_risk_scale=1.0`)
  - ship movement realism enabled

### Interpretation Note

- The minimal builder path is now validated as a real RL-capable search space for the current benchmark.
- The immediate blocker was selector behavior, not missing builder freedom.
- The next meaningful frontier is no longer selector repair. It is:
  - stronger reward design
  - broader attack-profile diversity
  - harder evaluation gates beyond matching the current heuristic baseline on one split

## Mixed-Convoy Diagnostic 1 - Baseline vs RL Before Phase 3B (2026-04-14)

- Baseline run:
  - `results/runs/baseline/20260414_224222_baseline_default`
  - `static_baseline.expected_hits = 2.565833333333333`
  - `heuristic_baseline.expected_hits = 2.5083333333333333` (winner)
- RL run:
  - `results/runs/rl/20260414_224410_rl_default`
  - `evaluation.expected_hits = 2.565`
  - `training.selected_action = rect_standard`
  - `training.mode = flat_action_menu`

### Conclusion

- On the first mixed-convoy pass, heuristic baseline remained best.
- RL slightly outperformed the static baseline but still trailed heuristic baseline by `0.05666666666666664` expected hits.
- This pass confirms that heterogeneous convoy composition changes the benchmark, but the current RL run is still using the older flat action menu rather than the newer builder path.

### Validity Notes

- Both runs use the same git SHA: `62c8ee49a5410a6af9dd6707901623e5df69a62f`.
- Both runs use the mixed convoy profile family, but this is **not** a strict apples-to-apples benchmark pair because the seed sets differ:
  - baseline train/eval seeds: `[1939, 1940, 1941]` / `[1942, 1943, 1944]`
  - RL train/eval seeds: `[301, 302, 303]` / `[401, 402, 403]`
- RL also used `mode = flat_action_menu`, not the current builder-mode canonical path.

### Interpretation Note

- Treat this as a mixed-convoy diagnostic pass, not as a new canonical benchmark.
- The main value of this run pair is:
  - mixed convoy support is now flowing through the baseline/RL workflows
  - heuristic still provides a stronger reference than current RL under the heterogeneous benchmark
  - Phase 3B is now better motivated, because the benchmark has mixed classes but the optimization objective is still effectively hits-centric and `value_lost` is not yet exposed in summaries
