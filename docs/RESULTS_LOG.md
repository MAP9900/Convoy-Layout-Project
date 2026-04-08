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
