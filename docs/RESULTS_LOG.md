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
  - `results/runs/baseline/20260401_203150_baseline_test1`
  - `static_baseline.expected_hits = 2.5225`
  - `heuristic_baseline.expected_hits = 2.495833333333333` (winner)
- RL run:
  - `results/runs/rl/20260401_203331_rl_test1`
  - `evaluation.expected_hits = 2.5225`
  - `training.selected_action = staggered_mid`

### Conclusion

- Under V2 realism defaults, heuristic baseline remained best in this run.
- RL evaluation matched static baseline and trailed heuristic baseline.

### Validity Notes

- Both runs use moving U-boat as default (`u_boat_mode=moving`).
- Both manifests stamp identical profile splits and seed sets.
- Both manifests include realism config stamps (noise/environment/ship-movement realism enabled).
