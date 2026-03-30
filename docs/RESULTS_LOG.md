# Results Log

## Test 1 - Baseline vs RL (2026-03-26)

- Baseline run:
  - `results/runs/baseline/20260326_213112_baseline_default`
  - `static_baseline.expected_hits = 2.7`
  - `heuristic_baseline.expected_hits = 2.2` (winner)
- RL run:
  - `results/runs/rl/20260326_213338_rl_default`
  - `evaluation.expected_hits = 2.7`
  - `training.selected_action = rect_standard`

### Conclusion

- RL matched static baseline (`2.7`) but underperformed heuristic baseline (`2.2`).
- Relative gap vs heuristic baseline: `+0.5` expected hits (~18.5% worse than 2.7->2.2 benchmark).

### Validity Notes

- Train/eval profile splits matched across runs.
- Eval seeds did not match across runs:
  - Baseline eval seeds: `[1942, 1943, 1944]`
  - RL eval seeds: `[401, 402, 403]`
- Next comparison should use identical eval seeds for strict apples-to-apples reporting.
