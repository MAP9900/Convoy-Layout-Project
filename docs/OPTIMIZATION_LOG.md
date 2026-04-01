# Optimization Log

Tracks *how* each test was produced (not just outcomes), including optimization method, objective, search space, and reproducibility details.

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

## V2-Realism Test 1 - Baseline vs RL (2026-04-01)

### Run References

- Baseline run dir: `results/runs/baseline/20260401_203150_baseline_test1`
- RL run dir: `results/runs/rl/20260401_203331_rl_test1`
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
