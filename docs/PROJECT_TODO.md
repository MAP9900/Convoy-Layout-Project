# Project TODO

Canonical project board for convoy RL/ML workflow.
Use this as the single operational list and keep detailed references in linked docs.

## Now

- [ ] Finalize attack profile quality gate and cleanup pass.
  - Owner: You
  - Best done by: You
  - Run: `python -m experiments.audit_attack_profiles --convoy-profile convoy_layout_1`
  - Notebook: `docs/notebooks/attack_profile_tests.ipynb` with `RUN_MODE='fast'`
  - Acceptance:
    - No unintended `implausible_geometry` profiles
    - Intentional misses documented and plausible

- [ ] Lock attack profile training/eval split + seeds manifest.
  - Owner: You
  - Best done by: You
  - Files: `convoy_sim/attack_profiles.py`, `docs/ML_RL_Checklist.md`
  - Acceptance:
    - Training distribution weights finalized
    - Held-out eval profile subset fixed
    - Fixed seed list + trial counts documented for eval

- [ ] Freeze runtime protocol (fast vs verify) and enforce it across notebook + scripts.
  - Owner: You
  - Best done by: You
  - Config: `RUN_MODE`, `HIT_DT`, `SIM_DURATION_S`, `PROFILE_LIMIT`, `N_TRIALS`
  - Acceptance:
    - Fast mode used for iteration/smoke
    - Verify mode used only for final artifacts/acceptance

- [ ] Define canonical trial schema + RL defaults.
  - Owner: You
  - Best done by: You
  - Docs: `docs/ML_RL_Checklist.md`
  - Acceptance:
    - Trial fields locked (including profile payload + seed)
    - Stochastic vs fixed variables listed
    - Motion model + hit slowdown defaults decided
    - Reward weights (hits vs value/risk penalties) decided

- [ ] Add per-profile performance report pipeline and overfitting checks.
  - Owner: CODEX
  - Best done by: CODEX
  - Output: reward/hits/survival split by `profile_id`
  - Acceptance:
    - Exported report artifact for each run
    - Automatic flag for profile collapse/overfitting patterns

- [ ] Add convoy composition assertions + visual regression captures.
  - Owner: CODEX
  - Best done by: CODEX
  - Scope: `small_demo` and `rl_large`
  - Acceptance:
    - Class-count assertions at layout build time
    - Reference captures with geometry summary checks

- [ ] Baseline benchmark pass before RL training.
  - Owner: You
  - Best done by: You
  - Runs: heuristic + random under same profile set and seeds
  - Acceptance:
    - Baseline metrics saved (hits, value, VaR/CVaR)

## Next

- [ ] Implement script-based batch renderer (notebook-equivalent) with timing logs.
  - Owner: CODEX
  - Best done by: CODEX
  - Goal: faster/reproducible long runs; easier automation

- [ ] Parallelize profile rendering/evaluation across CPU workers.
  - Owner: CODEX
  - Best done by: CODEX
  - Acceptance:
    - Deterministic per-profile seeding retained
    - Runtime reduced for full verify run

- [ ] Finalize `rl_large` composition and constraints.
  - Owner: You
  - Best done by: You
  - Files: `scenarios/convoy_profiles.py`, constraint docs

- [ ] Tune `scenario_rl` attacker params + metadata for first training pass.
  - Owner: You
  - Best done by: You
  - File: `scenarios/scenario_rl.py`

- [ ] Decide RL execution pathway and freeze first training config.
  - Owner: You
  - Best done by: You
  - Decision: `convoy_sim/rl_wrapper.py` vs `convoy_sim/rl_env.py`
  - Acceptance:
    - One pathway selected and documented
    - First hyperparameter config committed

- [ ] Attack profile generation track (new profile creation + repair loop).
  - Owner: CODEX
  - Best done by: CODEX
  - Scope:
    - Option A (first): constrained random generator with doctrine priors
    - Option B (second): evolutionary mutation/crossover on vetted profiles
    - Option C (optional): LLM-assisted template proposal + deterministic validator
  - Acceptance:
    - Generated profiles pass geometry audit gate
    - Auto-repair pass for minor bearing/spread errors
    - Output labeled as `credible_hit_threat` or `credible_near_miss`
    - Export artifacts: JSON library + audit CSV/JSON + sample frames

## Later

- [ ] Run first RL smoke training on fast tier.
  - Owner: You
  - Best done by: You

- [ ] Run full RL verify-tier training with checkpoints and 3+ seeds.
  - Owner: You
  - Best done by: You

- [ ] Optional surrogate/generative/evolutionary comparator track.
  - Owner: CODEX
  - Best done by: CODEX

- [ ] Resolve doc drift note in `docs/RL_Road_Map.md` about reset-time profile sampling.
  - Owner: CODEX
  - Best done by: CODEX
  - Note: current checklist says reset sampling wiring is already done

## Done

- [x] Dependency manifests added (`requirements*.txt`).
- [x] Import style normalized in core modules.
- [x] `PROJECT_CODE_REVIEW.md` + script/workflow doc refresh.
- [x] U-boat marker support in frame renderer.
- [x] `attack_profile_tests.ipynb` created and wired.
- [x] Frame exports updated to first/middle/last over 600s.
- [x] Dynamic hit tracking enabled in notebook frame renders.
- [x] Geometry plausibility audit module + CLI + tests added.
- [x] Runtime mode docs added across README/Visuals/SCRIPTS.
- [x] RL reset-time profile sampling wired in `convoy_sim/rl_wrapper.py`.

## References

- `docs/ML_RL_Checklist.md`
- `docs/RL_Road_Map.md`
- `docs/Visuals.md`
- `docs/SCRIPTS.md`
- `docs/notebooks/attack_profile_tests.ipynb`
