# PROJECT_GUIDE

Last updated: 2026-03-27


## Main Goal
Build a historically plausible convoy-defense research platform where reinforcement learning (RL) is the primary method, using a shared simulation/verification stack so conclusions are reproducible and defensible.

Primary success metric (V1): minimize `expected_hits`.
Guardrail metric: `CVaR_90` (avoid policies that only optimize mean outcomes).

## Scope Boundaries

### In Scope (V1)
- Defender RL against scripted attacker profiles.
- Historically plausible convoy constraints (layout geometry + movement limits).
- Shared sim and visualization logic for manual verification.
- Fixed evaluation protocol (profile suite, seeds, metrics, artifact format).
- Baseline non-RL optimization for comparison.

### In Scope (V2)
- Attacker RL against scripted defender.
- Alternating/bilevel training workflow.
- GenAI synthetic attack-profile dataset generation pipeline (simulation-backed, reproducible).
- Conditional generative adversary baseline (VAE) for attack-profile sampling.

### In Scope (V3)
- Multi-agent training and robustness suite.
- Domain randomization with controlled scenario distributions.
- Generative scenario/threat modeling extensions (e.g., diffusion variant, latent-space diagnostics).

### Out of Scope for Now
- Escort combat modeling.
- Full moving U-boat behavior in V1.
- New non-RL feature expansion not required for baseline comparisons.

### GenAI Final Project Scope (Planned)
- Goal: add a generative adversary path that complements (not replaces) canonical RL and baseline workflows.
- Data source: synthetic but simulation-grounded attack-profile datasets built with fixed seeds, feasibility checks, and manifest metadata.
- Core deliverable: conditional VAE that samples diverse, feasible attack profiles for robustness evaluation.
- Evaluation requirement: compare scripted/baseline vs generated attacks using shared metrics:
  - `expected_hits`
  - `CVaR_90`
  - `p_hit_ge_1`
- Stretch goals (time permitting):
  - latent-space exploration of vulnerability regions
  - diffusion-based generator variant

## Canonical Workflows

### Baseline Optimization Workflow (must mirror RL workflow style)
Single entrypoint, config-first, reproducible:
- Entrypoint target: `experiments/run_baseline_suite.py`.
- Run shape:
  1. load config
  2. set seeds/splits
  3. run static baseline
  4. run bounded heuristic search baseline
  5. evaluate on fixed benchmark suite
  6. write structured artifacts
- Required outputs per run:
  - `config_resolved.yaml`
  - `metrics_summary.json`
  - `per_profile_metrics.csv`
  - `run_manifest.json` (git SHA, seed sets, splits)

### RL Workflow (canonical)
Single entrypoint, config-first, reusable across milestones:
- Entrypoint target: `experiments/run_rl_train.py`.
- Run shape:
  1. load config
  2. set deterministic seeds
  3. build env + learner + opponent source
  4. train
  5. evaluate on fixed benchmark suite
  6. write checkpoints/metrics/config/manifest to structured run dir
- Milestones M1-M5 vary by config, not custom code paths.

## Verification Contract (non-negotiable)
- Same hit logic for simulation and visual diagnostics.
- Deterministic replay for fixed seed.
- Profile math checks before policy training.
- Manual verification outputs remain available (frames + key CSV diagnostics).

## Evaluation Contract
- Fixed train/eval profile splits.
- Fixed train/eval seed sets.
- Shared metrics across baseline and RL:
  - `expected_hits` (primary)
  - `CVaR_90` (guardrail)
  - `p_hit_ge_1`
  - `value_lost` (if enabled)
- Same artifact schema for baseline and RL runs.

## Repo Overhaul Strategy
The first major step is a complete repo reorg to remove/park non-essential paths and center the project around the canonical baseline + RL workflows.

Guiding principle: **do not break reproducibility while slimming**.

---

## Codex Phases

Status legend:
- `[x]` done
- `[ ]` not started
- `[-]` in progress
- `[!]` blocked / needs manual decision

### Phase 0: Direction Lock + North Star
Status: `[x]`
- Lock project direction to RL-first research platform.
- Define V1/V2/V3 scope and out-of-scope boundaries.
- Define primary metric and guardrails.

Exit criteria:
- Team agrees on scope boundaries and workflow style.

### Phase 1: Safety Backup + Freeze Point
Status: `[x]`
- External full workspace backup snapshot confirmed.
- Safety git branch/tag intentionally skipped per explicit user instruction for this execution.
- Capture current "known working" commands and outputs.

Exit criteria:
- Full rollback path exists.
- Freeze-point note committed.

### Phase 2: Repo Inventory + Keep/Archive/Delete Map
Status: `[x]`
- Inventory all scripts/modules/docs against V1/V2/V3.
- Categorize each item:
  - `core` (keep and maintain)
  - `baseline` (keep minimal)
  - `archive` (retain but off critical path)
  - `remove` (safe to delete)
- Produce a short migration map before changing files.

Exit criteria:
- Approved keep/archive/remove list exists.
- No large deletes before map approval.

### Phase 3: Structural Reorg (Core First)
Status: `[x]`
- Reorganize repo to emphasize:
  - core simulation + RL env
  - baseline suite
  - analysis/visual diagnostics
- Move non-core experiments into archive paths.
- Update imports and module boundaries.

Exit criteria:
- Core paths are obvious to new contributors.
- No broken imports in core workflows.

### Phase 4: Canonical Entrypoints
Status: `[x]`
- Establish one canonical baseline entrypoint.
- Establish one canonical RL entrypoint.
- Ensure both use config-first loading and same output structure.

Exit criteria:
- Both entrypoints run from docs without custom edits.
- Both emit required artifact schema.

### Phase 5: Baseline Workflow Consolidation
Status: `[ ]`
- Wire static baseline + bounded heuristic search into unified baseline suite.
- Align metrics and output schema with RL run outputs.
- Lock benchmark splits/seeds.

Exit criteria:
- Baseline suite reproducible and comparable to RL.

### Phase 6: Defender RL (V1 Core)
Status: `[ ]`
- Implement/clean defender RL run path with constraints.
- Train against scripted attacker profile source.
- Evaluate vs baseline on fixed suite.

Exit criteria:
- Defender RL outperforms static baseline on primary metric under fixed eval protocol.

### Phase 7: Verification + Diagnostics Hardening
Status: `[ ]`
- Tighten sim/visual consistency checks.
- Keep manual diagnostic outputs for hit verification.
- Add minimal smoke tests for critical paths.

Exit criteria:
- Regression checks catch sim/visual mismatches.
- Manual verification remains easy and trusted.

### Phase 8: Documentation Refresh
Status: `[ ]`
- Rewrite README around canonical entrypoints.
- Add quickstart for baseline and RL.
- Add architecture + dataflow diagram.

Exit criteria:
- New user can run baseline and RL workflows in one session.

### Phase 9: V2 Prep (Attacker RL + Alternating Training)
Status: `[ ]`
- Add attacker RL configs and runner extensions.
- Add alternating training protocol design.

Exit criteria:
- V2 plan executable with minimal codepath duplication.

### Phase 10: V3 Prep (Multi-agent + Robustness)
Status: `[ ]`
- Add robustness suite design and domain randomization protocol.
- Define experimental matrix and reporting templates.

Exit criteria:
- V3 roadmap ready for implementation without architecture churn.

---

## Manual Decision Points (Expected)
These require Matthew approval before Codex executes:
- Final keep/archive/remove list.
- Historical plausibility bounds for layout/action spaces.
- Train/eval profile split definitions.
- Seed protocols and benchmark suite locking.
- What to permanently delete vs archive.

## Working Rules for Overhaul
- Run tests/compilation checks after every structural change.
- Keep run artifact format stable during reorg.
- Do not add new features during structural phases unless required to restore broken workflows.

## Definition of "Repo Slimmed and Refocused"
- One obvious baseline entrypoint.
- One obvious RL entrypoint.
- Clear separation of core vs archive code.
- Reproducible evaluation protocol shared by baseline and RL.
- Documentation reflects actual run paths (no dead instructions).
