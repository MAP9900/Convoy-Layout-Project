# Clean-Up Plan

Goal: make the repository read like a polished, reproducible research project while preserving the old exploratory trail in the archive branch and/or external local storage.

Framing:
- Personal research project built for exploration, learning, and portfolio value.
- Data science emphasis: simulator outputs, experiment design, results, model comparisons, and reproducibility matter more than heavy software-engineering ceremony.
- Code should stay simple and reviewable where possible.
- Final presentation should feel clean, intentional, and lightweight.

## Guiding Rules

- [ ] Keep source code, configs, tests, and final reproducible workflows in the main repo.
- [ ] Keep only curated final outputs in Git.
- [ ] Move generated/historical outputs out of the main presentation path.
- [ ] Preserve old runs/results through the published archive branch and optionally a local external archive copy.
- [ ] Avoid major simulator rewrites unless cleanup exposes a clear, practical need.
- [ ] Prefer small, reviewable cleanup commits over one large deletion/reorganization commit.

## Branch And Safety

- [x] Create a cleanup branch.
- [x] Create and publish an archive branch preserving the pre-cleanup state.
- [x] Move generated historical artifacts into local ignored archive folder `Archive-June-23/`.
- [ ] Confirm all old run artifacts are recoverable from the archive branch before committing tracked generated-output deletions.
- [ ] Optionally copy heavy results/data/checkpoints to a local external archive folder outside the repo.
- [x] Before untracking files now covered by `.gitignore`, confirm they are either reproducible, preserved on the archive branch, or copied to local archive storage.

## Results And Artifact Layout

- [x] Replace `notebooks/results/` with `results/notebook-results/`.
- [x] Update every notebook output path that currently writes to `notebooks/results/`.
- [x] Update docs that reference `notebooks/results/`.
- [ ] Decide final public results structure, likely:

```text
results/
  final/
    figures/
    metrics/
    manifests/
  notebook-results/
```

- [ ] Keep `results/final/` small and curated.
- [x] Treat `results/notebook-results/` as generated output, usually ignored unless a specific final artifact is intentionally promoted.
- [x] Ignore future generated run folders such as `results/runs/`, `results/diag/`, `results/frames/`, and `results/notebook-results/`.

## Data Folder Policy

- [ ] Keep a `data/` folder because it is useful and expected in a data science project.
- [ ] Decide which data artifacts are source data versus generated data.
- [ ] Keep small, canonical seed/source data in Git if needed for reproducibility.
- [ ] Regenerate large or derived datasets during the final top-to-bottom rerun instead of keeping stale generated data in Git.
- [ ] Document how to regenerate final datasets from scripts/configs.
- [ ] Consider this structure:

```text
data/
  raw/          # manually curated or source-of-truth inputs, if any
  interim/      # generated intermediate datasets, ignored
  processed/    # final generated datasets, usually ignored unless small/canonical
```

- [x] Move current `data/attack_profiles/vae_candidates/` files to local archive as generated/stale experiment artifacts.
- [x] Move current `data/attack_profiles/synthetic/` files to local archive as generated/stale experiment artifacts.
- [ ] Decide whether any future small canonical attack-profile datasets should be promoted back into Git after the final rerun.

## Notebook Standardization

- [x] Standardize notebook names, opening order, and purpose headers.
- [x] Add a short top cell to each kept notebook with purpose, inputs, outputs, and expected runtime.
- [x] Standardize output paths to `results/notebook-results/<notebook-name>/`.
- [x] Standardize notebook repo-root bootstrap/import setup.
- [x] Standardize configuration section labels and `NOTEBOOK_NAME` variables.
- [ ] Clear or minimize bulky notebook outputs if they make diffs noisy.
- [ ] Keep only notebooks that support the final story or reproducibility.
- [ ] Archive/remove superseded exploratory notebooks after final decisions are made.

Decision needed:
- [ ] Decide whether final notebooks should keep representative executed outputs or be cleared before commit. Current cleanup has standardized live code paths, but old saved outputs may still contain historical absolute paths and stale run results.

Candidate final notebook categories:
- [ ] Profile/data generation QA.
- [ ] VAE training or VAE comparison.
- [ ] Candidate-pool generation/evaluation.
- [ ] POMDP candidate selector/evaluation.
- [ ] Manual visual verification, if it supports the final writeup.

## Markdown And Documentation Cleanup

- [ ] Archive or clean markdown files that are mostly old logs or working notes.
- [ ] Convert final public docs into a small, coherent set.
- [ ] Keep process-heavy docs out of the main reader path.
- [ ] Update README after cleanup so it points to the final workflow only.

Candidate final docs:
- [ ] `README.md`
- [ ] `docs/ARCHITECTURE.md`
- [ ] `docs/METHODS.md`
- [ ] `docs/EXPERIMENTS.md`
- [ ] `docs/RESULTS.md`
- [ ] `docs/REPRODUCING.md`

Docs likely to archive, rewrite, or consolidate:
- [ ] `NOTES.md`
- [ ] `docs/NOTES.md`
- [ ] `docs/TODO.md`
- [ ] `docs/RESULTS_LOG.md`
- [ ] `docs/OPTIMIZATION_LOG.md`
- [ ] `docs/Visuals.md`
- [ ] `docs/ADVERSARIAL_POMDP_OUTLINE.md`
- [ ] `docs/RL_PLAN.md`
- [ ] `docs/POMDP.md`
- [ ] `docs/VAE.md`
- [ ] `docs/SIM_FEATURES.md`
- [ ] `docs/SCRIPTS.md`
- [ ] `docs/PROTOCOL_V2_REALISM.md`

## Codebase Polish

- [ ] Keep simulator code stable unless there is a clear cleanup win.
- [ ] Add lightweight project tooling if useful: `pyproject.toml`, formatter/linter config, pytest config.
- [ ] Remove local junk files from the workspace and ignore them going forward.
- [ ] Consider adding simple command wrappers or a final workflow script only after the final rerun protocol is clear.
- [ ] Avoid splitting large modules just for aesthetics before final experiments are complete.

## Final Top-To-Bottom Rerun

- [ ] Define the final run order before deleting or regenerating data.
- [ ] Run tests from a clean branch state.
- [ ] Regenerate any required datasets.
- [ ] Run final baseline workflow.
- [ ] Run final VAE/candidate-pool workflow.
- [ ] Run final POMDP selector/evaluation workflow.
- [ ] Save only curated final summaries, figures, configs, and manifests to `results/final/`.
- [ ] Update final results documentation from the rerun.
- [ ] Confirm all README commands work from a fresh environment.

### Draft Regeneration Protocol

Goal: after cleanup, rerun the project from source code/configs and keep only the results that support the final project story. Treat this as the working protocol until the exact final commands are validated.

Environment notes:
- [ ] Use the project conda environment consistently. Likely commands should run with `/opt/homebrew/Caskroom/miniforge/base/envs/Python-DS/bin/python` for data/analysis work and `Python-ML` only where model training needs it.
- [ ] Record the exact environment used for the final rerun in `docs/REPRODUCING.md` or the final README.
- [ ] Before rerunning, confirm ignored/generated folders are empty or intentionally absent: `results/runs/`, `results/diag/`, `results/figures/`, `results/frames/`, `results/notebook-results/`, `data/attack_profiles/synthetic/`, and `data/attack_profiles/vae_candidates/`.

Recommended order:
- [ ] Run lightweight tests/import checks first so failures are caught before expensive notebook/model work.
- [ ] Regenerate synthetic attack-profile datasets using the current generation scripts/notebooks.
  - Candidate entry points: `experiments/generate_random_attack_profile_dataset.py`, `experiments/build_mixed_attack_profile_dataset.py`, `notebooks/profile_generation_tests.ipynb`, `notebooks/attack_profile_tests.ipynb`.
- [ ] Audit regenerated attack-profile geometry and labels before training.
  - Candidate entry points: `experiments/audit_attack_profiles.py`, `experiments/audit_attack_profile_dataset.py`, `notebooks/attack_manual_verification.ipynb`.
- [ ] Run baseline simulation/layout workflows and keep only useful summaries.
  - Candidate entry points: `experiments/run_baseline_suite.py`, `notebooks/torpedo_firing_doctrine_comparison.ipynb`.
- [ ] Train or rerun final VAE variants.
  - Candidate notebooks: `notebooks/vae_train.ipynb`, `notebooks/random_vae_train.ipynb`, `notebooks/mixed_vae_train.ipynb`.
- [ ] Generate VAE candidate pools from the final trained models.
  - Candidate entry points: `experiments/generate_vae_candidate_pool.py`, `notebooks/vae_candidate_pool.ipynb`.
- [ ] Evaluate candidate-pool quality and compare sources.
  - Candidate entry points: `experiments/evaluate_attack_candidate_pool.py`, `notebooks/attack_candidate_pool_eval.ipynb`, `notebooks/vae_source_comparison.ipynb`, `notebooks/vae_final_baseline_comparison.ipynb`.
- [ ] Run final POMDP candidate selector/evaluation work.
  - Candidate entry points: `experiments/run_pomdp_candidate_selector.py`, `notebooks/pomdp_candidate_selector.ipynb`, `notebooks/pomdp_candidate_eval.ipynb`, `notebooks/pomdp_fire_control_eval.ipynb`.
- [ ] Run final RL work only after the POMDP/RL scope is settled.
  - Candidate entry points: `experiments/run_rl_train.py`, `experiments/audit_rl_actions.py`.
- [ ] Promote only final, explainable artifacts into `results/final/`.
  - Suggested contents: selected figures, compact metrics tables, final configs, run manifests, and a short notes file explaining what each artifact supports.
- [ ] Update final docs from the rerun rather than preserving old working logs.
  - Likely docs: `README.md`, `docs/METHODS.md`, `docs/EXPERIMENTS.md`, `docs/RESULTS.md`, `docs/REPRODUCING.md`.

Open decisions before the rerun:
- [ ] Decide whether final notebooks should be committed with outputs cleared, representative outputs kept, or outputs omitted from Git entirely.
- [ ] Decide whether any regenerated dataset is small/canonical enough to track under `data/`, or whether all regenerated data remains ignored.
- [ ] Decide which final figures/metrics are portfolio-facing and deserve promotion to `results/final/`.
- [ ] Decide whether to create a small driver script or Makefile after the exact rerun order is proven manually.

## Git Ignore Updates

- [x] Ignore `.DS_Store`.
- [x] Ignore `.pytest_cache/`.
- [x] Ignore `.vscode/` unless intentionally shared.
- [x] Ignore generated result paths:

```gitignore
results/runs/
results/diag/
results/debug/
results/figures/
results/frames/
results/notebook-results/
notebooks/results/
```

- [x] Ignore generated model/checkpoint artifacts unless intentionally promoted:

```gitignore
*.pt
checkpoints/
```

- [x] Ignore generated datasets under `data/attack_profiles/` by default.
- [x] Ignore old one-off root-level plot artifacts.

## Delete Or Archive Candidates

Do not delete until the archive branch and/or local external archive is confirmed.

### Entire Folders Likely To Remove From Main Presentation

- [x] `results/runs/` moved to `Archive-June-23/results/runs/`
- [x] `results/diag/` moved to `Archive-June-23/results/diag/`
- [x] `results/debug/` moved to `Archive-June-23/results/debug/`
- [x] `results/figures/` moved to `Archive-June-23/results/figures/`; selected final figures should later be promoted to `results/final/figures/`
- [x] `results/frames/` moved to `Archive-June-23/results/frames/`
- [x] `results/notebook-results/` moved to `Archive-June-23/results/notebook-results/`
- [x] `notebooks/results/` after migrating paths to `results/notebook-results/`
- [ ] `.pytest_cache/`
- [ ] `.vscode/` unless intentionally keeping shared editor settings

### Data Folders To Review Before Removing

- [x] `data/attack_profiles/synthetic/` moved to `Archive-June-23/data/attack_profiles/synthetic/`
- [x] `data/attack_profiles/vae_candidates/` moved to `Archive-June-23/data/attack_profiles/vae_candidates/`

Decision needed: keep small canonical data, regenerate derived data, or move old snapshots to archive-only.

### Root-Level Files Likely To Move Or Remove

- [ ] `Mixed_Loss_Plot.png`
- [ ] `VAE_Curated.png`
- [ ] `VAE_Mixed.png`
- [ ] `VAE_Random.png`
- [ ] `final_comparison.png`
- [ ] `final_comparison_by_region.png`
- [ ] `NOTES.md`

Selected final plots should move to `results/final/figures/`; stale or intermediate plots should be removed from the main branch.

### Local Junk Files To Delete And Ignore

- [ ] `.DS_Store`
- [ ] `docs/.DS_Store`
- [ ] `configs/.DS_Store`
- [ ] `configs/archive/.DS_Store`
- [ ] `scenarios/.DS_Store`
- [ ] `data/.DS_Store`
- [ ] `data/attack_profiles/.DS_Store`
- [ ] `data/attack_profiles/synthetic/.DS_Store`
- [ ] `notebooks/.DS_Store`
- [ ] `notebooks/results/.DS_Store`
- [ ] `results/.DS_Store`
- [ ] any nested `.DS_Store` under generated results folders

### Markdown Files To Archive Or Rewrite

- [ ] `docs/TODO.md`
- [ ] `docs/RESULTS_LOG.md`
- [ ] `docs/OPTIMIZATION_LOG.md`
- [ ] `docs/NOTES.md`
- [ ] `docs/Visuals.md`
- [ ] `docs/ADVERSARIAL_POMDP_OUTLINE.md`
- [ ] `docs/RL_PLAN.md`
- [ ] `docs/POMDP.md`
- [ ] `docs/VAE.md`
- [ ] `docs/SIM_FEATURES.md`
- [ ] `docs/SCRIPTS.md`
- [ ] `docs/PROTOCOL_V2_REALISM.md`

These may not all be deleted. Some should be consolidated into final docs after the final rerun is defined.
