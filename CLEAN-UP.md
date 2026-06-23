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
- [ ] Confirm all old run artifacts are recoverable from the archive branch before deleting tracked generated outputs.
- [ ] Optionally copy heavy results/data/checkpoints to a local external archive folder outside the repo.
- [ ] Before untracking files now covered by `.gitignore`, confirm they are either reproducible, preserved on the archive branch, or copied to external archive storage.

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
- [ ] Treat `results/notebook-results/` as generated output, usually ignored unless a specific final artifact is intentionally promoted.
- [ ] Ignore future generated run folders such as `results/runs/`, `results/diag/`, `results/frames/`, and `results/notebook-results/`.

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

- [ ] Decide whether current `data/attack_profiles/vae_candidates/` files are final artifacts, reproducible generated artifacts, or archive-only outputs.
- [ ] Decide whether current `data/attack_profiles/synthetic/` files should be regenerated from scripts rather than tracked.

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

## Git Ignore Updates

- [x] Ignore `.DS_Store`.
- [x] Ignore `.pytest_cache/`.
- [x] Ignore `.vscode/` unless intentionally shared.
- [x] Ignore generated result paths:

```gitignore
results/runs/
results/diag/
results/debug/
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

- [ ] `results/runs/`
- [ ] `results/diag/`
- [ ] `results/debug/`
- [ ] `results/figures/` unless selected figures are promoted to `results/final/figures/`
- [ ] `results/frames/`
- [ ] `notebooks/results/` after migrating paths to `results/notebook-results/`
- [ ] `.pytest_cache/`
- [ ] `.vscode/` unless intentionally keeping shared editor settings

### Data Folders To Review Before Removing

- [ ] `data/attack_profiles/synthetic/`
- [ ] `data/attack_profiles/vae_candidates/`

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
