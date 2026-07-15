# Reproducing The Project

This is the operational runbook for regenerating data and results after cleanup.

Status: draft. The commands below reflect the current known entrypoints and should be tightened after the final top-to-bottom rerun.

## Environment

Run most commands from an already active `Python-DS` terminal, for example:

```text
(Python-DS) matthewplambeck@Matthews-MacBook-Pro-2 Convoy Layout Project %
```

Use plain `python` in that shell for data generation, audits, baseline runs, RL runs, and most notebook checks.

Switch to `Python-ML` for model-heavy VAE training/generation work when needed:

```bash
conda activate Python-ML
```

For plotting-heavy runs, set a writable Matplotlib cache directory if needed:

```bash
export MPLCONFIGDIR=.mplconfig
```

## Pre-Rerun Checks

Run these before expensive notebook/model work:

```bash
python -m pytest -q
python -m ruff check convoy_sim experiments scenarios tests --config pyproject.toml
```

Optional entrypoint smoke check:

```bash
for module in \
  audit_attack_profile_dataset \
  audit_attack_profiles \
  audit_rl_actions \
  build_mixed_attack_profile_dataset \
  evaluate_attack_candidate_pool \
  generate_attack_profile_scaffold \
  generate_random_attack_profile_dataset \
  generate_run_config \
  generate_vae_candidate_pool \
  plot_attack_once \
  plot_layout \
  render_attack_animation \
  render_attack_profile_previews \
  run_baseline_suite \
  run_pomdp_candidate_selector \
  run_rl_train
do
  python -m experiments.$module --help >/dev/null
done
```

## Canonical Baseline And RL Commands

Generate baseline config:

```bash
python -m experiments.generate_run_config \
  --template configs/templates/baseline.template.toml \
  --output configs/baseline/default.toml \
  --convoy-profile convoy_layout_1 \
  --split-seed 1945 \
  --n-total 30 \
  --n-train 20 \
  --train-seeds 1939,1940,1941 \
  --eval-seeds 1942,1943,1944
```

Generate RL config:

```bash
python -m experiments.generate_run_config \
  --template configs/templates/rl.template.toml \
  --output configs/rl/default.toml \
  --convoy-profile convoy_layout_1 \
  --split-seed 1945 \
  --n-total 30 \
  --n-train 20
```

Run baseline:

```bash
python -m experiments.run_baseline_suite \
  --config configs/baseline/default.toml
```

Run RL:

```bash
python -m experiments.run_rl_train \
  --config configs/rl/default.toml
```

## Regenerating Archived Data

The cleanup archived generated outputs from:

- `data/attack_profiles/synthetic/`
- `data/attack_profiles/vae_candidates/`
- `results/runs/`
- `results/diag/`
- `results/figures/`
- `results/frames/`
- `results/notebook-results/`

These should be regenerated from scripts/notebooks rather than restored into the main branch.

### Synthetic Attack-Profile Data

Curated tactical v4 dataset examples:

```bash
python -m experiments.generate_attack_profile_scaffold \
  --mode random_tactical_v4 \
  --count 100000 \
  --chunk-size 1000 \
  --seed 1945 \
  --start-index 1 \
  --convoy-profile convoy_layout_1 \
  --output data/attack_profiles/synthetic/train_random_tactical_v4_100k.jsonl

python -m experiments.generate_attack_profile_scaffold \
  --mode random_tactical_v4 \
  --count 5000 \
  --chunk-size 1000 \
  --seed 2945 \
  --start-index 100001 \
  --convoy-profile convoy_layout_1 \
  --output data/attack_profiles/synthetic/valid_random_tactical_v4_5k.jsonl

python -m experiments.generate_attack_profile_scaffold \
  --mode random_tactical_v4 \
  --count 25000 \
  --chunk-size 1000 \
  --seed 3945 \
  --start-index 105001 \
  --convoy-profile convoy_layout_1 \
  --output data/attack_profiles/synthetic/test_random_tactical_v4_25k.jsonl
```

The 5k validation split is evaluated each epoch. The 25k test split remains held out until the best validation checkpoint has been selected. VAE training reads only the eight model features from JSONL and does not retain the full audit records in memory.

Random baseline dataset examples:

```bash
python -m experiments.generate_random_attack_profile_dataset \
  --count 45000 \
  --seed 1945 \
  --convoy-profile convoy_layout_1 \
  --output data/attack_profiles/synthetic/train_random_profile_v1_45k.jsonl

python -m experiments.generate_random_attack_profile_dataset \
  --count 5000 \
  --seed 1946 \
  --convoy-profile convoy_layout_1 \
  --output data/attack_profiles/synthetic/valid_random_profile_v1_5k.jsonl
```

Mixed curated/random dataset example:

```bash
python -m experiments.build_mixed_attack_profile_dataset \
  --curated-train data/attack_profiles/synthetic/train_random_tactical_v4_45k.jsonl \
  --curated-valid data/attack_profiles/synthetic/valid_random_tactical_v4_5k.jsonl \
  --random-train data/attack_profiles/synthetic/train_random_profile_v1_45k.jsonl \
  --random-valid data/attack_profiles/synthetic/valid_random_profile_v1_5k.jsonl \
  --train-output data/attack_profiles/synthetic/train_mixed_curated70_random30_45k.jsonl \
  --valid-output data/attack_profiles/synthetic/valid_mixed_curated70_random30_5k.jsonl \
  --train-count 45000 \
  --valid-count 5000 \
  --curated-fraction 0.70 \
  --seed 1945 \
  --overwrite
```

Dataset audit example:

```bash
python -m experiments.audit_attack_profile_dataset \
  --input data/attack_profiles/synthetic/train_random_tactical_v4_100k.jsonl \
  --output-dir results/diag/attack_profile_dataset_audit/train_random_tactical_v4_100k
```

## Notebook Rerun Order

Run notebooks top-to-bottom after the generated datasets exist.

Use `Python-DS` for analysis/diagnostic notebooks and `Python-ML` for VAE training notebooks if the ML dependencies or model training stack are only installed there.

For VAE training, set `DATASET_SOURCE` in `notebooks/vae_train.ipynb` to `curated`, `random`, or `mixed`. The curated configuration uses 100,000 training, 5,000 validation, and 25,000 test records.

Recommended order:

1. `notebooks/profile_generation_tests.ipynb`
2. `notebooks/attack_profile_tests.ipynb`
3. `notebooks/visuals.ipynb`
4. `notebooks/vae_train.ipynb`
5. `notebooks/vae_candidate_pool.ipynb`
6. `notebooks/vae_source_comparison.ipynb`
7. `notebooks/vae_final_baseline_comparison.ipynb`
8. `notebooks/attack_candidate_pool_eval.ipynb`
9. `notebooks/pomdp_candidate_selector.ipynb`
10. `notebooks/pomdp_candidate_eval.ipynb`
11. `notebooks/pomdp_fire_control_eval.ipynb`

## VAE Candidate Pools

After VAE training notebooks produce run directories/checkpoints, generate candidate pools with `experiments.generate_vae_candidate_pool` or the corresponding notebooks.

Run this from `Python-ML` if the VAE checkpoint/model dependencies are installed there.

Example shape:

```bash
python -m experiments.generate_vae_candidate_pool \
  --run-dir results/runs/vae/<vae-run-dir> \
  --train-path data/attack_profiles/synthetic/<vae-train-dataset>.jsonl \
  --output results/notebook-results/vae_candidate_pool/<candidate-pool>.jsonl \
  --summary-output results/notebook-results/vae_candidate_pool/<candidate-pool>.summary.json \
  --sample-count 5000 \
  --keep-count 1000 \
  --device cpu \
  --sampling-method latent_bank \
  --latent-noise-scale 0.10 \
  --accepted-outcome credible_hit_threat
```

Replace placeholder paths with the actual final rerun artifacts.

## Final Output Policy

Generated outputs should remain ignored unless deliberately promoted.

Only promote curated final artifacts into:

```text
results/final/
  figures/
  metrics/
  manifests/
```

Final docs should cite the promoted artifacts, not old exploratory output folders.
