# Script Entrypoints

This file indexes runnable script entrypoints. Use `docs/REPRODUCING.md` for top-to-bottom rerun commands and `docs/PROJECT_MAP.md` for the broader codebase map.

Most commands support `--help`:

```bash
python -m experiments.run_baseline_suite --help
```

## Canonical Workflows

| Purpose | Entrypoint | Notes |
|---|---|---|
| Baseline suite | `python -m experiments.run_baseline_suite --config configs/baseline/default.toml` | Main baseline evaluation runner. |
| RL training/eval | `python -m experiments.run_rl_train --config configs/rl/default.toml` | Main RL layout runner. |
| Config generation | `python -m experiments.generate_run_config --template ... --output ...` | Creates reproducible TOML configs with split metadata. |
| RL action audit | `python -m experiments.audit_rl_actions --config configs/rl/default.toml` | Diagnostic audit for configured RL actions/layouts. |

## Attack-Profile Data

| Purpose | Entrypoint | Typical Outputs |
|---|---|---|
| Curated/random profile generation | `python -m experiments.generate_attack_profile_scaffold ...` | JSONL profile datasets or scaffold snippets. |
| Random baseline profile generation | `python -m experiments.generate_random_attack_profile_dataset ...` | Random-profile JSONL datasets for VAE comparison. |
| Mixed VAE dataset build | `python -m experiments.build_mixed_attack_profile_dataset --overwrite` | Mixed curated/random train-valid JSONL datasets. |
| Dataset audit | `python -m experiments.audit_attack_profile_dataset --input ...` | Flattened CSVs and distribution summaries. |
| Profile geometry audit | `python -m experiments.audit_attack_profiles ...` | Geometry/audit CSV and JSON diagnostics. |

Primary supporting modules:
- `convoy_sim/profile_audit.py`
- `convoy_sim/profile_outcome_audit.py`
- `convoy_sim/target_zones.py`
- `convoy_sim/attack_profiles.py`

## VAE And Candidate Pools

| Purpose | Entrypoint | Typical Outputs |
|---|---|---|
| VAE candidate-pool generation | `python -m experiments.generate_vae_candidate_pool --run-dir ... --train-path ... --output ...` | Filtered candidate-pool JSONL plus summary JSON. |
| Full-state candidate evaluation | `python -m experiments.evaluate_attack_candidate_pool --candidate-path ...` | Ranked candidate CSV, top candidate JSON, metrics, manifest. |

Primary supporting modules:
- `convoy_sim/vae.py`
- `convoy_sim/vae_diagnostics.py`
- `convoy_sim/workflows.py`

## POMDP / Adversarial Selection

| Purpose | Entrypoint | Typical Outputs |
|---|---|---|
| Belief-limited candidate selector | `python -m experiments.run_pomdp_candidate_selector --candidate-path ... --observation-preset good_contact` | Belief-ranked CSV/JSON and selected top-k JSONL pool. |
| POMDP fire-control rebuilds | `convoy_sim.pomdp_fire_control` module functions | Rebuilt candidate pools for notebook-driven evaluation. |

Primary supporting modules:
- `convoy_sim/pomdp_candidate_selector.py`
- `convoy_sim/pomdp_fire_control.py`
- `convoy_sim/fire_control.py`
- `convoy_sim/realism.py`

## Visualization And Manual QA

`notebooks/visuals.ipynb` is the notebook home for one-off visual checks. It calls these script entrypoints rather than duplicating plotting logic.

| Visual Check | Entrypoint | Default Outputs |
|---|---|---|
| Plan-view layouts | `python -m experiments.plot_layout` | `results/figures/rect_class.png`, `rect_value.png`, `staggered_class.png`, `staggered_value.png` |
| Static attack overlay | `python -m experiments.plot_attack_once` | `results/figures/attack_once.png`, `results/debug/attack_once.json` |
| Temporal attack frames | `python -m experiments.render_attack_animation` | `results/frames/demo_attack/frame_*.png`, optional MP4 |
| Attack-profile previews | `python -m experiments.render_attack_profile_previews ...` | `results/frames/attack_profile_previews/`, `results/diag/attack_profile_*` |

Plotting commands require Matplotlib. MP4 export requires animation/video support. If Matplotlib cache warnings appear, set `MPLCONFIGDIR` to a writable cache directory.

## Common Output Areas

Generated outputs are intentionally excluded from the clean repo presentation.

| Area | Use |
|---|---|
| `results/runs/` | Script-run metrics, manifests, and larger evaluation outputs. |
| `results/notebook-results/<notebook-name>/` | Notebook-owned summaries, figures, and compact artifacts. |
| `results/figures/` | Small script-generated figures. |
| `results/frames/` | Animation frames and videos. |
| `results/diag/` | Audit and diagnostic files. |

Archived pre-cleanup outputs live locally under `Archive-June-23/`.
