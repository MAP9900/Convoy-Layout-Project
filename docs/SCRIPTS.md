# Script And File Index

Current complete index of files in the repository, grouped by purpose.

Primary simulation feature reference:
- `docs/SIM_FEATURES.md`

## Canonical Workflows

- Baseline: `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- RL: `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Script Metadata Matrix

| Script | Entrypoint | Inputs | Outputs | Depends On | Runtime Estimate | Used By |
|---|---|---|---|---|---|---|
| `experiments/run_baseline_suite.py` | `python -m experiments.run_baseline_suite --config ...` | Baseline TOML config, scenario/profile refs | Baseline metrics/artifacts under `results/` | `convoy_sim.workflows`, scenarios, core sim modules | Medium | Baseline |
| `experiments/run_rl_train.py` | `python -m experiments.run_rl_train --config ...` | RL TOML config | RL training/eval artifacts and checkpoints under `results/` | `convoy_sim.rl_env`, `convoy_sim.rl_wrapper`, `convoy_sim.rl_layout_builder`, workflows | Long | RL |
| `experiments/audit_rl_actions.py` | `python -m experiments.audit_rl_actions --config ...` | RL TOML config | Per-action train/eval audit metrics, plots, and manifest under `results/` | `convoy_sim.workflows`, `convoy_sim.rl_layout_builder`, RL action/builder config, core sim modules | Medium-Long | RL diagnostic |
| `experiments/generate_run_config.py` | `python -m experiments.generate_run_config --template ... --output ...` | Template TOML, split seed, split sizes, optional seed/run-name overrides | Full generated TOML with reproducible splits and `split_meta` | Python `tomllib`, local config templates | Short | Baseline + RL config generation |
| `experiments/generate_attack_profile_scaffold.py` | `python -m experiments.generate_attack_profile_scaffold --count ...` | Mode, starting id, profile count, seed, convoy profile, output format/path | Curated scaffold payloads or JSONL dataset records with generation-time feasibility/audit filtering | `convoy_sim.attack_profiles`, `convoy_sim.profile_audit`, `convoy_sim.target_zones`, bounded realistic preset catalog | Short | Attack-profile authoring + dataset generation |
| `experiments/audit_attack_profile_dataset.py` | `python -m experiments.audit_attack_profile_dataset --input ...` | Dataset JSONL path, output dir | Flattened CSV plus distribution summaries for synthetic attack-profile datasets, including v3 target-zone fields when present | JSONL dataset records from `generate_attack_profile_scaffold --mode dataset/random_zones` | Short | Dataset audit / notebook prep |
| `experiments/audit_attack_profiles.py` | `python -m experiments.audit_attack_profiles ...` | Convoy profile id(s), audit parameters | Audit CSV/JSON diagnostics | `convoy_sim.profile_audit`, `convoy_sim.attack_profiles` | Short-Medium | Diagnostic |
| `experiments/render_attack_profile_previews.py` | `python -m experiments.render_attack_profile_previews ...` | Convoy profile id(s), run mode, rendering flags | Preview frames + geometry/hit diagnostics in `results/diag` and frame dirs | `convoy_sim.viz_attack`, profile audit helpers | Medium-Long | Manual verification |
| `experiments/render_attack_animation.py` | `python -m experiments.render_attack_animation ...` | Scenario/profile and render args | Animation frames/video outputs in `results/` | `convoy_sim.viz_attack`, simulation modules | Medium | Manual verification |
| `experiments/plot_attack_once.py` | `python -m experiments.plot_attack_once` | `small_demo` convoy profile + fixed fan spread | One static plot + debug JSON | `convoy_sim.viz_attack`, `scenarios.convoy_profiles` | Short | Diagnostic |
| `experiments/plot_layout.py` | `python -m experiments.plot_layout ...` | Layout/scenario args | Static layout figures | `convoy_sim.layouts`, `convoy_sim.viz` | Short | Diagnostic |

## Root Files

- `README.md` (`canonical`)
- `requirements.txt` (`canonical`)
- `requirements-dev.txt` (`canonical`)
- `requirements-ml.txt` (`canonical`)

## Configs (All)

- `configs/templates/baseline.template.toml` (`canonical`)
- `configs/templates/rl.template.toml` (`canonical`)
- `configs/baseline/default.toml` (`canonical`)
- `configs/rl/default.toml` (`canonical`)

## Experiment Scripts (All)

- `experiments/run_baseline_suite.py`: Canonical baseline runner (config-first, artifact schema output). (`canonical`)
- `experiments/run_rl_train.py`: Canonical RL runner (config-first train/eval + checkpoint output; supports flat action menu or builder mode). (`canonical`)
- `experiments/audit_rl_actions.py`: Direct audit of configured RL actions or builder-materialized layouts on train/eval splits. By default this now uses a staged screen-plus-promote funnel controlled by the RL `[runtime]` block. (`diagnostic`)
- `experiments/generate_run_config.py`: Reproducible full TOML generator for baseline/RL run configs and profile splits. (`canonical`)
- `experiments/generate_attack_profile_scaffold.py`: Shared attack-profile generator with legacy centroid modes, v3 target-zone modes, and v4 spawn-first tactical mode. `curated` preserves the current helper-style library workflow and can emit `_scaffolded_fan_profile(...)` calls for `attack_profiles.py`. `dataset` emits legacy centroid JSONL records for larger synthetic corpora and currently targets a `75% credible_hit_threat / 25% credible_near_miss` mix. `curated_zones` emits deterministic target-zone profiles for debugging. `random_zones` emits structured-random v3 JSONL records with explicit `intent` metadata for VAE training. `random_tactical_v4` samples U-boat spawn regions first, including perimeter, beam/ahead/astern, and inside-column positions, then selects an accessible target and a standard zig-zag lead/intercept aim point. v4 also adds deliberate meter-based aim offsets for `credible_near_miss` and `intentional_miss`, targets a `65/25/10` label mix, and rejects candidates that fail the moving zig-zag dynamic outcome gate. All modes sample `u_boat_initial_speed_mps` from `1.0` to `2.0` m/s in `0.1` steps, reject spawns that start within `250 m` of any convoy ship, and reject profiles that fail the selected geometry plausibility audit. (`canonical`)
- `experiments/generate_random_attack_profile_dataset.py`: Profile-first random baseline generator for VAE comparison. It samples U-boat spawn, reference point, bearing error, spread, timing, and speed directly; enforces the same minimum spawn clearance, moving U-boat, `uniform_divergent`, and moving zig-zag outcome audit; then labels accepted records from actual sim outcomes into the same `65/25/10` hit/near/intentional-miss quota. (`canonical`)
- `experiments/audit_attack_profile_dataset.py`: Audits JSONL synthetic attack-profile corpora, writes a flattened `profiles_flat.csv`, per-dimension count CSVs, and a summary JSON. v3 target-zone records add counts by `target_zone_kind`, `approach_side`, and `approach_lane`; v4 tactical records also add `spawn_region`, `inside_convoy_envelope`, target-aspect, target-score, nearest-ship clearance, lead-solution fields, and aim-offset fields. The module functions can also be imported directly into a notebook for ad hoc inspection. (`diagnostic`)
- `convoy_sim/profile_outcome_audit.py`: Reusable dynamic outcome audit for generated profile datasets. It converts records back into `AttackProfile`s, builds sim-native torpedoes, evaluates them against moving zig-zag convoy kinematics, and reports intended-target hits, other-ship hits, closest passes, and outcome-vs-intent agreement. It can also attach those outcome labels back onto JSONL-style records, filter records by the outcome gate, and support future adversarial candidate-selection work. Used by `random_tactical_v4` generation and `notebooks/profile_generation_tests.ipynb`. (`diagnostic`)
- `experiments/audit_attack_profiles.py`: Attack-profile geometry plausibility audit. (`diagnostic`)
- `experiments/render_attack_profile_previews.py`: Profile frame rendering + audit/hit CSV outputs. (`diagnostic`)
- `experiments/render_attack_animation.py`: Dynamic attack animation/frame generation demo. (`diagnostic`)
- `experiments/plot_attack_once.py`: Single static attack plot + debug JSON. (`diagnostic`)
- `experiments/plot_layout.py`: Static layout figure generator. (`diagnostic`)

## Scenarios (All)

- `scenarios/convoy_profiles.py` (`canonical`)
- `scenarios/scenario_base.py` (`supporting`)

## Core Python Modules (All)

- `convoy_sim/__init__.py` (`canonical`)
- `convoy_sim/attack_profiles.py` (`canonical`)
- `convoy_sim/attack_proposals.py` (`supporting`)
- `convoy_sim/attacker_opt.py` (`supporting`)
- `convoy_sim/attackers.py` (`supporting`)
- `convoy_sim/defender_opt.py` (`supporting`)
- `convoy_sim/defender_policy.py` (`canonical`)
- `convoy_sim/diagnostics.py` (`diagnostic`)
- `convoy_sim/dynamics.py` (`canonical`)
- `convoy_sim/entities.py` (`canonical`)
- `convoy_sim/fire_control.py` (`canonical`)
- `convoy_sim/feasibility.py` (`canonical`)
- `convoy_sim/game.py` (`supporting`)
- `convoy_sim/geometry.py` (`canonical`)
- `convoy_sim/layout_roles.py` (`canonical`)
- `convoy_sim/layouts.py` (`canonical`)
- `convoy_sim/noise.py` (`supporting`)
- `convoy_sim/objectives.py` (`canonical`)
- `convoy_sim/profile_audit.py` (`diagnostic`)
- `convoy_sim/profile_generation_viz.py` (`diagnostic`)
- `convoy_sim/realism.py` (`canonical`)
- `convoy_sim/risk.py` (`canonical`)
- `convoy_sim/rl_env.py` (`canonical`)
- `convoy_sim/rl_layout_builder.py` (`canonical`)
- `convoy_sim/rl_wrapper.py` (`canonical`)
- `convoy_sim/ship_catalog.py` (`canonical`)
- `convoy_sim/simulation.py` (`canonical`)
- `convoy_sim/target_zones.py` (`canonical`)
- `convoy_sim/trial_records.py` (`supporting`)
- `convoy_sim/viz.py` (`diagnostic`)
- `convoy_sim/viz_attack.py` (`diagnostic`)
- `convoy_sim/workflows.py` (`canonical`)

## Tests (All Files)

- `tests/conftest.py` (`canonical`)
- `tests/test_attack_frame_math.py` (`canonical`)
- `tests/test_attack_profiles.py` (`canonical`)
- `tests/test_attack_windows.py` (`canonical`)
- `tests/test_attacker_opt_smoke.py` (`supporting`)
- `tests/test_canonical_entrypoints.py` (`canonical`)
- `tests/test_defender_opt_smoke.py` (`supporting`)
- `tests/test_defender_policy.py` (`canonical`)
- `tests/test_detection_risk.py` (`canonical`)
- `tests/test_diagnostics_core.py` (`diagnostic`)
- `tests/test_dynamic_hit_events.py` (`canonical`)
- `tests/test_dynamics_models.py` (`canonical`)
- `tests/test_entities.py` (`canonical`)
- `tests/test_generate_run_config.py` (`canonical`)
- `tests/test_feasibility_checks.py` (`canonical`)
- `tests/test_feasibility_models.py` (`canonical`)
- `tests/test_formation_motion.py` (`canonical`)
- `tests/test_game_matrix_and_exploitability.py` (`supporting`)
- `tests/test_geometry.py` (`canonical`)
- `tests/test_layouts.py` (`canonical`)
- `tests/test_layouts_heterogeneous.py` (`canonical`)
- `tests/test_noise_regression.py` (`supporting`)
- `tests/test_objectives.py` (`canonical`)
- `tests/test_profile_audit.py` (`diagnostic`)
- `tests/test_realism_v2.py` (`canonical`)
- `tests/test_risk.py` (`canonical`)
- `tests/test_rl_layout_builder.py` (`canonical`)
- `tests/test_rl_wrapper.py` (`canonical`)
- `tests/test_sampler_with_constraints.py` (`canonical`)
- `tests/test_scenarios_smoke.py` (`canonical`)
- `tests/test_serialization_roundtrip.py` (`canonical`)
- `tests/test_ship_heterogeneity_models.py` (`canonical`)
- `tests/test_simulation.py` (`canonical`)
- `tests/test_simulation_semantics.py` (`canonical`)
- `tests/test_target_zones.py` (`canonical`)
- `tests/test_time_dependent_feasibility.py` (`canonical`)
- `tests/test_value_aimpoint.py` (`canonical`)
- `tests/test_value_scoring.py` (`canonical`)
- `tests/test_viz_attack_helpers.py` (`diagnostic`)
- `tests/test_viz_helpers.py` (`diagnostic`)
- `tests/test_viz_no_matplotlib_import.py` (`diagnostic`)
- `tests/tests2.ipynb` (`supporting`)
- `tests/tests_1.ipynb` (`supporting`)

## Notebooks (All)

- `notebooks/attack_profile_tests.ipynb` (`diagnostic`)
- `notebooks/attack_manual_verification.ipynb` (`diagnostic`)
- `notebooks/torpedo_firing_doctrine_comparison.ipynb` (`diagnostic`; saves PNGs to `notebooks/results/torpedo_firing_doctrine_comparison/` by default)
- `notebooks/vae_exploration.ipynb` (`canonical`; active notebook-first VAE training and analysis workflow)

## Archived

- `archive/vae/train_vae.py`: Archived terminal-first VAE runner kept for reference after moving the active VAE workflow into the notebook. (`archived-reference`)
- `archive/vae/test_train_vae.py`: Archived smoke test for the old terminal-first VAE runner. (`archived-reference`)

## Results Files (Tracked)

- `results/debug/attack_once.json` (`diagnostic`)
- `results/diag/attack_profile_geometry_audit.csv` (`diagnostic`)
- `results/diag/attack_profile_geometry_audit.json` (`diagnostic`)
- `results/figures/attack_once.png` (`diagnostic`)
- `results/figures/rect_class.png` (`diagnostic`)
- `results/figures/rect_value.png` (`diagnostic`)
- `results/figures/staggered_class.png` (`diagnostic`)
- `results/figures/staggered_value.png` (`diagnostic`)

## Reference Docs

- `docs/TODO.md` (`canonical`)
- `docs/SIM_FEATURES.md` (`canonical`)
- `docs/PROTOCOL_V2_REALISM.md` (`canonical`)
- `docs/NOTES.md` (`supporting`)
- `docs/RL_PLAN.md` (`canonical`)
- `docs/Visuals.md` (`supporting`)
- `docs/RESULTS_LOG.md` (`canonical`)
- `docs/OPTIMIZATION_LOG.md` (`canonical`)
- `docs/SCRIPTS.md` (`canonical`)

## Runtime Budget Notes

- Final benchmark eval still uses full `simulation.n_trials_per_seed`.
- Baseline heuristic train-search can use a cheaper budget with:
  - `runtime.baseline_search_n_trials_per_seed`
- RL train-time action ranking can use a cheaper budget with:
  - `runtime.rl_ranking_n_trials_per_seed`
- RL action audit now defaults to a staged funnel:
  - screen all actions with `runtime.audit_screen_n_trials_per_seed`
  - rerun only top-K train/eval candidates at full budget with `runtime.audit_top_k_full_eval`

To force the old full-fidelity audit behavior:
- set `runtime.audit_screen_n_trials_per_seed = simulation.n_trials_per_seed`
- set `runtime.audit_top_k_full_eval` to the full action count
  - current canonical builder count: `144`

## Lifecycle Tag Legend

- `canonical`: Primary paths and files to maintain and evolve.
- `supporting`: Auxiliary components that support canonical flows.
- `diagnostic`: Debug, plotting, or audit-only utilities and artifacts.
- `legacy-candidate`: Present but likely removable after replacement/confirmation.
