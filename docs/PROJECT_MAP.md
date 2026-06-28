# Project Map

This file is the compact codebase map for future development. Use it to decide where a change belongs before adding new modules, scripts, notebooks, or docs.

## Top-Level Layout

| Path | Role |
|---|---|
| `convoy_sim/` | Core simulation, physics, scoring, realism helpers, RL/POMDP wrappers, and VAE support code. |
| `experiments/` | Runnable command-line entrypoints for generation, audits, training, evaluation, and visual checks. |
| `notebooks/` | Notebook-first analysis workflows and final result regeneration. |
| `scenarios/` | Named convoy profiles and base scenario definitions. |
| `configs/` | Baseline/RL TOML configs and templates. |
| `tests/` | Regression, smoke, and behavior tests. |
| `docs/` | Methodology, runbooks, project map, roadmap, and cleaned reference docs. |
| `results/` | Generated outputs; ignored except for intentionally tracked lightweight examples, if any. |

## Core Simulation

| File | Purpose |
|---|---|
| `convoy_sim/entities.py` | Ship, torpedo, and domain data structures. |
| `convoy_sim/simulation.py` | Trial execution, event resolution, and hit/miss semantics. |
| `convoy_sim/dynamics.py` | Ship and U-boat kinematics, including zig-zag behavior. |
| `convoy_sim/geometry.py` | Geometry helpers for bearings, distances, intersections, and frame transforms. |
| `convoy_sim/fire_control.py` | Lightweight firing-solution helpers. |
| `convoy_sim/realism.py` | Realism presets, observation noise, and protocol helpers. |
| `convoy_sim/workflows.py` | Higher-level evaluation workflows used by scripts and notebooks. |

## Layouts, Feasibility, And Scoring

| File | Purpose |
|---|---|
| `convoy_sim/layouts.py` | Static convoy layout construction. |
| `convoy_sim/layout_roles.py` | Role/class metadata for heterogeneous convoy positions. |
| `convoy_sim/feasibility.py` | Layout and geometry validity checks. |
| `convoy_sim/objectives.py` | Defender/attacker scoring objectives. |
| `convoy_sim/risk.py` | Detection and risk utilities. |
| `convoy_sim/ship_catalog.py` | Ship classes, value, and physical metadata. |

## Attack Profiles

| File | Purpose |
|---|---|
| `convoy_sim/attack_profiles.py` | Hand-authored and generated attack-profile definitions. |
| `convoy_sim/attack_proposals.py` | Attack proposal helpers. |
| `convoy_sim/profile_audit.py` | Geometry plausibility checks for attack profiles. |
| `convoy_sim/profile_outcome_audit.py` | Dynamic outcome audit against moving convoy kinematics. |
| `convoy_sim/target_zones.py` | Target-zone and spawn-region logic for synthetic profile generation. |
| `convoy_sim/profile_generation_viz.py` | Visualization helpers for generated profile geometry. |

## Optimization And Learning

| File | Purpose |
|---|---|
| `convoy_sim/defender_policy.py` | Defender policy logic. |
| `convoy_sim/defender_opt.py` | Defender optimization helpers and smoke-tested strategy utilities. |
| `convoy_sim/attacker_opt.py` | Attacker optimization helpers. |
| `convoy_sim/game.py` | Game-matrix and exploitability helpers. |
| `convoy_sim/rl_env.py` | RL environment boundary. |
| `convoy_sim/rl_wrapper.py` | RL integration wrapper. |
| `convoy_sim/rl_layout_builder.py` | Layout builder used by RL actions and future freer layout generation. |
| `convoy_sim/vae.py` | Attack-profile VAE model and encoding/decoding logic. |
| `convoy_sim/vae_diagnostics.py` | VAE diagnostics, filtering, and feature inspection helpers. |
| `docs/REINFORCEMENT_LEARNING.md` | RL layout-optimization architecture and validation design. |

## POMDP

| File | Purpose |
|---|---|
| `convoy_sim/pomdp_candidate_selector.py` | Belief-limited candidate ranking from noisy observations. |
| `convoy_sim/pomdp_fire_control.py` | POMDP v2 fire-control rebuild helpers. |
| `docs/POMDP.md` | POMDP method and evaluation notes. |

## Visualization

| File | Purpose |
|---|---|
| `convoy_sim/viz.py` | General static layout plotting helpers. |
| `convoy_sim/viz_attack.py` | Attack overlay, animation, and frame rendering helpers. |
| `experiments/plot_layout.py` | Static layout figure script. |
| `experiments/plot_attack_once.py` | Single attack-overlay figure script. |
| `experiments/render_attack_animation.py` | Dynamic attack animation/frame script. |
| `experiments/render_attack_profile_previews.py` | Profile preview rendering and visual QA script. |

## Experiment Entrypoints

See `docs/SCRIPTS.md` for the runnable script index. The main groups are:

- baseline/RL runners
- attack-profile generation and audits
- VAE candidate-pool generation/evaluation
- POMDP candidate selection
- visualization/manual QA

## Notebooks

Notebook outputs should go under:

```text
results/notebook-results/<notebook-name>/
```

Current notebook groups:

| Group | Notebooks |
|---|---|
| Profile generation and manual checks | `attack_profile_tests.ipynb`, `attack_manual_verification.ipynb`, `profile_generation_tests.ipynb`, `visuals.ipynb` |
| VAE workflows | `vae_train.ipynb`, `random_vae_train.ipynb`, `mixed_vae_train.ipynb`, `vae_candidate_pool.ipynb`, `vae_source_comparison.ipynb`, `vae_final_baseline_comparison.ipynb` |
| Candidate/POMDP evaluation | `attack_candidate_pool_eval.ipynb`, `pomdp_candidate_selector.ipynb`, `pomdp_candidate_eval.ipynb`, `pomdp_fire_control_eval.ipynb` |
| Doctrine comparison | `torpedo_firing_doctrine_comparison.ipynb` |

## Tests

Use tests to locate expected behavior before editing core modules.

| Test Area | Examples |
|---|---|
| Simulation and geometry | `tests/test_simulation.py`, `tests/test_geometry.py`, `tests/test_dynamic_hit_events.py` |
| Attack profiles and generation | `tests/test_attack_profiles.py`, `tests/test_generate_attack_profile_scaffold.py`, `tests/test_profile_outcome_audit.py` |
| Layout and feasibility | `tests/test_layouts.py`, `tests/test_feasibility_checks.py`, `tests/test_rl_layout_builder.py` |
| VAE and candidate pools | `tests/test_vae.py`, `tests/test_generate_vae_candidate_pool.py`, `tests/test_evaluate_attack_candidate_pool.py` |
| POMDP | `tests/test_pomdp_candidate_selector.py`, `tests/test_pomdp_fire_control.py` |
| Visualization | `tests/test_viz_helpers.py`, `tests/test_viz_attack_helpers.py`, `tests/test_viz_no_matplotlib_import.py` |

## Change Routing

| If changing... | Start with... |
|---|---|
| Collision, hit/miss, movement, or zig-zag behavior | `convoy_sim/simulation.py`, `convoy_sim/dynamics.py`, `tests/test_simulation.py` |
| Layout spacing, bounds, or placement rules | `convoy_sim/layouts.py`, `convoy_sim/feasibility.py`, `convoy_sim/rl_layout_builder.py` |
| RL architecture, reward, or validation design | `docs/REINFORCEMENT_LEARNING.md`, `convoy_sim/rl_env.py`, `convoy_sim/rl_wrapper.py` |
| Attack-profile generation realism | `experiments/generate_attack_profile_scaffold.py`, `convoy_sim/target_zones.py`, `convoy_sim/profile_outcome_audit.py` |
| VAE features, training data, or decoding | `convoy_sim/vae.py`, `convoy_sim/vae_diagnostics.py`, VAE notebooks |
| POMDP observation or fire-control logic | `convoy_sim/pomdp_candidate_selector.py`, `convoy_sim/pomdp_fire_control.py`, `docs/POMDP.md` |
| Run order or regeneration instructions | `docs/REPRODUCING.md` |
| New runnable script | `experiments/`, `docs/SCRIPTS.md`, and an entrypoint smoke test if practical |
