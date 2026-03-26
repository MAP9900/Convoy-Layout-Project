# Script And File Index

Current complete index of files in the repository, grouped by purpose.

## Canonical Workflows

- Baseline: `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- RL: `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Script Metadata Matrix

| Script | Entrypoint | Inputs | Outputs | Depends On | Runtime Estimate | Used By |
|---|---|---|---|---|---|---|
| `experiments/run_baseline_suite.py` | `python -m experiments.run_baseline_suite --config ...` | Baseline TOML config, scenario/profile refs | Baseline metrics/artifacts under `results/` | `convoy_sim.workflows`, scenarios, core sim modules | Medium | Baseline |
| `experiments/run_rl_train.py` | `python -m experiments.run_rl_train --config ...` | RL TOML config | RL training/eval artifacts and checkpoints under `results/` | `convoy_sim.rl_env`, `convoy_sim.rl_wrapper`, workflows | Long | RL |
| `experiments/audit_attack_profiles.py` | `python -m experiments.audit_attack_profiles ...` | Convoy profile id(s), audit parameters | Audit CSV/JSON diagnostics | `convoy_sim.profile_audit`, `convoy_sim.attack_profiles` | Short-Medium | Diagnostic |
| `experiments/render_attack_profile_previews.py` | `python -m experiments.render_attack_profile_previews ...` | Convoy profile id(s), run mode, rendering flags | Preview frames + geometry/hit diagnostics in `results/diag` and frame dirs | `convoy_sim.viz_attack`, profile audit helpers | Medium-Long | Manual verification |
| `experiments/render_attack_animation.py` | `python -m experiments.render_attack_animation ...` | Scenario/profile and render args | Animation frames/video outputs in `results/` | `convoy_sim.viz_attack`, simulation modules | Medium | Manual verification |
| `experiments/plot_attack_once.py` | `python -m experiments.plot_attack_once ...` | Single scenario/profile invocation | One static plot + debug JSON | `convoy_sim.viz_attack`, geometry/sim helpers | Short | Diagnostic |
| `experiments/plot_layout.py` | `python -m experiments.plot_layout ...` | Layout/scenario args | Static layout figures | `convoy_sim.layouts`, `convoy_sim.viz` | Short | Diagnostic |

## Root Files

- `README.md` (`canonical`)
- `PROJECT_GUIDE.md` (`canonical`)
- `NOTES.md` (`supporting`)
- `IDEAS.md` (`supporting`)
- `requirements.txt` (`canonical`)
- `requirements-dev.txt` (`canonical`)
- `requirements-ml.txt` (`canonical`)

## Configs (All)

- `configs/baseline/default.toml` (`canonical`)
- `configs/rl/default.toml` (`canonical`)

## Experiment Scripts (All)

- `experiments/run_baseline_suite.py`: Canonical baseline runner (config-first, artifact schema output). (`canonical`)
- `experiments/run_rl_train.py`: Canonical RL runner (config-first train/eval + checkpoint output). (`canonical`)
- `experiments/audit_attack_profiles.py`: Attack-profile geometry plausibility audit. (`diagnostic`)
- `experiments/render_attack_profile_previews.py`: Profile frame rendering + audit/hit CSV outputs. (`diagnostic`)
- `experiments/render_attack_animation.py`: Dynamic attack animation/frame generation demo. (`diagnostic`)
- `experiments/plot_attack_once.py`: Single static attack plot + debug JSON. (`diagnostic`)
- `experiments/plot_layout.py`: Static layout figure generator. (`diagnostic`)

## Scenarios (All)

- `scenarios/scenario_base.py` (`canonical`)
- `scenarios/scenario_a.py` (`canonical`)
- `scenarios/scenario_a1_constraints.py` (`canonical`)
- `scenarios/scenario_rl.py` (`canonical`)
- `scenarios/convoy_profiles.py` (`canonical`)

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
- `convoy_sim/feasibility.py` (`canonical`)
- `convoy_sim/game.py` (`supporting`)
- `convoy_sim/geometry.py` (`canonical`)
- `convoy_sim/layout_roles.py` (`canonical`)
- `convoy_sim/layouts.py` (`canonical`)
- `convoy_sim/noise.py` (`supporting`)
- `convoy_sim/objectives.py` (`canonical`)
- `convoy_sim/profile_audit.py` (`diagnostic`)
- `convoy_sim/risk.py` (`canonical`)
- `convoy_sim/rl_env.py` (`canonical`)
- `convoy_sim/rl_wrapper.py` (`canonical`)
- `convoy_sim/ship_catalog.py` (`canonical`)
- `convoy_sim/simulation.py` (`canonical`)
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
- `tests/test_risk.py` (`canonical`)
- `tests/test_rl_wrapper.py` (`canonical`)
- `tests/test_sampler_with_constraints.py` (`canonical`)
- `tests/test_scenarios_smoke.py` (`canonical`)
- `tests/test_serialization_roundtrip.py` (`canonical`)
- `tests/test_ship_heterogeneity_models.py` (`canonical`)
- `tests/test_simulation.py` (`canonical`)
- `tests/test_simulation_semantics.py` (`canonical`)
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

- `docs/EXECUTION_PLAN.md` (`canonical`)
- `docs/Visuals.md` (`supporting`)
- `docs/REORG_PHASE2_4_AUDIT.md` (`supporting`)
- `docs/SCRIPTS.md` (`canonical`)

## Lifecycle Tag Legend

- `canonical`: Primary paths and files to maintain and evolve.
- `supporting`: Auxiliary components that support canonical flows.
- `diagnostic`: Debug, plotting, or audit-only utilities and artifacts.
- `legacy-candidate`: Present but likely removable after replacement/confirmation.
