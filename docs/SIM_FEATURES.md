# Simulation Features Reference

Last updated: 2026-04-07

This is the central technical reference for the convoy simulation stack.
It is intended to be the single place to answer:
- What does the simulator currently model?
- Which parameters affect which outputs?
- What is implemented vs planned?
- Where do baseline/RL workflows plug into simulation?
- What tests validate each major behavior?

---

## 1) Scope And Modeling Philosophy

### 1.1 Core Domain

The project models WWII-style convoy defense against torpedo attacks on a 2D Euclidean plane.

- Coordinates: meters
- Time: seconds
- Heading: radians (unless explicitly converted to degrees for reporting)
- Kinematics: deterministic torpedoes with optional gyro-deflection two-segment tracks + convoy-level route/zig-zag motion abstractions

### 1.2 Design Priorities

1. Reproducible experiment workflows (config-first baseline/RL entrypoints)
2. Explainable geometry and risk metrics
3. Incremental realism layers that preserve canonical output artifacts
4. Compatibility with existing profile libraries and diagnostics

### 1.3 Current Protocol Boundary

- Active track: `Protocol V2-Realism`
- Frozen reference track: `Protocol V1`
- Out of phase today:
  - full RL environment redesign
  - generative-model training pipelines

---

## 2) Architecture Map

## 2.1 Core Modules

- Geometry/math: `convoy_sim/geometry.py`
- Entities: `convoy_sim/entities.py`
- Layout generation: `convoy_sim/layouts.py`, `convoy_sim/ship_catalog.py`
- Convoy motion/dynamics: `convoy_sim/dynamics.py`
- Feasibility and environment risk: `convoy_sim/feasibility.py`
- Attack proposal generation: `convoy_sim/attack_proposals.py`
- Attack profile schema and sampling: `convoy_sim/attack_profiles.py`
- Fire-control baseline helpers: `convoy_sim/fire_control.py`
- Realism overlays: `convoy_sim/realism.py`
- Simulation kernels and Monte Carlo: `convoy_sim/simulation.py`
- Objectives and scoring: `convoy_sim/objectives.py`, `convoy_sim/risk.py`
- Game/RL wrappers: `convoy_sim/game.py`, `convoy_sim/rl_wrapper.py`, `convoy_sim/rl_env.py`
- Workflow utilities and artifact writers: `convoy_sim/workflows.py`
- Visualization: `convoy_sim/viz.py`, `convoy_sim/viz_attack.py`

## 2.2 Canonical Experiment Entrypoints

- Baseline workflow: `experiments/run_baseline_suite.py`
- RL workflow: `experiments/run_rl_train.py`
- Config generator: `experiments/generate_run_config.py`

---

## 3) State, Coordinates, And Frames

## 3.1 Coordinate System

- 2D Cartesian plane in meters
- Positive x/y directions are fixed global axes
- Ship/torpedo headings are radians in global frame

## 3.2 Convoy Pose And Frames

Convoy motion is represented with:
- global convoy pose (origin + heading)
- per-ship offsets in convoy frame (`ConvoyFormation.offsets_convoy_frame`)

Two motion semantics are available:
- `rigid`: ships preserve fixed formation offsets while convoy translates/rotates
- `independent` (default in many dynamic checks): each ship is integrated independently using convoy heading/speed schedule

## 3.3 Time Integration

- Dynamic stepping uses discrete `dt` integration
- `validate_dt(dt)` enforces `dt > 0`
- Smaller `dt` increases temporal precision at runtime cost

---

## 4) Entities And Physical Abstractions

## 4.1 Ship Model (`Ship`)

Each ship includes:
- `id`, `position`, `speed`, `heading_rad`
- hull geometry: `length`, `beam`
- class: `freighter|tanker|escort|decoy`
- `value_weight` for value-based scoring
- optional `hit_radius` override

Collision approximation:
- default effective hit radius: `sqrt((length/2)^2 + (beam/2)^2)`
- if `hit_radius` is set, that override is used

## 4.2 Torpedo Model (`Torpedo`)

Each torpedo includes:
- `id`, `launch_position`, `speed`, `heading_rad`, `max_run_time`
- `launch_delay`
- `is_dud`
- optional `launch_heading_rad`
- optional `gyro_turn_distance_m`

Behavior:
- waits at `launch_position` until `launch_delay`
- `max_run_time` is the torpedo's post-launch endurance, not a global simulation timestamp cutoff
- leaves the tube on `launch_heading_rad` when provided, otherwise on `heading_rad`
- after `gyro_turn_distance_m`, snaps to final `heading_rad`
- remains fixed-speed/fixed-course after the gyro deflection (no active guidance model)

## 4.3 Ship Class Catalog

Default class-level dimensions/values (used by `make_ship` unless overridden):
- freighter: `140 x 20 m`, value `1.0`
- tanker: `180 x 28 m`, value `1.5`
- escort: `90 x 12 m`, value `0.5`
- decoy: `120 x 18 m`, value `0.2`

Note: layout profiles can override dimensions at generation time.

---

## 5) Layout Generation

Primary layout builders:
- `make_rectangular_convoy`
- `make_staggered_convoy`
- `make_hexagonal_convoy`

Supported controls:
- row/column counts
- along/across spacing
- convoy speed and heading
- per-cell class map (`ship_class_map`)
- per-cell overrides (`ship_overrides_map`)
- optional Gaussian positional jitter

Scenario profile registry:
- `scenarios/convoy_profiles.py`
- currently includes `small_demo`, `convoy_layout_1`, `convoy_layout_2`

---

## 6) Convoy Dynamics

## 6.1 Route and Zig-Zag

`ConvoyKinematics` combines:
- `RoutePlan` (piecewise-constant heading, optional speed by leg)
- `ZigZagPlan` (sine or triangle heading perturbation)

## 6.2 Command Latency

`command_latency_s` shifts effective control time:
- heading/speed queries evaluate at `max(0, t - command_latency_s)`

Interpretation:
- simulates delayed order execution for convoy maneuvers

## 6.3 Hit Slowdown (Dynamic Visual/Step Context)

Optional `HitSlowdownSpec` can reduce post-hit ship speed exponentially during dynamic stepping/visualization contexts where hit state is advanced over time.

---

## 7) Attack Proposal And Feasibility Layer

## 7.1 AttackProposal

Proposal fields:
- U-boat position
- target point
- bearing
- approach mode (`abeam|bow_on|stern_chase`)
- salvo size
- launch time
- metadata

## 7.2 Feasibility Constraints

`AttackConstraints` supports:
- min/max range checks
- allowed approach modes
- escort exclusion/risk zones
- optional soft-risk threshold (`enable_soft_risk`, `max_allowed_risk`)

## 7.3 Environment In Feasibility

`Environment` fields:
- `time_of_day`
- `visibility_m`
- `sea_state`
- `detection_risk_scale`

Detection risk score combines:
- base scale
- visibility factor
- day/night factor
- sea-state factor
- escort proximity decay terms

Important boundary:
- environment currently affects feasibility/risk/observation context
- environment does **not** directly alter torpedo-vs-ship collision geometry in current physics kernel

---

## 8) Attack Profiles And Torpedo Generation

## 8.1 AttackProfile Library

`DEFAULT_ATTACK_PROFILE_LIBRARY` currently contains 30 scaffolded profiles (`P01`..`P30`) with weighted sampling support.

For faster library expansion, use:
- `python -m experiments.generate_attack_profile_scaffold --start-index 31 --count 30 --output results/diag/generated_attack_profiles.json`
- default output is a machine-readable JSON payload; `--format python` emits paste-ready `AttackProfile(...)` entries instead
- generated profiles always keep the modern explicit schema
- generated profiles sample `u_boat_initial_speed_mps` from `1.0` to `2.0` m/s in `0.1` steps
- generated profiles are filtered through the existing geometry plausibility audit against the selected convoy profile before being emitted

Profile supports:
- spread mode: `fan` or `parallel`
- torpedo count, speed, run time
- timing controls: launch delay + salvo interval
- U-boat motion controls
- partial-observation noise controls
- explicit fan spread doctrine controls

`fire_control.py` provides a deterministic attacker-side baseline that can turn noisy observation into a resolved `AttackProfile`.

## 8.2 V2 U-Boat Motion Controls

Fields include:
- `u_boat_mode = moving|static`
- initial position / heading / speed
- launch reference time
- motion legs (duration/heading/speed tuples)
- optional turn-rate / accel bounds

Canonical V2 baseline/RL runs currently stamp `u_boat_mode_default=moving` in their manifests. If a profile omits explicit motion overrides, it still uses:
- `u_boat_mode = "moving"`
- `u_boat_initial_speed_mps = 2.0`

## 8.3 Bow-Launch Realism (Current)

Current launch geometry behavior:
- launch position is tied to sub heading at each torpedo’s launch time
- launch origin defaults to bow point (`launch_from="bow"`)
- centerline attack intent must remain within `max_bow_offset_deg` of the submarine heading
- per-torpedo launch time updates launch position and bow heading when the submarine maneuvers between shots
- fan spread is now expressed through torpedo gyro deflection, not by rotating the submarine between shots

Additional launch geometry fields:
- `sub_length_m`, `sub_beam_m`
- `launch_from = bow|center`
- `max_bow_offset_deg`
- `gyro_straight_run_m`

## 8.4 U-Boat Firing Stability And Gyro Logic

The simulator now separates three distinct concepts:
- U-boat pose and bow heading at launch time
- intended attack centerline in world coordinates
- each torpedo's final post-gyro course

Source-of-truth order:
1. `UBoatMotionPlan.state_at(t)` determines the submarine center position and bow heading at each torpedo launch time.
2. `launch_from` determines whether torpedoes originate from the submarine center or bow point.
3. `base_bearing_rad` / `bearing_rad` defines the intended attack centerline.
4. fan-mode doctrine determines how per-torpedo final heading offsets are assigned around that centerline.
5. Each torpedo leaves on the bow heading, travels straight for `gyro_straight_run_m`, then turns to its own final heading.

Firing-stability doctrine:
- `require_stable_u_boat_during_salvo = true` by default
- the submarine is rejected if heading drifts materially across the salvo window
- the submarine is rejected if finite-difference turn rate at launch exceeds `max_u_boat_turn_rate_at_fire_rad_s`
- this models the doctrine assumption that torpedoes are fired while the boat is steady, not while the boat is actively swinging through a turn

Implication for profile authoring:
- `u_boat_initial_heading_rad` and any motion plan should describe how the submarine is pointed when firing starts
- `spread_rad` no longer requires rotating the submarine or consuming bow-tube arc across the full fan
- `spread_rad` is specifically the total fan width for `uniform_divergent`, not the universal spread mechanism
- default-library profiles keep their numeric spread values; those values now map to post-launch gyro deflection under `uniform_divergent`

## 8.5 Fan Spread Doctrines

Fan-mode attack profiles now support three explicit doctrines:

- `longitudinal`
  - all torpedoes share the same final heading
  - spacing comes only from launch timing and launch-position changes while the submarine holds course during the salvo
  - supported for completeness and backward compatibility, but rare/nonstandard for convoy attacks
- `uniform_divergent`
  - standard convoy doctrine
  - per-torpedo final headings are evenly spaced across total fan width `spread_rad`
  - submarine holds course during the salvo; only gyro-selected final headings differ
- `explicit_divergent`
  - advanced/manual convoy doctrine
  - uses `per_torpedo_heading_offsets_rad`
  - each torpedo gets an explicitly authored final offset from the centerline
  - submarine holds course during the salvo; only gyro-selected final headings differ

Important separation of concerns:
- `spread_doctrine` controls only the torpedoes' final post-gyro heading logic
- it does not tell the submarine to turn during the salvo
- non-steady firing windows are rejected by default for all three doctrines unless `require_stable_u_boat_during_salvo` is deliberately disabled for a diagnostic/special-case scenario

Backward compatibility:
- legacy fan profiles with `spread_rad > 0` behave as `uniform_divergent`
- legacy fan profiles with `spread_rad == 0` behave as `longitudinal`
- existing serialized profiles remain valid without adding new fields

Focused visual comparison:
- `notebooks/torpedo_firing_doctrine_comparison.ipynb` renders end-of-firing-cycle doctrine snapshots for static and moving U-boat cases using the same attack-profile semantics described above
- its `show_case(..., save=True)` and `show_summary_grid(save=True)` helpers save PNGs by default to `notebooks/results/torpedo_firing_doctrine_comparison/`
- its current longitudinal views are visualization aids for comparing spread geometry; the core simulator doctrine remains steady-boat timing spread, not a pivot or heading-sweep tactic

## 8.6 Fire Control Lite Baseline

`fire_control.py` implements a deterministic attacker-side firing solution baseline.

Role in the stack:
- `build_attacker_observation(...)` produces the attacker-facing estimate of convoy state
- `fire_control.py` consumes that estimate and turns it into a firing solution
- it does not replace the observation layer and it does not generate observation noise itself

Inputs:
- U-boat position
- U-boat bow heading
- noisy attacker observation:
  - estimated convoy bearing
  - estimated range
  - estimated convoy course
  - estimated convoy speed
  - observation-quality sigmas

Outputs:
- centerline bearing
- spread width
- salvo size
- G7a speed setting (`fast|medium|long_range`)
- torpedo speed and max run time
- estimated target point and metadata

Behavior:
- chooses G7a speed setting deterministically from estimated range
- computes a coarse intercept lead from estimated convoy lateral motion
- widens spread with observation uncertainty and range
- clips centerline to the configured bow-arc allowance relative to the submarine heading
- can emit a standard `AttackProfile` via `build_attack_profile_from_fire_control(...)`

Boundary:
- this is not a full TDC / fire-control computer simulation
- it is a coarse attacker-perspective baseline for:
  - realistic scripted attacks
  - future GenAI attack-distribution conditioning
  - eventual attacker-side baseline comparison

Conceptual split from partial observability:
- partial observability = sensing / information model
- `fire_control_lite` = firing-decision / solution model

---

## 9) Realism Overlays (`realism.py`)

## 9.1 Attacker Partial Observability

`build_attacker_observation(...)` provides attacker-facing noisy estimates:
- convoy bearing/range
- convoy heading/speed
- contact count
- class-confidence counts
- environment snapshot
- observation quality sigmas

This context can be passed into proposal metadata and consumed by bearing-resolution logic.

Role in the stack:
- starts from true convoy state
- applies uncertainty to produce what the attacker thinks it sees
- does not choose aimpoint, spread, gyro offsets, or torpedo speed setting

Relationship to `fire_control_lite`:
- partial observability answers: "what does the attacker know?"
- `fire_control_lite` answers: "given that estimate, how does the attacker shoot?"

## 9.2 Ship Movement Realism Overlay

`ShipMovementRealismConfig` supports bounded overlays:
- position jitter
- heading jitter
- deviation cap
- optional slot swaps with max fraction
- class-dependent scaling (freighter/tanker/escort/decoy)

Applied in simulation and strategy sampling paths after base layout generation.

## 9.3 U-Boat Motion Plan

`UBoatMotionPlan` is deterministic and supports:
- static or moving mode
- leg-based state evolution
- optional bounded transitions (turn/accel limits)
- `state_at(t)` and `position_at(t)`

It does not model hydrodynamic turn circles or tube-training mechanics.
Instead, it provides a deterministic heading/speed/position state that the attack-profile builder samples at each launch time.

---

## 10) Noise And Weapon Imperfection

`NoiseModel` currently includes:
- `sigma_heading_rad`
- `sigma_launch_delay`
- `sigma_speed_mps`
- `p_dud`

Applied via `apply_noise_to_torpedoes(...)`:
- heading perturbation
- speed perturbation (floored > 0)
- launch delay perturbation (floored >= 0)
- dud Bernoulli sampling

Not currently modeled:
- depth-fuze proxy
- wake-following/guided behavior
- continuous curved torpedo steering beyond a single gyro deflection

---

## 11) Simulation Kernels And Metrics

## 11.1 Static One-Off Simulation

`simulate_attack_once(...)`:
- deterministic hit count
- supports `max_hits_per_torpedo`

`simulate_attack_once_scored(...)`:
- adds value-destroyed and class breakdown metrics

## 11.2 Dynamic Simulation

Dynamic stepping supports:
- convoy motion over time
- time-aware torpedo launch windows
- per-step hit-state evolution (`DynamicHitState`, `HitEvent`)

## 11.3 Monte Carlo APIs

- `run_monte_carlo_attack`
- `run_monte_carlo_attack_scored`
- `run_monte_carlo_attack_dynamic`

Outputs include:
- expected hits
- hit probability (`P(hit >= 1)`)
- variance
- optional VaR/CVaR
- value-destroyed aggregates (scored path)

## 11.4 Risk Helpers

`risk.py` provides empirical:
- `VaR_alpha`
- `CVaR_alpha`

---

## 12) Objectives And Utility

`ObjectiveSpec` controls scalar scoring:
- value-weight and hit-count weight
- unique-ships-hit and repeat-hit weights
- class-specific value weights
- escort loss discount
- optional named objective presets with per-run overrides
- perspective mode (`defender_minimize` / `attacker_maximize`)
- optional risk alpha in aggregate objective

Default no-objective behavior in many wrappers:
- uses total value destroyed as baseline scalar

Named presets currently supported:
- `balanced_default`
  - canonical default for mixed-convoy RL/baseline tests
  - `w_total_value = 1.0`
  - `w_unique_ships_hit = 1.0`
  - `w_repeat_hits = 0.2`
- `protect_hulls`
  - stronger penalty on distributed damage and repeated hits
- `protect_value`
  - stronger emphasis on class-weighted value protection
- `accept_concentration`
  - intentionally more tolerant of multiple hits on the same ship

Config semantics:
- set `preset = "..."` under `[objective]` to choose a doctrine preset
- any explicit weight in `[objective]` overrides the preset value
- treat presets as starting points, not hard limits

---

## 13) Baseline Workflow Semantics

Entrypoint:
- `python -m experiments.run_baseline_suite --config ...`

Behavior:
1. Load split, simulation, and baseline layout config
2. Evaluate static baseline on eval split
3. Run bounded grid search on train split (heuristic baseline)
4. Re-evaluate best candidate on eval split
5. Emit canonical artifacts:
   - `config_resolved.yaml`
   - `metrics_summary.json`
   - `per_profile_metrics.csv`
   - `run_manifest.json`

Realism integration in baseline:
- noise model from config
- environment from config
- ship movement realism overlay
- manifest realism stamp (`u_boat_mode_default=moving` + config snapshot)

---

## 14) RL Workflow Semantics (Current)

Entrypoint:
- `python -m experiments.run_rl_train --config ...`

Current RL implementation is lightweight tabular action selection:
- discrete action list from `[[rl.actions]]`
- epsilon-greedy one-step training loop
- incremental Q-value updates
- selected best action evaluated on eval split

Not a full policy-gradient/deep RL environment today.

Realism integration in RL path:
- sampled attack profiles use V2 launch/motion realism
- noise/environment passed in evaluation
- ship movement realism applied in layout sampling path
- canonical checkpoint and manifest emitted

---

## 15) Visualization Features

## 15.1 Static Layout Views

`viz.py` and workflow plot writers support plan-view layout figures with class/value coloring and fixed style.

## 15.2 Dynamic Attack Views

`viz_attack.py` supports:
- frame rendering
- MP4 export (ffmpeg required)
- bent torpedo trails when gyro deflection is active
- trails, hit clipping, marker styling
- dynamic hit-state overlays
- optional U-boat rendering

Current U-boat rendering options:
- fixed `u_boat_position`
- time-aware `u_boat_position_fn(t)` for per-frame movement

---

## 16) Diagnostics And Audits

- Attack profile geometry audit: `convoy_sim/profile_audit.py`, `experiments/audit_attack_profiles.py`
- Preview frame renderer + hit/audit CSVs: `experiments/render_attack_profile_previews.py`
- Manual verification notebooks under `notebooks/`

Key manual verification notebook:
- `notebooks/attack_manual_verification.ipynb`
- `notebooks/torpedo_firing_doctrine_comparison.ipynb` for zoomed doctrine-comparison views and saved PNG outputs under `notebooks/results/torpedo_firing_doctrine_comparison/`

---

## 17) Configuration Surface (High-Value Fields)

Primary config families:
- `run`: naming/output roots
- `simulation`: time horizon, trials, hit cap
- `simulation.noise`
- `simulation.environment`
- `simulation.ship_movement_realism`
- `splits`: profile IDs and seed sets
- workflow-specific blocks (`baseline`, `training`, `rl.actions`, `plot`)

Reference defaults:
- `configs/baseline/default.toml`
- `configs/rl/default.toml`

---

## 18) Reproducibility And Artifact Schema

Canonical workflows are config-first and emit stable artifact files.

Run manifest includes:
- workflow name
- git SHA
- profile splits
- seed sets
- trial counts and horizon
- realism stamp (noise + environment + movement realism enabled)
- layout plot references

This schema is intended to remain stable across realism upgrades.

---

## 19) Validation And Test Coverage

High-value test groups include:
- geometry/entities/layout basics
- dynamics and formation motion
- feasibility checks and detection risk
- simulation semantics (static/dynamic/hits)
- visualization helpers (including no-matplotlib import guards)
- canonical workflow smoke tests
- V2 realism checks (`tests/test_realism_v2.py`)

Recommended routine before major merges:
1. `pytest -q tests/test_realism_v2.py tests/test_attack_profiles.py`
2. `pytest -q tests/test_canonical_entrypoints.py`
3. canonical baseline + RL commands on default configs
4. manual visual verification notebook pass for realism-sensitive changes

---

## 20) What Currently Affects What (Quick Matrix)

- `time_of_day / visibility / sea_state / detection_risk_scale`:
  - affects feasibility/risk + attacker observation context
  - does not directly change hit collision physics yet

- `simulation.noise`:
  - perturbs torpedo heading/speed/delay and dud outcomes

- `ship_movement_realism`:
  - perturbs ship initial placements/headings before trial simulation

- U-boat motion and launch geometry fields:
  - set launch origin/time/heading (including bow-point and tube arc limits)

- `max_hits_per_torpedo`:
  - controls whether one torpedo can register multiple hits

- `objective weights`:
  - change scalar optimization target, not raw collision outcomes

---

## 21) Known Gaps / Current Limits

- No depth-fuze proxy yet
- Torpedoes are straight-run after launch (no maneuver guidance)
- Environment does not yet modulate collision geometry directly
- Current RL training is simple tabular selection, not full modern RL stack
- Some legacy/supporting modules exist outside canonical path and may be retained for backward compatibility

---

## 22) Planned Extensions (From Active Backlog)

Representative planned realism/analysis areas:
- hard realism envelopes and additional doctrine constraints
- richer escort/search/reaction behavior
- detection stack decomposition (radar/sonar/visual/HF-DF)
- straggler and discipline dynamics
- value-aware and constrained RL redesign
- generative attack-profile model integration after realism baseline stabilizes

See `docs/TODO.md` and `docs/RL_PLAN.md` for prioritization and sequencing.

---

## 23) Related Docs

- Protocol boundary: `docs/PROTOCOL_V2_REALISM.md`
- Script/file index: `docs/SCRIPTS.md`
- Results records: `docs/RESULTS_LOG.md`
- Optimization process logs: `docs/OPTIMIZATION_LOG.md`
- Roadmap/backlog: `docs/TODO.md`
- RL design planning: `docs/RL_PLAN.md`
- Visual conventions and notes: `docs/Visuals.md`
