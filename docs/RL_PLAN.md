# RL Plan

Last updated: 2026-04-20

## Purpose

This document is the implementation plan for the RL overhaul after the V2 realism firing stack was stabilized.

It answers:
- what is wrong with the current RL setup
- what needs to change first
- what order the RL overhaul should happen in
- what is explicitly out of scope for this phase

## Current Diagnosis

Current canonical RL is too weak to be treated as a serious convoy-layout optimizer.

Main reasons:
- The learner is still a tabular one-step selector over predefined `[[rl.actions]]`.
- Even after the Phase 1 action-space fix, the flat action menu remained too coarse to be a strong optimizer.
- Training uses a very narrow threat setup.
- Reward is too coarse for the richer convoy behavior we now care about.
- The environment is still structured more like action selection than layout construction.

Implication:
- historical RL runs remain useful as workflow/pipeline benchmarks
- they are not strong evidence of RL optimization quality

## Design Goals

- Give RL meaningful control over convoy design.
- Keep outputs historically plausible through hard realism boundaries.
- Preserve canonical run artifacts and matched-seed comparability.
- Separate phases cleanly so failures are diagnosable.

## Core Principles

### 1. Freedom With Boundaries

RL should have more control, but only inside hard constraints:
- separation floors
- convoy footprint caps
- min/max rows
- min/max ships per row
- fixed fleet composition by class
- bounded speed / maneuver choices

### 2. Feasibility First

Invalid layouts should be masked before simulation where possible.

Fallback behavior:
- if masking is not possible, apply a heavy penalty
- but do not rely on penalties as the primary realism mechanism

### 3. Reward Must Reflect Defense Priorities

The objective should not only count hits.

The reward should distinguish between:
- total torpedo hits
- number of ships damaged
- value lost
- tail-risk outcomes

### 4. RL And Baseline Must Stay Comparable

If RL gains a richer control surface, the heuristic baseline must also expand in a bounded, interpretable way.

Otherwise comparisons become misleading.

## Target Control Surface

### Layout Geometry Controls

- layout family:
  - `rectangular`
  - `staggered`
  - `custom_row_counts`
- row pattern vectors, e.g. `[4,5,5,4]`
- `spacing_along`
- `spacing_across`
- bounded row offsets / skew

### Ship-Class Placement Controls

- class placement by slot/row within fixed composition rules
- protected-core vs perimeter escort patterns
- bounded high-value ship placement rules
- within-class hull heterogeneity:
  - not every freighter should share the same length/beam/value profile
  - not every tanker or escort should be identical
  - seeded hull-template sampling should define fleet realization, not RL layout control

### Kinematic / Doctrine Controls

- convoy speed within bounded realistic range
- zig-zag on/off
- bounded zig-zag parameters:
  - amplitude
  - period
  - phase
- optional route-leg change from a small bounded menu later

### Deferred / Optional Controls

- limited per-ship overrides for escorts/high-value ships only
- bounded local slot swaps
- no unconstrained free placement of every ship

## Constraint Model

Hard constraints should be explicit and configurable.

Minimum constraint families:
- minimum separation / collision safety
- maximum convoy length
- maximum convoy width
- minimum and maximum row count
- minimum and maximum ships per row
- fleet composition counts by class
- feasibility checks for station keeping / footprint realism
- escort-placement doctrine constraints where applicable
- seeded fleet-composition realization should be stable for matched baseline/RL comparisons

Constraint handling policy:
- mask invalid actions when possible
- otherwise assign a severe penalty and mark the sample invalid

## Reward Design

### Short-Term Reward

Use a simple but better reward than pure hit count:

`reward = - ship_loss_score - lambda_risk * CVaR_90_proxy - lambda_complexity * layout_complexity`

Where `ship_loss_score` should distinguish between:
- one ship hit multiple times
- many ships each hit once

Recommended direction:
- heavier penalty for number of unique ships damaged/destroyed
- lower marginal penalty for repeat hits on the same already-lost ship
- only treat value-weighted reward as canonical once heterogeneous convoy composition is active

Current implementation status:
- objective plumbing now supports:
  - `w_unique_ships_hit`
  - `w_repeat_hits`
  - class-weighted value loss
- baseline/RL workflows can now carry one shared `ObjectiveSpec`
- next step is benchmark/tuning, not more scoring-plumbing work

### Objective Presets

The reward layer now supports named doctrine presets plus explicit overrides.

Current presets:
- `balanced_default`
  - canonical default
  - favors lower value loss and fewer unique ships hit
  - tolerates some repeat hits on the same ship
- `protect_hulls`
  - stronger penalty on distributed damage across distinct ships
- `protect_value`
  - stronger protection priority for high-value classes
- `accept_concentration`
  - more tolerant of repeat hits on the same ship

Important rule:
- presets are starting points, not separate RL algorithms
- `[objective]` weights can still override preset values per experiment
- when comparing presets, rerun:
  - baseline
  - RL
  - direct RL action audit
  under the same seeds and convoy profile

### Current Preset Reference

Use these exact values when running an objective sweep.

`balanced_default`
- `w_total_value = 1.0`
- `w_total_hits = 0.0`
- `w_unique_ships_hit = 1.0`
- `w_repeat_hits = 0.2`
- `escort_loss_discount = 0.75`
- class weights:
  - `freighter = 1.0`
  - `tanker = 1.5`
  - `escort = 0.5`
  - `decoy = 0.2`

`protect_hulls`
- `w_total_value = 1.0`
- `w_total_hits = 0.0`
- `w_unique_ships_hit = 1.5`
- `w_repeat_hits = 0.5`
- `escort_loss_discount = 0.75`
- class weights:
  - `freighter = 1.0`
  - `tanker = 1.4`
  - `escort = 0.6`
  - `decoy = 0.2`

`protect_value`
- `w_total_value = 1.5`
- `w_total_hits = 0.0`
- `w_unique_ships_hit = 0.8`
- `w_repeat_hits = 0.2`
- `escort_loss_discount = 0.75`
- class weights:
  - `freighter = 1.0`
  - `tanker = 2.0`
  - `escort = 0.4`
  - `decoy = 0.2`

`accept_concentration`
- `w_total_value = 1.0`
- `w_total_hits = 0.0`
- `w_unique_ships_hit = 1.2`
- `w_repeat_hits = 0.1`
- `escort_loss_discount = 0.75`
- class weights:
  - `freighter = 1.0`
  - `tanker = 1.5`
  - `escort = 0.5`
  - `decoy = 0.2`

## Current RL Experiment Knobs

Use this as the running checklist when changing RL experiments.

### Benchmark Inputs

- convoy profile:
  - e.g. `convoy_layout_1`, `convoy_layout_mixed_1`
- fleet realization:
  - `fleet_profile`
  - `fleet_seed`
- profile split:
  - `train_profiles`
  - `eval_profiles`
- Monte Carlo seeds:
  - `train_seeds`
  - `eval_seeds`
- simulation budget:
  - `n_trials_per_seed`
  - `t_max`

### Objective Knobs

- objective preset:
  - `balanced_default`
  - `protect_hulls`
  - `protect_value`
  - `accept_concentration`
- explicit objective overrides:
  - `w_total_value`
  - `w_total_hits`
  - `w_unique_ships_hit`
  - `w_repeat_hits`
  - `escort_loss_discount`
  - `class_value_weights`

### RL Training Knobs

- `episodes`
- `epsilon`
- `epsilon_decay`
- `epsilon_min`
- `alpha`
- `seed`

### Runtime / Profiling Knobs

- timing summary in run artifacts:
  - `training_seconds`
  - `train_ranking_seconds`
  - `action_selection_seconds`
  - `eval_seconds`
  - `total_seconds`
- baseline timing summary:
  - `static_eval_seconds`
  - `heuristic_search_seconds`
  - `heuristic_eval_seconds`
  - `total_seconds`
- audit timing summary:
  - `evaluation_seconds`
  - `per_action_seconds`
  - `total_seconds`
- runtime budget knobs:
  - `baseline_search_n_trials_per_seed`
  - `rl_ranking_n_trials_per_seed`
  - `audit_screen_n_trials_per_seed`
  - `audit_top_k_full_eval`

### RL Builder Knobs

- `layout_families`
- `row_patterns`
- `row_offset_policies`
- `class_placement_policies`
- `spacing_along_options`
- `spacing_across_options`
- `family_complexity`
- `row_pattern_complexity`
- `row_offset_complexity`
- `class_placement_complexity`
- `spacing_along_complexity`
- `spacing_across_complexity`
- base convoy geometry:
  - `base_n_rows`
  - `base_n_cols`
  - `speed`
  - `heading_rad`
  - `length`
  - `beam`
  - `origin`

### RL Selection Knobs

- `risk_cvar_weight`
- `complexity_tiebreak_tolerance`

### Current Practical Testing Rule

When testing a change, change one family of knobs at a time:
- objective only
- builder space only
- training hyperparameters only
- attack-profile library only

Otherwise the result becomes hard to interpret.

For runtime work specifically:
- add instrumentation first
- rerun once
- optimize the biggest measured bucket, not the guessed one

Current canonical runtime behavior:
- final benchmark eval still uses full `simulation.n_trials_per_seed`
- RL train-time ranking can use a cheaper budget via `runtime.rl_ranking_n_trials_per_seed`
- baseline heuristic train-search can use a cheaper budget via `runtime.baseline_search_n_trials_per_seed`
- RL action audit now defaults to a staged funnel:
  - screen all actions at `runtime.audit_screen_n_trials_per_seed`
  - promote top-K train/eval candidates to full-budget reevaluation via `runtime.audit_top_k_full_eval`

To restore the old full-fidelity audit behavior:
- set `runtime.audit_screen_n_trials_per_seed = simulation.n_trials_per_seed`
- set `runtime.audit_top_k_full_eval` to the full candidate-action count
  - for the current canonical builder space, that is `144`

### Longer-Term Reward Additions

- value-weighted loss by ship class
- escort-loss penalties if desired
- exposure / detection-risk penalties if zig-zag and doctrine controls are added

## Environment / State Design

The current one-step wrapper is not enough for the final target.

Needed next:
- compact observation vector for:
  - current layout metrics
  - row pattern summary
  - footprint summary
  - class composition summary
  - threat/attack-profile context
- multi-step layout construction rather than only selecting one full predefined action

Suggested end state:
- RL chooses convoy design over several bounded decisions
- environment validates each step
- final completed layout is then evaluated against sampled attacks

## Threat Model For Training

RL should not train against one narrow threat family only.

Training should cover:
- multiple attack profiles from the library
- multiple seeds
- multiple attacker geometries / approach families where supported

Near-term requirement:
- preserve the sampled attack-profile training path
- avoid collapsing training to one synthetic threat label only

## Phase Plan

## Phase 1: Immediate Action-Space Fix

Goal:
- make the current canonical RL config non-degenerate without redesigning the whole environment yet

Scope:
- create genuinely different `[[rl.actions]]` in `configs/rl/default.toml`
- ensure layout family and geometry actually differ
- add a small bounded but meaningful action menu

Minimum action diversity:
- rectangular compact
- rectangular standard
- rectangular loose
- staggered compact
- staggered standard
- staggered loose

Optional immediate additions:
- one or two bounded row-pattern variants if implementation cost is low

Success criteria:
- RL actions no longer collapse to the same geometry
- selected action materially changes convoy layout
- rerun baseline vs RL and record a new benchmark

Recommended diagnostic immediately after Phase 1:
- run `python -m experiments.audit_rl_actions --config configs/rl/default.toml`
- compare all configured actions directly on train/eval splits
- use that result to decide whether the next bottleneck is:
  - weak action menu
  - weak reward/training setup
  - or both

Phase 1 status:
- completed
- action menu is now geometrically meaningful
- direct action audit showed the learner/selector mismatch more clearly than action degeneracy

## Phase 2: Constrained Layout-Builder Environment

Goal:
- move from “select one complete layout” to “construct a layout under constraints”

Scope:
- multi-step episode design
- bounded layout parameter choices
- action masking / invalid-action handling
- compact observation/state vector

Candidate decision sequence:
1. choose layout family
2. choose row pattern
3. choose spacing controls
4. choose stagger/skew controls
5. choose class placement policy
6. choose speed / zig-zag doctrine if included in this phase

Implemented builder slices:
- minimal slice:
  1. layout family
  2. along-spacing bucket
  3. across-spacing bucket
- bounded expansion slice:
  1. layout family
  2. row pattern
  3. row offset policy
  4. class placement policy
  5. along-spacing bucket
  6. across-spacing bucket
- canonical RL config now enables this builder path
- legacy flat `[[rl.actions]]` path remains supported as a fallback for regression checks

Next expansion inside Phase 2:
- add hard constraint metadata and masking
- add footprint and separation guardrails as explicit builder feasibility checks
- consider a second bounded row-offset family beyond `centered_alt`
- defer convoy speed / zig-zag controls until geometry and placement controls are stable

Phase 3A prerequisite now implemented:
- config-friendly seeded fleet profiles
- within-class hull heterogeneity support
- mixed-class convoy profile available for value-focused benchmarks

Success criteria:
- environment supports meaningful sequential design
- invalid layouts are mostly masked rather than merely punished

## Phase 3: Reward Redesign

Goal:
- make the reward reflect convoy defense priorities rather than only average hit count

Scope:
- unique ships hit vs total hits
- value-lost weighting
- tail-risk penalty (`CVaR_90` or proxy)
- optional complexity penalty

Success criteria:
- reward explains why one layout is better in tactically meaningful terms
- logs expose the contributing metrics clearly

## Phase 4: Learner Upgrade

Goal:
- replace the tabular selector with a learner that can exploit the richer environment

Preferred path:
- PPO

Requirements:
- preserve artifact schema
- keep tabular mode as a regression sanity fallback if possible

Success criteria:
- learner can improve over repeated training
- policy is stable enough to compare across seed sets

## Phase 5: Evaluation Hardening

Goal:
- make benchmark conclusions credible

Scope:
- repeated eval seed sets
- confidence intervals or repeated summaries
- stronger baseline comparison
- run-to-run comparator support

Acceptance gates:
- Gate 1: RL >= static baseline on matched eval seeds
- Gate 2: RL matches or beats heuristic baseline on matched seeds
- Gate 3: stable performance across at least 3 eval seed sets
- Gate 4: no guardrail regression (`CVaR_90` threshold respected)

## Non-Goals For This RL Phase

- Exact historical TDC / fire-control simulation
- Unconstrained per-ship free placement
- Realism-breaking convoy shapes
- Attacker-side GenAI fire-control work inside the defender RL overhaul
- High-fidelity submarine-turn kinematics work

## Open Decisions

- Should RL choose row counts directly, or choose from bounded row templates first?
- Should zig-zag controls be included in Phase 2 or deferred until geometry stabilizes?
- How strongly should reward penalize unique ships hit relative to total hits?
- What promotion threshold should replace the current informal benchmark standard?
- Should the baseline expand in lockstep with each new RL control, or lag by one phase?

## Recommended Immediate Next Step

Move to the next Phase 2 expansion step, not PPO yet.

Reason:
- the builder now covers family, row pattern, row offset, class placement, and spacing
- the next bottleneck is explicit feasibility handling and broader threat diversity
- reward redesign has already been wired and swept across presets against the builder path

## Logging Contract

For each RL overhaul test:
- record config path and exact command
- record run directory
- record seed/split sets
- record selected policy/action
- record key metrics and risk metrics
- summarize caveats in:
  - `docs/RESULTS_LOG.md`
  - `docs/OPTIMIZATION_LOG.md`
