# Reinforcement Learning Layout Optimization

This document describes the reinforcement-learning design for convoy layout optimization.

Historical phase notes, stale benchmark interpretations, and old run-specific details should stay in archived copies. This file should stay focused on architecture, design intent, validation criteria, and the remaining final-scope RL work.

## Current Role

The current RL system is a useful workflow and comparison baseline, but it is not yet the final layout optimizer.

Its current role is:

1. Exercise the baseline/RL artifact pipeline.
2. Compare learned or selected layouts against static and heuristic baselines.
3. Provide a controlled starting point for expanding layout freedom.
4. Establish final validation criteria before expensive reruns.

The final RL story should not depend on old run outputs. Final claims should come from a clean rerun after feasibility boundaries, validation protocol, and the freer layout generator are in place.

## Main Files

Core RL logic:
- `convoy_sim/rl_env.py`
- `convoy_sim/rl_wrapper.py`
- `convoy_sim/rl_layout_builder.py`

Layout, scoring, and simulation dependencies:
- `convoy_sim/layouts.py`
- `convoy_sim/feasibility.py`
- `convoy_sim/objectives.py`
- `convoy_sim/workflows.py`
- `convoy_sim/simulation.py`

Experiment entrypoints:
- `experiments/run_rl_train.py`
- `experiments/audit_rl_actions.py`
- `experiments/run_baseline_suite.py`

Configs and tests:
- `configs/rl/default.toml`
- `configs/templates/rl.template.toml`
- `tests/test_rl_layout_builder.py`
- `tests/test_rl_wrapper.py`

Related docs:
- `docs/TODO.md`
- `docs/REPRODUCING.md`
- `docs/SCRIPTS.md`
- `docs/PROJECT_MAP.md`
- `docs/SIMULATION_PHYSICS.md`

## Problem Framing

The RL agent is a defender-side layout optimizer. It proposes convoy layouts that are evaluated against sampled attacker profiles through the same simulation and scoring workflow used by baseline experiments.

The learning problem is not historical reconstruction. It is a constrained optimization problem:

- preserve a plausible convoy design envelope
- expose enough layout freedom for RL to discover non-obvious defensive patterns
- compare against interpretable baselines under matched attack profiles and seeds
- report results as simulator metrics, not historical sinking-rate claims

The important design tension is freedom versus feasibility. If the action space is too small, RL only selects from a hand-authored menu. If it is too unconstrained, it can exploit unrealistic geometry. The final design should give RL more freedom while enforcing hard placement boundaries.

## Current Architecture

The existing RL path supports bounded layout selection and builder-style layout construction. This is useful for regression tests and early comparisons, but it remains more constrained than the desired final optimizer.

Current assumptions:

- layout construction is config-driven
- baseline and RL runs share artifact conventions
- evaluation uses matched profiles and seeds where configured
- layouts are scored by existing objective plumbing
- old flat-action behavior remains useful as a sanity fallback

This architecture is good enough to preserve while expanding the layout generator. The next changes should extend the control surface rather than replacing the full experiment pipeline.

## Target Architecture

The final RL layout generator should allow much freer ship placement within convoy-frame bounds.

Target behavior:

1. Start from a fixed fleet composition.
2. Let the optimizer place or adjust ship positions within bounded sea-room limits.
3. Preserve ship class counts and ship identities.
4. Enforce non-overlap and minimum spacing.
5. Reject, mask, or repair invalid layouts before expensive simulation.
6. Evaluate valid layouts against held-out attack profiles and repeated seed sets.

The initial implementation does not need fully continuous unrestricted placement. A bounded grid, slot-offset, or per-ship delta model is acceptable if it creates materially more freedom than the current builder while remaining easy to inspect.

## Action Space

Near-term action-space priorities:

- per-ship or per-slot placement within convoy-frame bounds
- bounded movement in along-convoy and across-convoy axes
- preserved fleet composition and class labels
- optional row/column structure as a soft starting scaffold, not a hard final limit
- optional class-placement choices for high-value ships and escorts

Deferred action controls:

- convoy speed selection
- zig-zag parameter selection
- route-leg changes
- detailed escort/search behavior

Those deferred controls can be valuable, but adding them before layout feasibility and validation are stable would make results harder to interpret.

## Feasibility Constraints

Feasibility should be explicit and testable.

Minimum constraint families:

- minimum center-to-center or hull-aware ship separation
- maximum convoy length and width
- preserved fleet and class counts
- no overlapping ships
- optional route/sea-room bounds
- stable seeded fleet realization for matched baseline/RL comparisons

Preferred handling order:

1. Mask invalid actions when action structure supports it.
2. Repair simple invalid proposals when repair is deterministic and explainable.
3. Reject invalid full layouts before simulation.
4. Use penalties only as a fallback, not as the primary realism mechanism.

The final report should be able to state that RL was optimized inside a defined feasible layout space, not that unrealistic proposals were merely discouraged by reward penalties.

## Reward And Promotion Criteria

The reward should reflect defender priorities, not only raw hit count.

Candidate metrics:

- expected loss
- expected hits
- unique ships hit
- value-weighted loss
- tail-risk or high-loss guardrail
- layout complexity or footprint penalty, if needed

The final promotion rule should be decided before final reruns.

Recommended promotion standard:

- primary metric: lower expected defender loss on matched held-out evaluation profiles
- secondary guardrail: no worse high-loss tail behavior than the baseline
- robustness check: improvement persists across repeated eval seed sets
- interpretability check: selected layout remains feasible and explainable from visual/geometry diagnostics

RL should not be promoted as “better” based on one favorable run, one seed set, or a metric that ignores variance.

## Validation Protocol

Final RL validation should compare:

- static baseline layout
- current builder-mode RL
- freer-layout RL
- random or evolutionary feasible-layout search, if implemented

Minimum validation protocol:

1. Use regenerated clean inputs.
2. Train and evaluate on separated attack-profile splits.
3. Include repeated eval seeds.
4. Include a harder held-out profile pack.
5. Report mean and spread, not only best run.
6. Preserve visual comparisons of selected layouts.
7. Record feasibility and layout-diversity summaries.

Useful reporting metrics:

- expected loss
- expected hits
- unique ships hit
- value loss
- high-loss tail proxy
- convoy footprint
- spacing distribution
- class-placement summary

## Experiment Workflow

Operational commands belong in `docs/REPRODUCING.md`. At the design level, the RL workflow is:

1. Define or regenerate profile splits.
2. Train/evaluate baseline and RL under matched configs.
3. Audit available RL actions or generated layouts directly.
4. Compare selected layouts on held-out profiles and repeated seeds.
5. Save compact final artifacts under the agreed final results area.
6. Document caveats before promoting final conclusions.

The script entrypoints are indexed in `docs/SCRIPTS.md`.

## Current Limitations

- The current RL path still does not give every ship enough independent placement freedom.
- Final promotion thresholds are not yet locked.
- Final repeated-seed confidence summaries are not yet regenerated.
- The random/evolutionary feasible-layout comparator is not yet implemented.
- Escort search/reaction behavior is not modeled as an active tactical agent.
- RL currently optimizes defender layout under sampled attacker profiles; it is not a learned attacker policy.

## Near-Term Implementation Focus

The active RL workstream is tracked in `docs/TODO.md`.

Near-term order:

1. Define feasibility boundaries for freer layouts.
2. Add action masking, rejection, or deterministic repair.
3. Build the freer layout generator.
4. Add a simple random/evolutionary feasible-layout baseline.
5. Add layout novelty and diversity reporting.
6. Rerun RL under the final validation protocol.

This keeps the next RL work focused on the actual bottleneck: a richer but still defensible layout design space.
