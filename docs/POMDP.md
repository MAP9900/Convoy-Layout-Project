# POMDP Attacker Modeling

This document describes the partial-observation attacker layer used to evaluate convoy layouts against belief-limited U-boat attack selection and fire control.

Run-specific result tables, dated notebook outputs, and old benchmark interpretations should stay in archived copies or final result summaries. This file should stay focused on architecture, information boundaries, evaluation design, and interpretation rules.

## Current Role

The POMDP layer is an attacker-side model. It does not optimize defender layouts directly.

Its current role is:

1. Treat VAE-generated attack candidates as plausible U-boat opportunities.
2. Limit the attacker to noisy, incomplete observations of the convoy.
3. Compare belief-limited attack selection against a full-state oracle.
4. Rebuild firing solutions from partial observations for a more realistic attacker baseline.
5. Provide a stronger adversarial evaluation source for baseline and RL layout comparisons.

The POMDP work should support final layout evaluation without overstating historical fidelity. It is a compact, auditable partial-observation bridge, not a full human TDC/periscope simulation.

## Main Files

Core POMDP logic:
- `convoy_sim/pomdp_candidate_selector.py`
- `convoy_sim/pomdp_fire_control.py`
- `convoy_sim/realism.py`
- `convoy_sim/fire_control.py`

Candidate generation and evaluation:
- `experiments/generate_vae_candidate_pool.py`
- `experiments/evaluate_attack_candidate_pool.py`
- `experiments/run_pomdp_candidate_selector.py`

Notebook workflows:
- `notebooks/pomdp_candidate_selector.ipynb`
- `notebooks/pomdp_candidate_eval.ipynb`
- `notebooks/pomdp_fire_control_eval.ipynb`
- `notebooks/vae_candidate_pool.ipynb`
- `notebooks/vae_source_comparison.ipynb`

Related docs:
- `docs/VAE.md`
- `docs/REINFORCEMENT_LEARNING.md`
- `docs/SIMULATION_PHYSICS.md`
- `docs/REPRODUCING.md`
- `docs/SCRIPTS.md`
- `docs/PROJECT_MAP.md`

## Problem Framing

The attacker should not receive perfect information about the defender layout.

The full-state candidate evaluator is useful as an oracle: it ranks candidate attacks using simulator outcomes and defender loss. The POMDP layer asks a stricter question:

- if the attacker only has noisy contact information, which candidate would it choose?
- how much attack quality is lost relative to the full-state oracle?
- how does degraded observation quality affect selection and fire-control quality?

This is especially important for RL layout evaluation. If defender layouts are only tested against scripted or full-state attacks, the final result can miss the gap between omniscient adversaries and limited-information adversaries.

## Information Boundary

Hidden from the attacker:

- exact ship positions
- exact ship classes and value weights
- exact convoy formation geometry
- future movement jitter/evasion
- dynamic outcome labels
- Monte Carlo hit counts
- expected loss and full-state candidate ranks

Available to the attacker:

- own candidate U-boat location and approach context
- noisy range and bearing to the convoy/contact cluster
- noisy convoy heading and speed estimate
- noisy contact count
- noisy contact-detection fraction
- noisy formation width/depth estimate
- noisy contact-density estimate
- noisy class-confidence counts
- environment context such as night, visibility, and sea state

The selector may use candidate profile and intent metadata, but it should not use outcome labels or scored simulator results while choosing.

## Observation Presets

Observation presets live in `convoy_sim.realism`.

Active presets:

- `good_contact`: strong track or clearer contact picture
- `baseline_night`: moderate night-contact uncertainty
- `poor_contact`: degraded visibility/contact picture

Expected behavior:

```text
full-state oracle >= good_contact >= baseline_night >= poor_contact
```

The ordering is a design expectation, not a guarantee for every single run. Variable-location comparisons can be affected by which source locations each preset selects.

## Current Architecture

The current POMDP implementation has two layers.

POMDP v1:

- ranks complete VAE-generated attack candidates
- uses `belief_limited_heuristic_v1`
- keeps the original candidate firing parameters
- establishes the belief-limited selection baseline

POMDP v2:

- uses VAE candidates as plausible U-boat locations / approach contexts
- builds noisy attacker observations
- rebuilds firing solutions with `fire_control_lite`
- evaluates rebuilt profiles through the same Monte Carlo candidate-pool evaluator

The v2 fire-control model is convoy/formation-level, not ship-identification-level. It estimates the convoy picture, then fires into that estimated formation. It does not select one observed merchant ship and solve a precise ship-specific historical TDC shot.

## POMDP v1: Candidate Selection

POMDP v1 is an auditable selector over VAE candidates.

The heuristic scores:

- range suitability
- bearing alignment
- target aspect
- estimated formation span exposed to the firing direction
- estimated contact density
- spread suitability
- noisy contact count
- inside-envelope opportunity when contact quality supports it
- observation uncertainty penalties

This is not a learned POMDP policy. It is a transparent baseline that defines the information boundary and comparison pipeline before any learned attacker policy is attempted.

## POMDP v2: Fire-Control Rebuild

POMDP v2 keeps the VAE role narrower. The VAE supplies plausible tactical context; the attacker derives the firing solution from imperfect observation.

Current flow:

1. Load a VAE candidate pool.
2. Rank/select source candidates with the belief-limited selector.
3. Build noisy observations for each selected source candidate.
4. Point the U-boat toward the estimated convoy/contact bearing.
5. Use `fire_control_lite` to rebuild centerline bearing, spread, salvo size, torpedo speed, and max run time.
6. Evaluate rebuilt profiles with the shared candidate-pool evaluator.

V2 should not blindly copy the source candidate's original firing solution, outcome labels, or dynamic audit fields into the rebuilt candidate record. The partial-observation boundary is cleaner when the attacker gets a plausible position and derives fire control from the observed convoy picture.

## Fixed-Location Diagnostic

Final POMDP interpretation should include a fixed-location diagnostic.

Variable-location comparisons mix two effects:

- which source locations the preset selects
- how well fire control is rebuilt from noisy observations

The fixed-location diagnostic holds source locations constant across `good_contact`, `baseline_night`, and `poor_contact`, then rebuilds only the fire-control solution. This isolates observation-limited fire-control quality from location-selection composition effects.

Without this diagnostic, a poor-contact preset can appear unexpectedly strong or weak because it selected a different set of source locations, not because its fire-control reconstruction was better or worse.

## Evaluation Workflow

Operational commands and notebook order belong in `docs/REPRODUCING.md`. At the design level, the POMDP workflow is:

1. Generate or load VAE candidate pools.
2. Evaluate a full-state oracle candidate ranking.
3. Run POMDP v1 belief-limited candidate selection across observation presets.
4. Run POMDP v2 fire-control rebuilds from noisy observations.
5. Repeat across observation seeds.
6. Run the fixed-location diagnostic.
7. Compare expected hits, expected loss, unique ships hit, and tail-risk metrics.
8. Record caveats before using POMDP results in final RL/layout claims.

Useful comparisons:

- VAE candidate source versus full-state oracle
- full-state oracle versus POMDP v1 selection
- POMDP v1 original candidates versus POMDP v2 rebuilt fire control
- variable-location POMDP v2 versus fixed-location POMDP v2

## Metric Interpretation

POMDP metrics are comparative simulator metrics.

Important interpretation rules:

- `expected_hits` means simulated torpedo/ship hits, not confirmed sinkings.
- `expected_unique_ships_hit` means distinct ships hit in simulation trials.
- `expected_loss` is objective-weighted defender loss.
- Higher expected loss can occur even when raw hits are lower if hits affect higher-value or more distinct ships.
- Candidate-pool summaries average over selected high-threat profiles, not normal campaign-level convoy outcomes.

Reporting should avoid framing POMDP hit counts as calibrated historical sinking rates. A safer framing is:

```text
Under an attack-conditioned simulator that evaluates selected high-threat U-boat attack profiles, partial-observation fire control changes expected torpedo-hit and defender-loss metrics relative to full-state or VAE-original candidate selection.
```

## Current Limitations

- The current selector is heuristic, not a learned belief-state policy.
- The fire-control rebuild is formation-level, not full historical TDC modeling.
- The attacker does not yet perform bounded pre-attack repositioning.
- Escort search/reaction is not modeled as an active adversarial feedback loop.
- Final results need to be regenerated after the cleanup and final VAE/POMDP reruns.

## Near-Term Implementation Focus

The active POMDP workstream is tracked in `docs/TODO.md`.

Near-term order:

1. Regenerate VAE candidate pools from clean inputs.
2. Rerun POMDP v1 and v2 notebooks.
3. Preserve the fixed-location diagnostic.
4. Report robustness across `good_contact`, `baseline_night`, and `poor_contact`.
5. Decide whether POMDP v2 is final-scope or documented as near-final/future work.
6. Update final result summaries from regenerated outputs only.
