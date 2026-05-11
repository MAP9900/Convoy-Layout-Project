# Adversarial + POMDP Attacker (Brief Outline)

## Assumption / Starting Point

VAE-based attack profile generation is already implemented and available as an upstream candidate source.
This project starts after that stage.

## Goal

Train an attacker policy that:
- adapts to a defender/layout policy (adversarial objective)
- only uses noisy/partial observations (POMDP realism)
- selects or ranks VAE-generated attack candidates under feasibility constraints

Core question: does defender performance degrade under an adaptive, belief-limited red-team attacker compared with scripted and VAE-only attacks?

## Scope (MVP)

- Keep current simulator and realism stack as execution engine.
- Treat VAE output as a fixed candidate pool per scenario/seed.
- Add attacker policy layer (agent) on top of the existing attack profile/action interface.
- Observation input to attacker:
  - noisy bearing/range/course/speed/contact estimates
  - environment context already exposed by realism layer
- Action output from attacker:
  - select/rank feasible VAE candidates (optional light parameter refinement)
- Candidate metadata expected from the generation/audit pipeline:
  - tactical intent fields (`spawn_region`, `approach_side`, `target_zone_kind`, target ship ids)
  - firing-solution fields (`aim_point`, `aim_solution_kind`, lead distance/intercept time, aim-offset fields)
  - intended outcome labels (`credible_hit_threat`, `credible_near_miss`, `intentional_miss`)
  - dynamic outcome fields (`actual_outcome_label`, outcome-gate pass flag, intended-target hit, other-ship hit, closest-pass distances)
- Training setup:
  - phase 1: adversarial attacker against fixed defender with full-state input (debug/upper bound)
  - phase 2: belief-state attacker with partial observations (POMDP)
  - optional phase 3: alternating attacker/defender updates

## Method

1. Freeze the outcome-gated VAE candidate generation protocol for reproducibility.
   Compare curated v4, profile-first random baseline, and later curated/random mixed candidate pools before choosing the attacker source distribution.
2. Define attacker action as candidate selection/ranking from VAE outputs.
3. Train full-state strategic adversary policy as optimization baseline.
4. Replace full-state input with partial-observation + belief-state policy.
5. Evaluate on held-out profiles/seeds with matched artifact schema.

## Evaluation

Primary:
- expected hits
- unique ships hit
- weighted value loss
- CVaR_90

Comparisons:
- fixed/scripted attacks vs VAE-only sampling
- VAE-only sampling vs strategic adversary selector
- full-state adversary vs belief-state adversary

## Deliverables

- new experiment entrypoint for attacker training/eval
- run artifacts compatible with current baseline/RL reporting pattern
- brief results table + 1 figure showing robustness gap under adaptive attacker

## Stretch (Optional)

- bounded pre-attack U-boat reposition menu before candidate selection
- alternating minimax loop for attacker vs defender updates
