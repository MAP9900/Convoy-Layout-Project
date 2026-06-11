# Adversarial + POMDP Attacker (Brief Outline)

## Assumption / Starting Point

VAE-based attack profile generation is implemented as an upstream candidate source.
The first practical candidate source is a latent-bank sampled VAE pool filtered by the moving-convoy dynamic outcome audit.
After matched source comparison, the primary POMDP candidate source is the mixed `70%` curated v4 / `30%` random v1 VAE pool:
- `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl`

Random v1 remains useful as a stress comparison for unique-ship/value-spread behavior.
The curated v4 generator remains the historical-realism anchor used to train and interpret the VAE.

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
  - noisy formation width/depth and contact-density estimates
  - observation-quality presets: `good_contact`, `baseline_night`, `poor_contact`
  - environment context already exposed by realism layer
- Action output from attacker:
  - select/rank feasible VAE candidates (optional light parameter refinement)
- Current baseline selector:
  - `convoy_sim.pomdp_candidate_selector`
  - `experiments/run_pomdp_candidate_selector.py`
  - `notebooks/pomdp_candidate_selector.ipynb`
  - `notebooks/pomdp_candidate_eval.ipynb`
  - builds noisy attacker observations and ranks candidates with a transparent heuristic
  - deliberately avoids true simulation outcomes, hit counts, value loss, and full-state ranks when scoring
- Candidate metadata expected from the generation/audit pipeline:
  - derived tactical fields (`spawn_region`, `approach_side`, `target_zone_kind`, hit/closest ship ids)
  - firing-solution fields where available (`aim_point`, `aim_solution_kind`, lead distance/intercept time, aim-offset fields)
  - intended or derived outcome labels (`credible_hit_threat`, `credible_near_miss`, `intentional_miss`)
  - dynamic outcome fields (`actual_outcome_label`, outcome-gate pass flag, intended-target hit, other-ship hit, closest-pass distances)
- Training setup:
  - phase 1: adversarial attacker against fixed defender with full-state input (debug/upper bound)
  - phase 2: belief-state attacker with partial observations (POMDP)
  - optional phase 3: alternating attacker/defender updates

## Method

1. Freeze the outcome-gated VAE candidate generation protocol for reproducibility.
   The current GenAI source is `experiments/generate_vae_candidate_pool.py`, which loads a trained VAE checkpoint, samples decoded candidates, filters by clearance and dynamic outcome, and emits JSONL candidate pools.
   Use the completed VAE source comparison and final baseline comparison in `docs/VAE.md` as the source decision record.
   The active attacker source distribution is the mixed `70%` curated v4 / `30%` random v1 VAE pool; direct curated v4 remains the historical-realism baseline and random v1 remains a stress comparison.
2. Define attacker action as candidate selection/ranking from VAE outputs.
3. Establish the full-state strategic adversary baseline with `experiments/evaluate_attack_candidate_pool.py`.
   This ranks VAE candidates by defender loss or expected hits using the existing scored simulation and objective plumbing.
4. Replace full-state input with partial-observation + belief-state policy.
   The first implemented bridge is `belief_limited_heuristic_v1`, which scores candidates from noisy range/bearing/course/speed/contact observations plus candidate profile/intent fields.
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

- belief-selected top-k evaluation wrapper/notebook using the existing POMDP selector and scored candidate-pool evaluator
- run artifacts compatible with current baseline/RL reporting pattern
- matched comparison table for VAE-only sampling, full-state selector, and belief-limited selector
- 1 figure/table showing robustness gap under adaptive attacker

Current implemented bridge:
- `experiments/run_pomdp_candidate_selector.py`
- `notebooks/pomdp_candidate_selector.ipynb`
- `notebooks/pomdp_candidate_eval.ipynb`
- `docs/POMDP.md`
- smoke artifact: `results/runs/pomdp_candidate_selector/`

Next implementation target:
- evaluate the belief-selected top-k candidates through the same scored Monte Carlo pipeline
- compare VAE-only sampling, full-state selector, and belief-limited selector on expected loss/value/hits

## Stretch (Optional)

- bounded pre-attack U-boat reposition menu before candidate selection
- alternating minimax loop for attacker vs defender updates
