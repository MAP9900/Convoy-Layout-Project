# POMDP Attacker Notes

Status: active planning / MVP evaluation track as of 2026-06-10.

## Purpose

The POMDP layer models a U-boat commander selecting attacks without full knowledge of the convoy state.
The VAE provides realistic candidate attacks; the POMDP selector decides which candidate a limited-information attacker would choose.

The current primary candidate source is:

- `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl`

This is the mixed `70%` curated v4 / `30%` random v1 VAE candidate pool selected in `docs/VAE.md`.

## Information Boundary

Hidden from the attacker:

- exact ship positions
- exact ship classes and value weights
- exact convoy formation geometry
- exact future movement jitter/evasion
- true hit counts, expected loss, and full-state candidate ranks

Available to the attacker:

- own candidate attack geometry and firing parameters
- noisy range and bearing to the convoy/contact cluster
- noisy convoy heading and speed estimate
- noisy contact count
- noisy contact-detection fraction, so poor contact may only see a subset of the convoy
- noisy formation width/depth estimate
- noisy contact density estimate
- noisy class-confidence counts
- environment snapshot: time of day, visibility, sea state

The selector may use candidate profile and intent metadata, but it must not use dynamic outcome labels or Monte Carlo scores while choosing.

## Observation Presets

Observation presets live in `convoy_sim.realism`:

- `good_contact`:
  - default
  - represents a strong track or clearer contact picture
  - expected to perform closest to the full-state selector, but still below the oracle
- `baseline_night`:
  - moderate night-contact uncertainty
  - keeps the earlier range/bearing/speed/contact noise scale
- `poor_contact`:
  - degraded visibility/contact picture
  - should produce lower selection quality and lower oracle overlap

Use:

```python
from convoy_sim.realism import get_attacker_observation_config

cfg = get_attacker_observation_config("good_contact")
```

## Current Selector

Implemented bridge:

- `convoy_sim/pomdp_candidate_selector.py`
- `experiments/run_pomdp_candidate_selector.py`
- `notebooks/pomdp_candidate_selector.ipynb`
- `notebooks/pomdp_candidate_eval.ipynb`

Current method:

- `belief_limited_heuristic_v1`

The heuristic scores:

- range suitability
- bearing alignment with the candidate firing bearing
- target aspect
- estimated formation span exposed to the firing direction
- estimated contact density
- spread suitability
- noisy contact count
- inside-envelope opportunity when the contact picture is good enough
- observation uncertainty penalty
- additional inside-envelope uncertainty penalty, because internal convoy geometry is harder to resolve under poor contact

This is not a learned POMDP policy yet. It is an auditable baseline that establishes the information boundary and comparison pipeline before RL or belief-state training.

## First Eval Finding

The first single-seed evaluation showed the desired separation between belief-limited selection and the full-state oracle, but the three observation presets were too similar to each other.
The likely causes were:

- top-k selections were dominated by inside-convoy-envelope candidates
- formation-span score saturated near `1.0`
- contact count stayed near the full convoy size even under poor contact
- uncertainty penalty reduced scores but did not change candidate ordering enough

Current mitigation:

- prevent formation-span score from saturating as quickly
- model contact detection as a fraction of true convoy contacts
- make poor-contact observations see a smaller/noisier contact subset
- increase uncertainty penalties
- add a specific penalty for choosing inside-envelope attacks under high uncertainty
- evaluate each preset across multiple observation seeds and report mean/std

## Current Results - 2026-06-12

Latest notebook run:

- notebook: `notebooks/pomdp_candidate_eval.ipynb`
- candidate source: `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl`
- full-state comparison run: `results/runs/candidate_pool_eval/20260512_144030_vae_final_baseline_mixed_vae`
- observation seeds: `1945, 1946, 1947, 1948, 1949`
- top-k candidates evaluated per observation run: `25`

Aggregate results:

| Selector / preset | Expected hits | Expected loss | Unique ships hit | CVaR_90 loss |
|---|---:|---:|---:|---:|
| full-state oracle | `3.851` | `6.607` | `3.180` | `6.880` |
| `good_contact` | `3.449 +/- 0.063` | `4.096 +/- 0.173` | `1.865 +/- 0.095` | `4.581 +/- 0.193` |
| `baseline_night` | `3.466 +/- 0.112` | `3.967 +/- 0.249` | `1.791 +/- 0.146` | `4.424 +/- 0.291` |
| `poor_contact` | `3.031 +/- 0.156` | `3.561 +/- 0.181` | `1.617 +/- 0.084` | `4.095 +/- 0.161` |

Interpretation:

- The belief-limited selector is materially weaker than the full-state oracle, which is the expected POMDP behavior.
- `poor_contact` now clearly degrades attack quality relative to `good_contact` and `baseline_night`.
- `good_contact` and `baseline_night` remain close, which is acceptable because they are adjacent contact-quality regimes and use the same candidate source.
- `poor_contact` had `0%` top-25 overlap with the full-state oracle across the five observation seeds.
- Top-k composition shifted with degraded observation quality: `good_contact` mostly selected inside-convoy-envelope candidates, while `poor_contact` shifted toward `ahead_vae` and `astern_vae` candidates.

Saved reporting artifacts:

- `results/diag/pomdp_candidate_eval/pomdp_candidate_eval_summary.csv`
- `results/diag/pomdp_candidate_eval/pomdp_candidate_eval_summary.json`
- `results/diag/pomdp_candidate_eval/pomdp_candidate_eval_per_run.csv`
- `results/diag/pomdp_candidate_eval/pomdp_candidate_eval_oracle_overlap.csv`
- `results/diag/pomdp_candidate_eval/pomdp_candidate_eval_preset_pair_overlap.csv`

## Evaluation Pipeline

1. Load the mixed `70/30` VAE candidate pool.
2. Build noisy attacker-facing observations for each candidate under each preset and observation seed.
3. Rank candidates with `belief_limited_heuristic_v1`.
4. Write:
   - `belief_ranked_candidates.csv`
   - `top_belief_candidates.json`
   - `top_belief_candidate_pool.jsonl`
5. Evaluate each selected top-k JSONL pool with `experiments/evaluate_attack_candidate_pool.py`.
6. Aggregate expected hits/loss and overlap metrics across observation seeds.
7. Compare against the existing full-state mixed-VAE candidate-pool evaluation.

Expected ordering:

```text
full-state selector >= good_contact POMDP >= baseline_night POMDP >= poor_contact POMDP
```

The POMDP selector may still beat random VAE-only sampling because it is not random; it uses noisy but tactically meaningful estimates.

## Notebook

Use `notebooks/pomdp_candidate_eval.ipynb` for the current notebook-first workflow.

The notebook:

- runs belief selection for all three observation presets
- repeats selection over multiple observation seeds
- evaluates selected top-k candidates through the scored Monte Carlo simulator
- summarizes expected hits, expected loss, unique ships hit, and CVaR with mean/std by preset
- compares top-k overlap with the full-state oracle run when available

## Next Step

Run `notebooks/pomdp_candidate_eval.ipynb`, inspect whether metrics degrade as observation quality worsens, and then decide whether the heuristic is sufficient as a baseline or whether to replace it with an attacker policy learner.
