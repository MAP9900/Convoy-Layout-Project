# POMDP Attacker Notes

Status: POMDP v2 MVP complete as of 2026-06-20.

## Purpose

The POMDP layer models a U-boat commander selecting attacks without full knowledge of the convoy state.
The VAE provides realistic candidate attacks; the POMDP selector decides which candidate a limited-information attacker would choose.

The current primary candidate source is:

- `results/notebook-results/vae_source_comparison/candidate_pools/mixed_curated70_random30_hit_candidates.jsonl`

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
- `convoy_sim/pomdp_fire_control.py`
- `experiments/run_pomdp_candidate_selector.py`
- `notebooks/pomdp_candidate_selector.ipynb`
- `notebooks/pomdp_candidate_eval.ipynb`
- `notebooks/pomdp_fire_control_eval.ipynb`

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

## POMDP v2 Fire-Control Rebuild

POMDP v1 selects among complete VAE-generated `AttackProfile`s.
POMDP v2 keeps the VAE role narrower: it uses the VAE candidate as a plausible U-boat location / approach context, then rebuilds the firing solution from noisy observation.

The current v2 fire-control model is formation-level, not ship-identification-level.
The attacker estimates convoy/contact bearing, range, heading, speed, contact count, class mix, and formation dimensions, then fires a fan/spread into that estimated convoy picture.
It does not yet select one observed merchant ship and solve a precise intended-target torpedo shot against that ship.
This is intentional for now: it keeps the POMDP bridge compact, auditable, and useful for convoy-layout optimization without pretending to model a full human periscope/TDC workflow.

Current v2 flow:

1. Load the mixed `70/30` VAE candidate pool.
2. Run the belief-limited selector to choose top-k candidate locations.
3. For each selected location, build a noisy attacker observation with the selected preset.
4. Point the U-boat toward the estimated convoy/contact bearing.
5. Use `fire_control_lite` to rebuild:
   - centerline bearing
   - spread
   - salvo size
   - G7a speed setting
   - torpedo speed and max run time
6. Write rebuilt JSONL records under `results/notebook-results/pomdp_fire_control_eval/candidate_pools/`.
7. Evaluate rebuilt profiles with the same Monte Carlo candidate-pool evaluator.

V2 deliberately does not copy the source candidate's original `base_bearing_rad`, `spread_rad`, source outcome labels, or source audit fields into the rebuilt candidate record.
This keeps the partial-observability boundary cleaner: the attacker gets a plausible position, then derives the firing solution from imperfect information.

Notebook:

- `notebooks/pomdp_fire_control_eval.ipynb`

Expected comparison:

```text
full-state oracle
> POMDP v1 selected original VAE profiles
> POMDP v2 fire-control rebuilt profiles under good/baseline contact
> POMDP v2 fire-control rebuilt profiles under poor contact
```

Individual v2 profiles may occasionally beat their v1 source if `fire_control_lite` corrects a weak VAE firing parameter, so the intended comparison is aggregate behavior across presets and observation seeds.

Important comparison caveat:

- the main v2 notebook currently compares each preset's own selected locations and rebuilt fire-control solutions
- this combines location-selection quality and firing-solution quality
- the notebook also includes a fixed-location diagnostic that holds selected source locations fixed, then rebuilds only the fire-control solution under `good_contact`, `baseline_night`, and `poor_contact`
- that fixed-location comparison isolates how much degradation comes from observation-limited fire control rather than different spawn/location choices
- default fixed-location source: `good_contact`, observation seed `1945`, top-k `25`

## Current Results - 2026-06-20

Latest notebook run:

- notebook: `notebooks/pomdp_fire_control_eval.ipynb`
- candidate source: `results/notebook-results/vae_source_comparison/candidate_pools/mixed_curated70_random30_hit_candidates.jsonl`
- full-state comparison run: `results/runs/candidate_pool_eval/20260512_144030_vae_final_baseline_mixed_vae`
- observation seeds: `1945, 1946, 1947, 1948, 1949`
- top-k candidates evaluated per observation run: `25`
- POMDP v2 model: convoy/formation-level partial-observation fire control

Main variable-location comparison:

| Selector / preset | Expected hits | Expected loss | Unique ships hit | CVaR_90 loss |
|---|---:|---:|---:|---:|
| full-state oracle | `3.851` | `6.607` | `3.180` | `6.880` |
| v1 selected original VAE, `good_contact` | `3.449 +/- 0.063` | `4.096 +/- 0.173` | `1.865` | `4.581` |
| v1 selected original VAE, `baseline_night` | `3.466 +/- 0.112` | `3.967 +/- 0.249` | `1.791` | `4.424` |
| v1 selected original VAE, `poor_contact` | `3.031 +/- 0.156` | `3.561 +/- 0.181` | `1.617` | `4.095` |
| v2 fire-control rebuild, `good_contact` | `2.544 +/- 0.137` | `3.574 +/- 0.154` | `1.679` | `3.869` |
| v2 fire-control rebuild, `baseline_night` | `2.425 +/- 0.176` | `3.748 +/- 0.300` | `1.791` | `4.157` |
| v2 fire-control rebuild, `poor_contact` | `2.269 +/- 0.136` | `3.998 +/- 0.309` | `1.947` | `4.521` |

Interpretation:

- v2 reduces raw hit count relative to v1 because it no longer reuses the VAE candidate's original firing solution
- this is the expected realism penalty from partial-observation fire control
- the variable-location `poor_contact` case has lower expected hits but higher expected loss than `good_contact`; this comes from different location/target-composition selection, not from better poor-contact fire control
- the full-state oracle remains strongest, as expected

Fixed-location diagnostic:

- source location set: `good_contact`, observation seed `1945`, top-k `25`
- source composition: `23` inside-convoy-envelope candidates, `1` astern candidate, `1` ahead candidate
- each preset rebuilt fire control from those same `25` source locations across observation seeds `1945` to `1949`

| Fixed locations, rebuilt preset | Expected hits | Expected loss | Unique ships hit | CVaR_90 loss |
|---|---:|---:|---:|---:|
| `good_contact` | `2.700 +/- 0.086` | `3.816 +/- 0.219` | `1.795` | `4.074` |
| `baseline_night` | `2.557 +/- 0.088` | `3.805 +/- 0.187` | `1.803` | `4.194` |
| `poor_contact` | `2.430 +/- 0.111` | `3.640 +/- 0.163` | `1.724` | `3.989` |

Interpretation:

- holding locations fixed gives the expected monotonic hit-count degradation from `good_contact` to `poor_contact`
- the earlier variable-location v2 result where `poor_contact` had higher expected loss was mainly a location-selection/composition effect, not evidence that poor-contact fire control was better
- `good_contact` and `baseline_night` remain close, which is acceptable because they are adjacent observation-quality regimes

Saved reporting artifacts:

- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_eval_summary.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_eval_per_run.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_rebuilt_pools.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_fixed_location_summary.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_fixed_location_per_run.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_fixed_location_rebuilt_pools.csv`
- `results/notebook-results/pomdp_fire_control_eval/pomdp_fire_control_eval_summary.json`

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

## POMDP v1 Results - 2026-06-12

Latest notebook run:

- notebook: `notebooks/pomdp_candidate_eval.ipynb`
- candidate source: `results/notebook-results/vae_source_comparison/candidate_pools/mixed_curated70_random30_hit_candidates.jsonl`
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

- `results/notebook-results/pomdp_candidate_eval/pomdp_candidate_eval_summary.csv`
- `results/notebook-results/pomdp_candidate_eval/pomdp_candidate_eval_summary.json`
- `results/notebook-results/pomdp_candidate_eval/pomdp_candidate_eval_per_run.csv`
- `results/notebook-results/pomdp_candidate_eval/pomdp_candidate_eval_oracle_overlap.csv`
- `results/notebook-results/pomdp_candidate_eval/pomdp_candidate_eval_preset_pair_overlap.csv`

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

## Metric Interpretation and Historical Plausibility

The simulator metrics are useful for comparing layouts and attack-candidate policies, but they should not be read as direct historical ship-sinking rates.

Current metric units:

- `expected_hits`: mean simulated torpedo/ship hits, not confirmed sinkings
- `expected_unique_ships_hit`: mean number of distinct ships hit in a trial
- `expected_loss`: objective-weighted defender loss, so it can rise even when raw hit count falls if the hits land on higher-value or more distinct ships
- candidate-pool summaries average metrics across the selected top-k profiles, usually `25`, after each profile has been evaluated over multiple Monte Carlo trials

Historical sanity check:

- At the campaign level, only a minority of transatlantic convoys were attacked, and attacked convoys lost a minority of ships on average.
- Extreme early-war convoy disasters existed, especially before mature escort doctrine, radar, HF/DF, air cover, and support groups. Examples include HX 72, SC 7, and HX 79 in 1940.
- Those worst cases often involved multiple U-boats, confused escorts, night surface attacks, and U-boats penetrating the convoy columns. They are not a good baseline for a normal single-attack expected value.
- Later-war convoy defense became much more effective; by 1943 some heavily contested convoys lost few or no merchant ships while U-boats took severe losses.

Therefore, POMDP v2 values around `2.2` to `2.5` expected hits for selected high-quality attack profiles are plausible only as an optimistic, conditional, attack-event metric.
They are likely too high if described as normal ships sunk per U-boat contact, normal convoy battle losses, or campaign-level convoy attrition.

For reporting, phrase the result as:

```text
Under a deliberately attack-conditioned simulator that evaluates selected high-threat U-boat attack profiles, partial-observation fire control reduces expected torpedo hits from the original VAE-selected profiles. The raw hit counts are comparative simulator metrics rather than calibrated historical sinking rates.
```

Useful historical references for calibration:

- Battle of the Atlantic overview: <https://en.wikipedia.org/wiki/Battle_of_the_Atlantic>
- Convoy HX 72: <https://en.wikipedia.org/wiki/Convoy_HX_72>
- Convoy SC 7: <https://en.wikipedia.org/wiki/Convoy_SC_7>
- Convoy SC 130: <https://en.wikipedia.org/wiki/Convoy_SC_130>
- Time article on Western Approaches Tactical Unit convoy wargaming and internal-column U-boat attacks: <https://time.com/5772665/uboat-wargames/>

## Notebook

Use `notebooks/pomdp_candidate_eval.ipynb` for the current notebook-first workflow.

The notebook:

- runs belief selection for all three observation presets
- repeats selection over multiple observation seeds
- evaluates selected top-k candidates through the scored Monte Carlo simulator
- summarizes expected hits, expected loss, unique ships hit, and CVaR with mean/std by preset
- compares top-k overlap with the full-state oracle run when available
- runs a fixed-location POMDP v2 diagnostic so each preset rebuilds fire control from the same selected source locations
- writes fixed-location summary CSVs alongside the main POMDP v2 artifacts

## Next Step

Run `notebooks/pomdp_fire_control_eval.ipynb`, compare v1 selected profiles against v2 fire-control rebuilt profiles, and then pause for codebase audit/cleanup before any learned attacker policy work.
