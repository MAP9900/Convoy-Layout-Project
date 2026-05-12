# VAE Attack-Profile Workflow

This document summarizes the current VAE work for synthetic U-boat attack profile generation, candidate-pool construction, and downstream adversarial/POMDP convoy-layout evaluation.

## Current Role

The VAE is not the simulator and it is not the historical source of truth. Its current role is:

1. Learn a compact latent representation of realistic synthetic attack profiles.
2. Generate novel candidate profiles near that learned tactical manifold.
3. Feed those candidates through the existing moving-convoy simulation and outcome audit.
4. Provide a GenAI candidate source for later adversarial and POMDP attacker-selection work.

The curated generator still defines the intended realism envelope. The VAE is useful only when its decoded profiles remain physically safe, tactically plausible, and meaningful under the simulator.

## Main Files

Core VAE logic:
- `convoy_sim/vae.py`
- `convoy_sim/vae_diagnostics.py`

Training-data generation:
- `experiments/generate_attack_profile_scaffold.py`
- `experiments/generate_random_attack_profile_dataset.py`
- `experiments/build_mixed_attack_profile_dataset.py`
- `experiments/audit_attack_profile_dataset.py`
- `convoy_sim/profile_outcome_audit.py`
- `convoy_sim/target_zones.py`

VAE candidate generation and evaluation:
- `experiments/generate_vae_candidate_pool.py`
- `experiments/evaluate_attack_candidate_pool.py`

Notebook workflows:
- `notebooks/profile_generation_tests.ipynb`
- `notebooks/vae_train.ipynb`
- `notebooks/random_vae_train.ipynb`
- `notebooks/mixed_vae_train.ipynb`
- `notebooks/vae_candidate_pool.ipynb`
- `notebooks/vae_source_comparison.ipynb`
- `notebooks/vae_final_baseline_comparison.ipynb`
- `notebooks/attack_candidate_pool_eval.ipynb`

Related docs:
- `docs/SIM_FEATURES.md`
- `docs/SCRIPTS.md`
- `docs/ADVERSARIAL_POMDP_OUTLINE.md`
- `NOTES.md`

## Data-Manifold History

The first synthetic generator was too rigid for VAE training. It sampled from fixed approach presets and fixed centroid-based range bands. The audit also judged plausibility relative to the convoy centroid. In plots this produced formal spawn rings, clustered rows, and attacks pulled toward the middle of the convoy.

That was not a good training manifold. It taught the VAE a geometry that looked clean in code but was tactically artificial.

The current preferred training source is `random_tactical_v4`, implemented in `experiments/generate_attack_profile_scaffold.py`. It changed the data-generation frame from centroid-first to spawn-first and target-aware:

- samples U-boat spawn regions around and inside the convoy envelope
- allows perimeter, beam, ahead, astern, and inside-column tactical positions
- enforces a minimum spawn clearance of `250 m` from any ship
- chooses accessible target ships/zones instead of aiming blindly at the convoy centroid
- uses moving U-boats with `uniform_divergent` torpedo spreads
- audits against the moving zig-zag convoy simulation before accepting records
- supports an intended training mix of `65% credible_hit_threat`, `25% credible_near_miss`, and `10% intentional_miss`

This makes the training data more diverse and historically defensible: successful attacks can come from outside the convoy, near the convoy edge, or inside the convoy envelope, but they still must avoid impossible starting collisions.

## Random Baseline

The random comparison pipeline lives in `experiments/generate_random_attack_profile_dataset.py` and `notebooks/random_vae_train.ipynb`.

It is intentionally less curated than `random_tactical_v4`. The point is not to make the best possible U-boat doctrine. The point is to train and compare a VAE against a broader, profile-first source distribution using the same safety and moving-convoy outcome checks.

This gives three useful comparisons:

- curated VAE trained on `random_tactical_v4`
- random VAE trained on direct random profile samples
- mixed or downstream candidate pools built from both sources

The random baseline is useful for research comparison, but the curated v4 source remains the better realism baseline.

## Mixed Curated/Random Baseline

The mixed comparison pipeline lives in `experiments/build_mixed_attack_profile_dataset.py` and `notebooks/mixed_vae_train.ipynb`.

The default mixed split is `70%` curated v4 and `30%` random v1. The intent is to preserve curated v4 as the realism anchor while letting the random source broaden the training support.

Current mixed dataset files:

- `data/attack_profiles/synthetic/train_mixed_curated70_random30_45k.jsonl`
- `data/attack_profiles/synthetic/valid_mixed_curated70_random30_5k.jsonl`

Verified mixed dataset build:

- train records: `45,000`
- train source counts: `31,500` curated v4, `13,500` random v1
- train intended labels: `29,285` hit threats, `11,295` near misses, `4,420` intentional misses
- validation records: `5,000`
- validation source counts: `3,500` curated v4, `1,500` random v1
- validation intended labels: `3,220` hit threats, `1,275` near misses, `505` intentional misses

This mixed VAE is a comparison point, not an assumed improvement. If it blurs tactical modes or increases decoded clearance failures, the curated v4 VAE remains the primary GenAI source.

Verified mixed VAE training run:

- run directory: `results/runs/vae/20260511_195317_mixed_curated70_random30_v1_notebook`
- train set: `45,000` profiles
- validation set: `5,000` profiles
- input dimension: `8`
- latent dimension: `6`
- hidden dimension: `64`
- beta: `0.03`
- epochs: `60`
- best epoch: `60`
- best validation loss: `0.3324`
- best validation reconstruction loss: `0.1208`
- best validation KL loss: `7.0517`
- final training loss: `0.3316`
- training time in saved run: about `35.6 s`

Mixed VAE latent-bank notebook QA:

- samples: `1,000`
- actual dynamic labels: `905` hit threats, `76` near misses, `19` misses
- clearance pass rate: `0.913`
- minimum decoded clearance: `28.7 m`
- any-ship hit rate: `0.905`
- mean hits: `2.729`
- mean unique ships hit: `1.703`
- mean closest-ship pass distance: `44.9 m`

Interpretation at training time: the mixed VAE trained cleanly and samples remained broadly plausible, but training metrics alone did not prove it was better than curated v4. The later final baseline comparison showed the mixed VAE was competitive with direct curated v4 sampling under simulator scoring.

## VAE Feature Set

The current VAE learns an 8-field continuous vector:

- `u_pos_x`
- `u_pos_y`
- `sin_base_bearing_rad`
- `cos_base_bearing_rad`
- `spread_rad`
- `launch_delay_s`
- `salvo_interval_s`
- `u_boat_initial_speed_mps`

Several doctrine fields are fixed during decoding so the generated profiles remain compatible with the current moving-convoy test standard:

- moving U-boat
- `uniform_divergent` spread
- four torpedoes
- standard torpedo speed and runtime
- bow launch origin
- standard submarine dimensions

The VAE does not currently reconstruct rich tactical intent fields such as `spawn_region`, `target_zone_kind`, `target_ship_ids`, `aim_point`, or intended outcome label. Some of those fields are recovered later as derived metadata during dynamic audit and candidate-pool generation.

## Early VAE Issue

The first curated v4 VAE training run was numerically stable, but raw Gaussian-prior sampling was not good enough.

Observed problems from notebook analysis:

- decoded prior samples smeared through the convoy instead of preserving the tactical spawn distribution
- a noticeable fraction violated the `250 m` clearance expectation
- static centroid audit rejected most samples
- dynamic audit showed samples were heavily biased toward hit threats

The static audit rejection rate needed careful interpretation. The old static audit was partly unfair to decoded VAE profiles because the VAE does not decode the v4 `intent` metadata. Without target-zone fields, the audit falls back toward centroid-style logic. However, the clearance and moving-convoy outcome results still showed a real problem: raw prior samples were not a reliable attack-profile source.

## Latent-Bank Sampling

To separate poor Gaussian-prior coverage from poor learned reconstruction, `convoy_sim/vae.py` now supports empirical latent-bank sampling.

The latent-bank path:

1. encodes real training examples into latent vectors
2. stores either posterior means or sampled latent values
3. samples near those encoded training examples
4. decodes with a small noise scale, currently commonly `0.10`

This keeps VAE samples near the learned tactical manifold instead of asking a small VAE to make the full Gaussian prior match a multimodal tactical distribution.

Current conclusion: latent-bank sampling is the right immediate path for VAE-derived attack candidates. A CVAE may still be useful later, but it is not required before building the adversarial candidate-selection workflow.

## VAE Diagnostics

`convoy_sim/vae_diagnostics.py` provides VAE-specific audit helpers.

The important diagnostics are:

- minimum ship clearance
- dynamic moving-convoy outcome label
- any-ship hit rate
- intended/derived target hit information when available
- closest pass distances
- legacy static centroid label only as a comparison field

For decoded VAE profiles, the dynamic moving-convoy audit is more important than the old static centroid audit.

## Candidate-Pool Generation

`experiments/generate_vae_candidate_pool.py` turns a trained VAE run into a JSONL candidate pool for adversarial work.

Purpose: generate and audit VAE-derived attack profiles.

Typical workflow:

1. Load a trained VAE checkpoint and preprocessor.
2. Sample decoded profiles using `latent_bank` or `prior`.
3. Rebuild profiles into simulator-native `AttackProfile` objects.
4. Enforce minimum spawn clearance.
5. Run moving zig-zag outcome diagnostics.
6. Filter to selected actual outcomes, usually `credible_hit_threat`.
7. Attach derived metadata such as spawn region, closest ship, hit ships, and source diagnostics.
8. Write a JSONL candidate pool plus summary JSON.

The default adversarial candidate-pool use case intentionally favors successful attacks. This is different from preserving the original `65/25/10` training mix. For training the VAE, a mixed outcome distribution is useful. For red-team candidate selection, it is more useful to provide a diverse pool of credible attack threats and let the selector rank them.

Example command:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/Python-ML/bin/python -m experiments.generate_vae_candidate_pool \
  --run-dir results/runs/vae/20260511_005053_v4_notebook \
  --train-path data/attack_profiles/synthetic/train_random_tactical_v4_45k.jsonl \
  --output data/attack_profiles/vae_candidates/curated_v4_hit_candidates.jsonl \
  --sample-count 5000 \
  --keep-count 1000 \
  --device cpu \
  --sampling-method latent_bank \
  --latent-noise-scale 0.10 \
  --accepted-outcome credible_hit_threat
```

The notebook wrapper is `notebooks/vae_candidate_pool.ipynb`.

Use this notebook when the question is: "Can the trained VAE produce a safe, plausible, diverse pool of attack profiles?"

This notebook does not rank profiles strategically against a convoy layout. It only creates and audits the candidate source pool.

## Monte Carlo Candidate Evaluation

`experiments/evaluate_attack_candidate_pool.py` evaluates candidate profiles through the existing scored simulation pipeline and ranks them from the attacker perspective.

Purpose: evaluate an already-generated candidate pool and rank candidates by simulated attacker value.

The Monte Carlo setup keeps each candidate profile fixed, then reruns the simulator across multiple seeds and trials. The average over those trials becomes the candidate estimate:

- `expected_hits`
- `expected_unique_ships_hit`
- `expected_repeat_hits`
- `value_lost`
- `expected_loss`
- `CVaR_90`
- `CVaR_90_loss`

In the current notebook setup, the main source of run-to-run variation is convoy movement realism: position jitter, heading jitter, speed jitter, and evasive zig-zag behavior. Torpedo noise can also contribute if `NOISE_CFG` is enabled. The profile itself is fixed, but the realized ship positions and headings can differ across trials, which changes whether torpedoes hit, miss, or hit different ships.

This is why `expected_hits` can change between runs even for the same candidate. It is a Monte Carlo estimate of expected simulator outcome under stochastic convoy realism, not a deterministic property of the profile.

The evaluator uses the same scored simulation and objective plumbing as the rest of the project. A recent scoring fix ensures capped hit counts, unique ships hit, value loss, and class-value metrics are all derived from the same hit events when `max_hits_per_torpedo=1`.

The notebook wrapper is `notebooks/attack_candidate_pool_eval.ipynb`.

Use this notebook when the question is: "Given a candidate pool, which attacks are most dangerous to this convoy layout under the simulator?"

This notebook does not train the VAE and does not create new decoded VAE samples. It consumes a candidate JSONL file created by `notebooks/vae_candidate_pool.ipynb` or a comparable generator.

## Notebook Purpose Split

Current active notebooks:

- `notebooks/profile_generation_tests.ipynb`: inspect and validate curated/random synthetic training-data generation.
- `notebooks/vae_train.ipynb`: train the curated v4 VAE and sample from the trained model.
- `notebooks/random_vae_train.ipynb`: train the random-baseline VAE for comparison.
- `notebooks/mixed_vae_train.ipynb`: train the `70%` curated v4 / `30%` random v1 mixed VAE.
- `notebooks/vae_candidate_pool.ipynb`: turn a trained VAE into a filtered candidate-pool JSONL.
- `notebooks/vae_source_comparison.ipynb`: generate matched curated/random/mixed candidate pools, compare candidate-source summaries, and optionally run matched Monte Carlo evaluation.
- `notebooks/vae_final_baseline_comparison.ipynb`: final paper-facing baseline comparison of the original P01-P60 profile library, direct curated v4 synthetic hit-threat records, and the mixed 70/30 VAE-generated hit-threat pool.
- `notebooks/attack_candidate_pool_eval.ipynb`: evaluate and rank an existing candidate pool with the Monte Carlo simulator.

Legacy notebook:

- `vae_exploration.ipynb`: older exploratory notebook, now superseded by the focused notebooks above and archived outside the active workspace.

## Results Report

This section is the central reporting spot for notebook results. The most important final comparison is `notebooks/vae_final_baseline_comparison.ipynb`, because it compares the original hand-scaffolded profile library, direct synthetic generator output, and VAE-generated candidates through the same scored simulator.

### Top-Line Findings

- The project clearly improved beyond the original `P01-P60` profile library. The final synthetic/VAE pools are much more spatially diverse and produce stronger top-ranked adversarial candidates.
- The mixed `70%` curated / `30%` random VAE is competitive with direct curated-v4 synthetic sampling. It is not strictly dominant, but it preserves a high-quality attack manifold, has slightly broader spatial diversity, and produces the strongest pool-average expected loss in the final baseline comparison.
- Direct curated v4 synthetic sampling remains an excellent realism baseline and produced the single strongest candidate in the final run.
- The best honest paper claim is: the VAE converts a realistic synthetic attack-profile manifold into a compact generative candidate source that is substantially better than the original fixed library and competitive with direct generator sampling under matched simulation scoring.

### Final Baseline Comparison

Generated by `notebooks/vae_final_baseline_comparison.ipynb`.

Inputs:

- original baseline: `DEFAULT_ATTACK_PROFILE_LIBRARY` P01-P60 profiles
- direct synthetic baseline: `1,000` hit-threat records sampled from `data/attack_profiles/synthetic/train_random_tactical_v4_45k.jsonl`
- VAE baseline: `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl`
- generated comparison pools: `results/diag/vae_final_baseline_comparison/candidate_pools/`
- summary outputs: `results/diag/vae_final_baseline_comparison/final_baseline_*`
- convoy profile: `convoy_layout_1`
- evaluated profiles: all `60` P01-P60 profiles, all `1,000` direct curated v4 candidates, all `1,000` mixed VAE candidates
- trials per candidate: `30` (`3` seeds x `10` trials)
- top-k: `25`
- objective preset: `balanced_default`
- max hits per torpedo: `1`

Source diversity summary:

| Source | Profiles | Hit-threat records | Inside-envelope rate | Mean clearance | Median clearance | Min clearance | 250m bins | 500m bins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P01-P60 fixed library | `60` | `0` | `0.250` | `1049.2 m` | `768.0 m` | `244.1 m` | `48` | `41` |
| Direct curated v4 synthetic | `1,000` | `1,000` | `0.198` | `1546.1 m` | `1394.6 m` | `284.2 m` | `689` | `370` |
| Mixed 70/30 VAE candidates | `1,000` | `1,000` | `0.210` | `1564.9 m` | `1315.7 m` | `250.5 m` | `708` | `410` |

Spawn-region counts:

| Source | ahead | astern | beam attack | inside envelope | outside perimeter | port | starboard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct curated v4 synthetic | `215` | `106` | `147` | `198` | `334` | `0` | `0` |
| Mixed 70/30 VAE candidates | `297` | `181` | `0` | `210` | `0` | `154` | `158` |
| P01-P60 fixed library | `33` | `12` | `0` | `15` | `0` | `0` | `0` |

Note: direct curated v4 and mixed VAE use different region vocabularies. The direct generator records raw tactical regions like `outside_perimeter` and `beam_attack`; VAE candidate generation derives regions as `ahead`, `astern`, `port`, `starboard`, and `inside_convoy_envelope`.

Final Monte Carlo comparison:

| Source | Pool expected hits | Pool unique ships hit | Pool repeat hits | Pool value lost | Pool expected loss | Pool p(hit>=1) | Top-25 expected loss | Top-25 unique ships hit | Best candidate | Best expected loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| P01-P60 fixed library | `2.9256` | `1.6200` | `1.3056` | `1.6754` | `3.5565` | `0.9844` | `4.5715` | `2.1787` | `P27` | `6.7800` |
| Direct curated v4 synthetic | `3.1739` | `1.6206` | `1.5533` | `1.6752` | `3.6064` | `0.9991` | `6.5997` | `3.1680` | `T028529` | `8.0500` |
| Mixed 70/30 VAE candidates | `2.8688` | `1.7410` | `1.1278` | `1.7896` | `3.7561` | `0.9850` | `6.6073` | `3.1800` | `VAE000124` | `7.9817` |

Interpretation:

- P01-P60 is a useful historical baseline, but it is no longer competitive as a candidate source. It has only `41` occupied 500m spatial bins, compared with `370` for direct curated v4 and `410` for mixed VAE.
- Direct curated v4 has the highest pool expected hits and the best single candidate. This is expected because the curated generator is explicitly designed to produce tactically valid attacks.
- Mixed VAE has the highest pool expected loss, highest pool unique-ships-hit, lowest repeat-hit rate, slightly stronger top-25 expected loss, and the broadest 500m spatial-bin coverage.
- The VAE result is strongest when framed as generative candidate production, not as replacing the curated generator as the realism authority.

Paper-facing conclusion:

The final VAE does not simply win because it produces more profiles; the final baseline evaluated all available candidates for both large pools. Compared with the original P01-P60 library, the mixed VAE candidate pool is far more spatially diverse and produces much stronger top-ranked attacks. Compared with direct curated v4 synthetic sampling, the VAE is competitive rather than dominant: it gives slightly better pool expected loss, top-25 expected loss, unique-ship impact, and spatial coverage, while direct curated v4 keeps a small edge in raw hit count and best single candidate.

### VAE Training Runs

All current VAE runs use the same 8-field feature set, `latent_dim=6`, `hidden_dim=64`, `beta=0.03`, batch size `512`, learning rate `0.001`, and `60` epochs.

| Source | Run directory | Train | Valid | Best epoch | Best valid loss | Best valid recon | Best valid KL | Final valid loss | Training time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Curated v4 | `results/runs/vae/20260511_005053_v4_notebook` | `45,000` | `5,000` | `53` | `0.3312` | `0.1201` | `7.0368` | `0.3319` | `34.7 s` |
| Random v1 | `results/runs/vae/20260511_161735_random_v1_notebook` | `45,000` | `5,000` | `53` | `0.3220` | `0.1096` | `7.0779` | `0.3225` | `32.0 s` |
| Mixed 70/30 | `results/runs/vae/20260511_195317_mixed_curated70_random30_v1_notebook` | `45,000` | `5,000` | `60` | `0.3324` | `0.1208` | `7.0517` | `0.3324` | `35.6 s` |

Training interpretation:

- All three VAEs trained stably.
- The random v1 model has the lowest reconstruction loss, but that does not make it the most historically realistic source.
- Training loss alone is not the decisive metric. Candidate-pool audit and simulator scoring matter more because the research goal is realistic adversarial attack-profile generation.

### VAE Candidate-Pool Generation

Generated by `notebooks/vae_candidate_pool.ipynb` and `notebooks/vae_source_comparison.ipynb`.

Matched candidate-generation settings:

- decoded samples per source: `5,000`
- kept candidates per source: `1,000`
- sampling method: `latent_bank`
- latent noise scale: `0.10`
- accepted outcome filter: `credible_hit_threat`
- minimum accepted clearance: `250 m`

Candidate-pool summary:

| Source | Pre-filter clearance ok | Pre-filter hit rate | Pre-filter mean hits | Accepted mean clearance | Accepted median clearance | Accepted mean hits | Accepted mean unique ships hit | Inside-envelope rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Curated v4 VAE | `0.9008` | `0.9166` | `2.8112` | `1287.7 m` | `1264.7 m` | `3.030` | `1.836` | `0.180` |
| Random v1 VAE | `0.9170` | `0.8734` | `2.5412` | `2201.1 m` | `2051.5 m` | `2.894` | `2.013` | `0.189` |
| Mixed 70/30 VAE | `0.8970` | `0.9094` | `2.7816` | `1564.9 m` | `1315.7 m` | `3.041` | `1.896` | `0.210` |

Accepted spawn-region counts:

| Source | ahead | astern | inside envelope | port | starboard |
|---|---:|---:|---:|---:|---:|
| Curated v4 VAE | `285` | `202` | `180` | `161` | `172` |
| Random v1 VAE | `270` | `213` | `189` | `170` | `158` |
| Mixed 70/30 VAE | `297` | `181` | `210` | `154` | `158` |

Accepted hit-count counts:

| Source | 1 hit | 2 hits | 3 hits | 4 hits |
|---|---:|---:|---:|---:|
| Curated v4 VAE | `72` | `214` | `326` | `388` |
| Random v1 VAE | `80` | `249` | `368` | `303` |
| Mixed 70/30 VAE | `72` | `221` | `301` | `406` |

Candidate-generation interpretation:

- All three VAE sources can produce `1,000` safe hit-threat candidates from `5,000` latent-bank decoded samples.
- Mixed 70/30 produces the highest accepted mean hit count and the largest inside-envelope share.
- Random v1 produces higher accepted mean clearance and higher mean unique ships hit, but lower pre-filter hit rate and lower accepted mean hits.
- Curated v4 remains the cleanest realism anchor because its training source is the most tactically controlled.

### VAE Source Comparison

Generated by `notebooks/vae_source_comparison.ipynb`.

This comparison evaluated the first `100` VAE candidates per source with `30` Monte Carlo trials per candidate. It is useful for source selection, but the final baseline comparison above is stronger because it evaluates all `1,000` candidates for the direct curated and mixed VAE pools.

| Source | Pool expected hits | Pool unique ships hit | Pool expected loss | Pool value lost | Pool p(hit>=1) | Top-25 expected loss | Top-25 unique ships hit | Best candidate | Best expected loss |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Curated v4 VAE | `2.9223` | `1.6927` | `3.6732` | `1.7346` | `0.9787` | `5.1983` | `2.4747` | `VAE000046` | `6.4267` |
| Random v1 VAE | `2.6923` | `1.8080` | `3.8420` | `1.8571` | `0.9693` | `5.7451` | `2.8053` | `VAE000018` | `7.3717` |
| Mixed 70/30 VAE | `2.9820` | `1.8177` | `3.9180` | `1.8675` | `0.9997` | `5.6694` | `2.7240` | `VAE000124` | `7.9817` |

Source-comparison interpretation:

- Mixed 70/30 was strongest on pool expected loss, pool expected hits, hit reliability, and best single candidate in the 100-candidate source comparison.
- Random v1 was strongest on top-25 expected loss and top-25 unique ships hit, suggesting its candidates may be more dispersed across target ships.
- Curated v4 remained the realism anchor but was not the strongest adversarial source under this scoring run.

## Relationship To CVAE

A conditional VAE remains a valid future direction, especially if we need direct control over:

- outcome label
- spawn region
- inside/outside convoy envelope
- target zone
- target class

However, the current latent-bank VAE plus dynamic filtering is already useful for the immediate adversarial candidate-pool stage. The CVAE is therefore deferred unless the regular VAE fails the next comparison step.

## Relationship To POMDP Work

The current VAE candidate selector is full-state. It ranks candidates using true simulator evaluation and is best understood as an attacker upper bound.

The intended POMDP progression is:

1. Use the VAE to produce a fixed candidate pool.
2. Rank candidates with full-state simulation to establish an upper-bound red-team baseline.
3. Replace full-state information with noisy observations and belief-state features.
4. Train or evaluate a belief-limited attacker that selects among feasible VAE candidates.
5. Compare defender robustness under scripted, VAE-only, full-state adversarial, and belief-limited adversarial attacks.

This keeps the GenAI piece meaningful: the VAE proposes plausible attack candidates, while the adversarial/POMDP layer decides which candidate a limited-information attacker would choose.

First implemented POMDP bridge:

- module: `convoy_sim/pomdp_candidate_selector.py`
- entrypoint: `experiments/run_pomdp_candidate_selector.py`
- notebook: `notebooks/pomdp_candidate_selector.ipynb`
- primary candidate source: `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl`
- selector method: `belief_limited_heuristic_v1`
- smoke run best candidate: `VAE000009`
- smoke run best belief score: `0.9678`

The belief selector uses noisy attacker-facing estimates and candidate profile/intent fields. It deliberately does not use true dynamic hit counts, value loss, expected loss, or full-state ranks while scoring.

## Known Limitations

- The VAE only decodes 8 continuous fields, so tactical metadata must be derived after sampling.
- Raw Gaussian-prior sampling is still weak for this multimodal tactical manifold.
- Latent-bank sampling is more reliable but stays close to the training distribution by design.
- The accepted VAE candidate pool is intentionally hit-heavy, so it should not be described as the full distribution of historical U-boat attacks.
- Current candidate ranking is full-state and should not be described as POMDP behavior yet.
- Most current results are centered on `convoy_layout_1`; generalization across layouts still needs testing.
- Direct curated v4 and VAE candidates use different spawn-region vocabularies, so region counts should be interpreted as tactical coverage, not one-to-one category equivalence.
- Direct curated v4 records used in the final baseline do not store source-audit `n_hits` in the same place as VAE candidate records. Use the Monte Carlo scoring table for hit/loss comparisons.
- The final baseline uses a reasonable notebook-scale budget of `30` trials per candidate. Stronger publication-grade confidence would require more seeds/trials and additional convoy layouts.

## Possible Next Steps

1. Use the `Results Report` section above as the source of truth for the VAE final paper.
2. Frame the main result as "VAE is competitive with direct curated generation and clearly improves beyond P01-P60," not as "VAE replaces the curated generator."
3. Keep `data/attack_profiles/vae_candidates/mixed_curated70_random30_hit_candidates.jsonl` as the primary GenAI candidate source for later adversarial/POMDP work.
4. Keep direct curated v4 synthetic sampling as the realism baseline in the paper.
5. If more runtime is available, rerun the final baseline on additional convoy layouts or with more Monte Carlo trials per candidate.

## Practical Notebook Order

For the current notebook-first workflow:

1. `notebooks/profile_generation_tests.ipynb`
2. `notebooks/vae_train.ipynb`
3. `notebooks/vae_candidate_pool.ipynb`
4. `notebooks/random_vae_train.ipynb`
5. `notebooks/mixed_vae_train.ipynb`
6. `notebooks/vae_source_comparison.ipynb`
7. `notebooks/vae_final_baseline_comparison.ipynb` for final paper-facing comparison against P01-P60 and direct synthetic baselines
8. `notebooks/attack_candidate_pool_eval.ipynb` for single-pool drilldowns after the source comparison
