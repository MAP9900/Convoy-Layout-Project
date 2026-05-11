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
- `notebooks/vae_candidate_pool.ipynb`
- `notebooks/attack_candidate_pool_eval.ipynb`
- `notebooks/vae_exploration.ipynb` is older exploratory work retained for reference.

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
- `notebooks/vae_candidate_pool.ipynb`: turn a trained VAE into a filtered candidate-pool JSONL.
- `notebooks/attack_candidate_pool_eval.ipynb`: evaluate and rank an existing candidate pool with the Monte Carlo simulator.

Legacy notebook:

- `notebooks/vae_exploration.ipynb`: older exploratory notebook. It is useful for historical context, but it is no longer the clean active workflow. It is safe to archive once any outputs you want to preserve have been captured elsewhere.

## Current Results Snapshot

The latest curated latent-bank candidate-pool workflow produced a pool of accepted VAE hit-threat candidates with a much more plausible spawn distribution than raw prior sampling.

Verified curated v4 VAE training run:

- run directory: `results/runs/vae/20260511_005053_v4_notebook`
- train set: `45,000` profiles
- validation set: `5,000` profiles
- input dimension: `8`
- latent dimension: `6`
- hidden dimension: `64`
- beta: `0.03`
- epochs: `60`
- best epoch: `53`
- best validation loss: `0.3312`
- final validation loss: `0.3319`
- final validation reconstruction loss: `0.1193`
- final validation KL loss: `7.0866`
- training time in saved run: about `34.7 s`

Verified VAE candidate-pool generation:

- source run: `results/runs/vae/20260511_005053_v4_notebook`
- candidate file: `data/attack_profiles/vae_candidates/curated_v4_hit_candidates.jsonl`
- sampling method: `latent_bank`
- latent noise scale: `0.10`
- decoded samples: `5,000`
- kept candidates: `1,000`
- accepted outcome filter: `credible_hit_threat`
- minimum accepted clearance: `250 m`
- pre-filter dynamic labels: `4,583` hit threats, `347` near misses, `70` misses
- pre-filter clearance pass rate: `0.9008`
- pre-filter any-ship hit rate: `0.9166`
- accepted spawn regions:
  - `ahead_vae`: `285`
  - `astern_vae`: `202`
  - `inside_convoy_envelope`: `180`
  - `starboard_vae`: `172`
  - `port_vae`: `161`

One candidate-pool evaluation run used:

- `convoy_layout_1`
- moving convoy with realism jitter/evasive behavior
- VAE hit-threat candidate pool
- `100` evaluated candidates
- `30` Monte Carlo trials per candidate
- full-state candidate ranking by attacker objective

Observed aggregate result:

- candidate-pool average expected hits: `2.9223`
- candidate-pool average expected unique ships hit: `1.6927`
- candidate-pool average expected loss: `3.6732`
- candidate-pool average `p_hit_ge_1`: `0.9787`
- top-25 expected hits: `3.2387`
- top-25 expected unique ships hit: `2.4747`
- top-25 expected loss: `5.1983`
- best candidate: `VAE000046`
- best candidate spawn region: `ahead_vae`
- best candidate expected hits: `4.0000`
- best candidate expected unique ships hit: `3.0667`
- best candidate expected loss: `6.4267`

Interpretation: the full-state selector is finding more dangerous VAE-derived profiles than random sampling from the candidate pool. That is a useful upper-bound red-team baseline before adding partial observations and belief-state limitations.

This should not yet be treated as the final research result. Before final reporting, rerun with a larger evaluated pool and matched baselines.

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

## Known Limitations

- The VAE only decodes 8 continuous fields, so tactical metadata must be derived after sampling.
- Raw Gaussian-prior sampling is still weak for this multimodal tactical manifold.
- Latent-bank sampling is more reliable but stays close to the training distribution by design.
- The accepted VAE candidate pool is intentionally hit-heavy, so it should not be described as the full distribution of historical U-boat attacks.
- Current candidate ranking is full-state and should not be described as POMDP behavior yet.
- Most current results are centered on `convoy_layout_1`; generalization across layouts still needs testing.
- Final claims should use matched baselines and a larger evaluation budget.

## Recommended Next Steps

1. Compare VAE candidate pools against curated v4, random baseline, and scripted profiles using the same Monte Carlo budget.
2. Increase candidate-pool evaluation beyond the current notebook smoke settings before reporting final numbers.
3. Add a baseline-comparison notebook once the comparison script is in place.
4. Keep CVAE as a second-stage improvement only if latent-bank VAE candidates fail the matched baseline comparison.
5. Begin POMDP work by turning candidate metadata into noisy-observation and belief-state features.

## Practical Notebook Order

For the current notebook-first workflow:

1. `notebooks/profile_generation_tests.ipynb`
2. `notebooks/vae_train.ipynb`
3. `notebooks/vae_candidate_pool.ipynb`
4. `notebooks/attack_candidate_pool_eval.ipynb`
5. future baseline-comparison notebook
