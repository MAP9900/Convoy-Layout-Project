# Notes
- For Personal Use!



# Remove Plot Edge Lines! 
`for spine in plt.gca().spines.values():
    spine.set_visible(False)`

# Baseline logic in run_baseline_suite.py:
- Loads fixed train/eval profile splits from default.toml.
- Evaluates one static layout on eval profiles.
- Runs a grid search over spacing params on train profiles.
- Picks best train candidate by lowest expected_hits.
- Re-evaluates that winner on eval profiles.
- Writes canonical artifacts: config_resolved.yaml, metrics_summary.json, per_profile_metrics.csv, run_manifest.json.


3/26 Work:

LOCK in default.toml Via:

Baseline:

python -m experiments.generate_run_config --template configs/templates/baseline.template.toml --output configs/baseline/default.toml --convoy-profile convoy_layout_1 --split-seed 1945 --n-total 30 --n-train 20 --train-seeds 1939,1940,1941 --eval-seeds 1942,1943,1944

RL
python -m experiments.generate_run_config --template configs/templates/rl.template.toml --output configs/rl/default.toml --convoy-profile convoy_layout_1 --split-seed 1945 --n-total 30 --n-train 20

Run baseline optimization

python -m experiments.run_baseline_suite --config configs/baseline/default.toml

Run Base RL optimization

python -m experiments.run_rl_train --config configs/rl/default.toml


3/30

Re Ran With Seed Fix and Added Final Layout Plotting. Overides prevoius Test 1

python -m experiments.run_baseline_suite --config configs/baseline/default.toml
python -m experiments.run_rl_train --config configs/rl/default.toml


# --- Artifact and Metric Verification ---
def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

b_metrics = _load_json(BASELINE_RUN / 'metrics_summary.json')
r_metrics = _load_json(RL_RUN / 'metrics_summary.json')
b_manifest = _load_json(BASELINE_RUN / 'run_manifest.json')
r_manifest = _load_json(RL_RUN / 'run_manifest.json')

print('Baseline expected hits:')
print('  static   =', b_metrics['static_baseline']['expected_hits'])
print('  heuristic=', b_metrics['heuristic_baseline']['expected_hits'])
print('RL expected hits:', r_metrics['evaluation']['expected_hits'])

assert b_manifest['realism']['u_boat_mode_default'] == 'moving'
assert r_manifest['realism']['u_boat_mode_default'] == 'moving'
assert b_manifest['seed_sets'] == r_manifest['seed_sets']
assert b_manifest['profile_splits'] == r_manifest['profile_splits']

print('Manifest checks passed: moving-default + split/seed alignment confirmed.')

# Exact-metric check against prior V2-realism values
assert abs(b_metrics['static_baseline']['expected_hits'] - 2.5225) < 1e-12
assert abs(b_metrics['heuristic_baseline']['expected_hits'] - 2.495833333333333) < 1e-12
assert abs(r_metrics['evaluation']['expected_hits'] - 2.5225) < 1e-12
print('Metric values match prior V2-realism logged values exactly.')



## Curated V4 VAE Findings - 2026-05-11

Context:
- Training data used outcome-gated `random_tactical_v4` curated synthetic profiles.
- Dataset size was `45k` train / `5k` valid.
- Intended label mix was `65% credible_hit_threat / 25% credible_near_miss / 10% intentional_miss`.
- V4 data generation itself looked good: minimum spawn clearance was enforced, moving zig-zag outcome labels matched intent, and spawn distributions were tactically structured.

Observed VAE training behavior:
- Training was numerically stable.
- Train and validation losses tracked closely; no obvious overfit from the loss curves.
- Best validation epoch was around epoch `53`.
- Final/best losses were not the main failure signal.

Decoded prior-sample issue:
- Raw VAE prior samples did not preserve the curated tactical manifold.
- In the notebook static audit, only about `7.9%` of decoded samples passed the coarse geometry gate.
- About `117 / 1000` decoded samples violated the `250 m` minimum ship-clearance expectation.
- The decoded spawn plot smeared samples through the middle of the convoy instead of preserving the tactical spawn bands/envelope.
- Dynamic profile-first sim audit on `500` decoded samples produced roughly:
  - `448` hit threats
  - `43` near misses
  - `9` misses
  - `any_ship_hit_rate ~= 0.896`
- This means decoded samples were heavily biased toward over-hitting and did not preserve the intended `65/25/10` outcome mix.

Important audit nuance:
- The static audit is partly unfair to VAE samples because decoded profiles do not include v4 `intent`, `target_zone_kind`, `spawn_region`, `target_ship_ids`, or `aim_point`.
- Without those fields, static audit falls back toward older centroid-style geometry checks.
- However, the dynamic sim audit still shows a real issue: unconditional prior samples are not a reliable curated attack-profile generator.

Likely cause:
- The current VAE learns only 8 continuous fields:
  - `u_pos_x`
  - `u_pos_y`
  - `sin(base_bearing_rad)`
  - `cos(base_bearing_rad)`
  - `spread_rad`
  - `launch_delay_s`
  - `salvo_interval_s`
  - `u_boat_initial_speed_mps`
- It does not condition on or reconstruct key tactical variables:
  - intended outcome label
  - spawn region
  - target zone kind
  - inside/outside convoy flag
  - target point / aim point
  - outcome label
- The curated v4 distribution is multimodal. An unconditional Gaussian-prior VAE averages across modes and produces hybrid samples that are numerically plausible but tactically incoherent.

Current conclusion:
- Do not use raw unconditional VAE prior samples for convoy layout optimization yet.
- Keep curated v4 synthetic data as the source-of-truth attack distribution for now.
- The next VAE design should be conditional/outcome-aware rather than simply trained longer.

Recommended next VAE direction:
- Build a conditional VAE that conditions on tactical metadata such as:
  - intended label
  - spawn region
  - target zone kind
  - inside convoy envelope
- Prefer modeling/reconstructing target/aim context as well, or at least preserving it in decoded outputs.
- Evaluate decoded samples with the same moving zig-zag dynamic outcome audit, not only static centroid geometry.


VAE 5/10 Key summary

{'dataset': {'train_samples': 45000,
  'valid_samples': 5000,
  'input_dim': 8,
  'feature_names': ['u_pos_x',
   'u_pos_y',
   'sin_base_bearing_rad',
   'cos_base_bearing_rad',
   'spread_rad',
   'launch_delay_s',
   'salvo_interval_s',
   'u_boat_initial_speed_mps']},
 'model': {'latent_dim': 6,
  'hidden_dim': 64,
  'parameter_count': 10644,
  'beta': 0.03},
 'training': {'epochs': 60,
  'batch_size': 512,
  'learning_rate': 0.001,
  'best_epoch': 53,
  'final_train_loss': 0.32973783869634976,
  'final_train_recon_loss': 0.11938681665130636,
  'final_train_kl_loss': 7.011700830676339,
  'final_valid_loss': 0.3319258153438568,
  'final_valid_recon_loss': 0.11932632029056549,
  'final_valid_kl_loss': 7.086649942398071,
  'best_valid_loss': 0.3311635583639145,
  'best_valid_recon_loss': 0.12006057053804398,
  'best_valid_kl_loss': 7.036766624450683},
 'samples': {'count': 1000, 'output_file': 'sampled_profiles.json'},
 'timing': {'dataset_load_seconds': 2.819287625141442,
  'training_seconds': 34.67270758282393,
  'sample_decode_seconds': 0.21951066702604294,
  'total_seconds': 39.77002716716379}}


  {'samples': 1000,
 'label_counts': {'implausible_geometry': 922, 'credible_hit_threat': 78},
 'passes_gate_rate': 0.076,
 'clearance_ok_rate': 0.905,
 'min_clearance_m': 51.073124926132344}