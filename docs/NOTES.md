




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