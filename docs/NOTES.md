




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


