




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


LOCK in default.toml Via:

python -m experiments.generate_run_config --template configs/templates/baseline.template.toml --output configs/baseline/default.toml --convoy-profile convoy_layout_1 --split-seed 1945 --n-total 30 --n-train 20 --train-seeds 1939,1940,1941 --eval-seeds 1942,1943,1944

