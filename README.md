# Optimal Convoy Layout Project

Research-focused simulator for WWII-style convoys and straight-running torpedoes.

## Surrogate Training (Phase 7)

Install extra dependencies:

```bash
pip install scikit-learn joblib
```

Train baseline surrogates from a generated dataset:

```bash
python experiments/make_dataset.py --samples 200 --trials 50 --seed 0
python experiments/train_surrogate.py --data results/datasets/dataset.csv --target expected_hits
```

Outputs:
- `results/surrogate_report_expected_hits.json`
- `results/randomforestregressor_expected_hits.joblib`
- `results/gradientboostingregressor_expected_hits.joblib`
