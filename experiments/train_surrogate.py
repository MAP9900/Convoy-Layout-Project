"""Train simple surrogate models on generated datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate regressors")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/datasets/dataset.csv"),
        help="Dataset CSV path",
    )
    parser.add_argument("--target", default="expected_hits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return np.array(rows, dtype=float), header


def main() -> None:
    args = parse_args()
    data, header = load_csv(args.data)
    if args.target not in header:
        raise ValueError(f"Target '{args.target}' not found in dataset")

    target_idx = header.index(args.target)
    feature_indices = [idx for idx, name in enumerate(header) if name != args.target]
    feature_names = [header[idx] for idx in feature_indices]

    X = data[:, feature_indices]
    y = data[:, target_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    models: dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=args.seed,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=args.seed),
    }

    report: dict[str, Any] = {
        "target": args.target,
        "feature_names": feature_names,
        "metrics": {},
    }

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(mean_squared_error(y_test, preds, squared=False))
        r2 = float(r2_score(y_test, preds))
        report["metrics"][name] = {"rmse": rmse, "r2": r2}
        model_path = output_dir / f"{name.lower()}_{args.target}.joblib"
        joblib.dump(model, model_path)

    report_path = output_dir / f"surrogate_report_{args.target}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
