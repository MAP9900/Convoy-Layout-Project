"""Solve approximate Nash for a saved payoff matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convoy_sim.nash import fictitious_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve approximate Nash from matrix JSON")
    parser.add_argument("matrix_json", type=Path, help="Path to game_matrix.json")
    parser.add_argument("--iters", type=int, default=500, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nash_solution.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.matrix_json.read_text())
    matrix = data.get("matrix", {})
    m = np.array(matrix.get("matrix_mean_loss", []), dtype=float)
    if m.size == 0:
        raise ValueError("matrix_mean_loss not found in JSON")

    result = fictitious_play(m, n_iters=args.iters, rng_seed=args.seed)

    payload = {
        "p": result["p"].tolist(),
        "q": result["q"].tolist(),
        "exploitability": result["exploitability"],
        "history": result["history"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"Wrote results to {args.output}")
    print("p:", result["p"])
    print("q:", result["q"])
    print("exploitability:", result["exploitability"])


if __name__ == "__main__":
    main()
