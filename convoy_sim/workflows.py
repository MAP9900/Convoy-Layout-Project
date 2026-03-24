"""Shared workflow utilities for canonical baseline and RL runners."""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from convoy_sim.attack_profiles import AttackProfileLibrary
from convoy_sim.risk import empirical_cvar, empirical_var
from convoy_sim.simulation import run_monte_carlo_attack


LayoutFn = Callable[..., list[Any]]


@dataclass(frozen=True)
class ProfileEvalRow:
    model_name: str
    profile_id: str
    samples: int
    expected_hits: float
    cvar_90: float
    p_hit_ge_1: float
    value_lost: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "profile_id": self.profile_id,
            "samples": int(self.samples),
            "expected_hits": float(self.expected_hits),
            "CVaR_90": float(self.cvar_90),
            "p_hit_ge_1": float(self.p_hit_ge_1),
            "value_lost": self.value_lost,
        }


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML config requires PyYAML; use .toml or .json config instead") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if loaded is None else loaded
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_to_yaml_lines(payload)) + "\n", encoding="utf-8")


def _to_yaml_lines(value: Any, *, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return lines or [f"{pad}{{}}"]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_to_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines or [f"{pad}[]"]
    return [f"{pad}{_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def git_sha(cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_run_dir(output_root: Path, workflow_name: str, run_name: str | None = None) -> Path:
    suffix = f"_{run_name}" if run_name else ""
    run_dir = output_root / workflow_name / f"{now_stamp()}{suffix}"
    ensure_dir(run_dir)
    return run_dir


def profile_lookup(library: AttackProfileLibrary) -> dict[str, Any]:
    return {profile.profile_id: profile for profile in library.profiles}


def evaluate_layout_over_profiles(
    *,
    model_name: str,
    layout_fn: LayoutFn,
    layout_kwargs: dict[str, Any],
    library: AttackProfileLibrary,
    profile_ids: list[str],
    seeds: list[int],
    n_trials_per_seed: int,
    t_max: float,
    noise_model: Any | None = None,
    max_hits_per_torpedo: int | None = None,
) -> list[ProfileEvalRow]:
    lookup = profile_lookup(library)
    rows: list[ProfileEvalRow] = []

    for profile_id in profile_ids:
        if profile_id not in lookup:
            raise ValueError(f"Unknown profile id in split: {profile_id}")
        profile = lookup[profile_id]
        all_hits: list[float] = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed))

            def sampler(generator: np.random.Generator):
                return profile.build_torpedoes(generator)

            result = run_monte_carlo_attack(
                layout_fn=layout_fn,
                layout_kwargs=layout_kwargs,
                torpedo_sampler=sampler,
                n_trials=n_trials_per_seed,
                t_max=t_max,
                rng=rng,
                noise_model=noise_model,
                max_hits_per_torpedo=max_hits_per_torpedo,
            )
            all_hits.extend(np.asarray(result["hits_per_trial"], dtype=float).tolist())

        hits = np.asarray(all_hits, dtype=float)
        rows.append(
            ProfileEvalRow(
                model_name=model_name,
                profile_id=profile_id,
                samples=int(hits.size),
                expected_hits=float(np.mean(hits)),
                cvar_90=float(empirical_cvar(hits, 0.9)),
                p_hit_ge_1=float(np.mean(hits >= 1.0)),
                value_lost=None,
            )
        )

    return rows


def summarize_profile_rows(rows: list[ProfileEvalRow]) -> dict[str, Any]:
    if not rows:
        return {
            "profiles": 0,
            "samples": 0,
            "expected_hits": 0.0,
            "CVaR_90": 0.0,
            "VaR_90": 0.0,
            "p_hit_ge_1": 0.0,
            "value_lost": None,
        }

    expected_hits = np.array([row.expected_hits for row in rows], dtype=float)
    cvar_90 = np.array([row.cvar_90 for row in rows], dtype=float)
    p_hit = np.array([row.p_hit_ge_1 for row in rows], dtype=float)

    return {
        "profiles": len(rows),
        "samples": int(sum(row.samples for row in rows)),
        "expected_hits": float(np.mean(expected_hits)),
        "CVaR_90": float(np.mean(cvar_90)),
        "VaR_90": float(empirical_var(expected_hits, 0.9)),
        "p_hit_ge_1": float(np.mean(p_hit)),
        "value_lost": None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_profile_rows_csv(path: Path, rows: list[ProfileEvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "profile_id", "samples", "expected_hits", "CVaR_90", "p_hit_ge_1", "value_lost"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def iter_param_overrides(grid: dict[str, list[Any]], *, max_candidates: int | None = None) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*[grid[key] for key in keys])
    rows: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        if max_candidates is not None and idx >= max_candidates:
            break
        rows.append(dict(zip(keys, combo)))
    return rows
