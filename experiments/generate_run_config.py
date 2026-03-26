"""Generate full baseline/RL TOML run configs with reproducible profile splits.

Usage examples:
  python -m experiments.generate_run_config \
    --template configs/baseline/default.toml \
    --output configs/baseline/default.toml \
    --split-seed 42 --n-total 30 --n-train 20

  python -m experiments.generate_run_config \
    --template configs/rl/default.toml \
    --output configs/rl/default.toml \
    --split-seed 42 --n-total 30 --n-train 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full TOML config with reproducible train/eval profile split")
    parser.add_argument("--template", type=Path, required=True, help="Template TOML to read")
    parser.add_argument("--output", type=Path, required=True, help="Output TOML to write")
    parser.add_argument("--split-seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--n-total", type=int, default=30, help="Total profile count (P01..Pnn)")
    parser.add_argument("--n-train", type=int, default=20, help="Train profile count")
    parser.add_argument("--profile-prefix", type=str, default="P", help="Profile id prefix")
    parser.add_argument("--profile-width", type=int, default=2, help="Zero-padding width for profile ids")
    parser.add_argument(
        "--train-seeds",
        type=str,
        default="",
        help='Optional comma-separated train seeds override, e.g. "1939,1940,1941"',
    )
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="",
        help='Optional comma-separated eval seeds override, e.g. "1942,1943,1944"',
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run.name override (if omitted, preserves template value)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help=(
            "Override config leaf values via dotted path, e.g. "
            "--set baseline.static_layout.spacing_along=500.0 "
            "--set rl.actions[0].type=staggered"
        ),
    )
    parser.add_argument(
        "--convoy-profile",
        choices=list_convoy_layout_profiles(),
        default="",
        help=(
            "Optional convoy profile to inject into layout sections. "
            "Updates baseline.static_layout and rl.actions[*] geometry fields."
        ),
    )
    return parser.parse_args()


def _parse_seed_list(raw: str) -> list[int] | None:
    text = raw.strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _layout_type_from_fn_name(fn_name: str) -> str:
    if fn_name == "make_rectangular_convoy":
        return "rectangular"
    if fn_name == "make_staggered_convoy":
        return "staggered"
    raise ValueError(f"Unsupported layout function for TOML generation: {fn_name}")


def _to_toml_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _apply_convoy_profile(cfg: dict[str, Any], profile_name: str) -> None:
    if not profile_name:
        return
    profile = get_convoy_layout_profile(profile_name)
    layout_type = _layout_type_from_fn_name(getattr(profile.layout_fn, "__name__", str(profile.layout_fn)))
    layout_fields = {
        k: _to_toml_safe(v)
        for k, v in profile.layout_kwargs.items()
        if not callable(v)
    }

    if "baseline" in cfg and isinstance(cfg["baseline"], dict):
        baseline = cfg["baseline"]
        static_layout = dict(baseline.get("static_layout", {}))
        static_layout["type"] = layout_type
        static_layout.update(layout_fields)
        baseline["static_layout"] = static_layout
        cfg["baseline"] = baseline

    if "rl" in cfg and isinstance(cfg["rl"], dict):
        rl = cfg["rl"]
        actions = list(rl.get("actions", []))
        updated_actions: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            row = dict(action)
            row["type"] = layout_type
            row.update(layout_fields)
            updated_actions.append(row)
        if updated_actions:
            rl["actions"] = updated_actions
            cfg["rl"] = rl


def _profile_ids(prefix: str, width: int, n_total: int) -> list[str]:
    return [f"{prefix}{i:0{width}d}" for i in range(1, n_total + 1)]


def _random_partition(ids: list[str], n_train: int, split_seed: int) -> tuple[list[str], list[str]]:
    pool = list(ids)
    rng = random.Random(split_seed)
    rng.shuffle(pool)
    train = sorted(pool[:n_train], key=lambda pid: int(pid[1:]))
    eval_ = sorted(pool[n_train:], key=lambda pid: int(pid[1:]))
    return train, eval_


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value is None:
        return '""'
    raise TypeError(f"Unsupported scalar type for TOML serialization: {type(value)!r}")


def _toml_array(values: list[Any]) -> str:
    return "[" + ", ".join(_toml_value(v) for v in values) + "]"


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return _toml_array(value)
    return _toml_scalar(value)


def _emit_table(path: list[str], table: dict[str, Any], out: list[str]) -> None:
    if path:
        out.append(f"[{'.'.join(path)}]")
    for key, value in table.items():
        if isinstance(value, dict) or (isinstance(value, list) and value and all(isinstance(x, dict) for x in value)):
            continue
        out.append(f"{key} = {_toml_value(value)}")

    for key, value in table.items():
        if isinstance(value, dict):
            out.append("")
            _emit_table(path + [key], value, out)
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            for row in value:
                out.append("")
                out.append(f"[[{'.'.join(path + [key])}]]")
                for row_key, row_val in row.items():
                    if isinstance(row_val, dict):
                        raise TypeError("Nested dicts inside array-of-table rows are not supported by this generator")
                    if isinstance(row_val, list) and any(isinstance(item, (dict, list)) for item in row_val):
                        raise TypeError("Nested compound lists inside array-of-table rows are not supported by this generator")
                    out.append(f"{row_key} = {_toml_value(row_val)}")


def dump_toml(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _emit_table([], payload, lines)
    return "\n".join(lines).strip() + "\n"


def load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


_PATH_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def _parse_override_value(text: str) -> Any:
    raw = text.strip()
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    if raw and raw[0] in "[{\"'":
        try:
            return json.loads(raw)
        except Exception:
            pass
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for segment in path.split("."):
        if not segment:
            raise ValueError(f"Invalid empty path segment in override path: {path}")
        idx = 0
        while idx < len(segment):
            m = _PATH_TOKEN_RE.match(segment, idx)
            if m is None:
                raise ValueError(f"Invalid path segment '{segment}' in override path: {path}")
            key, list_idx = m.group(1), m.group(2)
            if key is not None:
                tokens.append(key)
            else:
                tokens.append(int(list_idx))
            idx = m.end()
    return tokens


def _apply_override(cfg: dict[str, Any], *, path: str, value: Any) -> None:
    tokens = _parse_path(path)
    target: Any = cfg
    for token in tokens[:-1]:
        if isinstance(token, str):
            if not isinstance(target, dict) or token not in target:
                raise KeyError(f"Override path not found: {path}")
            target = target[token]
        else:
            if not isinstance(target, list) or token < 0 or token >= len(target):
                raise KeyError(f"Override list index out of range in path: {path}")
            target = target[token]

    leaf = tokens[-1]
    if isinstance(leaf, str):
        if not isinstance(target, dict) or leaf not in target:
            raise KeyError(f"Override leaf not found: {path}")
        target[leaf] = value
    else:
        if not isinstance(target, list) or leaf < 0 or leaf >= len(target):
            raise KeyError(f"Override list leaf index out of range: {path}")
        target[leaf] = value


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override (expected key=value): {item}")
        key, raw_val = item.split("=", 1)
        _apply_override(cfg, path=key.strip(), value=_parse_override_value(raw_val))


def main() -> None:
    args = parse_args()
    if args.n_total <= 0:
        raise ValueError("--n-total must be positive")
    if args.n_train <= 0 or args.n_train >= args.n_total:
        raise ValueError("--n-train must be between 1 and n_total-1")

    cfg = load_toml(args.template)
    splits = dict(cfg.get("splits", {}))

    ids = _profile_ids(args.profile_prefix, args.profile_width, args.n_total)
    train_profiles, eval_profiles = _random_partition(ids, args.n_train, args.split_seed)

    splits["train_profiles"] = train_profiles
    splits["eval_profiles"] = eval_profiles

    train_seed_override = _parse_seed_list(args.train_seeds)
    eval_seed_override = _parse_seed_list(args.eval_seeds)
    if train_seed_override is not None:
        splits["train_seeds"] = train_seed_override
    if eval_seed_override is not None:
        splits["eval_seeds"] = eval_seed_override

    cfg["splits"] = splits
    cfg["split_meta"] = {
        "method": "random_partition",
        "split_seed": int(args.split_seed),
        "n_total": int(args.n_total),
        "n_train": int(args.n_train),
        "n_eval": int(args.n_total - args.n_train),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile_prefix": args.profile_prefix,
        "profile_width": int(args.profile_width),
    }

    if args.run_name.strip():
        run = dict(cfg.get("run", {}))
        run["name"] = args.run_name.strip()
        cfg["run"] = run

    _apply_convoy_profile(cfg, args.convoy_profile)
    _apply_overrides(cfg, list(args.set))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump_toml(cfg), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"train_profiles={train_profiles}")
    print(f"eval_profiles={eval_profiles}")


if __name__ == "__main__":
    main()
