from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_generate_run_config_full_partition_and_meta(tmp_path: Path) -> None:
    template = tmp_path / "baseline_template.toml"
    output = tmp_path / "baseline_generated.toml"
    template.write_text(
        """
[run]
name = "baseline_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [11, 12, 13]
eval_seeds = [21, 22, 23]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    train = cfg["splits"]["train_profiles"]
    eval_ = cfg["splits"]["eval_profiles"]

    assert len(train) == 20
    assert len(eval_) == 10
    assert set(train).isdisjoint(set(eval_))
    assert len(set(train) | set(eval_)) == 30

    meta = cfg["split_meta"]
    assert meta["method"] == "random_partition"
    assert meta["split_seed"] == 42
    assert meta["n_total"] == 30
    assert meta["n_train"] == 20
    assert meta["n_eval"] == 10


def test_generate_run_config_preserves_rl_actions_table(tmp_path: Path) -> None:
    template = tmp_path / "rl_template.toml"
    output = tmp_path / "rl_generated.toml"
    template.write_text(
        """
[run]
name = "rl_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [301, 302, 303]
eval_seeds = [401, 402, 403]

[training]
episodes = 10
epsilon = 0.25
epsilon_decay = 0.99
epsilon_min = 0.02
alpha = 0.1
seed = 7

[rl]

[[rl.actions]]
name = "rect_standard"
type = "rectangular"
complexity_cost = 1.0
n_rows = 6
n_cols = 7
spacing_along = 457.2
spacing_across = 1371.6
speed = 5.0
heading_rad = 0.0
length = 150.0
beam = 20.0
origin = [0.0, 0.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "99",
            "--n-total",
            "30",
            "--n-train",
            "20",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    actions = cfg["rl"]["actions"]
    assert len(actions) == 1
    assert actions[0]["name"] == "rect_standard"
    assert actions[0]["origin"] == [0.0, 0.0]


def test_generate_run_config_supports_layout_overrides(tmp_path: Path) -> None:
    template = tmp_path / "baseline_template.toml"
    output = tmp_path / "baseline_generated.toml"
    template.write_text(
        """
[run]
name = "baseline_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [11, 12, 13]
eval_seeds = [21, 22, 23]

[baseline.static_layout]
type = "rectangular"
n_rows = 6
n_cols = 7
spacing_along = 457.2
spacing_across = 1371.6
speed = 5.0
heading_rad = 0.0
length = 150.0
beam = 20.0
origin = [0.0, 0.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
            "--set",
            "baseline.static_layout.type=staggered",
            "--set",
            "baseline.static_layout.spacing_along=500.0",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    layout = cfg["baseline"]["static_layout"]
    assert layout["type"] == "staggered"
    assert layout["spacing_along"] == 500.0


def test_generate_run_config_injects_mixed_convoy_profile_metadata(tmp_path: Path) -> None:
    template = tmp_path / "baseline_template.toml"
    output = tmp_path / "baseline_generated.toml"
    template.write_text(
        """
[run]
name = "baseline_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [11, 12, 13]
eval_seeds = [21, 22, 23]

[baseline.static_layout]
type = "rectangular"
n_rows = 6
n_cols = 7
spacing_along = 457.2
spacing_across = 1371.6
speed = 5.0
heading_rad = 0.0
length = 150.0
beam = 20.0
origin = [0.0, 0.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
            "--convoy-profile",
            "convoy_layout_mixed_1",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    layout = cfg["baseline"]["static_layout"]
    assert layout["fleet_profile"] == "mixed_convoy_v1"
    assert layout["fleet_seed"] == 1947


def test_generate_run_config_injects_convoy_profile_into_baseline(tmp_path: Path) -> None:
    template = tmp_path / "baseline_template.toml"
    output = tmp_path / "baseline_generated.toml"
    template.write_text(
        """
[run]
name = "baseline_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [11, 12, 13]
eval_seeds = [21, 22, 23]

[baseline.static_layout]
type = "rectangular"
n_rows = 3
n_cols = 4
spacing_along = 100.0
spacing_across = 100.0
speed = 4.0
heading_rad = 0.0
length = 120.0
beam = 18.0
origin = [0.0, 0.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
            "--convoy-profile",
            "convoy_layout_2",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    layout = cfg["baseline"]["static_layout"]
    assert layout["type"] == "rectangular"
    assert layout["n_rows"] == 7
    assert layout["n_cols"] == 7
    assert layout["spacing_along"] == 731.52
    assert layout["spacing_across"] == 914.4


def test_generate_run_config_injects_common_convoy_profile_fields_into_rl_actions_without_flattening_geometry(tmp_path: Path) -> None:
    template = tmp_path / "rl_template.toml"
    output = tmp_path / "rl_generated.toml"
    template.write_text(
        """
[run]
name = "rl_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [301, 302, 303]
eval_seeds = [401, 402, 403]

[training]
episodes = 10
epsilon = 0.25
epsilon_decay = 0.99
epsilon_min = 0.02
alpha = 0.1
seed = 7

[rl]

[[rl.actions]]
name = "action_1"
type = "rectangular"
complexity_cost = 1.0
n_rows = 3
n_cols = 4
spacing_along = 100.0
spacing_across = 100.0
speed = 4.0
heading_rad = 0.0
length = 120.0
beam = 18.0
origin = [0.0, 0.0]

[[rl.actions]]
name = "action_2"
type = "rectangular"
complexity_cost = 1.1
n_rows = 3
n_cols = 4
spacing_along = 90.0
spacing_across = 90.0
speed = 4.0
heading_rad = 0.0
length = 120.0
beam = 18.0
origin = [0.0, 0.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
            "--convoy-profile",
            "convoy_layout_1",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    actions = cfg["rl"]["actions"]
    assert len(actions) == 2
    assert actions[0]["name"] == "action_1"
    assert actions[1]["name"] == "action_2"
    assert actions[0]["n_rows"] == 6
    assert actions[0]["n_cols"] == 7
    assert actions[0]["spacing_along"] == 100.0
    assert actions[0]["spacing_across"] == 100.0
    assert actions[1]["n_rows"] == 6
    assert actions[1]["n_cols"] == 7
    assert actions[1]["spacing_along"] == 90.0
    assert actions[1]["spacing_across"] == 90.0
    assert actions[0]["fleet_profile"] == "freighter_heterogeneous_v1"
    assert actions[1]["fleet_profile"] == "freighter_heterogeneous_v1"
    assert actions[0]["fleet_seed"] == 1945
    assert actions[1]["fleet_seed"] == 1945


def test_generate_run_config_preserves_rl_builder_mode_when_injecting_convoy_profile(tmp_path: Path) -> None:
    template = tmp_path / "rl_builder_template.toml"
    output = tmp_path / "rl_builder_generated.toml"
    template.write_text(
        """
[run]
name = "rl_default"
output_root = "results/runs"

[simulation]
t_max = 400.0
n_trials_per_seed = 40
max_hits_per_torpedo = 1

[splits]
train_profiles = ["P01"]
eval_profiles = ["P02"]
train_seeds = [301, 302, 303]
eval_seeds = [401, 402, 403]

[training]
episodes = 10
epsilon = 0.25
epsilon_decay = 0.99
epsilon_min = 0.02
alpha = 0.1
seed = 7

[rl]

[rl.builder]
enabled = true
base_n_rows = 3
base_n_cols = 4
speed = 4.0
heading_rad = 0.0
length = 120.0
beam = 18.0
origin = [0.0, 0.0]
layout_families = ["rectangular", "staggered"]

[rl.builder.spacing_along_options]
compact = 90.0
loose = 110.0

[rl.builder.spacing_across_options]
compact = 80.0
loose = 100.0

[rl.builder.family_complexity]
rectangular = 1.0
staggered = 1.2

[rl.builder.spacing_along_complexity]
compact = 0.0
loose = 0.1

[rl.builder.spacing_across_complexity]
compact = 0.0
loose = 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.generate_run_config",
            "--template",
            str(template),
            "--output",
            str(output),
            "--split-seed",
            "42",
            "--n-total",
            "30",
            "--n-train",
            "20",
            "--convoy-profile",
            "convoy_layout_mixed_1",
        ],
        check=True,
    )

    cfg = _read_toml(output)
    builder = cfg["rl"]["builder"]
    assert builder["enabled"] is True
    assert builder["base_n_rows"] == 6
    assert builder["base_n_cols"] == 7
    assert builder["fleet_profile"] == "mixed_convoy_v1"
    assert builder["fleet_seed"] == 1947
    assert builder["spacing_along_options"]["compact"] == 90.0
    assert builder["spacing_across_options"]["compact"] == 80.0
