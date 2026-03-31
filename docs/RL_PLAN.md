# RL Plan

Last updated: 2026-03-30

## Current RL Status

- Current canonical RL is a tabular one-step selector over predefined `[[rl.actions]]`.
- It is useful as a proof-of-concept and artifact-contract anchor.
- Next phase is moving to a constrained multi-step layout-construction policy.

## Goals

- Make RL meaningfully control convoy design.
- Keep decisions historically plausible via explicit hard boundaries.
- Preserve canonical run artifacts for apples-to-apples comparisons.

## Control Surface (Planned)

### Layout Geometry Controls

- Layout family/mode (`rectangular`, `staggered`, `custom_row_counts`).
- Row pattern vectors (e.g., `[4,5,5,4]`).
- `spacing_along`, `spacing_across`.
- Row offsets/skew (bounded).

### Ship-Class Placement Controls

- Assign class by row/slot (freighter/tanker/escort/decoy).
- Fixed fleet composition constraints.
- Protected-core/perimeter escort patterns.

### Kinematic/Doctrine Controls

- Convoy speed within bounded realistic range.
- Zig-zag on/off and parameters (amplitude, period, phase).
- Optional route-leg changes (bounded schedule set).

### Future Optional Controls

- Limited per-ship overrides for high-value or escort vessels only.
- Bounded local offsets / slot swaps with strict realism constraints.

## Constraint Model (Hard Boundaries)

- Minimum separation and collision safety.
- Maximum convoy footprint (length/width).
- Min/max rows and ships-per-row.
- Fleet composition counts by class.
- Station-keeping feasibility checks.
- Optional escort-zone/defensive doctrine constraints.

Invalid actions/layouts should be masked where possible; otherwise penalized heavily.

## Reward Design (Planned)

- Primary: minimize hit impact.
- Base form:
  - `reward = - expected_hits`
- With risk guardrail:
  - `reward = - expected_hits - lambda_risk * CVaR_90_proxy`
- With value weighting:
  - `reward = - value_lost_weighted - lambda_risk * CVaR_90_proxy - lambda_complexity * layout_complexity`

## Algorithm Roadmap

### Phase A: Env Upgrade

- Move from one-step to multi-step episode dynamics in RL env/wrapper.
- Add compact observation vector for geometry/class/threat context.

### Phase B: Learner Upgrade

- Replace tabular selector with production learner:
  - PPO preferred for flexible policy learning.
  - Keep tabular mode as fallback for regression sanity.

### Phase C: Evaluation Hardening

- Compare against static and heuristic baseline on identical eval seeds/splits.
- Keep artifact schema unchanged:
  - `config_resolved.yaml`
  - `metrics_summary.json`
  - `per_profile_metrics.csv`
  - `run_manifest.json`
  - `checkpoints/policy_latest.json`

## Acceptance Gates

- Gate 1: RL >= static baseline on matched eval seeds.
- Gate 2: RL matches or beats heuristic baseline on matched seeds.
- Gate 3: Stable performance across at least 3 eval seed sets.
- Gate 4: No guardrail regression (`CVaR_90` threshold respected).

## Experiment Logging Contract

For each RL test:
- record config path and exact generation command
- record run directory
- record seed/split sets
- log selected policy and main metrics
- summarize conclusion and caveats in:
  - `docs/RESULTS_LOG.md`
  - `docs/OPTIMIZATION_LOG.md`
