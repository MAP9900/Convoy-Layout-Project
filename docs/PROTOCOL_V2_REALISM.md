# Protocol V2-Realism

Status: active as of 2026-04-01.

Primary technical reference: `docs/SIM_FEATURES.md`.

## Boundary

- `Protocol V1` is frozen as reference behavior.
- `Protocol V2-Realism` is the active implementation track.
- RL redesign and GenAI model training are out of scope for this phase.

## Defaults

- `u_boat_mode = moving` is the default.
- `u_boat_mode = static` remains supported for compatibility/regression checks.
- Torpedo imperfection includes:
  - heading noise
  - launch delay noise
  - speed variance
  - dud probability
- Depth/fuze proxy remains deferred.

## Implemented Realism Features

- Deterministic U-boat motion plan with optional turn-rate/acceleration bounds.
- Time-aware launch geometry integration for moving U-boat.
- Bow-fire launch constraint:
  - launch position tied to submarine heading at each torpedo launch time
  - configurable bow tube arc limit (`max_bow_offset_deg`)
  - launch point configurable (`center` or `bow`), defaulting to bow-point realism
- Gyro-angle torpedo logic:
  - torpedo exits along the submarine bow heading
  - after a short straight run, it turns to a preset final course
  - fan spread is now produced by per-torpedo gyro deflection rather than forcing the submarine to yaw through the spread
- Firing-stability constraint:
  - U-boat salvos are rejected by default if the boat is materially turning during the firing window
  - opt-out remains available for explicit what-if or backward-compatibility scenarios
- Attacker partial-observability layer (noisy estimated bearing/range/course/speed/contacts + environment).
- Ship movement realism overlay:
  - bounded position/heading jitter
  - class-dependent cohesion scaling
  - bounded optional slot swaps
- Command/response latency in convoy kinematics.

## Validation

- Unit tests in `tests/test_realism_v2.py` cover:
  - motion determinism and bounds
  - moving/static compatibility
  - observation reproducibility
  - ship movement overlay bounds
  - torpedo imperfection sampling incl. dud
- Canonical smoke tests in `tests/test_canonical_entrypoints.py` validate baseline/RL artifact production with V2 defaults.
- V2 run records:
  - baseline: `results/runs/baseline/20260401_203150_baseline_test1`
  - rl: `results/runs/rl/20260401_203331_rl_test1`
