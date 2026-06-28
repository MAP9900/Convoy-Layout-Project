# Simulation Physics

This document describes the physical and behavioral model implemented by the Python simulator. It focuses on what the simulation represents: ships, torpedoes, convoy motion, U-boat motion, firing geometry, noise, and current modeling limits.

Workflow commands live in `docs/REPRODUCING.md`. Script entrypoints live in `docs/SCRIPTS.md`. The codebase map lives in `docs/PROJECT_MAP.md`. RL and POMDP methodology live in `docs/REINFORCEMENT_LEARNING.md` and `docs/POMDP.md`.

## Scope

The simulator models WWII-style convoy defense against torpedo attacks on a 2D Euclidean plane.

- Coordinates: meters
- Time: seconds
- Heading: radians unless a plot/report explicitly converts to degrees
- Core abstraction: deterministic geometry and kinematics with optional realism/noise overlays

The simulator is not a full naval hydrodynamics model, sonar model, escort tactics model, or historical fire-control computer. Those are modeled only through bounded abstractions where needed for experiments.

## Core Modules

- Geometry/math: `convoy_sim/geometry.py`
- Entities: `convoy_sim/entities.py`
- Layout generation: `convoy_sim/layouts.py`, `convoy_sim/ship_catalog.py`
- Convoy motion/dynamics: `convoy_sim/dynamics.py`
- Attack profiles and torpedo construction: `convoy_sim/attack_profiles.py`
- Feasibility/risk context: `convoy_sim/feasibility.py`
- Observation and movement realism: `convoy_sim/realism.py`
- Fire-control baseline: `convoy_sim/fire_control.py`
- Simulation kernels: `convoy_sim/simulation.py`
- Visualization helpers: `convoy_sim/viz.py`, `convoy_sim/viz_attack.py`

## Coordinate Frames

The global frame is a fixed 2D Cartesian coordinate system.

Convoy motion can also use a convoy frame:

- convoy origin
- convoy heading
- per-ship offsets from the convoy origin

Two motion semantics are used in different contexts:

- `rigid`: ships preserve fixed offsets while the convoy translates/rotates
- `independent`: each ship is integrated independently using the current convoy heading/speed schedule

Dynamic stepping uses a positive `dt`. Smaller `dt` improves temporal precision at higher runtime cost.

## Ship Model

Each `Ship` has:

- `id`
- `position`
- `speed`
- `heading_rad`
- `length`
- `beam`
- `ship_class`
- `value_weight`
- optional `hit_radius`

Default hit geometry uses an effective circular hit radius:

```text
sqrt((length / 2)^2 + (beam / 2)^2)
```

If `hit_radius` is set, that override is used.

Default class dimensions and values are defined in `convoy_sim/ship_catalog.py`. Layout profiles can override dimensions and classes at generation time.

## Torpedo Model

Each `Torpedo` has:

- `id`
- `launch_position`
- `speed`
- `heading_rad`
- `max_run_time`
- `launch_delay`
- `is_dud`
- optional `launch_heading_rad`
- optional `gyro_turn_distance_m`

Behavior:

- waits at the launch position until `launch_delay`
- travels for `max_run_time` after launch
- exits on `launch_heading_rad` when provided, otherwise on `heading_rad`
- after `gyro_turn_distance_m`, snaps to final `heading_rad`
- remains fixed-speed and fixed-course after gyro deflection

The simulator does not currently model continuous torpedo steering, acoustic homing, wake-following, depth setting, or fuze/depth failures.

## Layout Generation

Primary layout builders:

- `make_rectangular_convoy`
- `make_staggered_convoy`
- `make_hexagonal_convoy`

Supported layout controls include:

- row/column counts
- along/across spacing
- convoy speed and heading
- per-cell ship-class maps
- per-cell ship overrides
- optional Gaussian positional jitter

Scenario profiles live in `scenarios/convoy_profiles.py`.

## Convoy Dynamics

`ConvoyKinematics` combines:

- `RoutePlan`: piecewise heading and optional speed by leg
- `ZigZagPlan`: sine or triangle heading perturbation

Command latency is represented with `command_latency_s`, which shifts effective control time:

```text
effective_t = max(0, t - command_latency_s)
```

This approximates delayed order execution for convoy maneuvers.

`HitSlowdownSpec` can reduce post-hit ship speed in dynamic visual/step contexts where hit state is advanced over time.

## U-Boat Motion

`UBoatMotionPlan` supports:

- static or moving mode
- initial position, heading, and speed
- leg-based motion
- optional turn-rate and acceleration bounds
- `state_at(t)` and `position_at(t)`

The U-boat motion model is deterministic. It provides the submarine pose sampled by attack-profile construction at each torpedo launch time. It does not model hydrodynamic turn circles or tube-training mechanics.

## Launch Geometry

The simulator separates:

- U-boat center position and bow heading at launch time
- torpedo launch origin
- intended attack centerline
- final post-gyro torpedo course

Launch process:

1. `UBoatMotionPlan.state_at(t)` determines submarine center position and bow heading.
2. `launch_from` chooses bow or center launch origin.
3. `base_bearing_rad` defines the attack centerline.
4. spread doctrine assigns per-torpedo final heading offsets.
5. each torpedo leaves along the bow heading, travels straight for `gyro_straight_run_m`, then turns to final heading.

Relevant fields:

- `launch_from = bow|center`
- `sub_length_m`
- `sub_beam_m`
- `max_bow_offset_deg`
- `gyro_straight_run_m`

## Firing Stability

Attack profiles require stable U-boat firing by default:

- `require_stable_u_boat_during_salvo = true`
- heading drift across the salvo window is rejected when too large
- launch-time turn rate is bounded by `max_u_boat_turn_rate_at_fire_rad_s`

This models the doctrine assumption that salvos are fired from a steady boat, not while actively swinging through a turn. The guardrail can be disabled for controlled diagnostics or counterfactual cases.

## Spread Doctrines

Fan-mode attack profiles support three spread doctrines:

- `longitudinal`: all torpedoes share the same final heading; spacing comes from launch timing and launch-position changes.
- `uniform_divergent`: torpedo final headings are evenly spaced across total fan width `spread_rad`.
- `explicit_divergent`: each torpedo receives an authored final offset from the centerline.

Important boundary:

- spread doctrine controls torpedo final headings
- it does not command the submarine to turn during the salvo
- non-steady firing windows are rejected unless the stability guardrail is deliberately disabled

Backward compatibility:

- legacy fan profiles with `spread_rad > 0` behave as `uniform_divergent`
- legacy fan profiles with `spread_rad == 0` behave as `longitudinal`

`notebooks/torpedo_firing_doctrine_comparison.ipynb` provides visual checks for these semantics.

## Fire-Control Lite

`convoy_sim/fire_control.py` implements a deterministic attacker-side firing solution baseline.

Inputs:

- U-boat position
- U-boat bow heading
- noisy attacker observation of convoy bearing, range, course, speed, and uncertainty

Outputs:

- centerline bearing
- spread width
- salvo size
- G7a speed setting
- torpedo speed and run time
- estimated target point and metadata

Behavior:

- chooses speed setting from estimated range
- computes a coarse intercept lead from estimated convoy lateral motion
- widens spread with observation uncertainty and range
- clips centerline to the allowed bow arc
- can emit a standard `AttackProfile`

Boundary:

- partial observability answers "what does the attacker know?"
- `fire_control_lite` answers "given that estimate, how does the attacker shoot?"

`convoy_sim/pomdp_fire_control.py` uses this bridge for POMDP v2 by rebuilding firing solutions from noisy observations rather than reusing a VAE candidate's original bearing/spread.

## Partial Observation

`build_attacker_observation(...)` creates attacker-facing estimates:

- convoy bearing and range
- convoy heading and speed
- contact count
- contact-detection fraction
- formation width/depth
- contact density
- class-confidence counts
- environment snapshot
- observation quality sigmas

Observation presets:

- `good_contact`
- `baseline_night`
- `poor_contact`

Poorer presets increase uncertainty and reduce visible contacts. These presets support POMDP and fire-control evaluation; they do not directly change torpedo collision geometry.

## Ship Movement Realism

`ShipMovementRealismConfig` supports bounded overlays:

- position jitter
- heading jitter
- deviation cap
- optional slot swaps
- class-dependent cohesion scaling

The overlay perturbs layouts before simulation/evaluation so the same nominal convoy can produce stochastic realizations.

## Weapon Noise

`NoiseModel` supports:

- heading noise
- launch-delay noise
- speed noise
- dud probability

`apply_noise_to_torpedoes(...)` perturbs torpedo heading, speed, launch delay, and dud state. Speed is kept positive and launch delay is kept nonnegative.

## Simulation Kernels

Static simulation:

- evaluates ship/torpedo geometry without convoy motion over time
- supports capped hits per torpedo

Dynamic simulation:

- advances convoy and torpedo state over time
- respects launch delays
- tracks hit events through `DynamicHitState` / `HitEvent`

Monte Carlo wrappers exist for repeated stochastic evaluation, but experiment-level Monte Carlo protocol belongs in `docs/REPRODUCING.md`, `docs/REINFORCEMENT_LEARNING.md`, and `docs/POMDP.md`.

## What Affects Collision Physics

Directly affects hit geometry:

- ship position, length, beam, and hit radius
- torpedo launch position, speed, heading, launch delay, run time
- U-boat launch pose and bow/gyro settings
- convoy dynamics and ship movement overlays
- weapon noise and dud state
- `max_hits_per_torpedo`

Does not directly affect hit geometry today:

- time of day
- visibility
- sea state
- detection risk scale
- objective weights

Those fields affect feasibility, observation quality, scoring, or experiment interpretation.

## Current Limits

- No depth/fuze proxy yet.
- No acoustic, wake-homing, or guided torpedoes.
- Torpedoes are straight-run after the gyro transition.
- Environment does not directly alter collision geometry.
- Escort search/reaction behavior is not modeled as an active tactical agent.
- Current U-boat motion is deterministic kinematics, not a hydrodynamic submarine model.
- Full historical TDC/fire-control procedure is approximated only through `fire_control_lite`.

## Related Docs

- Reproduction/runbook: `docs/REPRODUCING.md`
- Script index: `docs/SCRIPTS.md`
- Codebase map: `docs/PROJECT_MAP.md`
- POMDP/fire-control evaluation: `docs/POMDP.md`
- RL methodology: `docs/REINFORCEMENT_LEARNING.md`
