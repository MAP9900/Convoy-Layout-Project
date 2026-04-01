# TODO

Last updated: 2026-04-01

## Current Workflow (Canonical)

- Generate baseline config:
  - `python -m experiments.generate_run_config --template configs/templates/baseline.template.toml --output configs/baseline/default.toml --convoy-profile convoy_layout_1 --split-seed 1945 --n-total 30 --n-train 20 --train-seeds 1939,1940,1941 --eval-seeds 1942,1943,1944`
- Generate RL config:
  - `python -m experiments.generate_run_config --template configs/templates/rl.template.toml --output configs/rl/default.toml --convoy-profile convoy_layout_1 --split-seed 1945 --n-total 30 --n-train 20`
- Run baseline:
  - `python -m experiments.run_baseline_suite --config configs/baseline/default.toml`
- Run RL:
  - `python -m experiments.run_rl_train --config configs/rl/default.toml`

## Now

- [x] Protocol v2 foundation: freeze current stack as V1 baseline/RL reference and start V2 realism track.
- [x] Add moving U-boat support (deterministic motion first) with config toggle:
  - `u_boat_mode = static | moving`
  - maintain backward compatibility for static profiles.
- [x] Define U-boat motion schema in attack profiles:
  - initial position/heading/speed
  - leg/waypoint plan
  - launch timing/window fields
  - optional turn-rate / kinematic bounds
- [x] Integrate U-boat motion into attack proposal + simulation paths (not just plotting).
- [x] Add diagnostics/visuals for U-boat launch context in canonical run manifests.
- [x] Add tests for motion realism:
  - deterministic replay under fixed seeds
  - kinematic-feasibility checks
  - regression vs static mode when `u_boat_mode=static`
- [x] Re-lock benchmark protocol for V2 after movement integration (new seed/split stamp).
- [x] Align eval seeds between baseline and RL for strict apples-to-apples comparison.
- [x] Run baseline + RL pair with matched eval seeds.
- [x] Record V2-Realism Test 1 in `docs/RESULTS_LOG.md` and `docs/OPTIMIZATION_LOG.md`.
- [ ] Define RL promotion threshold (`expected_hits` primary, `CVaR_90` guardrail).
- [ ] Define GenAI attack-profile dataset schema (context + attack vector + outcomes).
- [ ] Add reproducible synthetic attack-dataset generation config/script (fixed seeds + manifest metadata).
- [ ] Generate first dataset snapshot for GenAI model training.

## Next

- [ ] Replace tabular RL selector with stronger learner while preserving artifact schema.
- [ ] Add constrained RL layout-builder action space:
  - row pattern vectors (e.g., `4,5,5,4`)
  - spacing controls
  - ship-class placement controls
  - convoy speed and zig-zag controls
- [ ] Define and enforce hard layout boundaries in config (separation, footprint, class counts, feasibility).
- [ ] Add value-weighted objective support (e.g., tanker > freighter) with risk guardrail.
- [ ] Expand baseline search space beyond spacing-only (bounded, interpretable knobs).
- [ ] Add confidence intervals or repeated-seed summaries to run metrics.
- [ ] Add run-to-run comparator script for baseline vs RL outputs.
- [ ] Add CI smoke checks for canonical config generation + baseline + RL entrypoints.
- [ ] Train/evaluate conditional VAE generative adversary baseline on synthetic attack-profile data.
- [ ] Add generation-time feasibility filtering and duplicate controls for sampled attack profiles.
- [ ] Integrate generated attack profiles into canonical defender evaluation flow.
- [ ] Add generated-vs-scripted comparison outputs for `expected_hits`, `CVaR_90`, and `p_hit_ge_1`.

## Later

- [ ] Attacker RL and alternating training protocol.
- [ ] Add limited per-ship override controls for high-value/escort ships (bounded offsets/swap rules only).
- [ ] Multi-agent robustness and domain-randomization matrix.
- [ ] Add latent-space diagnostics/visualization for generated attack-profile families and failure regions.
- [ ] Add diffusion-model variant as optional stretch after VAE baseline is stable.

## RL Design Docs

- [ ] Keep `docs/RL_PLAN.md` updated whenever RL state/action/reward/constraint design changes.

## Historical Realism Backlog

- [x] Add bounded station-keeping randomness:
  - per-ship positional jitter
  - heading jitter
  - class-dependent cohesion (escort tighter than freighter)
- [x] Add torpedo imperfection model:
  - heading error / drift over time (bend-like behavior via controlled error, not arbitrary curves)
  - speed variance
  - dud probability
  - depth/fuze error proxy (deferred)
- [x] Add ship-movement realism progression:
  - formation-level motion as primary control
  - limited per-ship deviation overlays (bounded offsets/swap rules)
  - avoid unconstrained fully independent ship control by default
- [x] Add command/response latency:
  - delayed execution of heading/zig-zag orders
  - class-dependent response lag (future refinement)
- [x] Add environment-driven observability:
  - visibility, sea state, time-of-day effects on detection and attack setup quality
- [ ] Add hard realism envelopes:
  - separation floors
  - turn-rate limits
  - class-based speed bounds
  - convoy footprint caps
- [ ] Add attack timing doctrine:
  - night surface approach vs day submerged approach modes
  - mode-specific speed/detection constraints
- [ ] Add approach-geometry doctrine:
  - abeam/intercept preference
  - stern-chase effectiveness penalties where historically appropriate
- [ ] Add salvo doctrine realism:
  - bow/stern tube limits
  - reload delays
  - partial-salvo decisions
  - abort behavior
- [ ] Add weapon-era configuration variants:
  - straight-run
  - pattern-running
  - acoustic torpedoes (era-gated config)
- [ ] Add escort behavior model:
  - sector screen assignment
  - sweep/search behavior
  - reaction delay after contact
  - disruption windows after counterattack
- [ ] Add detection stack model:
  - visual / radar / sonar / HF-DF probability components
  - range/weather/light conditioning
- [ ] Add convoy discipline factors:
  - blackout/radio-silence effects
  - zig-zag compliance variability
  - signaling delays
- [ ] Add straggler dynamics:
  - damage/slowdown induced formation dropouts
  - elevated vulnerability for stragglers
- [ ] Add sea-room and routing constraints:
  - lane width / route corridor limits
  - hazard/chokepoint effects
- [ ] Add proficiency variability:
  - attacker and escort skill distributions
  - aiming/detection/coordination variance
- [ ] Add persistence/fatigue limits:
  - battery/fuel/attack-window constraints over long engagements
- [x] Add intelligence uncertainty model:
  - attacker decisions based on partial/noisy belief state
  - no direct use of full true convoy state for attacker/generative conditioning
