


# Sample README:

# # Convoy Layout Project — Development Checklist

# This is the roadmap for building the minimal simulation (Phase 1) and realism extensions (Phase 2).  
# Each box can be checked off as tasks are completed. The idea is to keep this project modular and reusable for both defense (layouts) and attack (salvos).

# ---

# ##  Phase 1 — Minimal Simulation (MVP)

# - [ ] **Scaffold repo**
#   - Create folders: `convoy/`, `notebooks/`, `scripts/`, `tests/`
#   - Add empty files: `layouts.py`, `attacks.py`, `physics.py`, `simulate.py`, `metrics.py`
#   - Commit: "scaffold project"

# - [ ] **Physics math & unit tests**
#   - Plan 3 tests: Static line test, Time-gating test, Max-run test
#   - Implement collision check for ship vs torpedo (continuous-time closest approach)
#   - Ensure all tests pass

# - [ ] **Scenario A spec**
#   - 8×8 grid, 1000×600 yd spacing, heading 0°, ship speed 8 kn, beam 15 yd
#   - Attack: 4 torpedoes, 8° spread, 2s between shots, 30 kn speed, 4000 yd max run, abeam start
#   - Document expected outputs: hits, hit flags, hit times

# - [ ] **Episode runner**
#   - Build ships (layout)
#   - Build torpedo rays (attack)
#   - Run collisions
#   - Return hits, torpedo flags, ship IDs
#   - Validate on Scenario A (deterministic run)

# - [ ] **Monte Carlo wrapper**
#   - Add randomness (aim error, small bearing jitter)
#   - Run N episodes, return mean hits, std, quantiles
#   - Support CRN (common random numbers) for comparisons
#   - Convergence check: Scenario A stable at N=2000 (±5%)

# - [ ] **Baseline results**
#   - Run Scenario A (deterministic + noisy)
#   - Save metrics & at least one episode plot

# ---

# ##  Phase 2 — Calibration & Realism

# - [ ] **Add distributions**
#   - Attack params: torpedo speed (Normal), aim error, salvo size (categorical)
#   - Episode config samples params per run

# - [ ] **Layout families**
#   - Add `"staggered"` (offset = sx/2)
#   - Add `"hex"`
#   - Add `"jitter"` with clamp to maintain min separation
#   - Enforce minimum separation rule

# - [ ] **Torpedo realism toggles**
#   - Dud probability
#   - Δt noise per shot
#   - Speed noise %
#   - Keep defaults = off for MVP

# - [ ] **Approach variations**
#   - Bearing mixture (e.g., abeam ±15°)
#   - Escort exclusion mask (forbidden zones)

# - [ ] **Extended metrics**
#   - q90, q95, CVaR@90
#   - Convoy footprint area
#   - Layout complexity score (unique offsets, jitter)

# - [ ] **Scenario set**
#   - **A-Robust**: Scenario A + distributions
#   - **B-Angles**: Approach mixtures
#   - **C-Timing**: Shot spacing sweeps
#   - **D-LayoutFamily**: Rect vs staggered vs hex

# - [ ] **Analysis runs**
#   - Spacing sweep: sx ∈ [600..1400], sy ∈ [400..1000]
#   - Spread angle sweep: α ∈ [0°..15°]
#   - Shot spacing sweep: Δt ∈ [0..12s]
#   - Layout A/B test: Rect vs staggered (equal footprint)

# ---

# ##  Shared Architecture (for ML later)

# - [ ] **Dataclasses/configs**
#   - `ConvoyLayout`, `ShipParams`, `AttackPlan`, `WorldConfig`, `EpisodeConfig`, `SimResult`

# - [ ] **Pure sim function**
#   - `simulate(layout, attack, world, episode) -> SimResult`
#   - `evaluate(layout, attack, world, episode, n_mc) -> Metrics`

# - [ ] **Agents**
#   - `DefenderAgent` → emits ConvoyLayout
#   - `AttackerAgent` → emits AttackPlan

# - [ ] **Match driver**
#   - `run_match(defender, attacker, world, episode_cfg, n_mc) -> Metrics`
#   - Logs configs, seeds, metrics for every run

# ---

# ##  Exit Criteria (Phase 2)

# - Deterministic Scenario A produces plausible hits & plan-view plot
# - Monte Carlo Scenario A stable mean ± CI across seeds
# - Physics unit tests pass
# - First sensitivity sweep (spacing or spread) shows sensible trend
# - Layout family (rect vs stagger) comparison produces explainable difference