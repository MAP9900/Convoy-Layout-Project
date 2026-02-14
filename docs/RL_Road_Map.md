RL Road Map

Goal
Build a realistic, reproducible RL workflow for convoy layout optimization with sim-aligned visuals and risk-aware evaluation.

Phase 0: Readiness (pre-RL)
- Lock baseline scenarios, seeds, and sim settings.
- Decide static vs dynamic sim as the training default.
- Confirm visuals and sim share the same hit logic.
- Set motion model defaults (independent motion, zig-zag settings).
- Decide whether hit-driven slowdown is enabled for training.

Phase 1: Baselines (non-RL)
- Create heuristic baselines (rectangular, staggered, fixed spacing tweaks).
- Run Monte Carlo evaluations for each baseline.
- Track expected hits, VaR, CVaR, and variance.
- Save results as the initial reference point.

Phase 2: RL Problem Definition
- Agent: start with defender-only.
- Action space: layout parameter changes (spacing, rows/cols, offsets).
- Observation: layout metrics + attack parameters + environment flags.
- Reward: negative expected hits with optional CVaR penalty.
- Episode length: 1-step layout selection or short horizon with adjustments.

Phase 3: Attacker Library (fixed threat model)
- Build and maintain attacker profiles in `convoy_sim/attack_profiles.py`.
  - Current implementation is a flat schema with sim-native fields:
  - `mode`, `u_pos`, `n`, `speed`, `max_run_time`, `base_bearing_rad`, `spread_rad`, `bearing_rad`, `lateral_spacing`, `launch_delay_s`, `salvo_interval_s`.
  - Use `AttackProfile.build_torpedoes(...)` to generate torpedoes for simulation.
- Use weighted sampling with `AttackProfileLibrary.sample_profile(...)` during training.
- Reserve a held-out set for evaluation.
- Integration note: RL wrapper still needs explicit reset-time profile sampling wiring.

Phase 4: RL Training
- Train on the attacker library distribution.
- Log expected hits and risk metrics per episode.
- Check convergence vs baselines.
- Use periodic evaluation on held-out attacks.

Phase 5: Risk-Aware Evaluation
- Re-run Monte Carlo at higher trial counts.
- Report mean, VaR, CVaR, and variance.
- Compare to baseline layouts.

Phase 6: Optional ML Add-ons
- Train a surrogate model to predict expected hits and CVaR.
- Use surrogate for fast screening of layout candidates.
- Pretrain RL policy from the surrogate, then fine-tune in sim.

Phase 7: Two-Agent or Self-Play (optional)
- Add attacker agent if needed.
- Start from fixed library, then co-evolve.
- Keep a stable evaluation set to avoid "moving target" drift.

Deliverables
- Reproducible training configs (seeded).
- Baseline and RL results with MC + VaR/CVaR.
- Visual verification runs for selected policies.
- Final policy summary and stress-test report.

Notes
- Avoid single fixed attack profiles (trivial solutions).
- Prefer risk-aware metrics for realism.
- Keep sim + viz aligned for debugging and validation.
