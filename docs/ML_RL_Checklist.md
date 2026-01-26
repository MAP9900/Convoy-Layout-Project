Block C Readiness Checklist (ML/RL)

Core Sim Correctness
- Determinism: all stochastic paths use explicit RNGs and propagate seeds.
- Time stepping: consistent dt usage across dynamics, tactics, and feasibility.
- API clarity: static vs dynamic simulation paths are cleanly separated.
- Outcome invariants: hit/value metrics match across static/dynamic when equivalent.

Data & Logging for ML/RL
- Trial schema: standardized JSON record (seed, config, policy/plan, outcomes).
- Scenario serialization: config can be fully reproduced from a JSON blob.
- Feature snapshot: layout metrics + threat/plan parameters are recorded per trial.

Objectives & Rewards
- Unified loss/utility functions for defender and attacker.
- Optional tail-risk terms (VaR/CVaR) can be toggled and logged.
- Reward sign conventions are documented and consistent.

Performance & Scaling
- Batch evaluation hooks for dataset generation.
- Early-stop or confidence bounds available for search loops.
- Runtime caps or budget controls in optimizers.

Robustness & Uncertainty
- Noise models apply consistently to static, dynamic, and plan-based sims.
- Environment variability has a standard sampler/conditioning interface.
- Sensitivity toggles are centralized and logged per run.

RL Integration Hooks
- Episode-style step API (even minimal) exists for each side.
- Action spaces are explicit and mappable to discrete indices.
- Observation schema is stable and documented.

Suggested Next Steps
- Decide a canonical trial JSON schema.
- Add a minimal episode wrapper for defender and attacker.
- Add a dataset export helper used by all runners.
