Block C Readiness Checklist (ML/RL)

Codex Checkable Items 
- [DONE] Determinism: audit RNG propagation and add repeatability tests.
- [DONE] Time stepping: add a single dt helper or guardrails for dynamics/tactics.
- [DONE] API clarity: clean static vs dynamic simulation entry points.
- [DONE] Outcome invariants: add regression tests for static vs dynamic equivalence.
- [DONE] Trial schema: implement a standard JSON record and helper.
- [DONE] Scenario serialization: add to_dict/from_dict hooks where feasible.
- [DONE] Feature snapshot logging: record layout metrics + threat/plan params per trial.
- [DONE] Reward sign conventions: codify in helpers and add doc comments.
- [DONE] Batch evaluation hooks: reusable batch runner utilities.
- [DONE] Early-stop/budget controls: optional caps in optimizer loops.
- [DONE] Noise model consistency: ensure all sim paths use same noise hooks.
- [DONE] Episode-style step API: minimal wrapper for RL compatibility.
- [DONE] Action space mapping: discrete indexing for layouts/plans.
- [DONE] Observation schema: stable, versioned dict schema.

Manual Checkable Items
- [TODO] Canonical trial JSON schema fields/size limits.
- [TODO] Which environment variables are stochastic vs fixed.
- [TODO] Preferred objective weights for defender vs attacker.
- [TODO] Fidelity vs speed tradeoffs for ML training.
- [TODO] RL framing: single-agent vs two-agent/self-play.
- [TODO] Dataset sizes and compute budgets.

Suggested Next Steps (proposed batch)
- Decide a canonical trial JSON schema.
- Pick one sim pathway (static or dynamic) as the ML default.
- Approve a minimal episode wrapper for both sides.
