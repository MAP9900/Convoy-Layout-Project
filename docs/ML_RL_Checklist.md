Block C & E Readiness Checklist (ML/RL)

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
- [DONE] Determinism parity across static/dynamic runs for matched setups.
- [DONE] Noise regression: zero-noise matches baseline across multiple scenarios.
- [DONE] Attack visualization math: hit markers appear only after launch delay.
- [DONE] Visual vs sim alignment: hit markers/torpedo stops share sim hit logic.
- [DONE] Zig-zag motion realism: smooth heading oscillation with forward integration.
- [DONE] Dynamic hit persistence: hits persist across frames after impact.
- [DONE] Optional hit slowdown: exponential decay model implemented (toggleable).
- [DONE] RL wrapper step: stable observation schema + reward sign consistency.
- [DONE] Serialization round-trip: scenarios + noise + policy/plan configs.


Manual Checkable Items
- [TODO] Canonical trial JSON schema fields/size limits.
    -Need to define standard consistent JSON Record (layout params, attack params, hits, seed, timestamps, etc.)
    -Current Framework:
        - `convoy_sim/trial_records.py`: `make_trial_record(...)` used in `convoy_sim/defender_policy.py`.
        - `convoy_sim/rl_wrapper.py`: episode + observation serialization.
        - `scenarios/scenario_base.py`: scenario `to_dict/from_dict`.
        - `convoy_sim/simulation.py`: Monte Carlo outputs (`hits_per_trial`, `expected_hits`, `var_hits`, etc.).
- [TODO] Which environment variables are stochastic (change randomly) vs fixed.
    -Dud probability 
    -Convoy Speed (?)
    -ETC
- [TODO] Preferred objective weights for defender vs attacker.
    -Measure rewards as number of hits or values. Values comes from hits on specific types of ships. EX Escorts have higher value than merchant. 
- [TODO] Fidelity vs speed tradeoffs for ML training.
    -The time over which the sim takes place, like in render_attack_animation
    -Larger dt = fewer steps → faster runs but less precise hit timing and motion.
    -Smaller dt = more steps → more accurate but slower.

- [TODO] RL framing: single-agent vs two-agent/self-play.
    -Start with single agent but have library of attack profiles. One attack profile would see RL simply move ships out of the torpedo vectors.
    -Alternatively, have two agents, the ships and submarine attacking 
- [TODO] Dataset sizes and compute budgets.
    -How many Monte Carlo trials per evaluation
    -How many RL episodes per training run
    -Total runtime or cost budget
    -Whether it needs smaller/faster configs for iteration and larger/high‑fidelity configs for final results.
- [TODO] Decide whether to enable hit-driven slowdown for training.
    -Yes, will most likely add this. 
- [TODO] Confirm per-ship independent motion default for RL scenarios.
    -Whether or not ships move as one or individually 
- [TODO] Create Starting Convoy Layout
    -Decide how many ships, if there are different types of ships, shape of convoy, etc
    -Look into historic shapes to determine this. 
    -50 ships, mix of ship type (use values instead of hits then), start with no zig-zag but independent shape movements (hit ships slow down)
- [TODO] Create Layout Constraints 
- [TODO] Build Attack Profiles



Suggested Next Steps (proposed batch)
- Decide a canonical trial JSON schema.
- Pick one sim pathway (static or dynamic) as the ML default.
- Approve a minimal episode wrapper for both sides.
- Lock a baseline scenario + seeds for reproducible RL/ML runs.
