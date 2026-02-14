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
    -Current code status:
        - Flat attack profile schema implemented in `convoy_sim/attack_profiles.py`.
        - 25 explicit profile stubs scaffolded (`P01` to `P25`).
        - Profile fields are aligned with sampler args in `convoy_sim/attackers.py`.
    -Attack profile schema (per profile, sim-native names):
        - `profile_id`, `name`, `weight`
        - `mode` (`fan` or `parallel`)
        - `u_pos`, `n`, `speed`, `max_run_time`
        - fan params: `base_bearing_rad`, `spread_rad`
        - parallel params: `bearing_rad`, `lateral_spacing`
        - launch timing: `launch_delay_s`, `salvo_interval_s`
    -Build behavior:
        - Use `AttackProfile.build_torpedoes(...)` to convert profile directly into torpedoes.
        - `fan` mode launches from one origin (`u_pos`) with heading spread.
        - `parallel` mode uses laterally offset launch positions.
    -Starter library (first 5 profiles):
        - `P1_straight_bow_shot`: medium range, bow aspect, straight run, no decoy.
        - `P2_beam_snap_shot`: closer beam aspect, short launch delay, quick egress turn.
        - `P3_stern_shadow`: long trailing approach, slower closure, late salvo.
        - `P4_evasive_weave`: pre-launch zig-zag approach, moderate range, staggered fire.
        - `P5_decoy_first`: medium range, decoy/jammer first, delayed salvo, hard break-away.
    -Sampling policy for RL episodes:
        - Sample exactly one attack profile at `env.reset()` (not per step).
        - Use weighted random sampling via `AttackProfileLibrary.sample_profile(...)`.
        - Initial weights example: `P1=0.30, P2=0.20, P3=0.20, P4=0.15, P5=0.15`.
        - Keep RNG seeded for reproducibility and log selected `profile_id` each episode.
        - NOTE: RL wrapper is not yet auto-wired to sample profile library at reset; add this integration step in `convoy_sim/rl_wrapper.py`.
    -Curriculum / training schedule:
        - Phase 1: `P1`, `P2` only (stability + baseline learning).
        - Phase 2: all profiles with weighted sampling.
        - Phase 3: stress mix with heavier `P4`/`P5` plus unseen parameter combinations.
    -Evaluation requirements:
        - Report reward, survival rate, and hit metrics broken out by `profile_id`.
        - Maintain a held-out eval distribution distinct from training distribution.
        - Confirm no profile collapse (agent overfits to most common profile).



Suggested Next Steps (proposed batch)
- Decide a canonical trial JSON schema.
- Pick one sim pathway (static or dynamic) as the ML default.
- Approve a minimal episode wrapper for both sides.
- Lock a baseline scenario + seeds for reproducible RL/ML runs.

