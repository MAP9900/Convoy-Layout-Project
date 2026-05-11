# Notes

## Curated V4 VAE Findings - 2026-05-11

Context:
- Training data used outcome-gated `random_tactical_v4` curated synthetic profiles.
- Dataset size was `45k` train / `5k` valid.
- Intended label mix was `65% credible_hit_threat / 25% credible_near_miss / 10% intentional_miss`.
- V4 data generation itself looked good: minimum spawn clearance was enforced, moving zig-zag outcome labels matched intent, and spawn distributions were tactically structured.

Observed VAE training behavior:
- Training was numerically stable.
- Train and validation losses tracked closely; no obvious overfit from the loss curves.
- Best validation epoch was around epoch `53`.
- Final/best losses were not the main failure signal.

Decoded prior-sample issue:
- Raw VAE prior samples did not preserve the curated tactical manifold.
- In the notebook static audit, only about `7.9%` of decoded samples passed the coarse geometry gate.
- About `117 / 1000` decoded samples violated the `250 m` minimum ship-clearance expectation.
- The decoded spawn plot smeared samples through the middle of the convoy instead of preserving the tactical spawn bands/envelope.
- Dynamic profile-first sim audit on `500` decoded samples produced roughly:
  - `448` hit threats
  - `43` near misses
  - `9` misses
  - `any_ship_hit_rate ~= 0.896`
- This means decoded samples were heavily biased toward over-hitting and did not preserve the intended `65/25/10` outcome mix.

Important audit nuance:
- The static audit is partly unfair to VAE samples because decoded profiles do not include v4 `intent`, `target_zone_kind`, `spawn_region`, `target_ship_ids`, or `aim_point`.
- Without those fields, static audit falls back toward older centroid-style geometry checks.
- However, the dynamic sim audit still shows a real issue: unconditional prior samples are not a reliable curated attack-profile generator.
- This should be treated as two separate issues:
  - The regular VAE sample path needs improvement because it does not preserve the curated multimodal tactical distribution.
  - The regular VAE audit path also needs improvement because it evaluates decoded profiles without the intent metadata used by the v4 generator.
- A high static rejection rate alone is therefore not enough to diagnose model failure. It should be paired with clearance checks and dynamic moving-convoy outcome audit results.

Likely cause:
- The current VAE learns only 8 continuous fields:
  - `u_pos_x`
  - `u_pos_y`
  - `sin(base_bearing_rad)`
  - `cos(base_bearing_rad)`
  - `spread_rad`
  - `launch_delay_s`
  - `salvo_interval_s`
  - `u_boat_initial_speed_mps`
- It does not condition on or reconstruct key tactical variables:
  - intended outcome label
  - spawn region
  - target zone kind
  - inside/outside convoy flag
  - target point / aim point
  - outcome label
- The curated v4 distribution is multimodal. An unconditional Gaussian-prior VAE averages across modes and produces hybrid samples that are numerically plausible but tactically incoherent.

Current conclusion:
- Do not use raw unconditional VAE prior samples for convoy layout optimization yet.
- Keep curated v4 synthetic data as the source-of-truth attack distribution for now.
- The next VAE design should be conditional/outcome-aware rather than simply trained longer.

Recommended next VAE direction:
- Build a conditional VAE that conditions on tactical metadata such as:
  - intended label
  - spawn region
  - target zone kind
  - inside convoy envelope
- Prefer modeling/reconstructing target/aim context as well, or at least preserving it in decoded outputs.
- Evaluate decoded samples with the same moving zig-zag dynamic outcome audit, not only static centroid geometry.

Implementation follow-up:
- Regular VAE diagnostics now support an empirical latent-bank sampler in addition to the standard Gaussian-prior sampler.
- The latent-bank sampler draws near encoded training examples and is meant to test whether failures are coming from bad prior coverage versus poor reconstruction/manifold learning.
- Decoded VAE QA now uses a VAE-specific diagnostic helper that reports:
  - minimum spawn clearance
  - profile-first dynamic moving-convoy outcome labels
  - hit rates and closest-ship distances
  - legacy centroid-static labels only as comparison fields
- The notebook summary should therefore treat `passes_safety_gate` / clearance and dynamic outcome distribution as primary metrics, not the older centroid-static rejection rate.

Adversarial/POMDP candidate-pool direction:
- CVAE is not required for the immediate adversarial/POMDP phase.
- Use the regular VAE as the GenAI source by sampling with the latent-bank method, then filter decoded profiles through the moving-convoy dynamic outcome audit.
- The first candidate-pool pipeline should prioritize safe, successful attacks by default, while preserving enough derived metadata for later attacker candidate selection.
- This is a cleaner research story than forcing the VAE to preserve the original training label mix: the curated generator defines a realistic source manifold, the VAE generates nearby novel candidates, and the simulator/audit stack keeps only physically meaningful candidates.
- The next adversarial baseline is full-state candidate ranking: evaluate each VAE candidate against a convoy layout with the scored Monte Carlo simulation, then select the candidate with highest defender loss or expected hits.
- This full-state selector is not yet POMDP, but it gives the upper-bound red-team baseline needed before adding noisy observations and belief-state limitations.
