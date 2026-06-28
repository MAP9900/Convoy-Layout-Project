# TODO

Last updated: 2026-06-28

Operational commands live in `docs/REPRODUCING.md`. Completed historical planning notes were moved out of the active docs during cleanup.

Current project focus: finish the POMDP story, then expand RL layout freedom and final testing.

## Workstream 1: POMDP Finalization

- [ ] Rerun POMDP v1/v2 notebooks from clean regenerated inputs.
- [ ] Preserve the fixed-location diagnostic when comparing observation presets.
- [ ] Define POMDP success metrics:
  - gap/regret versus full-state oracle
  - expected loss
  - expected hits
  - robustness across `good_contact`, `baseline_night`, and `poor_contact`
- [ ] Add a short POMDP interpretation checklist:
  - describe `expected_hits` as a comparative simulator metric
  - do not frame simulator hit counts as historical sinking rates
  - separate location-selection quality from fire-control reconstruction quality
- [ ] Decide whether POMDP v2 fire-control is final-scope or documented as near-final/future work.
- [ ] Update `docs/POMDP.md` after final POMDP reruns.

## Workstream 2: RL Layout Freedom

- [ ] Define the RL promotion threshold:
  - primary metric
  - risk/variance guardrail
  - minimum improvement needed over baseline
- [ ] Add final RL validation protocol:
  - repeated eval seeds
  - harder held-out attack-profile pack
  - confidence intervals or repeated-seed summaries
- [ ] Define and enforce layout feasibility boundaries:
  - minimum ship separation
  - maximum convoy footprint
  - class counts preserved
  - optional route/sea-room bounds
- [ ] Add action masking, rejection, or repair for invalid RL layouts.
- [ ] Design a freer RL layout generator:
  - allow per-ship placement within convoy-frame bounds
  - prevent overlapping ships
  - preserve valid ship classes and fleet counts
  - start with bounded continuous or grid-like movement before fully arbitrary placement
- [ ] Add a simple random/evolutionary feasible-layout baseline.
- [ ] Add layout novelty/diversity reporting:
  - footprint size
  - spacing distribution
  - class placement
  - visual comparison against baseline and current builder layouts
- [ ] Rerun RL after the freer layout generator exists and compare against:
  - static baseline
  - current builder-mode RL
  - random/evolutionary feasible search

## Final Testing And Results

- [ ] Regenerate archived datasets and run outputs from `docs/REPRODUCING.md`.
- [ ] Run full tests before final experiments.
- [ ] Run final baseline, VAE, POMDP, and RL workflows.
- [ ] Promote only curated final artifacts to `results/final/`.
- [ ] Create a compact final method/result summary:
  - method
  - assumptions
  - observation level
  - action space
  - metric
  - result
  - caveat
- [ ] Create final visual comparison artifacts:
  - baseline layout
  - current builder RL layout
  - freer RL layout
  - random/evolutionary best layout, if used

## Later / Stretch

- [ ] Conditional VAE if tactical conditioning becomes necessary.
- [ ] Learned attacker policy after POMDP baselines are stable.
- [ ] Multi-layout domain randomization.
- [ ] More detailed escort/search/reaction behavior.
- [ ] Diffusion or other generative attack-profile variants.
