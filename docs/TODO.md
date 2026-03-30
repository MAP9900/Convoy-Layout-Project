# TODO

Last updated: 2026-03-30

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

- [ ] Align eval seeds between baseline and RL for strict apples-to-apples comparison.
- [ ] Run baseline + RL pair with matched eval seeds.
- [ ] Record Test 2 in `docs/RESULTS_LOG.md` and `docs/OPTIMIZATION_LOG.md`.
- [ ] Define RL promotion threshold (`expected_hits` primary, `CVaR_90` guardrail).
- [ ] Define GenAI attack-profile dataset schema (context + attack vector + outcomes).
- [ ] Add reproducible synthetic attack-dataset generation config/script (fixed seeds + manifest metadata).
- [ ] Generate first dataset snapshot for GenAI model training.

## Next

- [ ] Replace tabular RL selector with stronger learner while preserving artifact schema.
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
- [ ] Multi-agent robustness and domain-randomization matrix.
- [ ] Add latent-space diagnostics/visualization for generated attack-profile families and failure regions.
- [ ] Add diffusion-model variant as optional stretch after VAE baseline is stable.

