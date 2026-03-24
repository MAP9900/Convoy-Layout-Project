# Project TODO

## Now

- [x] Complete Phase 2–4 repo refocus (audit map + reorg + canonical entrypoints).
- [x] Establish canonical baseline runner (`experiments.run_baseline_suite`).
- [x] Establish canonical RL runner (`experiments.run_rl_train`).
- [x] Move non-critical-path scripts to `experiments/archive/`.
- [ ] Finalize locked train/eval profile split policy for publication-grade runs.
- [ ] Freeze seed protocol for official benchmark suite.
- [ ] Define promotion thresholds for RL policy acceptance.

## Next

- [ ] Replace tabular RL learner with production RL algorithm while preserving output schema.
- [ ] Add per-profile performance dashboard and collapse/overfitting flags.
- [ ] Add canonical run comparison utility (baseline vs RL manifests).
- [ ] Add CI target for canonical entrypoint smoke tests.

## Later

- [ ] V2: attacker RL + alternating training protocol.
- [ ] V3: multi-agent robustness and domain randomization matrix.

## References

- `PROJECT_GUIDE.md`
- `docs/REORG_PHASE2_4_AUDIT.md`
- `docs/RL_Road_Map.md`
- `docs/ML_RL_Checklist.md`
