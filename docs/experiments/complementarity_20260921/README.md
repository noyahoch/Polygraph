# Complementarity and LogitDynamics decomposition

**Completed September 21, 2026.** All planned fits, 2,000 bootstrap draws,
checks and private backup finished. No further jobs are pending.

- [Concise Hebrew findings](SUMMARY_HE.md)
- [Full results, all seeds and diagnostics](results/report/REPORT.md)
- [Immutable backup and validation scope](PRESERVATION.md)

Read [PROTOCOL.md](PROTOCOL.md) for the frozen scientific definitions and
[SESSION.md](SESSION.md) for current ownership and restart instructions.

The implementation is in `pilots/complementarity_20260921`. It adds two
comparisons to the preserved September 19 experiment:

- Fusion: nine two-score logistic regressions, fitted on 400 photographs and
  assessed on the other 400 photographs from the existing development cohort.
- Decomposition: nine A/B/C linear readouts using subsets of the original LD
  feature recipe, assessed on all 800 original development photographs. The
  original full D is reused.

These are different assessment cohorts. Keep their result tables separate.

## Recorded execution order (completed)

Every numerical command, including preparation of split/draw manifests and
synthetic tests, runs inside a Slurm allocation. The entrypoint is
`python -B -m pilots.complementarity_20260921.run STAGE --root EXPERIMENT_ROOT`.
Do not execute it on a laptop or a cluster login node.

1. `prepare`: bind the original release and data, freeze the exact 400/400 split
   and all 2,000 bootstrap draws. Additional arguments identify the parent
   experiment, parent release and protocol file.
2. `tests-cpu`: semantic fixtures and weighted/duplicated statistical tests.
3. `tests`, then `cache`: CUDA continuation checks, feature-cache generation,
   semantic checks, and original D replay from the saved new cache. The cache
   completion receipt must pass before any A/B/C training.
4. Three `fit-ablation --seed SEED` jobs for seeds 7, 17 and 27, plus the CPU
   `fusion-fit` command. Each seed job fits A, B and C for exactly 100 epochs.
5. `freeze-ablation`, then `freeze`: require all planned fitted models and seal
   their artifacts before assessment.
6. `predict-ablation` on CUDA and `fusion-predict` on CPU.
7. `statistics --stop-after 50`: weighted versus explicit-duplication gate and
   retained first 50 draws. Use timing only to check the remaining budget.
8. `statistics`: resume the same draw IDs through 1,999. Failed draws remain
   identified; affected confidence intervals are withheld.
9. `report`, private backup, immutable-revision checksum verification and
   independent review.

The operator records exact source/configuration hashes, submission intents,
job identities, resource reservations and completion receipts. Do not submit
jobs using this overview alone or assume that an existing job ID is current.
Read the newest operator handoff and reconcile the server first.

## Status and claims

Source code, a protocol commit or a submitted job is not a scientific result.
Only validated completion receipts establish that a stage passed. A local
syntax review does not replace the required Slurm tests.

The final report must distinguish original-fit point estimates from bootstrap
intervals, and retain the limitations in the protocol: development-data reuse,
different conditional uncertainty in the two experiments, class-identity
information in D, and no claim that graph topology is necessary.

The old LD experiment, existing detector predictions and Overleaf document stay
unchanged. Commits are local; no push or merge is part of this task.
