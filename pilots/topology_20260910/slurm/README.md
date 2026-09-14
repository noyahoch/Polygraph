# Slurm ownership and recovery

Only the designated operations agent submits jobs or transfers files. Worktree code is staged
under a new immutable release at the experiment root; previous September artifacts stay intact.
No Git commit/push is part of this workflow.

Remote experiment root:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910`

## Files and operation

- `data_fetch.py` / `data_fetch.sbatch`: allocated CPU-only official data download and hash checks.
  No classifier inference. The output is `data/download_complete.json`.
- `stage.py` / `stage.sbatch`: execute a command from the immutable `manifests/workflow.json`,
  verify its source hash binding, write running/completed/failed execution receipts, and propagate failures.
- `submit.py`: submit explicitly approved stages in topological order with `afterok` dependencies.
  `release_review.json` must bind the exact workflow SHA and record engineering/science GO.
  Repeated calls do not duplicate recorded jobs. If a crash leaves a scheduler job without a receipt,
  reconcile it rather than resubmitting. No automatic scientific repair or new candidate is allowed.

The workflow records the seven approved arms crossed with seeds 1, 2, 7, 17, 27 (35 fits).
Use one GPU per fit, initially one pilot, then a two-worker I/O check. Production concurrency is at
most eight and starts only after those checks and the associated release review. CPU publication
and statistics jobs do not reserve GPUs. Do not launch overlapping GPU arrays beyond the shared
eight-GPU limit. Every numerical or data-processing check runs on Slurm.

## Durable sequence

Data and protocol checks → small separate-cache GPU smoke → development extraction → validation
and rewiring → isolated one-epoch pilot → two-worker I/O check → approved production fit array →
complete-set checkpoint freeze → one final evaluation → statistics/report/publication.

Exact commands and receipt paths are filled from the engineer/publisher interfaces in the reviewed
workflow; this README does not authorize empty placeholder commands. Failed dependencies prevent
downstream scientific jobs. Hourly/finished-fit publication must go through one central locked
publisher, never concurrent writes from each GPU. Authentication may delay upload without preventing
local-on-cluster checkpoints.

## Monitoring and recovery

Read `manifests/submissions.json` and `manifests/execution/` alongside date/name-matched `sacct`
and `squeue --me`. Queue absence alone is not completion. Persist actual Slurm elapsed time and
GPU allocation for accounting. Track one project-wide active-agent wall clock; parallel agents
count once. GPU jobs may run while the app is asleep and do not consume active-agent hours.

Resume training from each epoch's `latest.pt` only through the same arm/seed/config contract.
The engineer's CLI verifies this contract. A new repair needs an explicit recorded review and a
distinct stage/attempt; maximum two targeted technical repairs per stage. Never bypass validation
or repurpose old final-test scores as validation. Keep all heavy arrays and model artifacts on Slurm.
