# Complementarity follow-up — September 21, 2026

**Current state: COMPLETED. No numerical work or further submission is pending.**
Both authorized experiments, their fixed 2,000 bootstrap draws, report and
verified private backup finished on September 21. Read [SUMMARY_HE.md](SUMMARY_HE.md)
for the findings and [PRESERVATION.md](PRESERVATION.md) for artifacts and limits.
The implementation and execution checkpoints below are historical; they are
not instructions to resume the completed DAG.

Omri explicitly authorized the two-experiment plan. Both scientific definitions
were frozen before new result inspection. PROTOCOL.md remains unchanged.

## Completion checkpoint — September 21, 2026

- All initial jobs 915915–915924 completed. The durable controller submitted
  915946 (remaining statistics), 915947 (report), and 915948 (backup) once at
  14:12:13 Israel after its timing gate. These also completed successfully.
- The final backup verification finished at 14:22:42 Israel; the recorded queue
  was empty at 14:23:30. Historical IDs are bound to user `omrifahn`, account
  `gpu-students`, prefix `comp0921-135802` and exact timestamps in
  [resource_ledger_final.json](results/ops/resource_ledger_final.json).
- Total elapsed time from the first job was 17m24s. Consumed resources were
  382 GPU-seconds and 816 cumulative CPU-job wall-seconds, below the respective
  7,200 and 14,400 second ceilings. No retries, changed settings or replacement
  draws were required.
- All nine A/B/C readouts completed 100 epochs; all nine fusion models fitted.
  Semantic cache checks, original D replay from the new cache, weighted/literal
  duplication, serialization and continuation fixtures passed in Slurm.
  All 2,000 planned bootstrap draws are valid for every reported contrast.
- Private HF revision `769e8b48c9e33edaebac81e97405945197054c11` was downloaded
  and verified, including all 2,429 archive members. All 1,391 previous Hub files
  retained their identities. See the [publication receipt](results/publication/receipt.json).
- Post-run artifact reviews found no blocking discrepancy. These inspect code,
  receipts, hashes and saved results; they do not constitute independent
  numerical recomputation or fresh-environment/raw-image reproduction.
- The `monitor-polygraph-slurm-continuation` heartbeat has been paused after
  completion and review. No new experimental work launches automatically.
- [Full results](results/report/REPORT.md) retain the 400-photo fusion and
  800-photo decomposition tables separately. The main findings are DG−D
  +0.007203, DG−DS +0.000043 (no established advantage), and C−B +0.004135.
  Keep the frozen uncertainty definitions and development-data limitations.

The compact closure receipt was produced before final local reviews and still
labels root review as pending. This session and the subsequent review notes
record that later work; the original receipt is preserved byte-for-byte.
Local completion notes/reviews postdate the immutable backup and are not claimed
to be inside it. Commits remain local; no push, merge or Overleaf edit occurred.

## Ownership

- Root: orchestration, protocol, independent integration review and local commits.
- replay_completion_engineer: new common/cache/ablation implementation and tests.
- english_report_factcheck: new fusion/statistics/report implementation and tests;
  cross-review the other engineer's code before launch.
- replay_completion_ops: sole remote operator, readiness, submissions, resource
  ledger, receipts, durable progression and private backup.

All source is new under pilots/complementarity_20260921. Do not mutate the
completed logit_dynamics_20260919 package or data. All tests/numerical execution
remain on Slurm; Mac is for editing and coordination only.

Operator state:
`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/complementarity-20260921/ops/`.

The parent is the completed September 19 LD campaign with the September 21
replay audit. Original model snapshot
`86605781c0528e286483e305072f7787393da860` and completed replay addendum
`7656f8b19e4b51d12ea2ed21ae3bc04a4dcc67fd` remain immutable.
Do not resume the superseded September 10 training matrix or old closed audit jobs.

New resource ceiling: 2 cumulative GPU-hours, maximum 3 GPUs; statistics 4
cumulative CPU-job wall-hours. Preserve original and new failure evidence,
fit/draw IDs and checkpoints. All commits stay local on September-10th;
no push or merge.

Preserve unrelated pre-existing untracked directories:
docs/experiments/final_comparison_20260916/results/ and pilots/results_final_20260917/.

## Restart discipline

Read this session, PROTOCOL.md, then newest operator HANDOFF/approval/submission/
status receipts. Match names, account and submission times as well as job IDs.
Continue from verified completion markers; never submit from this note alone.
Access failures require precise user action while independent coding continues.

## Implementation checkpoint — before any submission

- Scientific definitions were committed locally as `47179a9` before new effects.
- The new cache, A/B/C readouts, fusion, statistical analysis, tests, reporting
  and Slurm helpers have been implemented. Two engineers cross-reviewed the
  scientific modules; root reviewed integration and operational boundaries.
- Static Python AST parsing passed. This is not a numerical or runtime test.
  All required Slurm checks are still pending; no new result is available.
- Both bounded SSH attempts to `c-003.cs.tau.ac.il:22` timed out before
  authentication. No remote staging, readiness script or job submission occurred.
  The user has been asked to reconnect GlobalProtect to `vpn.tau.ac.il`.
- The sole operator records access checks and exact local release/configuration
  proposals in `work/complementarity-20260921/ops/HANDOFF.md` under the
  coordination workspace above. A proposal is not a submission receipt.
- Proposed full reservations are 105 GPU-minutes and 230 CPU-job wall-minutes.
  These are ceilings, not measured runtime estimates. Actual new GPU/statistical
  runtime is zero at this checkpoint. Reconcile before relying on this note.

## Execution checkpoint — September 21, 14:05 Israel

Access returned after Omri confirmed the VPN reconnection. Read-only readiness
found an empty queue and no previous new-namespace submissions. The operator
verified the original 133 source files, original scores, all 450 CLS shards and
required model artifacts, then staged and verified all 160 release files.

Remote root:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/complementarity_20260921_135802`.

The frozen source checkpoint is `45b0c4f`. Configuration SHA-256:
`19c04dc936a958fa9864a008cf01171d4446a9b55734681294b38b1f6cbe31bd`.
Source-manifest SHA-256:
`4f2debe7aaa861c44538240ce624d4ff1ad4e4160dbe538f1f1298e24db08fed`.

All initial jobs were submitted once at 14:05:18 Israel under user `omrifahn`,
account `gpu-students`, with job-name prefix `comp0921-135802`:

| Stage | Job ID |
|---|---:|
| Prepare and CPU checks | 915915 |
| CUDA checks and new-cache compatibility gate | 915916 |
| A/B/C fits, seed 7 | 915917 |
| A/B/C fits, seed 17 | 915918 |
| A/B/C fits, seed 27 | 915919 |
| Fusion fits | 915920 |
| Freeze all fitted models | 915921 |
| Ablation predictions | 915922 |
| Fusion predictions | 915923 |
| Retained first 50 bootstrap draws | 915924 |

At 14:05:52, preparation was running on `rack-iscb-31`; all descendants were
pending on dependencies. This is not evidence that numerical tests passed or
that the experiment completed. Match timestamps/account/names as well as IDs.

The durable timing controller owns submission of the remaining 1,950 draws,
report and private backup. Do not submit that phase independently. The existing
`monitor-polygraph-slurm-continuation` heartbeat is now active for this new
scope only and should stay quiet on unchanged state. This historical instruction
was fulfilled: the monitor is now paused following completion and review.

Submission, staging and scheduler evidence are in the operator directory:
`submission_stdout_v1.jsonl`, `staging_receipt.json`, and `status_initial.json`.
Next: inspect completion receipts and early failures, continue only within the
frozen protocol, and account for actual consumed plus outstanding reserved
resources before any mechanical recovery. No new scientific approval is needed.
