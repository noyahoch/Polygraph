# Complementarity follow-up — September 21, 2026

Omri explicitly authorized implementing the final two-experiment plan. Read
PROTOCOL.md before any numerical work. Both scientific definitions must be
frozen before new result inspection. No experiment has been submitted at this
initial documentation checkpoint; reconcile live operator records before acting.

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

Next: restore access, reconcile the original sources and live scheduler, review
the exact release/configuration hashes, then launch the gated Slurm workflow.
The plan is already authorized; another scientific approval is not required.
