# Completing saved-input replay verification — September 21, 2026

## Scope and reason

The original CPU audit stopped at its first score-tolerance failure, before
seed 27 validation and every development-score replay. The September 19
diagnostic characterized one seed-17 validation mismatch and verified saved
metrics and bootstrap aggregation. It did not complete the remaining replay
or independently recompute every weighted bootstrap draw.

On September 21 Omri asked why those checks were left incomplete, and then
confirmed reconnection to the university VPN. We will finish these checks on
Slurm using the existing frozen source, models, scalers, cached CLS inputs and
reference predictions. Nothing is trained again or substituted into the
original scientific results.

## Fixed verification contract

1. Verify source/artifact identities and the pinned HF portable weights and
   scalers. Use seeds 7, 17, 27 and the same 3,600 validation / 7,200 development
   records per seed, with exact source-image and outcome alignment.
2. Replay all six seed/role combinations on CPU. Keep the original absolute
   and relative score tolerances at `1e-4`. Collect all mismatch counts, IDs,
   maximum errors and score-metric differences; a score mismatch must not stop
   later seed/role checks. Hash, schema and nonfinite-value failures remain
   errors. Store new outputs separately; keep the old failed audit unchanged.
3. Replay the same six combinations on CUDA after loading the published
   safetensors and scalers. Match the original inference code, batch size 512
   and precision settings. Record actual hardware and environment. Using CUDA
   does not imply the exact same physical GPU as the original job.
4. On CPU, independently recompute all 2,000 paired bootstrap draws for all
   three seed pairs with scikit-learn weighted AUROC, using the saved shared
   source-photo multiplicities. Compare per-draw differences, means and the
   resulting interval against the original stored values. Do not generate new
   resamples or change the statistical estimand.
5. Report execution completeness separately from numerical tolerance success.
   Completing all checks cannot convert a failed threshold into a pass.

This completes **portable checkpoint-to-score replay from saved CLS inputs**
and the independent stored-bootstrap calculation. It does not perform a fresh
dependency installation, raw-image-to-output inference, training replication
or replication of the published paper's ViT-L experiments. Those are separate
verification or experimental scopes and must not be claimed as completed.

## Resource limits and ownership

- CPU allocation: at most 20 minutes, six CPUs, 16 GB.
- CUDA allocation: at most 15 minutes, one GPU, six CPUs, 16 GB.
- At most one new GPU is needed; there is no model/parameter search.
- Historical allocated GPU time is 14,376 seconds (3:59:36). This single new
  GPU reservation adds at most 900 seconds, remaining below the existing
  twelve-hour ceiling. Account for actual elapsed allocations on completion.
- Engineer: `replay_completion_engineer`, owns the new audit script only.
- Operator: `replay_completion_ops`, sole submitter and keeper of new receipts.
- Independent reviewer: `english_report_factcheck`, read-only coverage review.
- Root: scope approval, reporting, documentation and local commits.

All numerical work, including CPU checks, runs in Slurm allocations. The Mac
is used only for source/document editing and coordination. Local commits only;
no push, merge, Overleaf edits, paid hardware or old training-matrix resumption.
Preserve the unrelated pre-existing untracked results directories.

## Resume safely

New operator records:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/logit-dynamics-20260919/ops/replay_completion_20260921/`

Existing remote experiment root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500`

At preparation time, the first SSH check timed out before authentication.
Omri then confirmed VPN reconnection. Reconcile fresh connectivity, source and
data availability, existing intents and exact job identities before submitting.
The new audit script requires root review of its exact hash and commands before
launch. The previous continuation heartbeat remains paused.

Read-only reconciliation at 12:20:11 Israel confirmed restored access and an
empty user queue, with no September 21 audit namespace or prior submission.
All 133 frozen source files and all 44 published input files matched their
recorded hashes. The original checkpoints, reference predictions, scores,
bootstrap artifact, Python environment and 28,800-row CLS index were available.
This is readiness evidence, not evidence that the new replay has run.

## Submitted — September 21, 12:25:09 Israel

The reviewed audit was committed locally as `42ef421` and submitted once.
Do not submit it again from this note; reconcile the receipts and live queue.

| Verification | Slurm job | Allocation | Started |
| --- | --- | --- | --- |
| Six CPU replays and independent bootstrap | 915652 | 20 minutes, 6 CPUs, 16 GB | 12:25:10, rack-iscb-31 |
| Six CUDA replays | 915653 | 15 minutes, 1 RTX 2080 GPU, 6 CPUs, 16 GB | 12:25:09, s-005 |

Both jobs belong to user `omrifahn`, account `gpu-students`, and exclude s-004.
These are submission/start facts, not completion claims. The exact job names
and submission records are in the operator namespace above.

- Audit source SHA-256: `582a308a4b8b0ba4f1ef0f0114129761989434d2768f5d663e8bc589ab3dde55`.
- Approved configuration SHA-256: `8f03bcfeeba2bef3e8544d46ec12be37336000eb1da70dfbe74759c989f137d7`.
- New remote receipts: `audit/replay_completion_20260921_cpu_v1.json` and
  `audit/replay_completion_20260921_cuda_v1.json` under the experiment root.
- Original scientific outputs, failed audit and diagnostic remain unchanged.
