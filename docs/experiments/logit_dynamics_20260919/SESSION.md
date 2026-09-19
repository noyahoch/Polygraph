# LogitDynamics follow-up — September 19, 2026

## Authorization and scope

Omri authorized implementing and running the proposed LogitDynamics comparison on September 19: “Okay, let's do it … I'm connected to the VPN.” The new arm extends the completed September 17 development comparison. Historical instructions to keep all experiments paused are superseded for this arm only.

- Frozen ViT and existing G/H/S/O/L models and predictions remain unchanged.
- All extraction, tests, fitting, prediction and statistics run as Slurm jobs. The Mac is for source editing and coordination only.
- Local commits only; no push, merge or communication to colleagues.
- Preserve all existing artifacts, including two pre-existing untracked directories: `docs/experiments/final_comparison_20260916/results/` and `pilots/results_final_20260917/`.
- New namespace: `logit_dynamics_20260919`. Exact scientific settings are in `PROTOCOL.md`; no settings may be selected using development-evaluation scores.
- This is a fixed-configuration adaptation of a published method to our backbone and grouped corruption benchmark, not a reproduction of the paper's entire experimental protocol.

## Roles

- Root: orchestration, protocol approval, progress communication, local commits.
- `logit_scientist_sep19`: protocol and independent methodological review.
- `logit_engineer_sep19`: new baseline implementation and remote-test fixes; owns `pilots/logit_dynamics_20260919/`.
- `logit_slurm_ops_sep19`: sole job submitter; owns operational state and durable server-side execution.

## Current operational facts

Read-only access succeeded on September 19 at approximately 11:37 Israel time; Omri's queue was empty. Existing final campaign and feature caches were found. GPU availability is a snapshot, not a scheduling guarantee.

The ops state is under:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/logit-dynamics-20260919/ops/`

Start with `readiness.json` and the latest submission/status files there when resuming. Reconcile remote job IDs and completion receipts before submitting anything again. Do not infer live status from this note.

The initial operational envelope is at most three concurrent GPU allocations and twelve aggregate allocated GPU-hours for this follow-up, including preflight and retries. This is a conservative ceiling, not a runtime estimate or an instruction to consume it. Preflight measures extraction/fit rates before the full submission.

## Execution and completion

Prepare reviewed code and a frozen source snapshot; run correctness/parity checks on Slurm; then use durable dependencies for full CLS extraction, three-seed fitting, frozen prediction, paired analysis and preservation. Failed correctness checks block dependent fitting/evaluation. Mechanical fixes may be made within the approved method; scientific changes must be documented and reconsidered before evaluation.

Completed results must include all three seeds, exact code/configuration identities, group assignments, training/checkpoint histories, model weights, aligned predictions and uncertainty with explicit development-set reuse. Download only compact results to the Mac. Heavy artifacts stay on Slurm; use the previously authorized private Hugging Face destination if authenticated and within quota. Backup failure must not invalidate already completed science or trigger duplicate training.

The absence of a fresh independent evaluation and the different allocation of auxiliary-head versus detector supervision must be stated in any report. This arm cannot be silently added to Yishai's differently sampled main table.
