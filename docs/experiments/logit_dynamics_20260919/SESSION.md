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

### Submitted workflow — September 19, approximately 12:03 Israel

The complete dependency chain is submitted. **Do not resubmit it from this note.**
Read the live queue and stage receipts first; the states below are a dated snapshot.

Remote experiment root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500`

| Stage | Slurm job | State at snapshot |
| --- | --- | --- |
| Full CLS extraction and parity | 910058 | Running on s-004 |
| Fit seed 7 | 910059 | Pending extraction success |
| Fit seed 17 | 910060 | Pending extraction success |
| Fit seed 27 | 910061 | Pending extraction success |
| Freeze all three completed fits | 910062 | Pending all fit successes |
| Three-seed predictions | 910063 | Pending freeze success |
| Paired analysis | 910064 | Pending prediction success |
| Private Hugging Face backup | 910065 | Pending analysis success |

GPU preflight 910039 completed successfully with 339 allocated seconds: 11
tests plus three subtests passed, 96 records reproduced historical logits and
the checked CLS states with maximum absolute difference zero, and the tiny
training/reload/next-optimizer-step checks passed. CPU validation 910041 failed
because its selector included a GPU-only integration test; this failed attempt
is preserved. Corrected CPU validation 910046 completed in 24 seconds with 13
science tests plus three subtests and verified the 7,200-row, 69-score-column
historical prediction schema and 800 source groups of nine views.

Full source release: `release-400f1fbbc105a370`, 133 files, archive SHA-256
`07f81a1dcd5fe0c0e24e490f023de6c436218b88465be94d365073a3c0fcf2b2`.
Full configuration SHA-256:
`6f9db29db035974c6dc08fc249f694ccea1273abde8078de20a8d96d4fb94194`.
Scientific implementation commit: `4310b9d`; initial protocol/session commit:
`347dacd`. The frozen release predates the operational-documentation commit;
use its source manifest for exact run reproduction.

Caps are 270 minutes for extraction, 120 minutes for each of three parallel
fits, and 30 minutes for prediction. Including the original 15-minute preflight
reservation, these total 675 GPU-minutes (11.25 GPU-hours), within the 12-hour
ceiling. These caps are not estimates of elapsed runtime. At 12:02, extraction
had completed 448 of 28,800 records, with the five most recent 64-record shard
intervals between approximately 8.8 and 10.2 seconds. A roughly 70-minute
capture projection assumes that warm rate persists and excludes later fitting
and analysis; the short preflight's much longer extrapolation repeatedly counted
fixed initialization overhead. Re-estimate from live progress if needed.

All stages use successful-completion dependencies and invalid-dependency
cancellation. They continue on the server if the laptop or VPN disconnects;
new agent decisions wait for an active application session. The existing
`monitor-polygraph-slurm-continuation` heartbeat was updated to this campaign
and activated every ten minutes. It must remain quiet on unchanged status,
avoid duplicate submissions, and be paused after verified completion is
reported. The job chain does not depend on this heartbeat.

Preserved operator instructions and launch records are under
`pilots/logit_dynamics_20260919/ops/OPERATIONS.md` and this directory's
`run_records/`. Live receipts remain under the remote `ops/` directory and the
coordination workspace above. Scientific results are not yet available at this
snapshot; passing readiness checks is not completion of the comparison.

## Execution and completion

Prepare reviewed code and a frozen source snapshot; run correctness/parity checks on Slurm; then use durable dependencies for full CLS extraction, three-seed fitting, frozen prediction, paired analysis and preservation. Failed correctness checks block dependent fitting/evaluation. Mechanical fixes may be made within the approved method; scientific changes must be documented and reconsidered before evaluation.

Completed results must include all three seeds, exact code/configuration identities, group assignments, training/checkpoint histories, model weights, aligned predictions and uncertainty with explicit development-set reuse. Download only compact results to the Mac. Heavy artifacts stay on Slurm; use the previously authorized private Hugging Face destination if authenticated and within quota. Backup failure must not invalidate already completed science or trigger duplicate training.

The absence of a fresh independent evaluation and the different allocation of auxiliary-head versus detector supervision must be stated in any report. This arm cannot be silently added to Yishai's differently sampled main table.
