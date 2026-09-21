# LogitDynamics follow-up — September 19, 2026

## Reopened verification — September 21, 2026

Omri challenged the incomplete CPU replay and asked why it was not completed.
He subsequently confirmed the requested VPN reconnection. This authorizes
finishing the bounded verification of the existing experiment; it does not
authorize new models, hyperparameters, training or changes to reported scores.

Current scope and restart entry point: [REPLAY_COMPLETION_20260921.md](REPLAY_COMPLETION_20260921.md).
The operator must reconcile current jobs and new submission receipts before
launching anything. Historical job IDs below must not be treated as live jobs.
The September 19 closure remains preserved below as a dated record.

## Completed — September 19, 16:15 Israel

**The scientific comparison, independent saved-result review, private model and
review preservation are complete. No further numerical jobs are planned.
The continuation heartbeat has been paused.**

- Recovery predictions 910571/910572/910573 completed; analysis 910574 finished
  at 15:34:44 and backup 910575 at 15:35:05 Israel. GPU allocations total
  14,376 seconds (3:59:36), including the original prediction timeout.
- LD mean AUROC is 0.8980659074; historical G_mean is 0.8888946659.
  Prespecified G_mean minus LD is −0.0091712415, with paired source-photo
  95% interval [−0.0158737735, −0.0023608076]. Read `RESULTS_HE.md`, the original
  `results/report.json` and `AUDIT_SCIENCE.md` for interpretation and limitations.
- Private HF artifact revision is `86605781c0528e286483e305072f7787393da860`,
  prefix `logit_dynamics_20260919_114500/snapshot_910575` in
  `omrifahn/polygraph-experiments`. All 295 manifest-listed paths and the exact
  backup manifest were verified. Portable models/scalers were freshly fetched
  to the server; all 44 fetched files and native/portable model tensors matched.
- CPU audit **910647 failed** at 15:55:32 after 115 allocated seconds.
  Source hashes, roles, complete histories, selected epochs and training scaler
  reconstructions passed. Seed 17 validation scores exceeded the fixed
  `atol=rtol=1e-4`; preserve `audit/reproducibility_cpu_v1.json` as FAILED.
- Targeted CPU diagnostic **910651 completed** at 16:06:09, using 92 allocated
  seconds. Independent metadata, historical-score identity, all 14 AUROC/AP
  vectors, seed means/SDs, primary estimate and saved-bootstrap aggregation/
  interval checks passed. It did not regenerate weighted AUROC for every draw.
- Exactly one of 3,600 seed 17 validation rows violated CPU replay tolerance:
  record 15321 / photograph 659, maximum difference 0.0007970333. FP32/FP64 CPU
  head arithmetic changes top-five membership at layer 4 with boundary margin
  2.384e-7; the FP64-head diagnostic matches the stored GPU score at the
  original tolerance. This supports numerical ranking sensitivity, but absent
  original GPU intermediates does not establish its exact historical cause.
- The failed strict CPU replay is **not cleared**. Seed 27 validation and full
  development CPU replay were not completed. Scientific results stay exactly
  as originally computed on GPU. No tolerance, settings, weights or predictions
  were changed; no general CPU portability or clean-install claim is supported.
- Root and the independent scientific reviewer close the comparison with this
  explicit portability limitation; no extra GPU experiment is needed for the
  scoped scientific claim. See `AUDIT_REPRODUCIBILITY.md` and `REUSE.md`.
- Original results/private-backup/accounting receipts are locally committed as
  `93b9341`. Preserve two unrelated untracked results directories. No push or
  merge. Ops remains the sole remote operator for final documentation backup.
- Private review addendum was verified at 16:14:23 Israel: immutable revision
  `075380b8975fb29f95fa8727234f98cf00e43eaa`, prefix
  `logit_dynamics_20260919_114500/review_20260919` in the same private repository.
  All 38 addendum files passed checksum verification. All 296 original
  snapshot Git blobs (including its manifest) and the repository root README
  remained unchanged. Review manifest SHA-256:
  `8ddb24b79f4607ff7233eab20e3bb768e5db8410b0c075467e01b878d33a64fd`.
  Exact receipt: `run_records/review_addendum_publication_receipt.json`.
- Additional local commits: `01f7224` records the findings/scientific audit;
  `f54e42c` preserves the CPU audit, diagnostic, receipts and reuse guide. The
  closing documentation commit records the addendum receipt and this status.
- `monitor-polygraph-slurm-continuation` is PAUSED by the app automation tool.
  No new training, inference or experiment is queued by this task. Future
  scientific work requires a new scoped request; never resume the superseded
  September 10 matrix. The dated sections below are historical snapshots.


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

### Prediction recovery prepared — September 19, approximately 15:19 Israel

Extraction completed at 14:32:40 with all 28,800 records and exact full-export
logit/CLS12 parity, using 9,242 allocated GPU-seconds. All three fits completed
their required 16 head epochs and 100 probe epochs. Seed 7 used 1,306 allocated
seconds; seeds 17 and 27 used 573 each. Freeze job 910062 completed successfully
at 14:54:34. Original prediction job 910063 started at 14:54:37.

At 15:12, that prediction job was still on its first command (seed 7), with no
prediction output or numerical error. A bounded read-only process probe found
the process waiting in `rpc_wait_bit_killable`, supporting network-filesystem
waiting; the exact waiting path and underlying cause were not established.
Slurm denied the normal owner request to increase this job's limit from 30 to
75 minutes. Its limit remained 30 minutes, with scheduled expiry at 15:24:37.
The denied attempt and diagnostic evidence are preserved in `run_records/`.

A prediction-only recovery is prepared and **conditionally approved, not yet
submitted at this snapshot**. Its exact configuration SHA-256 is
`9d28852754f236f3820a015d6478e94bf1eb682a41e89fc19c42c8965080d181`;
operator helper SHA-256 is
`6e00f870501eb8025f0ce7e843590105fa68b9e60f6277808fba2c8e91794228`.
The configuration is `run_records/config_prediction_recovery_v1.json`.

This recovery runs the original prediction command separately for each of
seeds 7/17/27, with 45 minutes and one GPU per job, at most three concurrently.
It reuses the completed freeze and immutable scientific source, weights and
settings. Completed seed predictions verify and skip; an incomplete seed
recomputes predictions only. Excluding s-004 is an operational workaround, not
a demonstrated diagnosis. Analysis waits for all three prediction jobs, then
private backup follows. Neither training nor freeze is resubmitted.

The sole Slurm operator may submit the approved configuration without renewed
permission **only after** verifying that original job 910063 terminated
unsuccessfully, its processes are gone, original analysis/backup 910064/910065
cannot run, the frozen gate/source still match, and no recovery submission
already exists. If 910063 succeeds, abandon the recovery. Preserve every
original attempt and receipt. Read current remote state before deciding;
this note is not evidence that the original job has timed out.

Budget review uses completed actual allocations, not unused expired caps:
12,033 completed GPU-seconds, plus at most 1,800 for original predictions,
plus 8,100 for the three recovery reservations, total **21,933 seconds
(6:05:33)**. This remains below the unchanged 12 cumulative GPU-hour ceiling.
No scientific result has been used to choose this operational recovery.

### Recovery submitted — September 19, 15:31:26 Israel

Original predictions 910063 reached TIMEOUT at 15:24:56, using 1,819 allocated
seconds including termination grace. Seed 7 had completed all 7,200 prediction
records and its valid completion receipt is preserved. Original downstream
analysis/backup 910064/910065 were cancelled. CPU-only diagnostic 910564
confirmed that original prediction processes were gone on s-004; it ran for
one allocated second and used no GPU.

The first recovery submission was rejected before any job ID was returned:
`allocation failure: Job dependency problem`. The old completed freeze job
910062 was no longer a valid scheduler dependency, although accounting and the
sealed gate confirmed its success. The rejected `predict7_retry1` intent is
preserved. Scheduler validation succeeded after removing only that obsolete
dependency. The replacement submit helper verifies the complete frozen gate
and its hash; unchanged prediction commands verify it again in Slurm. Analysis
still waits for all three new successes. No scientific setting changed.

Root approved recovery configuration SHA-256
`eae1ca980507606b495fa2aeb03f794dfd87820ca169abe1bab4c7d0d3adb7bc`
and helper SHA-256
`3a23fda0ba088fe585998c1e80ea5542dd8f0a41f51b23e782d1cbe2cb948d25`.
This **supersedes the prepared v1 configuration**. Do not submit v1 or duplicate
the following jobs, submitted at 15:31:26:

| Stage | Slurm job |
| --- | --- |
| Seed 7 prediction verification / skip completed output | 910571 |
| Seed 17 predictions | 910572 |
| Seed 27 predictions | 910573 |
| Analysis after all three succeed | 910574 |
| Private backup after analysis succeeds | 910575 |

Each prediction job retains its 45-minute cap, same scientific command and
immutable release, with s-004 excluded. The actual-plus-reserved worst-case
GPU budget is **21,952 seconds (6:05:52)** after including termination grace,
within the unchanged 12-hour ceiling. Read live states and submission receipts
from the ops folder; these IDs record submission, not completion. Completed
models and seed-7 predictions must not be retrained or needlessly recomputed.

## Execution and completion

Prepare reviewed code and a frozen source snapshot; run correctness/parity checks on Slurm; then use durable dependencies for full CLS extraction, three-seed fitting, frozen prediction, paired analysis and preservation. Failed correctness checks block dependent fitting/evaluation. Mechanical fixes may be made within the approved method; scientific changes must be documented and reconsidered before evaluation.

Completed results must include all three seeds, exact code/configuration identities, group assignments, training/checkpoint histories, model weights, aligned predictions and uncertainty with explicit development-set reuse. Download only compact results to the Mac. Heavy artifacts stay on Slurm; use the previously authorized private Hugging Face destination if authenticated and within quota. Backup failure must not invalidate already completed science or trigger duplicate training.

The absence of a fresh independent evaluation and the different allocation of auxiliary-head versus detector supervision must be stated in any report. This arm cannot be silently added to Yishai's differently sampled main table.
