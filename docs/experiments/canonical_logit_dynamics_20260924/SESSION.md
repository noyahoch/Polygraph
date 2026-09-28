# Canonical Polygraph / LogitDynamics — September 24 continuation

## Current status

**COMPLETE — September 24, 2026.** All submitted stages 924822–924831 completed successfully; the last backup finished at **10:25:27 Israel**. All three seeds, checkpoint freeze, canonical predictions, 2,000 paired-photo bootstrap draws and private remote archive verification are complete. The independent final scientific audit found no blocking issue. There are no outstanding jobs or reservations. Do not resubmit this campaign.

Mean AUROC: **Polygraph 0.89390638; LogitDynamics 0.89029565**. The mean within-seed difference is **+0.00361072**, with conditional 95% paired-photo interval **[−0.00212575, 0.00932902]**. This does not establish superiority or equivalence. Polygraph here is Yishai's preserved full conditional output-MLP plus edge-gated GNN mixture. See [RESULTS.md](RESULTS.md), [SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md) and [final original evidence](run_records/final_comparison/INDEX.md).

Total campaign consumption, including failed attempts: **3,279 GPU-seconds (54m39s)** and **957 CPU-job wall-seconds (15m57s)**. The submitted full chain took **33m06s elapsed**, from 09:52:21 to 10:25:27. Its private Hugging Face archive is verified at immutable revision `903edb4f409a8a1af7617a991d2f3462dce9fb6d`, with all 2,361 members checked after download. See [PRESERVATION_REVIEW.md](PRESERVATION_REVIEW.md). No new scientific run or manuscript edit is pending in this authorization. After completing the report and independent review, the app confirmed heartbeat `monitor-polygraph-slurm-continuation` **PAUSED**. The scientific reviewer also checked the curated RESULTS.md against the original evidence and found no material discrepancy.

The sections below retain historical milestones and release decisions. Their old running/pending descriptions are not the current state.

### Historical milestone: full-cohort parity, September 24 at 10:19

The frozen full manifest records `mode: full`, `complete: true` and all **71,991** required rows. Maximum absolute confidence difference is `5.5730342864990234e-05`; maximum probability-margin difference is `1.1068582534790039e-04`, both below `2e-4`. Original class predictions and labels match exactly. This is the full extraction gate, not just the earlier 496-row panel.

Scheduler elapsed time was **26m45s / 1,605 GPU-seconds**. Nested manifest timers record decode 29.6618 seconds, inference 361.1612 seconds, parity 2.3698 seconds, model loading 82.3305 seconds and total extractor invocation 1,272.7933 seconds. They exclude/include different setup costs and must not be added as independent full-job timings. Completed source/model/processor provenance, the manifest and scheduler receipts are preserved in workspace `work/canonical-bundle-20260924/runtime_extraction_complete/`; compact Git evidence is in `run_records/extraction_complete/`.

At the milestone check, all three fits were running under their original identities, and downstream stages were pending dependencies. No training/validation scores were inspected or used to change the recipe. Completed GPU use is now **2,619 seconds (43m39s)** including failed attempts. Adding the full remaining fit/prediction reservations gives **3h13m39s** GPU; CPU actual plus remaining reservations is still **75m21s**. Refresh live elapsed time and remaining caps before any later repair. No additional submission or scientific change accompanied this milestone.

CPU repair job 924767 passed all 18 checks in 33 allocated seconds. The unchanged checker/data ran with the existing scikit-learn environment overlay correctly selected. All 75,000 plan records exist in the 1,010,000-record scan; all four photograph roles are disjoint; 17,000 test rows have exact ordered key, historical error-target and confidence agreement. Preserved predictions reproduce mean seed AUROC 0.8939063783160323, AP 0.874839083136075 and AURC 0.20710602279320625, matching the published rounded table. See `run_records/input_validation/validation_report_repair1.json`. Failed attempt 924758 remains preserved (24 allocated seconds).

The user supplied Yishai's four-artifact reproduction bundle, asked whether anything was missing, then explicitly asked us to continue instead of stopping at the readiness report. Continue the agreed canonical-main comparison autonomously within the documented resource ceiling; report concrete access or scientific blockers. No new architectures, sweeps, weather expansion, Overleaf edits or remote Git push.

Local branch: `revision/canonical-polygraph-logitdynamics`.
Starting commit: `5db4a6068458a608d6d7c2b1ba678210a41cf8aa`.
Unrelated untracked historical result directories remain untouched.

## Source artifacts and coordination

Incoming bundle, preserved unchanged:
`/Users/omrifahn/Downloads/poly24/logit_dynamics_reproduction_20260923`.

Local receipts and validation preparation:
`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924`.

Remote namespace:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`.

One remote operator only: agent `bundle_slurm_validation`. Scientific protocol/review: `bundle_protocol_audit`. Implementation: `ld_adaptation_readiness`. Root coordinates, reviews, records release decisions and commits. If any agent is interrupted, first reconcile receipts, remote markers, Slurm user/account/job name/submission time and actual hashes before resubmitting. Job IDs alone are insufficient.

All numerical checks, extraction, fitting, inference, statistics and heavy storage remain in Slurm allocations. Mac use is limited to source edits, inspection and coordination. Preserve every failed attempt and resume under identical frozen identities; do not erase a failure or replace a seed.

## Resource boundary

Campaign ceiling: 12 cumulative GPU-hours including preflights and failed attempts, at most three simultaneous GPUs; two cumulative CPU-job wall-hours. This is a ceiling, not an elapsed-time estimate. The initial ten-minute preflight cap was extended through the documented same-source 45-minute retry after measured fixed I/O caused a timeout. That retry passed in 406 allocated seconds. Record consumed time plus outstanding reservations. Do not expand the campaign ceiling or change the scientific method silently if the measured work does not fit.

## Full release — September 24, 09:52:21 Israel

Preflight **924803 completed at 09:42:38 Israel** in 406 GPU-seconds. All 496 fixed panel records passed exact labels and predicted classes. Maximum absolute confidence difference was `1.0132789611816406e-05`, and margin difference `1.913309097290039e-05`, below the frozen `2e-4` tolerance. Six feature/normalizer fixtures, semantic order, finite gradients, synthetic exact optimizer/RNG resume, save/load, changed-resume-identity rejection and weighted-AUROC literal-duplication fixtures passed. See `run_records/preflight_passed/INDEX.md` and `SCIENTIFIC_REVIEW.md`. This is panel success, not full-cohort parity or a final result.

The independent scientific reviewer gave runtime GO. Root reviewed actual receipts, exact configuration/source identity, zero prior full-phase submissions, measured throughput and remaining reservations, then explicitly approved **phase `full`** of the existing v4 configuration:

- Configuration SHA-256: `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
- Source manifest SHA-256: `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
- Campaign SHA-256: `f489a75f558d7588b946ebf5f5586e418b5473c156c6a5b88a4a071118a65268`.

| Stage | Job ID | Required predecessor / initial state |
| --- | --- | --- |
| Full extraction and every-row parity | 924822 | Sealed successful preflight; RUNNING on s-005 |
| Fresh LD fit, seed 1 | 924823 | Extraction success; pending dependency |
| Fresh LD fit, seed 2 | 924824 | Extraction success; pending dependency |
| Fresh LD fit, seed 7 | 924825 | Extraction success; pending dependency |
| Freeze all selected checkpoints | 924826 | All three fits; pending dependency |
| Test predictions, seed 1 | 924827 | Freeze; pending dependency |
| Test predictions, seed 2 | 924828 | Freeze; pending dependency |
| Test predictions, seed 7 | 924829 | Freeze; pending dependency |
| Canonical comparison and paired-photo bootstrap | 924830 | All three predictions; pending dependency |
| Private archive and immutable remote verification | 924831 | Evaluation; pending dependency |

Every job was checked against user `omrifahn`, account `gpu-students`, `canonical0924-085200-` names and the submission receipts. The chain is already submitted: **do not submit it again**. Full extraction must pass the same gates on all 71,991 rows before any fitting. Later stages require all seeds, not a successful subset. Jobs continue independently of the laptop; new scientific decisions still require agent review.

At submission, completed use was **1,014 GPU-seconds and 801 CPU-job wall-seconds**, including failures. Remaining GPU reservations total 570 minutes, so actual past use plus all outstanding reservations is **9h46m54s GPU**. CPU actual plus remaining 62 minutes is **75m21s**. Both remain below their ceilings; the submitter's more conservative configured budget also passes. These values must be refreshed before any repair.

Measured panel inference was 17.415 seconds for 496 rows, projecting about 42 minutes for the full cohort. Scaling even the entire 11.558-second panel decode timer adds about 28 minutes and conservatively repeats fixed table-loading cost; checksum/model/import startup is accounted for separately. Operational expectation is about **1–2 hours extraction plus 1–2 hours downstream**, excluding queues and failures. This is an estimate, not a guarantee or a seven-hour expected extraction time. Stage wall limits remain extraction 420 minutes, each parallel fit 30, each parallel prediction 20, freeze 2 CPU minutes, evaluation 30 and backup 30. Their sequential cap sum is 8h52, not the expected completion time.

Exact root approval, intents, submissions and initial queue/hash checks are in workspace `work/canonical-bundle-20260924/full_phase_receipts/` and `full_phase_submission_receipt.txt`; the current operator `HANDOFF.md` points to subsequent snapshots. Preserve all failed attempts and update this status from server evidence, never from stale in-process receipts.

First input validation reservation: two CPU minutes, two CPUs, 2 GB, zero GPUs. CPU job 924758 submitted September 24 at 08:52:34 Israel under `gpu-students`, name `canonical-0924-validate`. All structural checks passed before a missing scikit-learn import; wrapper-only repair 924767 passed with the existing overlay. Both attempts are preserved.

## Submitted preparation phase — September 24, approximately 09:14 Israel

- Dataset fetch 924773 failed after 151 CPU-job wall-seconds because the library selected a home-directory Xet cache with insufficient quota. Mechanical repair 924774 redirected runtime caches into the new project namespace and completed in 447 seconds. Source revisions and scientific inputs stayed unchanged; no old data or cache was removed.
- Downloaded source revisions: CIFAR-100 `aadb3af77e9048adbea6b47c21a81e47dd092ae5`; corruption dataset `a12f0bcc1da33fa26d8c76ce8c1fb32e6f913bea`. These are the downloaded revisions, not a claim that historical revisions had been recorded.
- Preparation/import/split validation: job **924782**. GPU preflight: **924783**, dependent on preparation. The already completed fetch is checked by its sealed completion receipt instead of an expired scheduler dependency.
- Approved configuration: `config_science_v1.json`, SHA-256 `ab7434d04b7a08f65098352a5176b794ae96bb999d0e53b6aed8fa5eaa03579a`.
- Released source manifest: `releases/science-source-v1/source_manifest.json`, SHA-256 `eebb28f35b3e138afb659e385f9e0893f9df2eb1c7262c6e47455d3e38ff0180`.
- Compact receipts are in workspace `work/canonical-bundle-20260924/reconciled_0914/` and `preflight_phase_submission_receipt.txt`; exact account/user/name/time checks remain required. No full extraction or LD fitting has been submitted at this checkpoint.
- Consumed CPU-job wall-time before preparation: 655 seconds including failed attempts. Preparation reserves up to 300 further CPU seconds; preflight reserves up to 600 GPU seconds. All remaining proposed stages plus these actual costs fit within 77m55s CPU wall-time and 9h40m GPU reservations, beneath the campaign ceilings. These reservations are not completion-time estimates.

The root approved only this bounded phase after independent static scientific review. Full DAG release follows actual parity/test results and measured throughput. Distinguish fixed model/parquet startup from per-row inference in the timing estimate; do not extrapolate an entire cold-start panel linearly.

Preparation attempt 924782 failed after 21 seconds while recording runtime versions because the old NumPy-based environment has no pandas distribution. The canonical parquet path also requires pyarrow. Core AST parsing/imports had succeeded, but the full runtime/reader gate had not. Preserve this failure and cancelled dependent job 924783. Root authorized preparation repair 1: an isolated pinned pandas/pyarrow overlay installed and tested in a CPU allocation of at most ten minutes, without altering the historical environment or scientific method. Reconcile the new release/configuration and successful CPU reader/import checks before a GPU retry. Cumulative CPU-job wall-time at this failure: 676 seconds; cumulative GPU time: zero.

### Successful replacement preparation

Dependency repair job **924784** passed in 82 seconds. It verifies pinned wheel hashes, imports and actual paths, canonical parquet row decoding/label agreement and the pinned cached fast processor producing finite FP32 `[1, 3, 224, 224]` output. The isolated overlay uses pandas 2.3.3 and pyarrow 21.0.0; the prior environment is unchanged. Complete-overlay manifest SHA-256: `b8e316b7e55718669695297cd1322fefcfcbbbe00697da2e59ed3439bc8e0008`.

Root reviewed and approved replacement phase `preflight` on September 24 at approximately 09:22 Israel:

- Configuration `config_science_v2.json`: SHA-256 `abb3ddf5aaa740cdf84cd2ec4b77e84d12c7697aaf9ce2cc930308b8a0fed75a`.
- Source `releases/science-source-v2/source_manifest.json`: SHA-256 `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
- Changes are limited to binding verified dependency bytes and preserving dependency wheels/manifests in backup. No scientific source, parity tolerance or model setting changed.
- **924785** `prepare_r1`, submitted 09:22:49 Israel, completed in 43 seconds. Sixteen AST files and eight core imports pass; campaign and role manifests are frozen.
- **924786** `preflight_r1`, submitted at the same time after 924785; latest verified state RUNNING on s-005. Full extraction/training remains gated on the actual preflight.
- Cumulative consumed CPU-job wall-time after preparation is **801 seconds**, including all failed attempts. Outstanding preflight reserves at most 600 GPU seconds.
- Local workspace runtime receipts: `runtime_prepare_r1/`, `reconciled_0922/` and `reconciled_0924/`, plus `preflight_r1_phase_submission_receipt.txt`.

### I/O timeout and same-source preflight recovery

Preflight **924786** timed out on September 24 at 09:33:40 Israel, consuming **608 GPU-seconds** including termination overhead. It produced no classifier parity result and no `preflight_cls` output. Administrative inspection showed an active process waiting for filesystem reads; a measured 421,788,404-byte read increase over 71 seconds was approximately 5.94 MB/s. The 96 parquet files total 2,151,903,646 bytes. Static review found two complete checksum passes before extraction output is created, approximately 4.30 GB of fixed I/O. A small panel therefore does not avoid this setup cost. No numerical deadlock or parity failure has been demonstrated.

The scheduler denied increasing the running job's time limit; preserve the amendment attempt and original timeout. Root then authorized a **45-minute total preflight retry**, with identical core source, classifier, inputs, tolerances, runtime and campaign. The old allocation is terminal and absent from the queue; the process-owned lock also prevents concurrent execution. No lock file is deleted. Draft v3 (30 minutes) was reviewed but never submitted and is retained separately.

Approved recovery configuration **`config_science_v4.json`**, SHA-256:
`38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
Same source manifest:
`7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
The approval is for **phase `preflight` / stage `preflight_r2` only**. The full phase still requires completed parity/numerical gates, independent review and measured feasibility. The new configuration retains the failed 608 GPU-seconds; all planned GPU reservations total **10h25m08s**, below the 12-hour ceiling. Consumed CPU-job wall-time remains 801 seconds, and conservative configured CPU reservations remain below two hours.

The repeated hashing is inefficient but the released scientific code is preserved for this retry. Do not silently optimize it inside the sealed release or delete provenance checks. Separate fixed cold-start I/O from per-image work in any eventual extraction estimate.

Recovery submission **924803**, name `canonical0924-085200-preflight_r2`, uses the sealed successful preparation receipt for 924785. Its immediate status was RUNNING on s-005, one GPU / four CPUs / 16 GB, excluding s-004. No full-phase jobs have been submitted. Exact submission and approval receipts are in the workspace operator directory; consult `HANDOFF.md` for the latest retrieval path.

The existing heartbeat `monitor-polygraph-slurm-continuation` is now **ACTIVE**, renamed **Polygraph — canonical LogitDynamics comparison**, at its existing ten-minute cadence. Its old September 21 prompt was replaced by this campaign's source paths, identities, budget and explicit preflight/full-release gates. It must stay quiet on unchanged state and pause after validated results, verified preservation and the final Hebrew report. Slurm jobs themselves continue when the laptop is closed; new agent review/submission decisions require the app's agents to be available.

## Required progression

1. Complete independent bundle validation, including reproducing the original per-seed and mean Polygraph metrics.
2. Freeze the new LD role allocation and full 85-feature recipe, preserving original targets and exact test membership; fit fresh heads/probe/scalers only.
3. Review implementation and run bounded Slurm preflight for raw-input/classifier parity, feature semantics, role isolation, save/reload and resumability. Stop on a semantic mismatch rather than changing targets or tolerances.
4. Review measured throughput and exact source/configuration hashes; submit a durable dependency chain only when all gates pass within the ceiling.
5. Freeze all three fitted seeds before comparison; evaluate using canonical metrics and paired source-photo uncertainty, with variable numbers of views per photograph.
6. Preserve compact results locally, complete code/environment/model provenance and private Hugging Face backup, independently review the comparison, and report plainly in Hebrew. No manuscript update is part of this phase.

Historical comparison methods remain unchanged. The supplied final predictions represent Polygraph's conditional output-plus-edge-gated mixture, not our later graph ensemble and not the standalone GNN. The original test has already been used in project development; this comparison is not a pristine new confirmatory test.
