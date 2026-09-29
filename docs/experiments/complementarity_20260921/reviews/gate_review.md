# September 21 complementarity: executed-gate review

**Verdict: ACCEPT the cache, fitting, freeze and prediction gates on the saved evidence.** No blocking discrepancy was found. This is a post-execution review of receipt contents, source bindings and chronology, not an independent numerical rerun or full end-to-end reproduction. No numerical analysis, model import, SSH connection or job submission was performed for this review.

Reviewed on September 21, 2026. Local evidence root (all relative receipt paths below resolve here):

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/complementarity-20260921/ops/gates_snapshot/`

Frozen specification:

`/Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/complementarity_20260921/PROTOCOL.md`

## Provenance and scope

The local protocol SHA-256 is `1bf40959841d7a967263913e2e90a42f795844459d1c0ae1e155cf98786730ac`, matching `campaign.json`. All ten local scientific package source hashes match its `source_files` mapping. The experiment root is `complementarity_20260921_135802`; its parent is the preserved September 19 LD experiment and frozen `release-400f1fbbc105a370`.

Direct file hashing confirmed the campaign digest `52a9cad66bebfd276ea536d2389a8d5acc04309ed8ae36c0636e94def6671588` and global evaluation-gate digest `fb696c94828ad2a31c81e3ae896178e596dd7a46d011d88b9f87a9f54c1cb4a2`. The cache, ablation freeze, fusion freeze, nine individual fit receipts, all 27 downloaded fit config/history/normalizer files, fusion model JSON and inspected cache sidecar match the corresponding saved gate hashes. Heavy arrays and weights were not downloaded or rehashed in this review.

Executed stage receipts consistently name source manifest `4f2debe7aaa861c44538240ce624d4ff1ad4e4160dbe538f1f1298e24db08fed` and operational configuration `19c04dc936a958fa9864a008cf01171d4446a9b55734681294b38b1f6cbe31bd`. These are recorded execution identities; this review did not independently reconstruct the whole server release.

## Cache and preflight evidence

`cache/complete.json` contains exactly the planned seed/role combinations: seeds 7, 17, 27, each with probe_train 7,200, probe_val 3,600 and dev_eval 7,200 records. Every entry reports exact feature names and independent semantic maximum absolute difference 0. Training normalizer means and scales also report difference 0 for all three seeds.

The full semantic list in `cache/seed7/probe_train.json` agrees with the frozen 85-feature order: twelve six-value auxiliary blocks, the six-value original-classifier block, then `top1_switch_rate`, `topk_weighted_jaccard`, `unique_topk_count`, `top1_mode_frequency`, `top1_entropy`, `top1_unique_count`, `top1_commitment_depth`. All nine cache checks bind A=72:78, B=66:78, C=0:78, D=0:85. The nine readout configs preserve those exact slices.

Semantic records are 18–26 for training, 14400–14408 plus the known record 15321 for validation, and 21600–21608 for development, for every seed. The frozen code checks the minimum record in each declared condition and performs per-photo unique-record/nine-condition checks before issuing the cache receipt. Acceptance of actual metadata coverage here relies on that source-bound executed gate; it was not recalculated from arrays locally.

All six replay entries explicitly use the newly saved cache and the original D model/scaler:

| Seed | Role | Records | Maximum absolute difference | Violations | Tolerance |
|---:|---|---:|---:|---|---|
| 7 | probe_val | 3,600 | 0 | none | atol=rtol=0.0001 |
| 7 | dev_eval | 7,200 | 0 | none | atol=rtol=0.0001 |
| 17 | probe_val | 3,600 | 0 | none | atol=rtol=0.0001 |
| 17 | dev_eval | 7,200 | 0 | none | atol=rtol=0.0001 |
| 27 | probe_val | 3,600 | 0 | none | atol=rtol=0.0001 |
| 27 | dev_eval | 7,200 | 0 | none | atol=rtol=0.0001 |

`preflight/cpu_tests.json` records passed names/slices, tied-logit semantics, population/zero-scale normalization, semantic-row selection and exact nine-condition fixtures. `preflight/gpu_tests.json` additionally records exact resumed next update, sampler restoration and portable-model reload. This is a synthetic continuation fixture using production save/restore functions, not a forced interruption of all nine scientific runs. `preflight/statistics_tests.json` records passed weighted/literal duplication, AUROC/AP ties and zero weights, undefined-class handling, model JSON reload, within-photo ranking, exact quantiles/no dropped failures, and fixed-ID continuation. All are campaign-bound Slurm results.

## Fits and freeze chronology

All nine `runs/ablation/{A,B,C}/seed{7,17,27}/complete.json` receipts state 100 epochs and successful selected portable reload. Their histories start at epoch 1 and end at 100. All normalizers specify probe_train, 7,200 records and ddof=0. All configs retain the original head-constructor RNG prefix, batch 256, learning rate 0.001, weight decay 0.01, weighted BCE convention, no dropout and strict-best validation AP with earliest exact tie.

| Arm | Selected epoch, seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| A | 99 | 98 | 100 |
| B | 99 | 96 | 100 |
| C | 100 | 100 | 100 |

The maximum-AP selection itself was not numerically recomputed in this review. The selected epochs above are direct receipt values.

`fusion/models.json` contains all nine DG/DS/DDprime fits, each converged with fit weight 3,600. The recipe is float64, fit-only standardization, L2 (`l1_ratio=0` in installed sklearn 1.8.0), C=1, lbfgs, max_iter=1000, tol=1e-8, intercept, no class weighting and no warm start. Pairing is 7→17, 17→27, 27→7. `fusion/freeze.json` states 400 fitting photographs and that assessment metrics were not observed.

Submission receipts bind freeze job 915921 to successful jobs 915917/915918/915919/915920. Both prediction jobs depend on 915921. Source-bound execution timestamps, all UTC on September 21, show:

- Last readout fitting completed at 11:10:38.154715; fusion fitting had completed at 11:09:50.471246.
- Global freeze completed at 11:10:52.734215.
- Fusion prediction started at 11:10:53.760296; ablation prediction started at 11:10:54.032181.

Both prediction receipts bind the same evaluation-gate digest. Ablation has 12 model scores on 800 photographs/7,200 records, including reused D; fusion has 400 assessment photographs/3,600 records. This supports the required joint freeze before final predictions and the separate assessment cohorts.

## Not certified by this bounded review

This note does not certify final effect sizes, all 2,000 bootstrap draws, final report interpretation or the immutable private HF backup; those require their separate completion receipts/reviews. It does not claim new raw-image extraction, clean-environment recreation, independent retraining or a second numerical implementation of the entire pipeline. The first-50 timing receipt records zero failed draws and zero unexpected-error draws, but is not a substitute for final statistical completion.

No required pre-prediction gate is missing from the reviewed evidence. No additional experiment, tolerance change or refit is indicated by this gate review.
