# Canonical Polygraph versus LogitDynamics — completed September 24, 2026

The comparison is complete on **the exact 17,000 original test views from 1,998 source photographs**, using seeds **1, 2 and 7**. Polygraph's mean AUROC is 0.89390638; the fixed LogitDynamics adaptation reaches 0.89029565. The difference is small and its paired-photo confidence interval crosses zero. These results establish neither superiority nor equivalence.

## Why we ran this comparison

The team wanted a literature-based LogitDynamics baseline on the original Polygraph benchmark. Omri's earlier LD comparisons used his separate experimental protocol and could not directly answer this question. Yishai supplied the original train/evaluation plans, scan records and final Polygraph scores, allowing us to recover and verify the exact original examples, splits, targets and reference result.

The reference is **Yishai's complete conditional output-MLP plus edge-gated GNN mixture**, not a standalone GNN and not Omri's later layer ensemble. Each supplied seed score already averages five gate-fold models. We reused those scores without retraining Polygraph, and trained fresh LD auxiliary heads and error detectors under the frozen [protocol](PROTOCOL.md). The ViT remained frozen. This is a fixed ViT-B adaptation of LD, not a full replication of the original paper's experiments or search.

## What was checked and run

- Input validation reproduced the supplied Polygraph result and confirmed original row identities, ordered test membership, labels and disjoint source-photo roles.
- We extracted the required 71,991 views: 52,000 detector-training rows, 2,991 checkpoint-selection rows and 17,000 test rows. The original 3,009 meta-validation rows were left unused by LD. The 52,000 rows were split by source photo into 31,328 auxiliary-head training rows and 20,672 error-probe training rows.
- Full extraction reproduced every original class prediction and label. Maximum confidence and probability-margin absolute differences were 0.00005573034 and 0.00011068583, respectively, within the predeclared 0.0002 bound. No tolerance was widened.
- Each seed completed 16 epochs for its 12 auxiliary linear classifiers and 100 epochs for the 85-feature linear error detector. Validation AP selected probe epochs 78, 95 and 96 for seeds 1, 2 and 7. All selected models were frozen together before the final comparison.
- Feature semantics, train-only normalization, finite gradients, save/load, synthetic optimizer/RNG resume, and weighted-versus-duplicated AUROC checks passed. These checks and successful artifact preservation do not constitute a new clean-environment, raw-image end-to-end replay.

## Results

AUROC and average precision (AP) are higher-is-better; AURC is lower-is-better. The main table reports averages of per-seed metrics, not metrics of cross-seed averaged predictions.

| Method | Mean AUROC | AUROC seed SD | Mean AP | Mean AURC |
|---|---:|---:|---:|---:|
| Polygraph full mixture | 0.89390638 | 0.00045790 | 0.87483908 | 0.20710602 |
| LogitDynamics, fixed adaptation | 0.89029565 | 0.00053918 | 0.87330330 | 0.20995887 |

| Seed | Polygraph AUROC | LD AUROC | Polygraph minus LD |
|---|---:|---:|---:|
| 1 | 0.89388170 | 0.89016422 | +0.00371747 |
| 2 | 0.89437612 | 0.88983434 | +0.00454178 |
| 7 | 0.89346132 | 0.89088840 | +0.00257292 |

The predeclared primary contrast is the **mean within-seed AUROC difference: +0.00361072**. Its conditional 95% paired-photo percentile interval is **[−0.00212575, 0.00932902]**. All 2,000 predetermined draws completed, with no invalid or replacement draws. Each resampled source photograph retains all its selected views; paired weights are shared by methods and seeds. The RNG seed is 20260924 and endpoints are the 0.025 and 0.975 quantiles with linear interpolation. AP and AURC are descriptive secondary metrics.

Full precision, per-seed AP/AURC, input hashes and bootstrap metadata are in the unmodified [machine-readable report](run_records/final_comparison/evaluation/report.json). The independent checks are documented in [SCIENTIFIC_REVIEW.md](SCIENTIFIC_REVIEW.md).

## What this supports, and what it does not

This supplies the previously missing LD comparison on the team's exact original evaluation examples. The preserved Polygraph mixture has a numerically higher AUROC in all three matched seeds, but the photo-resampling interval includes zero. We cannot conclude a reliable advantage or equivalence, and this comparison does not isolate the contribution or necessity of graph topology.

The evaluation rows and error labels match exactly; the complete training procedures do not. LD uses a separate auxiliary-head/probe supervision allocation, whereas Polygraph has its own detector and gate training stages. This is not a matched-compute or identical-stage-supervision experiment.

The benchmark had already been used in project development. Its selection by classifier correctness constructs **50% errors**: AP and AURC describe that selected benchmark, not natural deployment prevalence. The uncertainty interval conditions on the fitted models and does not account for prior configuration selection, development-data exposure, or variability from training future seeds. The method tested is the specified ViT-B adaptation; no universal claim about LD or GNNs follows.

## Timing and preservation

All times below are September 24 in Israel. CPU durations are allocated job wall-time, not the sum across CPU cores.

| Stage | Time window | Allocated duration |
|---|---|---|
| Full extraction | 09:52:21–10:19:06 | 26m45s, one GPU |
| Three parallel fits | 10:19:07–10:21:52 | 2m45s / 2m42s / 2m44s, one GPU each |
| Joint freeze | 10:21:52–10:22:13 | 21s, CPU |
| Three parallel predictions | 10:22:13–10:23:10 | 56s / 56s / 57s, one GPU each |
| Evaluation and bootstrap | 10:23:12–10:23:49 | 37s, CPU |
| Private backup and downloaded-byte verification | 10:23:49–10:25:27 | 1m38s, CPU |

The full submitted chain took **33m06s** elapsed. Total campaign use including earlier preparation, preflights and failures was **54m39s of GPU time** and **15m57s of CPU-job wall-time**, below the 12-GPU-hour / 2-CPU-job-hour ceilings. No jobs remain active.

The fast fitting stage operated on cached CLS features with linear heads and probes; it did not retrain the ViT or run a new graph extraction/training pipeline. Scheduler caps and the earlier conservative estimates were not actual durations.

Exact code, environment metadata, configurations, original input plans/scores, split manifests, selected and resumable model states, predictions and all bootstrap outputs are preserved in the [private immutable Hugging Face snapshot](https://huggingface.co/omrifahn/polygraph-experiments/tree/903edb4f409a8a1af7617a991d2f3462dce9fb6d/canonical_ld_20260924_085200/snapshot_v1). Verification downloaded the six snapshot files and checked the archive and all **2,361 member hashes**. Raw image files and full CLS shards remain on Slurm; their manifests and reconstruction code are in the archive. See [preservation review](PRESERVATION_REVIEW.md) and [original receipts](run_records/final_comparison/INDEX.md).

The final human review and this report were written after that immutable snapshot and are preserved in local Git. No push, merge, new experiment or manuscript modification is part of this completion.
