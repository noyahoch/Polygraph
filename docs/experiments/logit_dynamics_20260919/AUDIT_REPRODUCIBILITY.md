# Engineering and reproducibility audit — September 19–21, 2026

The completed fixed LogitDynamics run has all three training pipelines, a common
freeze, paired predictions, evaluation and a private Hugging Face snapshot.
The source and saved JSON records support the provenance and completion findings
below. The September 21 completion audit replayed every validation/development
score from the published portable weights on CPU and CUDA. CUDA scores matched
the original values exactly; CPU retained one validation tolerance violation.
All development CPU scores passed tolerance. Every bootstrap draw was also
independently recomputed. No local tensor loading or numerical testing was used.

## Current disposition — September 21

Jobs **915652** (CPU) and **915653** (CUDA) completed at 12:29:49 and 12:29:52
Israel, using 279 and 283 allocated seconds respectively. Each loaded the six
published safetensors and three scalers inside the existing server environment;
native and portable parameter tensors matched exactly. The CUDA allocation
reported an NVIDIA GeForce RTX 2080 Ti. Both audits executed all six cases;
the score tolerance remained `atol=rtol=1e-4`.

| Seed / role | Records | CPU maximum absolute difference | CPU violations | CUDA maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| 7 / validation | 3,600 | 3.4570693969726562e-6 | 0 | 0 |
| 17 / validation | 3,600 | 0.0007970333099365234 | 1 | 0 |
| 27 / validation | 3,600 | 2.384185791015625e-6 | 0 | 0 |
| 7 / development | 7,200 | 3.337860107421875e-6 | 0 | 0 |
| 17 / development | 7,200 | 2.4437904357910156e-6 | 0 | 0 |
| 27 / development | 7,200 | 3.814697265625e-6 | 0 | 0 |

CUDA had exact score-value equality for all cases (`np.array_equal`, not a
separate byte-representation comparison). CPU execution completed, but its
aggregate `tolerance_passed` is false. The sole failing row remains validation
record 15321 / image 659 for seed 17: saved score −1.778078317642212, CPU replay
−1.7788753509521484. The old failure was not erased or cleared. CPU development
metrics are not universally identical: seed 7 AUROC is 0.8991769878197307 versus
saved 0.8991770324874779; AP is 0.8000408005318231 versus 0.8000409413412997.
These new CPU scores were not substituted into the scientific results.

The CPU job independently recomputed all 2,000 shared source-photo bootstrap
draws for all three seed pairs using `sklearn.metrics.roc_auc_score` with photo
multiplicity weights, rather than the production `WeightedAUC` implementation.
All 12,000 weighted AUROC calculations completed. The maximum per-seed paired
difference discrepancy was 3.3306690738754696e-16; no draw was undefined. The
recomputed interval was [−0.015873773489983915, −0.002360807640635412], agreeing
with the saved interval at absolute tolerance 1e-12. All 14 saved point-metric
vectors and seeded source multiplicities were independently checked as well.

Exact receipts and direct post-run source/result-preservation evidence are in
`run_records/replay_completion_20260921/`. The audit script is
`pilots/logit_dynamics_20260919/ops/audit_replay_completion.py`, SHA-256
`582a308a4b8b0ba4f1ef0f0114129761989434d2768f5d663e8bc589ab3dde55`.
The original report, scores, bootstrap, freeze gate, source and failed audit
remain unchanged. See [the completion record](REPLAY_COMPLETION_20260921.md).

**The previously unexecuted replay cases are now complete.** Portable GPU
checkpoint-to-score replay from the cached CLS inputs is verified in this
environment. The one CPU tolerance failure remains a documented limitation.
This does not test a clean installation, fresh raw-image-to-output pipeline,
full training rerun or replication of the paper's broader experimental study.
The detailed September 19 record below is retained as historical evidence.

## Artifact identity

- Experiment: `logit_dynamics_20260919_114500`.
- Executed source: `release-400f1fbbc105a370`, 133 source files.
- Source-manifest SHA-256: `d78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3`.
- Campaign SHA-256: `f133671b92871129a0e859e47188c316e8ef535634a1fda33a9efa287dfa094f`.
- Role-map SHA-256: `7df831d2e9de37fe2a5792a409949cbd7d46a9f24adc4ed16239f2b91fdeca3b`.
- CLS-manifest SHA-256: `9358a39f19c00068aa3f29030e90a3a53fede91172b81077822a053430ff5877`.
- All-three-seed gate SHA-256: `8918ff80c58a74e53a747fb0c054fe18b93e0f3dd06d64ebb1ff1d42741c4ab0`.
- Private model repository: `omrifahn/polygraph-experiments`.
- Published weight revision: `86605781c0528e286483e305072f7787393da860`.
- Published prefix: `logit_dynamics_20260919_114500/snapshot_910575/`.
- Backup-manifest SHA-256: `f2e17250d314568bf7b5815b5eea7d79720e8c9bb24d35289f58530f974ed572`.

The original frozen scientific release remains unchanged. Later audit and
operational documentation are separate artifacts. The ops transfer receipt
records 44 newly fetched published files, all matching the pinned authenticated
backup manifest. This establishes transferred byte identity; model execution is
addressed separately by the CPU replay.

## Training, selection and data use

The preserved receipts record:

| Seed | Auxiliary epochs / checkpoint | Probe epochs | Selected probe epoch | Selected validation AP |
| --- | --- | --- | --- | --- |
| 7 | 16 / 16 | 100 | 100 | 0.7941545257603159 |
| 17 | 16 / 16 | 100 | 95 | 0.7945446561607754 |
| 27 | 16 / 16 | 100 | 100 | 0.8019755629531333 |

The CPU audit checked every epoch entry and reconstructed the strict-greater,
earliest-tie AP selection from each entire 100-epoch validation history. It also
compared that choice with the native checkpoint's embedded epoch and identity,
and recomputed AP from the stored validation predictions. Selecting epoch 100
for two seeds is an endpoint observation, not evidence of convergence; no epoch
budget was extended.

`train.fit_heads()` loads only `head_train`, uses true CIFAR-100 class targets,
and keeps the final epoch-16 heads. `train.fit_probe()` loads only `probe_train`
and `probe_val`; its normalizer and positive weight derive from `probe_train`.
The normalizer files identify 7,200 fitting rows, population standard deviation
(`ddof=0`), and the role `probe_train`. The probe configuration binds the exact
normalizer hash. The source split is shared across seeds and separates
1,200/800/400/800 source photographs, with all nine views together. The final
audit independently checked the saved role/index membership and repeated scaler
construction from all probe-training CLS features with the fixed published
heads. All three scalers passed the fixed tolerance. It fitted no head or probe.

The historical G/S/O predictions are read from the immutable baseline NPZ.
`evaluate.align_metadata()` demands exact equality of all eight integer metadata
columns, including record ID, photograph ID, condition, label, classifier
prediction and error label. Independent checks against the frozen CLS index,
the historical score vectors, and the final 7,200-by-14 score matrix were not
reached by the failed first audit. The separate bounded diagnostic completed
these checks successfully before investigating the CPU discrepancy.

## Completion gates and recovery

`protocol.freeze()` requires complete heads and probes for all three seeds,
verifies their files, and seals their hashes in `evaluation_gate.json` before
any development prediction is permitted. Every prediction rechecks this gate.
The three prediction receipts bind the same gate and their respective selected
epochs. Seed 7's completed output from original job 910063 was retained and
verified by recovery job 910571; seeds 17/27 completed in 910572/910573. Recovery
changed scheduler routing and prediction scheduling only. It did not retrain,
replace weights, alter source, change scoring or select new settings.

All final evaluation files are hash-bound by `evaluation/complete.json`.
Published copies of weights, histories, configurations, scalers and predictions
are bound by the backup manifest. These receipts prove the checks recorded by
the producing jobs; the additional CPU audit provides a separate reload and
replay check of the published portable files.

## Numerical checks and their limits

The full extraction receipt reports exact labels and predicted classes and
maximum absolute differences of **zero** for both final classifier logits and
FP16 CLS12 over all 28,800 exported records. The upfront allowed tolerances were
`atol=rtol=1e-5` for FP32 logits and `atol=1e-4, rtol=1e-3` for both quantized
CLS12 tensors promoted to FP32. Intermediate blocks 1–11 were shape/finite
checked; there is no historical all-layer reference comparison to claim.

The preflight checked gradients, trained-parameter changes, serialization,
next-step optimizer restoration, feature formulas, source grouping, and paired
bootstrap semantics. Production prediction reloaded native `.pt` checkpoints.
The `.safetensors` versions were saved and hashed but were not used by production
prediction; this is why the final portable reload audit is necessary.

The final CPU audit has the following fixed scope:

1. Verify the frozen release, role/index bindings, gate, all complete-fit files,
   and all 44 freshly fetched published files against both manifests and the
   original artifacts.
2. Load the six published safetensors files and three published scalers. Require
   exact equality of every native and portable parameter tensor, including
   dtype, shape and keys; verify the full training histories and selected AP.
3. Rebuild every seed's scaler from all 7,200 probe-training rows and replay all
   3,600 validation and 7,200 development scores on CPU. Apply the predeclared
   `atol=rtol=1e-4` to scaler values and CPU-versus-GPU score comparisons. Preserve
   any mismatch; do not relax tolerance or change the primary result afterward.
4. Independently recompute AUROC and average precision with scikit-learn for
   all 14 saved score vectors, and check the paired primary point estimate.
5. Verify 2,000-by-800 source multiplicities, three-seed draw means, undefined
   draw inventory and stored-draw percentile limits. It does **not** recompute
   every bootstrap draw's weighted AUROC, nor rerun the raw-image backbone.

This tests freshly downloaded Hugging Face artifacts inside the **existing
pinned server environment**. It does not establish a clean installation on a
new machine, complete environment portability, bitwise CPU/GPU equality, or
arbitrary-image end-to-end inference. The pinned source uses absolute campaign
paths and existing baseline/cache resources. See [REUSE.md](REUSE.md) for the
precise artifact layout and practical boundaries.

## Original replay and diagnostic — September 19 (historical)

CPU Slurm job **910647** failed after 115 allocated CPU-job seconds (30-minute
outer cap, 25-minute internal limit, six CPU threads, 16 GB; no GPU or fitting).
The audit script is
`pilots/logit_dynamics_20260919/ops/audit_reproducibility.py`, SHA-256
`acec13a4899263c2069faca3e45c9367956b308ba56fe5bdc965b02287ff9b0e`.
Its receipt is written separately at
`audit/reproducibility_cpu_v1.json`; it never overwrites scientific outputs.
The exact execution configuration hash is
`dda0c746fc9aa9398480e0916917058d1f3dd87d9e30268263cd4d917410b6b3`.

The preserved failure receipt has SHA-256
`68caa27e659bbe46f7d71c2e5e8d6bcf2a2f15065a32662c0194fe369e5c0d53`.
It records successful verification of all 133 source files, 44 freshly fetched
published files, exact native-versus-portable equality for all six weight files,
all three complete histories and selected validation AP values, and all three
train-only scaler reconstructions. Seed 7 validation replay passed, with a
maximum absolute score difference of `3.4570693969726562e-06`.

Seed 17 validation replay failed the predeclared `atol=rtol=1e-4`, with maximum
absolute difference `0.0007970333099365234`. The first audit did not record the
violation count or affected IDs. Seed 27 validation, all development-score
replays, the independent 14-vector metric checks and stored-bootstrap
aggregation checks were not reached. No tolerance, weight, source, reference
score or primary result was changed.

A separate diagnostic script, `ops/audit_cpu_diagnostic.py`, SHA-256
`7b1ac1267bafb5c13c621a7f487cc723029c253df0206d713f1a7a8d3a3cae1d`,
completed as job **910651** in 92 allocated CPU-job seconds, within one
ten-minute allocation with six threads and 16 GB. Its separate receipt is
`audit/cpu_precision_diagnostic_v1.json`, SHA-256
`9c6ddea512a82f0624eed238d8100d23e0c0b1e4eb016c49563bd380a82c2343`.
It reports:

- Exact saved metadata alignment, exact historical score vectors, all 14
  independent AUROC/AP calculations, seed means/sample standard deviations,
  error prevalence and the paired primary point estimate passed. Metric
  comparisons used absolute tolerance `1e-12`.
- The 2,000 source-count vectors, within-seed draw averaging, undefined-draw
  inventory and stored-draw percentile interval passed. Individual weighted
  AUROC bootstrap draws were not recomputed.
- Exactly **one of 3,600** seed 17 validation records exceeded the original
  CPU replay tolerance: role row 921, record ID 15321, photograph ID 659.
- On that row, FP32 and FP64 CPU head calculations change depth 4 top-five
  membership at a fifth/sixth logit margin of `2.384185791015625e-07`.
  Maximum head-logit difference is `3.053481968606775e-06`; top-one classes do
  not change. The `unique_topk_count` and `topk_weighted_jaccard` contributions
  explain the CPU precision sensitivity: their weighted differences are
  `-0.043774851674839965` and `0.04297739644706127`, respectively. The combined
  dynamics contribution is `-0.0007974552277786942`; numeric-feature
  contribution is `4.6178694634651466e-07`.
- FP64 CPU heads with the unchanged FP32 probe reproduce this saved GPU score
  exactly (`-1.778078317642212`). Promoting only the probe arithmetic leaves the
  score near the differing FP32 CPU result. These diagnostic values were not
  substituted into the scientific outputs.

This establishes a near-tie sensitivity in the CPU calculations. Historical
GPU head logits and feature vectors were not saved, so it does **not** prove
the exact historical GPU ranking responsible for the discrepancy. The receipt
explicitly retains `original_CPU_replay_failure_cleared: false`.

At the September 19 closure, the completed original GPU comparison and independently checked saved metrics
remain unchanged. Published portable weights loaded exactly; full CPU score
replay **did not pass**. Seed 27 validation and all development scores remain
unverified by fresh CPU inference. No additional GPU replay, fitting, threshold
change or scientific rerun is pending. The audit closes with these explicit
portability limits. The September 21 section above supersedes the statements
about unexecuted cases and no pending replay; it preserves the original failure.
