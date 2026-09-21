# LogitDynamics — fixed ViT-B adaptation, September 19, 2026

This card is the Git entrypoint for the three completed Polygraph error detectors
trained with seeds **7, 17, 27**. They predict errors of one frozen CIFAR-100
classifier. This is a course-project research artifact, not a language-model
hallucination detector or a calibrated probability service.

## Immutable artifact location

- Private Hub repository: `omrifahn/polygraph-experiments`.
- Artifact revision: `86605781c0528e286483e305072f7787393da860`.
- Snapshot prefix: `logit_dynamics_20260919_114500/snapshot_910575`.
- [Browse the pinned snapshot](https://huggingface.co/omrifahn/polygraph-experiments/tree/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575).
- Backup manifest SHA-256: `f2e17250d314568bf7b5815b5eea7d79720e8c9bb24d35289f58530f974ed572`.

Access requires permission to the private repository. Use this immutable
revision rather than `main`. The snapshot contains 295 manifest-listed files;
the private status, revision, manifest digest and listed paths were independently
verified after upload. Later review documents do not change this result snapshot.

The final reviews, reuse guide, audit scripts and preserved failure/diagnostic
receipts are independently backed up at immutable review revision
`075380b8975fb29f95fa8727234f98cf00e43eaa`, prefix
`logit_dynamics_20260919_114500/review_20260919`.
[Browse the reviewed documentation](https://huggingface.co/omrifahn/polygraph-experiments/tree/075380b8975fb29f95fa8727234f98cf00e43eaa/logit_dynamics_20260919_114500/review_20260919).
All 38 review files were checksum-verified, and the original model snapshot's
296 Git blobs remained unchanged. Use the original artifact revision above for
model loading. The closing Git receipt/session update follows this immutable
review snapshot and records its revision; it does not change the scientific
results or the audit disposition.

For each `SEED` in `7`, `17`, `27`, paths relative to the prefix include:

- `runs/seedSEED/heads/model.safetensors`: twelve auxiliary class heads.
- `runs/seedSEED/probe/model.safetensors`: selected error probe.
- `runs/seedSEED/probe/normalizer.json`: that seed's training-only scaler.
- Corresponding `config.json`, `history.json`, `complete.json`, native
  `checkpoint.pt`, and optimizer/RNG continuation `resume.pt` files.
- `runs/seedSEED/predictions/dev_eval.npz`: frozen evaluation predictions.
- `source/`: exact source snapshot and protocol; `source/source_manifest.json`
  identifies the preserved source files.
- `campaign.json`, `evaluation_gate.json`, `role_map.json`, `evaluation/report.json`,
  `evaluation/scores.npz`, and `evaluation/bootstrap.npz`: provenance and results.

Raw images and extracted CLS tensors are intentionally absent from the Hub
snapshot and remain on Slurm. Replaying raw-image extraction requires the pinned
classifier, processor, source manifests and documented input data. See
[reuse guidance](REUSE.md) for loading/reproduction boundaries and the
[reproducibility audit](AUDIT_REPRODUCIBILITY.md) for what was actually tested.

## Method and training

The method adapts [LogitDynamics v1](https://arxiv.org/html/2604.10643v1) to
`edumunozsala/vit_base-224-in21k-ft-cifar100` revision
`b0c51e4a5e5bda35cc922419a28df93bb87e6efa`, with processor
`google/vit-base-patch16-224-in21k` revision
`b4569560a39a0f1af58e3ddaf17facf20ab919b0`.

The backbone stays frozen. Raw post-block CLS states from all twelve blocks are
stored in FP16 and promoted for FP32 model computation. Twelve independent
linear 768-to-100 heads are fitted on 1,200 source photographs (all nine views),
using AdamW, learning rate 0.001, weight decay zero, batch 512, exactly 16 epochs,
and the final epoch. Together they have 922,800 parameters.

The original classifier's final logits append a thirteenth trajectory element.
Five competitor logits per depth plus the predicted-class logit, together with
seven identity/commitment dynamics, yield 85 features. A separate linear error
probe has 86 parameters: **922,886 trainable parameters per seed in total**,
excluding the shared frozen backbone. Greater raw probe score means greater
predicted error; sigmoid output is not established to be calibrated.

The probe is trained on 800 other source photographs; 400 additional photographs
select its checkpoint. Features are standardized only on probe training data.
Training uses weighted BCE, AdamW, learning rate 0.001, weight decay 0.01, batch
256 and exactly 100 epochs, choosing maximum validation average precision,
earliest on an exact tie. Selected epochs are 100, 95 and 100 for seeds 7, 17 and
27. All views of one source remain within its role. No settings were selected
after seeing the new evaluation scores. Full conventions, including top-K,
ties, role hashing and bootstrap, are in [PROTOCOL.md](PROTOCOL.md).

## Completed development comparison

Evaluation uses the same 800 source photographs and nine views as the historical
Polygraph comparison: 7,200 records, 2,271 classifier errors (31.54%).

| Seed | LD AUROC | Historical G_mean AUROC | LD average precision |
| --- | ---: | ---: | ---: |
| 7 | 0.899177 | 0.891193 | 0.800041 |
| 17 | 0.896264 | 0.891829 | 0.793786 |
| 27 | 0.898756 | 0.883662 | 0.803290 |
| Mean | 0.898066 | 0.888895 | 0.799039 |

Historical G_mean mean AP is 0.779509. The prespecified mean within-seed paired
AUROC contrast, **G_mean minus LD**, is **−0.0091712415**, with a 95% paired
source-photograph bootstrap interval **[−0.0158737735, −0.0023608076]** from
2,000 shared draws. No draw was undefined. Seed metrics are averaged; scores
are not ensembled across seeds. AP means sklearn average precision, not
trapezoidal PR area. Secondary descriptive results and all exact values are
preserved in the original report.

## Interpretation and limits

LD outperformed this historical graph ensemble in all three fitted seed pairs.
The comparison therefore does not support superiority of our GNN over this
published-method adaptation. It does not prove that topology is useless or
isolate the benefit of the seven dynamics features.

This is a retrospective development comparison: the evaluation photographs
were previously examined and G_mean was chosen with historical results known.
The interval is conditional on these fits, not an independent confirmatory
test or a characterization of all future training runs. LD and G_mean differ
in supervision allocation, capacity, information and training budget. This is
a fixed ViT-B adaptation, not the paper's full ViT-L hyperparameter study.
Two selected probes reach the fixed epoch limit; no convergence or optimality
claim is made. Do not transfer these numbers into a colleague's differently
sampled table without a distinct protocol explanation.

## Provenance and preservation

The immutable scientific release is `release-400f1fbbc105a370`; its archive
SHA-256 is `07f81a1dcd5fe0c0e24e490f023de6c436218b88465be94d365073a3c0fcf2b2`.
The scientific implementation was locally committed as `4310b9d` on
`September-10th`. Subsequent local commits preserve operations and review
documents. No branch push or merge occurred; the private Hub source copy makes
the trained models independent of unpublished files on the Mac.

All compute ran on Slurm. The original September 19 GPU allocations, including
the prediction timeout and successful recovery, totaled 14,376 seconds (3:59:36).
The recovery changed scheduling only; all failed attempts and source identities
remain preserved. Final CPU artifact checks are documented separately and add
no GPU training. See [SESSION.md](SESSION.md) for operational receipts and
[AUDIT_SCIENCE.md](AUDIT_SCIENCE.md) for the independent source/report review.

The initial portable CPU replay (910647) passed downloaded-file integrity,
exact portable/native weights, histories, source roles and training scalers,
but exceeded its fixed `atol=rtol=1e-4` for seed 17 validation scores
(maximum absolute difference 0.0007970333). This audit failure is preserved;
the CPU replay is not certified by the existence of the backup. Diagnostic
910651 isolated one of 3,600 seed 17 validation rows and showed
FP32/FP64 CPU top-five ranking sensitivity at a near tie; the higher-precision
head calculation matched the saved GPU score. Historical GPU intermediates
were not saved, so the exact original cause is not established. The separate
saved-metric audit passed for all 14 vectors, seed summaries, metadata and
stored-bootstrap aggregation. The original strict CPU failure is not cleared.
On September 21, completion audits 915652 (CPU) and 915653 (CUDA) ran all six
seed/role replays from portable weights and cached CLS inputs. CUDA reproduced
all original score values exactly. CPU retained only the same validation
violation; all development scores and seeds 7/27 validation passed the unchanged
tolerance. Small CPU metric differences are recorded in the detailed audit.
All 2,000 bootstrap draws were independently recomputed using scikit-learn,
agreeing with paired differences to 3.33e-16. The GPU replay added 283 allocated
seconds, bringing total GPU usage to 14,659 seconds (4:04:19). The CPU companion
used 279 allocated seconds without a GPU. The existing environment and saved
CLS inputs were used; a clean installation or fresh raw-image pipeline was not
tested. Original GPU scores and the completed scientific comparison remain
unchanged. Consult
[AUDIT_REPRODUCIBILITY.md](AUDIT_REPRODUCIBILITY.md)
for the final status before claiming cross-device reproducibility.
