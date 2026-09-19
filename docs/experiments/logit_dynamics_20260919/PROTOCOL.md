# Fixed LogitDynamics adaptation — September 19, 2026

Scientific contract approved by the coordinating agent on September 19 before
new fitting or development readouts. No experiment was run while preparing
this document. Execution also requires a separately recorded, bounded Slurm
resource reservation and completion gates.

## Question and scope

How does a fixed, practical LogitDynamics adaptation compare with the completed
September 17 GNN mean ensemble on the same frozen ViT-B, source photographs,
corruption views and error labels?

Call the method **LogitDynamics, fixed ViT-B adaptation (LD)**. This is neither
a numerical replication of the paper's ViT-L results nor an equal-supervision,
equal-parameter or equal-compute comparison. It is a retrospective development
comparison: the 800 evaluation photographs and historical G/S/H results have
already been examined, and G mean was chosen with those historical results
known. No new untouched-test or literature-priority claim is supported.

The method reference is [LogitDynamics v1, Sections 3 and Appendix A.2–A.3](https://arxiv.org/html/2604.10643v1#A1.SS3).
The paper separates auxiliary class-head fitting from error-probe fitting and
validation, freezes the backbone, uses a contiguous suffix of CLS states,
includes the original classifier as the final trajectory element, and applies
probe-training normalization. Its hyperparameter search and original ViT-L
are not reproduced here. The specific allocation and single configuration
below are project decisions made before observing LD performance.

Historical reference:
[September 16 protocol](../final_comparison_20260916/PROTOCOL.md),
[September 17 synthesis](../scientific_synthesis_20260917/HANDOFF.md), and the
preserved completed prediction artifacts. Reuse all historical fitted models
and predictions read-only; no G/S/H/O retraining or checkpoint reselection.

## Frozen source allocation

Keep the original cohort, nine views per source image, classifier, processor,
raw-image decoding, class order and error definition unchanged. Error label is
one when the frozen classifier's predicted class differs from the true class.
The grouping key is **image_id**, not the condition/source_id field.

| New role | Existing source photographs | Photographs | Records | Permitted use |
| --- | --- | ---: | ---: | --- |
| head_train | Fixed subset of base_train | 1,200 | 10,800 | Auxiliary class-head parameters |
| probe_train | Remaining base_train plus all meta | 400 + 400 | 7,200 | Error probe, scaler and positive-class weight |
| probe_val | All checkpoint | 400 | 3,600 | Error-probe epoch selection only |
| evaluation | All dev_eval | 800 | 7,200 | Frozen final readout only |

Within the existing 1,600-source base_train role, sort unique integer image_id
values by the ascending hexadecimal SHA-256 digest of the UTF-8 string
`logit_dynamics_20260919:head_split:v1:<image_id>`, with image_id rendered as its
ordinary decimal integer without leading zeroes. Break a hypothetical digest
tie by ascending numeric image_id. Assign the first 1,200 to head_train and
the other 400 to probe_train. Join all 400 existing meta sources to probe_train.
Write and checksum the explicit source/record manifest before any fitting.
No seed changes this membership; no outcome-based reshuffle is permitted.

This fixed hash split is an adaptation, not the paper's error-stratified split.
All nine views follow their source. Verify source disjointness, expected counts,
the 100-class head-training coverage, and both error labels in the probe fit
and validation pools on Slurm before fitting. A failed required coverage or
identity check stops the run for a documented amendment; do not silently
resample, borrow evaluation sources, or assume an additional clean-image pool.

This uses the same total 2,000 parameter-fitting sources as historical stacked
G/S/H, but supervision differs: LD uses 1,200 source class labels for head
fitting and 800 source error labels for the probe; the historical base detectors
used 1,600 source error labels, with 400 additional sources for stack fitting.
G mean itself does not fit on meta. The common 400-source checkpoint role has
also previously served historical-model selection. Report these distinctions
alongside results. Views are repeated measurements, not additional independent
photographs.

## One implementation and optimization configuration

Use training seeds **7, 17, 27**, with a fully separate auxiliary-head/probe
pipeline per seed and common source allocation. Each pipeline uses all its
training rows each epoch, including the final incomplete batch, with shuffled
training batches and no resampling to balance labels. Use FP32 model computation
(feature statistics and normalization may accumulate in FP64 before returning
FP32 features),
no dropout, no augmentation beyond the nine frozen views, no learning-rate
schedule and no early termination based on performance. Record initialization,
sampler/RNG state and package versions. The backbone stays in evaluation mode
with all its gradients disabled.

| Setting | Frozen value |
| --- | --- |
| Layers L | 12 contiguous blocks: human layers 1 through 12 |
| Competitor count K | 5 |
| Auxiliary model | Independent linear 768-to-100 map with bias per layer |
| Auxiliary optimizer | AdamW; lr 0.001; weight decay 0; betas (0.9, 0.999); eps 1e-8 |
| Auxiliary objective/batch/epochs | Unweighted multiclass cross-entropy; 512 rows; exactly 16 epochs |
| Auxiliary checkpoint | Final epoch 16 for every layer; no layer/head selection |
| Error model | Linear map with bias to one error logit |
| Error optimizer | AdamW; lr 0.001; weight decay 0.01; betas (0.9, 0.999); eps 1e-8 |
| Error objective/batch/epochs | Weighted binary cross-entropy with logits; 256 rows; exactly 100 epochs |
| Positive weight | Number of negative / positive rows in probe_train only |
| Probe checkpoint selection | Greatest probe_val average precision; earliest epoch on exact tie |

The paper does not specify error-probe weight decay or an epoch-level
checkpoint rule. Weight decay 0.01 is an explicit AdamW-default resolution;
the stated validation checkpoint rule is an explicit local choice. Neither is
searched. AP selection follows the paper's PR emphasis, while AUROC is the
primary comparison metric for continuity with the historical benchmark.

Report **922,800 auxiliary-head parameters** (12 × (768 × 100 + 100)) plus
**86 error-probe parameters**, for **922,886 trainable parameters per seed**.
The frozen ViT is additional shared inference cost. Describing the complete
LD method as an 86-parameter detector would omit its learned feature extractor.
Report extraction and fitting cost separately; parameters are not matched to G.

Extract CLS from the exact returned `hidden_states[t][:, 0, :]` for t=1..12;
the embedding output at index 0 is not a transformer block. Do not substitute
the historical sparse 3/6/9/12 layers, patch-token pooling, the final classifier
weights, or per-layer error heads. Do not insert an extra final LayerNorm into
the chosen raw hidden states. The original frozen classifier logits remain a
separate terminal vector. Preserve this implementation convention as part of
the adaptation's identity.

Build an 85-dimensional feature vector: 78 ordered numeric trajectory values
and seven dynamics measurements. The numeric block at each of the 13 depths
starts with the original classifier's predicted-class logit, followed by the
five largest other-class logits in descending order. The seven dynamics
measurements follow Appendix A.2 on full-class top-five identities, including
the predicted class when it belongs to that set. Use restricted, stable softmax
for their weighted overlap; natural-log identity entropy; and the appendix's
normalized commitment depth. Ties in class logits use lower class index first.
Keep a fixed named feature order and validate these distinctions on synthetic
cases in Slurm before fitting.

Fit the mean and population standard deviation of each feature on probe_train
only, separately for each seed's trained heads. Replace an exactly zero
standard deviation by one. Apply this scaler unchanged to probe_val and
evaluation. Neither class-head fitting nor normalization sees probe_val or
evaluation labels. Keep raw feature construction separate from standardization;
dynamics are calculated from raw logits. Rank using the selected probe's raw
error logits, which are monotonic in sigmoid probability and avoid saturation
ties from a finite-precision sigmoid. Increasing score means greater predicted
error; weighted training does not establish probability calibration.

## Compatibility gates and completion

Before training, bind the historical cohort, checkpoint, processor and model
revisions, class order and record IDs. Confirm the new extraction reproduces
existing frozen logits within absolute/relative tolerances 1e-5, with exact
predicted classes and exact labels, on the declared preflight records. Compare
overlapping H12 CLS states after casting the new extraction to FP16 and
promoting both new and stored values to FP32, with absolute tolerance 1e-4
and relative tolerance 1e-3. This upfront storage-precision allowance covers
an adjacent FP16 representable-value step and a small absolute floor near
zero; it replaces the historical extractor's unnecessarily strict 1e-5
comparison between quantized values. Record maximum absolute H12 and logit
differences. These tolerances are fixed before preflight. Any semantic or
numerical parity failure stops before head fitting. Extraction must retain checks for every exported
record; never publish a mismatched sidecar.

Before any evaluation readout require completed 16-epoch heads, completed
100-epoch probes, selected checkpoints, train-only scalers and convergence/
finite-value checks for all three seeds. Save epoch histories, chosen epochs,
weights, manifests, hashes and checkpoint predictions. These are completion
evidence; they do not authorize additional fits, altered settings, or post-dev
adaptation. A numerical failure or missing seed cannot be replaced by a shorter
fit, a different seed or a best-seed report.

All numerical testing, extraction, fitting, inference and statistics run in
Slurm allocations. Local work is limited to code, documents and reading already
produced artifacts. Follow the coordinating agent's independent baseline
resource cap; no automatic stronger hardware, scientific retry, sweep or
deadline extension. Preserve the previous experiment namespace and caches.

## Evaluation and permitted claims

The sole primary contrast is **G_mean AUROC minus LD AUROC**, computed per
corresponding seed and then averaged arithmetically over seeds 7/17/27. G_mean
is the historical equal mean of four within-seed sigmoid detector scores at
layers 3/6/9/12. Match prediction records by immutable record_id and verify
image_id, condition and error labels before comparing. Do not average LD or G
scores across training seeds and then score the resulting ensemble.

Use **2,000 paired image_id bootstrap draws**, RNG seed **20260919**, sampling
800 source photographs with replacement per draw and retaining all nine views
and common multiplicities across both methods and all three seeds. Each draw
recomputes the arithmetic mean of the three within-seed AUROC differences.
Report the ordinary 95% percentile interval with NumPy linear percentiles and
half credit for tied AUROC scores. No multiplicity adjustment is needed for
this one predeclared primary contrast. Record undefined draws without
replacement or redraw; withhold the interval if any draw is undefined.

Report LD and G_mean per-seed AUROC, mean AUROC and sample seed SD (ddof=1),
plus per-seed and mean **average precision**, specifically the threshold-grouped
average-precision definition used by scikit-learn, not trapezoidal PR area.
Do not silently equate this implementation with the paper's unspecified
numerical AUCPR integration convention. Include the fixed evaluation error
prevalence to make AP interpretable.

Present existing **S_mean, O, MSP and entropy** as secondary descriptive
comparators on the identical records; reuse predictions and report their
matching AUROC/AP summaries. Do not choose a different primary G recipe,
claim a statistically selected winner among this secondary set, add optional
ablation searches, or tune an alarm threshold on evaluation labels.

The primary interval describes source-sampling uncertainty conditional on the
three fitted seed pairs. It does not account for adaptive development-data
reuse, all future training seeds, or configuration selection in the original
paper and project. Positive, negative and inconclusive results all belong in
the report. A G_mean advantage would establish an advantage for these complete
recipes under their stated budgets, not graph necessity or superiority over a
fully tuned paper replication. An LD advantage would likewise not establish
that graph information is useless. No equivalence claim is predeclared.
