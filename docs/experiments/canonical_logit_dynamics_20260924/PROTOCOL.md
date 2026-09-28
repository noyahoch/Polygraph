# Canonical Polygraph versus fixed LogitDynamics — September 24, 2026

## Scientific freeze and scope

This specification fixes the scientific choices before any new LogitDynamics
training or evaluation results. The user requested continuation on September 24.
Execution uses the separately recorded Slurm resource reservation and validated
input artifacts; this document does not create an additional permission step.

Question: how does the existing, complete Polygraph main-benchmark method compare
with the project's fixed full LogitDynamics adaptation on the **same original
test records and original classifier-error targets**?

Run one fresh three-seed LD experiment. Do not retrain Polygraph, its experts,
its gate, or the ViT. Do not add fusion, decompositions, architecture searches,
cross-fitting, weather experiments, or post-result configuration changes.
This is a ViT-B adaptation of LD, not a replication of the paper's ViT-L result
or an equal-supervision/equal-compute comparison.

## Authoritative inputs and Polygraph identity

The supplied `logit_dynamics_reproduction_20260923` bundle contains these original
artifacts, identified here by repository-relative paths and SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `runs/research_20260830/combiners/strict/main/detector_train_plan.json` | `44e1aef95e78be4d82d5582d6629582e09165b996c3f8b4da653ef5bb9be35be` |
| `runs/research_20260830/combiners/strict/main/combiner_eval_plan.json` | `e599e06cf2e7a7cad8652e9fc29feae1eca78479c2830263b2a09424d21d2091` |
| `data/graph_dataset/scan_records.jsonl` | `a999d53f7a71f79794f882809fdfa6541126ec55c8c347752d520b7de88dfe4f` |
| `runs/topology_depth_last4_20260905/final/best_scores_main.npz` | `186a7b76ad21a4f11fe59f46fbdc9763a3362e157e3be592732d4d0c3e74e737` |

Preserve the bundle bytes and independently verify these identities in the
operational intake. Record its README's assertions separately from our checks.
Canonical code reference is `9af8890405297c0f37dbaa54f6be29a84cb0c37e`.

The comparator is the **conditional output-MLP plus edge-gated GNN mixture**,
selected by mean out-of-fold meta-validation AUROC. Its NPZ directory-derived
name `A_edge_gated_mean` must not be mistaken for the standalone GNN. Canonical
`scripts/finalize_topology_depth_last4.py:157–186` writes these selected gate
scores. Use `score_seed1`, `score_seed2`, and `score_seed7` exactly as supplied.
Each already represents the corresponding five-fold gate ensemble. Do not
average scores across training seeds to manufacture a different comparator.

Recompute the preserved comparator's metrics as an integrity check against the
published main-benchmark table: mean AUROC 0.89391, AP 0.87484, AURC 0.20711
(rounded publication values). This check does not authorize changing the
selected comparator. Report any discrepancy before comparing new LD results.

## Records and stage allocation

A record is `(source, severity, base_index)`. Its source-photograph key is the
canonical `RecordKey.group_id`: `test:{base_index}` for clean-test and all
corruptions; official clean-training images would have `train:{base_index}`.
Require that the supplied main plans contain no `clean_train` records.
Require uniqueness of each record within its role and source-photo disjointness
across the four original roles. Check the two plans' test lists are identical
in order, and detector-validation equals combiner nominal-training in order.

| Original role | Rows | Source photographs | New permitted use |
| --- | ---: | ---: | --- |
| Detector train | 52,000 | 6,985 | Subdivide into class-head and error-probe fitting pools |
| Detector validation / base-validation | 2,991 | 498 | Error-probe checkpoint selection only |
| Combiner validation / meta-validation | 3,009 | 499 | Preserve and validate identity; no new fitting, selection, or calibration |
| Original test | 17,000 | 1,998 | Fixed final comparison only |

The combiner plan's 2,991 nominal `train` rows are not the historical final
gate's fitting pool. `scripts/create_strict_meta_plans.py:46–53` defines this
nominal field for the evaluator; `scripts/evaluate_strict_gate.py:33–63` fits
the final gate in five folds on its 3,009 meta-validation rows.

Within the original detector-training pool only, sort the unique photograph
keys by ascending hexadecimal SHA-256 of the UTF-8 string
`canonical-logit-dynamics-20260924:head-split:v1:{group_id}`. Break a hypothetical
digest tie by lexicographic `group_id`. The first `floor(3*N/5)` groups are
`head_train`; the rest are `probe_train`. With the supplied N=6,985 this fixes
4,191 head-training and 2,794 probe-training photographs. All originally
selected views follow their photograph; do not fill in other corruptions,
severities, clean images, or omitted rows. Retain original plan order within
each derived role. There is no seed-dependent split or outcome-based reshuffle.

This 60/40 allocation preserves the previous LD experiment's proportion of
total fitting photographs, not its former base/meta-role assignment. Row
counts need not follow 60/40 because photographs have varying selected views.
Save exact derived photo/record lists and their hashes before fitting. Require
all 100 class labels in head training, both error classes in probe training and
validation, and the expected original role sizes. A failed required check stops
the stage; do not resample or borrow validation/test photographs.

The LD class heads and error probe have disjoint fitting photographs. Their
combined fitting pool is the original 52,000-row training cohort, while
Polygraph's error experts used this full pool and its gate additionally learned
from meta-validation. These differences are explicit; identical test data do
not imply equal supervision at every stage. Do not assert otherwise.

## Frozen classifier, raw inputs, and parity gates

Use `edumunozsala/vit_base-224-in21k-ft-cifar100` revision
`b0c51e4a5e5bda35cc922419a28df93bb87e6efa` and the fast
`google/vit-base-patch16-224-in21k` processor revision
`b4569560a39a0f1af58e3ddaf17facf20ab919b0`. Bind weight/configuration and
processor hashes, class-label order, package versions, and inference settings.
Use the canonical source recipe: `uoft-cs/cifar100` and
`WNJXYK/TTA-CIFAR-100-C`, with source/severity/base-index mapping and RGB decoding
from `polygraph/data/sources.py`. Record actual dataset revisions and source-file
hashes; the historical code did not pin them, so do not claim a historical
dataset revision that was not supplied.

For each requested raw row, verify its source identity, index, and true label
against the original scan. If substituting an existing local image dataset or
feature cache, establish exact decoded uint8 RGB pixel identity with the
canonical reader for every reused row, plus compatible model/processor/CLS
extraction identity. Dataset names or matching dimensions alone do not suffice.
Using the canonical reader directly requires no second alternate data source.

Frozen inference uses evaluation mode, eager attention, FP32 computation, no
autocast, and disabled gradients. Do not train or fine-tune the ViT. Extract the
raw post-block CLS states `hidden_states[t][:, 0, :]` for t=1..12; index zero is
the embedding output, not a block. Do not insert another LayerNorm. Store CLS
in FP16 and promote for FP32 head computation, preserving the previous recipe;
retain original-classifier logits in FP32. No graph extraction is required.

Before full extraction, select a deterministic preflight panel: within each
occupied `(original_role, source, severity)` cell for detector train, base-val,
and test, take the first two available keys sorted by SHA-256 of UTF-8
`canonical-logit-dynamics-20260924:preflight:v1:{source}:{severity}:{base_index}`,
breaking a digest tie by the canonical record tuple. Use the sorted unique union.
This panel is fixed from identities without inspecting outcomes. Its readout is
classifier parity, not LD performance or configuration selection.

On the preflight panel, and again on **every exported row** before any LD fitting:

1. Verify finite values, exact true labels, unique aligned keys and expected shapes.
2. Require the newly computed classifier argmax to equal the original scan's
   predicted class exactly. Require scan `correct == (pred == label)`.
3. Compare fresh softmax maximum confidence and top-one-minus-top-two probability
   margin to the original scan. Each maximum absolute difference must be at most
   **2e-4**, with no relative allowance (`rtol=0`). This is the existing canonical
   `capture_logits` probability criterion (`polygraph/data/pipeline.py:119`),
   adopted before new measurements. Save the maxima and all discrepancies.
4. Use original scan `y = 1-correct` as the authoritative historical target.
   Because class equality has passed, the LD feature constructor's fresh final
   argmax also denotes the historical predicted class. Never silently substitute
   a historical class into inconsistent new logits or redefine error targets.

Any predicted-class mismatch or probability-gate failure stops before fitting
and requires a documented diagnosis; tolerances do not widen automatically.
Historical graph extraction allowed limited class drift, but this new protocol
does not silently inherit that allowance. A potential amendment must be resolved
before new LD metrics, with its effect on comparability stated explicitly.

Extract only the 71,991 required rows (52,000 train + 2,991 base-val + 17,000 test).
Preserve the 3,009 meta-val rows in source manifests without extracting features
or using their outcomes. The million-row original scan is metadata, not a reason
to rerun inference for every scanned presentation.

## Fixed full LD85 method

Reuse the full feature construction and training recipe documented in
`../logit_dynamics_20260919/PROTOCOL.md` and its tested implementation, with only
the new cohort, source-photo roles, seed identities and reference comparator.
The retained definition is below so this protocol is self-contained.

Fresh training seeds are **1, 2, 7**. Every seed fits its own twelve independent
linear 768-to-100 auxiliary heads and its own error probe. Do not reuse fitted
September 19/21 heads, probes or scalers: their photograph allocation was not
defined relative to this canonical test. Preserve all historical models intact.

| Setting | Fixed value |
| --- | --- |
| Auxiliary objective | Unweighted multiclass cross-entropy on `head_train` |
| Auxiliary optimizer | AdamW; lr 0.001; weight decay 0; betas (0.9, 0.999); eps 1e-8 |
| Auxiliary batch / epochs | 512 rows / exactly 16 epochs; keep the final epoch for every head |
| Error model | One linear map with bias from 85 standardized features to an error logit |
| Error optimizer | AdamW; lr 0.001; weight decay 0.01; betas (0.9, 0.999); eps 1e-8 |
| Error batch / epochs | 256 rows / exactly 100 epochs |
| Error objective | BCE-with-logits, positive weight `negative_rows / positive_rows` from `probe_train` only |
| Error checkpoint | Greatest base-val average precision; earliest epoch on exact tie |

Train on all assigned rows each epoch, including incomplete final batches, with
shuffled training batches and no balancing resample. No dropout, augmentation,
learning-rate schedule, performance-based early exit or hyperparameter search.
Use FP32 model computation. Record initialization and sampler/RNG states.

The trajectory consists of twelve auxiliary 100-class logit vectors followed
by the frozen original classifier's vector. At each of these thirteen depths,
the numeric block holds the original classifier's final predicted-class logit
followed by the five largest other-class logits in descending order. Stable
ties favor the lower class index. These 78 numbers are followed, in order, by:
`top1_switch_rate`, `topk_weighted_jaccard`, `unique_topk_count`,
`top1_mode_frequency`, `top1_entropy`, `top1_unique_count`,
`top1_commitment_depth`.

Retain the exact semantics of
`pilots/logit_dynamics_20260919/features.py`: top-five class identities over the
full class vector (including the final predicted class if it belongs there),
restricted stable softmax for weighted overlap, natural-log identity entropy,
and the appendix-normalized commitment depth. Do not replace these identity
features by statistics of the six selected numeric logits. Full D has 85
features; this run does not include the later A/B/C feature ablations.

Fit population mean and standard deviation (`ddof=0`) on `probe_train` only,
separately per seed; replace zero standard deviation by one. Statistics may
accumulate in FP64, then return standardized FP32 features. Calculate dynamics
from raw logits before normalization. Rank evaluation examples by raw selected
probe error logits, not finite-precision sigmoid probabilities.

The complete method learns 922,800 auxiliary-head parameters plus 86 probe
parameters per seed. Do not describe it as merely an 86-parameter method or as
matched to Polygraph in capacity or compute.

## Completion, held-fixed evaluation, and statistics

Only inspect new test metrics after all three seeds have completed the exact
16 head epochs and 100 probe epochs, passed finite-value/checkpoint tests, and
frozen their chosen checkpoints and scalers. Preserve each epoch's history and
resumable optimizer/RNG state. Require exact row/label alignment with the supplied
Polygraph NPZ before calculating a contrast. Missing or failed seeds yield an
incomplete comparison, not a best-seed or partially trained substitute.

The sole primary contrast is the arithmetic mean across seeds 1/2/7 of
**Polygraph AUROC minus LD AUROC**. Compute each seed's AUROC separately; do not
compute AUROC from scores averaged across training seeds. Matching seed labels
does not establish identical random perturbations between architectures.

Use canonical `polygraph.training.evaluate.detector_metrics` definitions:
high score means higher error risk; AUROC has half credit for score ties;
AP is scikit-learn's `average_precision_score`, not trapezoidal PR area;
AURC is the mean cumulative error over coverages after stable ascending-risk
sorting. Preserve the original ordered test list, relevant for tied-score AURC.
Report every seed and arithmetic mean for AUROC, AP and AURC, sample seed SD
(`ddof=1`), test record/photo counts and error prevalence.

Primary uncertainty uses exactly **2,000** paired source-photo bootstrap draws,
`numpy.random.default_rng(20260924)`. Sort the original test `group_id` strings
lexicographically, sample N=1,998 indices with replacement in each draw, and
retain every originally selected row of each sampled photo with its integer
multiplicity. Photographs have variable numbers of selected views: do not impose
the old nine-view layout or resample rows independently. Use identical photo
multiplicities for both methods and all seeds. Recompute all six AUROCs and the
mean paired contrast for each draw. Integer row weights inherited from photo
multiplicities may replace literal duplication for AUROC, after an equivalence
test in Slurm, including score ties and zero weights.

Use ordinary 95% percentile endpoints **0.025 and 0.975**, with linear NumPy
quantile interpolation. There is one predeclared primary contrast, so no
multiplicity adjustment is required. AP and AURC are descriptive secondary
point estimates, not additional significance tests. Fixed draw IDs 0–1999 and
their index/multiplicity manifests must survive resumption. Retain failures and
warnings; never replace or selectively discard an undefined draw. Withhold the
interval if any required draw is undefined or incomplete.

The interval conditions on these fitted models, fixed stage allocation and
observed benchmark. It does not incorporate future training-seed variation,
the original model-selection process, or prior exposure to these project data.
The original benchmark was correctness-stratified and capped: the validated
17,000-row test contains 8,500 errors, so AP/AURC describe this constructed 50%
error prevalence, not natural dataset frequency or probability calibration.
The canonical test has already informed project discussion, and our earlier
campaigns may overlap its source photos. Fresh fitting prevents direct new
training/test leakage, but does not make this an untouched confirmatory test.

A difference supports a comparison of these complete stated recipes. It does
not establish graph necessity, fairness under all possible resource definitions,
equivalence from nonsignificance, or superiority to a fully tuned LD paper
replication. Report negative and inconclusive results unchanged.

## Execution and preservation contract

All numerical checks, data preparation, inference, fitting, statistics and heavy
storage remain inside Slurm allocations. Mac work is static editing, reading
and coordination. One operator reconciles live jobs and receipts before any
submission; no duplicate or superseded-campaign submissions. The measured
resource reservation is an operational gate outside this scientific protocol;
do not assume the earlier campaign's time or memory budget applies.

Preserve input bundle identities, derived splits, selected preflight keys,
source/model/environment manifests, exact executable code, checkpoints, scalers,
histories, all predictions, bootstrap IDs/results, logs, warnings and failures.
Check save/load and interrupted-run continuation in Slurm. Pin the final code
and private artifact-backup revisions, and verify uploaded files/hashes before
calling the package backed up. An upload failure must never restart completed
scientific work. No automatic paid resources, pushes, merges or paper edits.
