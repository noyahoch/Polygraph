# Scientific review — canonical LD comparison, September 24

## Decision

Proceed with one fixed main-benchmark LD comparison after artifact and classifier
compatibility checks. The newly supplied original split plans, scan targets and
per-example final Polygraph scores remove the previously identified missing-data
prerequisite. No additional colleague file is presently required for this scope.
Raw source images remain retrievable through the known canonical public recipe.
The separate weather benchmark is not included.

The protocol is fixed in `PROTOCOL.md` before new LD results. This document is
static scientific review, not a claim that its runtime gates have already passed.
The Slurm operator records numerical checks, budgets and execution receipts.

## Why the proposed allocation is defensible

The selected 60/40 subdivision by source photograph gives LD a class-head pool
and a disjoint error-probe pool entirely inside Polygraph's original training
cohort. It keeps the same overall fitting-photo proportion as our prior fixed
LD recipe and avoids choosing a ratio based on new performance. Hash allocation
does not use labels or outcomes, every originally selected view follows its
photograph, and all original evaluation rows remain fixed. Variable views mean
60/40 photos does not guarantee 60/40 rows; the manifest must report both.

The original 2,991 base-validation rows select the error-probe epoch. The original
3,009 meta-validation rows have no new LD role. Keeping them unused is a simple,
conservative choice for standalone LD; adding a new gate or using them to tune
LD would create another decision not needed for the requested comparison.

This does not equalize the two algorithms' stage-wise supervision. LD's heads
learn class labels on 60% of the training photos; its probe learns error labels
on the other 40%. Polygraph's base error experts learned from the full original
training cohort, and its final gate used meta-validation. Both complete methods
must be described with these allocations. No claim of equal compute, parameters
or supervised examples follows from sharing a benchmark.

## Critical checks and resolved ambiguities

1. The final NPZ's `A_edge_gated_mean` identifies a gate directory. Its scores
   belong to the final **output-plus-edge-gated mixture**, not the standalone
   edge-gated GNN. Canonical finalization selects among gate directories by mean
   OOF meta-validation performance and copies their test scores. Recompute the
   reported mean AUROC 0.89391 as an integrity check, without changing selection.
2. The bundle README calls the combiner plan's 2,991 nominal training rows
   “train/meta-fit.” The code clarifies the final gate actually fitted its five
   folds on the 3,009 meta-validation rows. The new allocation uses code-defined
   roles, not that shorthand.
3. Original scan error labels remain authoritative. Exact new argmax agreement
   is required before LD fitting, with confidence/margin absolute tolerance
   2e-4 inherited from canonical `capture_logits`. This prevents either silent
   target drift or choosing the historical class from incompatible new logits.
   The probability tolerance is fixed before measurement; a mismatch is a
   concrete diagnostic stop, not permission to widen it until extraction passes.
4. Existing September 19 heads, probes and scalers cannot be presumed clean with
   respect to these canonical test photos. The new experiment trains all LD
   components fresh. Only verified compatible raw/feature caches may be reused.
5. Canonical source identifiers and original row order are preserved. The older
   experiment's nine-view layout is inapplicable here. Paired uncertainty samples
   source photographs while retaining their variable numbers of original views.
6. The per-seed mean is the performance estimand. Across-seed score averaging
   would create a new ensemble and would not reproduce the published comparator.

## Changes from the September 19 campaign

| Item | Previous LD campaign | This protocol |
| --- | --- | --- |
| Comparator | Omri's four-layer GNN mean ensemble | Yishai's frozen canonical output-plus-GNN gate |
| Test | 800 photos with nine views each | Original 17,000 selected rows / 1,998 photos |
| New fitting roles | 1,200 head photos; 800 probe photos from previous base/meta roles | 60/40 split of the original 6,985 training photos |
| Checkpoint selection | Previous 400-photo checkpoint group | Original 2,991 base-val rows / 498 photos |
| Seeds | 7, 17, 27 | 1, 2, 7 |
| Bootstrap grouping | Eight hundred fixed nine-view photo groups | 1,998 groups with variable selected-view counts |
| Method | Full LD85, twelve linear heads, linear error probe | Same fixed feature definition and optimization recipe, freshly fitted |

The prior experiment's measured performance cannot be inserted into the
canonical paper table. The newly aligned experiment can supply the missing
descriptive external-method comparison and paired uncertainty on the canonical
test, conditional on the stated training adaptation and compatibility checks.

## Interpretation and remaining work

There is one primary question: mean per-seed AUROC difference, Polygraph minus
LD. AP and AURC are secondary point estimates using canonical definitions.
Two thousand paired-photo bootstrap draws characterize sampling uncertainty
conditional on the fitted models and chosen partition. They do not erase prior
development exposure, estimate all future training-seed variability, or prove
equivalence when an interval includes zero.
The original correctness-stratified, capped test has 8,500 errors among 17,000
rows; AP/AURC therefore describe its constructed 50% error prevalence, not
natural dataset frequency or probability calibration.

A positive result would support this complete Polygraph recipe against this
fixed LD adaptation under disclosed stage allocations. A negative result would
favor LD on this comparison. Neither result alone proves graph topology is
necessary or useless. No further search is triggered by the sign of the result.

Remaining execution work is concrete: validate bundle identities and role/target
alignment; verify raw-input/classifier compatibility; measure extraction/fitting
throughput and reserve bounded Slurm resources; complete fresh LD fitting and
sealed evaluation; perform paired analysis; preserve and independently review
the result. No original graph regeneration or Polygraph retraining is required.

## Independent implementation review before Slurm preflight

Static review of the new implementation supports **GO for bounded Slurm
preflight**, not a claim that numerical checks or the comparison have passed.
The reviewer did not run numerical code locally or submit jobs.

Checked in code: original role/photo counts and group separation; fixed hash
subdivision; probe-only scaler fitting; unchanged full LD85 construction;
16/100-epoch training and earliest-AP-tie checkpoint selection; test-loading
gate after all three completed seeds; exact metadata and predicted-class
alignment; per-seed metric averaging; paired variable-view photograph bootstrap;
and checkpoint/RNG identity checks.

Focused issues found during review were resolved before preflight: the extractor
now requires the pinned dataset manifest instead of an unpinned download
fallback, records model/processor byte hashes and class mappings, and applies
the stated 2e-4 absolute probability tolerance; completion checks verify seed
and stage identities; actual bootstrap group order and draw multiplicities are
retained with checksums; transitive reused feature/test sources are identified.
Remaining runtime failures must be diagnosed from Slurm receipts without
silently changing the method or its acceptance criteria.

## Independent runtime review before full-DAG release

**Decision: GO for the planned full DAG, subject to the coordinating agent's
separate operational budget/throughput review.** No scientific setting or
tolerance changed to obtain this decision. This is clearance to execute the
remaining gated stages, not a completed LD comparison or reproduction claim.

The successful prepare job was `924785`. Its source/configuration and dependency
overlay matched the approved identities; syntax/import checks and preparation
completed in Slurm. The derived frozen rows are 31,328 for heads, 20,672 for the
probe, 2,991 for probe validation, 3,009 unused meta rows, and 17,000 test rows
with 8,500 original scan errors. The downloaded role-map and preflight-list
hashes agree with `campaign.json`; the approved protocol SHA-256 is
`a4d48caac007b9237324d5b572f44d8f48f11aa1fb7c45d27d0c631444edaaaa`.

The first GPU preflight, `924786`, timed out after 608 allocation-seconds during
shared-storage verification and produced no scientific gate result. Its stale
wrapper “running” receipt is superseded by the terminal scheduler TIMEOUT.
The successful retry, `924803`, completed on `s-005` at 09:42:38 Israel time on
September 24, taking 406 allocation-seconds, with these exact reviewed identities:

- Source manifest: `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
- Configuration: `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
- Campaign: `f489a75f558d7588b946ebf5f5586e418b5473c156c6a5b88a4a071118a65268`.
- Capture manifest: `600cd5dd00c7a5981926791fa0ea1013ebd51970e415867b8d97e1bf3bbe522e`.

Actual completed checks, rather than inferred readiness from code:

- All **496** predetermined panel rows completed the exact true-label and
  predicted-class checks. Maximum absolute confidence difference was
  `1.0132789611816406e-05`; maximum margin difference was
  `1.913309097290039e-05`, both below the fixed `2e-4` absolute-only threshold.
- The six preserved feature/normalizer tests passed, including all seven
  dynamics, class-logit ties, top-k identity semantics and frozen validation
  normalization. Semantic feature positions and auxiliary parameter count passed.
- Finite-gradient/update checks passed. The synthetic optimizer/RNG continuation
  test reproduced the next step exactly, save/load passed, and an altered resume
  identity was rejected. This is a preflight fixture, not a claim that an entire
  completed training campaign was independently replayed.
- Weighted AUROC matched literal duplication for the predefined integer-weight
  fixtures, including tied scores, unequal view counts and zero weights.
- Locally retrieved campaign, capture, index, processor and model-provenance
  files passed cryptographic identity checks against the sealed receipts.
  The reviewer read existing server results; no numerical experiment ran locally.

The preflight covers a panel, **not all 71,991 required extraction rows**. The
full extraction still must pass the same classifier/target gates on every row
before fitting begins. All three LD fits, final checkpoint freeze, predictions,
paired analysis and verified private preservation remain subsequent work.

Receipt locations on the coordinator's Mac:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/runtime_prepare_r1/`

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/runtime_preflight_r1_failed/`

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/runtime_preflight_r2_complete/`

## Independent full-extraction gate review

**The full-cohort gate is consistent with the frozen protocol.** Extraction
`924822` (`canonical0924-085200-extract`, user `omrifahn`, account
`gpu-students`) completed on September 24 at 10:19:06 Israel time after 1,605
GPU allocation-seconds. The stage receipt records exit code 0 and the same
approved source, configuration, campaign and dependency-overlay identities as
the successful preflight.

The full cache manifest declares `mode=full`, `complete=true`, and exactly
**71,991 required and completed rows**, with all 71,991 newly extracted in this
invocation. The gated extractor checked original true labels, original error
targets and fresh classifier argmax exactly on every row. Maximum absolute
confidence difference was `5.5730342864990234e-05`; maximum margin difference
was `0.00011068582534790039`. Both pass the unchanged `2e-4` absolute tolerance
with relative tolerance zero. Model, processor and raw-data provenance files
are byte-identical to those reviewed for the successful preflight.

The downloaded full manifest SHA-256 is
`d79c0bff547e702187e95e237d48010cf92f7703ef35b77c3072480f754b86bd`;
it binds the full index SHA-256
`d41b760159179ff1c1db63cfa05ea2394a225aae2f930c4cb81362d1b4b85468`
and individual shard hashes. Before fitting, the loader requires a complete
full cache belonging to the frozen campaign and verifies index/shard hashes,
frozen role order, stored metadata, finite values and original-prediction
agreement. The scheduler receipt shows all three authorized fits starting at
10:19:07, after extraction completed.

This audit examined compact server-generated receipts and static gate code;
it did not download/recompute tensors, inspect training scores or perform local
numerical work. Training, prediction, paired analysis and verified backup are
not declared complete by this milestone. Reviewed receipts are under:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/runtime_extraction_complete/`

## Independent final scientific audit — September 24, 2026

**Decision: GO to report the completed, fixed canonical comparison. No blocking
scientific concern was found in the reviewed evidence.** This audit inspected
server-generated results, manifests and execution receipts, checked their file
hashes, and reviewed the corresponding fixed analysis code. It did not rerun
training, inference or numerical analysis locally, and is not an independent
end-to-end numerical reproduction. The coordinating agent reviews private
backup completeness separately.

The final evaluation job `924830` completed under the same approved identities:

- Source manifest: `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
- Configuration: `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
- Campaign: `f489a75f558d7588b946ebf5f5586e418b5473c156c6a5b88a4a071118a65268`.
- Full CLS manifest: `d79c0bff547e702187e95e237d48010cf92f7703ef35b77c3072480f754b86bd`.
- Frozen evaluation gate: `427ac424bb9cf3b7426f19f356c4f1579a3699f0dc1bb8046774afe4bac06a9d`.
- Final machine-readable report: `8a87c4744a23ad961be76dc12f636f14dfdc0eb75b15ee24dff7a722dc93f454`.

All three seeds, **1, 2 and 7**, have complete heads, probe and prediction
receipts. Each auxiliary-head history contains exactly 16 epochs, with epoch
16 retained as specified. Each probe history contains exactly 100 epochs;
validation-AP checkpoint selection retained epochs **78, 95 and 96**,
respectively. The final histories, completion receipts and prediction receipts
agree on those selections. The configurations retain the prescribed separate
`head_train`, `probe_train` and `probe_val` roles, 85 probe features and
train-fitted normalization. No seed or alternative checkpoint was selected
using the test comparison. The freeze completed before the three prediction
jobs began.

Each prediction receipt contains **17,000 canonical test rows**. The evaluation
completed the fixed code's exact record-metadata and original-label alignment
checks against the original Polygraph score artifact and all three LD files;
it also reproduced the supplied historical Polygraph mean AUROC, AP and AURC.
Locally retrieved prediction, freeze, configuration, history and result-file
hashes agree with their completion receipts. **The comparator named
“Polygraph” is Yishai's full conditional output-MLP + edge-gated GNN mixture,
not the standalone GNN.**

The original fitted-model point estimates are:

| Seed | Polygraph AUROC | LD AUROC | Polygraph minus LD |
|---|---:|---:|---:|
| 1 | 0.8938816955 | 0.8901642215 | 0.0037174740 |
| 2 | 0.8943761176 | 0.8898343391 | 0.0045417785 |
| 7 | 0.8934613218 | 0.8908884014 | 0.0025729204 |
| Mean | 0.8939063783 | 0.8902956540 | 0.0036107243 |

The primary estimate is the **mean within-seed AUROC difference**, rather than
AUROC calculated from seed-averaged predictions or a bootstrap-average point
estimate. The paired-photo 95% percentile interval is
**[-0.0021257526, 0.0093290213]**. Its fixed manifest contains all **2,000**
draw IDs `0000`–`1999`, with no invalid draw IDs. Sampling uses **1,998**
source photographs, the recorded lexicographic group order, RNG seed
`20260924`, shared multiplicities across methods/seeds, each photograph's
actual variable number of views, and linear percentile endpoints 0.025/0.975.
The executed weighted-AUROC fixtures passed literal-duplication checks with
ties and unequal view counts. The interval is conditional on these fitted
models; it does not estimate uncertainty from new training seeds or prior
configuration selection.

Polygraph is numerically higher in AUROC in each of these three fits, but the
paired interval crosses zero: this comparison **does not establish superiority
or equivalence**. Its scope remains the previously used canonical development
benchmark, with correctness-stratified selection and 8,500 errors among 17,000
views. AP and AURC therefore describe the constructed 50% error prevalence,
not natural deployment prevalence or calibration. LD uses fresh auxiliary
heads and a different stage-wise supervision allocation from Polygraph; the
comparison is not a matched-supervision claim. This is a fixed ViT-B
adaptation of LogitDynamics rather than a full reproduction of the paper's
search. Comparison of the complete recipes does not establish that graph
topology is necessary.

The original reviewed evidence is at:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/final_results/`

The repository evidence location designated for the coordinator's compact
copies is `docs/experiments/canonical_logit_dynamics_20260924/run_records/final_comparison/`.
The authoritative reviewed items are `evaluation/report.json`,
`evaluation/complete.json`, `evaluation/bootstrap_draw_manifest.json`,
`evaluation/bootstrap_groups.json`, `evaluation_gate.json`, the three seed
configurations/histories/completion receipts, and the terminal/stage receipts.
