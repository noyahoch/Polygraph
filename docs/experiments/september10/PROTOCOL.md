# September 10 experiment: attention incidence and message passing

Protocol registered for implementation on 2026-09-11. Experiment identifier: `topology_20260910`; cohort identifier: `polygraph-20260911`. The branch name and cohort namespace intentionally have different dates. This protocol describes authorized new work, not a report that it has completed.

Starting code: `noyahoch/Polygraph`, commit `9af8890405297c0f37dbaa54f6be29a84cb0c37e`, local branch `September-10th`. The implementation, machine protocol, environment and artifact manifests must be pinned before final evaluation. Amendments require a dated record of the change, reason, available evidence and whether any test results were already visible. A technical completion marker is not a scientific result.

## 1. Question and permissible claims

The task is to predict whether a **frozen CIFAR-100 classifier is wrong**, with error as the positive class. A corrupted image is not automatically an error. This is image-classification error detection, not an established benchmark of language-model hallucinations.

The primary question is whether the selected edge-gated MPNN improves on a detector receiving the same internal node and edge values without incidence. An endpoint-set control distinguishes independent processing of attributed relationships from iterative neighborhood aggregation. A rewired MPNN tests sensitivity to the original attributed connections under matched architecture and capacity.

The second experiment asks whether the graph-versus-set difference changes between attention-only and full internal features. This directly tests the suggestion that rich representations reduce the marginal benefit of graph processing. It does not separately identify the effects of hidden states and class conditioning.

Neither parameter matching nor these controls can prove that every possible non-GNN is inferior. Endpoint records retain relational information and can, in principle, encode topology; call that control **no explicit message passing**, not topology-free. Rewiring changes destination/attribute alignment and paths, not higher-order topology alone.

## 2. Cohort, classifier and features

### Immutable cohort

- Start from the 10,000 official CIFAR-100 test photographs, using their original zero-based `base_index`.
- For each index, hash the exact ASCII string `polygraph-20260911:{base_index}`, with the index in ordinary unpadded decimal, using SHA-256. Sort ascending by digest, with `base_index` as a deterministic tie-breaker. Select the first 4,000.
- Partition this ordering into the first 2,400 source photographs for detector training, the next 800 for validation, and the final 800 for test. All versions of one source photograph belong to its source split.
- Every photograph contributes nine records: clean; Gaussian noise at severity 3 and 5; motion blur at 3 and 5; fog at 3 and 5; JPEG compression at 3 and 5. Canonical corruption names are `gaussian_noise`, `motion_blur`, `fog`, and `jpeg_compression`; clean has severity 0.
- There are 36,000 records: 21,600 training, 7,200 validation, and 7,200 test. Preserve all records regardless of classifier correctness. No balanced sampling of errors and correct predictions is applied.
- Record keys include source photograph, corruption name and severity. Verify correspondence between clean and corrupted source indices, not merely equal array lengths. Persist the exact selection, ordering, source revisions and checksums in immutable `cohort.json` and cache manifests.

This is an unbalanced correctness distribution **within a chosen nine-condition mixture**, not a claim about deployment prevalence. Each corruption condition occurs in all three source-disjoint splits. The experiment does not test unseen corruption families. The team's prior use of CIFAR-100 means these photographs are not guaranteed globally untouched; report this as a new fixed cohort and detector comparison on an established benchmark, not independent confirmation on a new benchmark.

All seven detector conditions below train from scratch on this split. Historical detector checkpoints cannot be used as contestants because their training sources can overlap the new test set. The frozen classifier and exactly compatible deterministic features may be reused.

### Frozen classifier and extraction

- Classifier: `edumunozsala/vit_base-224-in21k-ft-cifar100`, revision `b0c51e4a5e5bda35cc922419a28df93bb87e6efa`.
- Pin the image-processor configuration and resolved revision. The fixed fallback processor is `google/vit-base-patch16-224-in21k`, revision `b4569560a39a0f1af58e3ddaf17facf20ab919b0`; do not resolve a moving default during a resumed run. Its exact configuration and the actually used processor are manifest fields.
- The ViT remains in evaluation mode with frozen weights, using eager attention. Extract attention from block 11 and the full hidden-token representation `hidden_states[12]`. There are 197 tokens, including CLS, and 12 attention heads.
- A directed edge is key/source `j` to query/target `i`. Keep **every non-self edge** with `max_head(attention[i,j]) > 0.02`. There is **no top-K cap**. The 256-edge cap from a different overnight experiment does not apply.
- Raw nodes contain 16 features: patch row, patch column, CLS indicator, layer position, and 12 per-head diagonal self-attention values. Raw edges contain their 12 attention weights.
- Full nodes append all 768 hidden features, giving 784 dimensions. Full edges contain 36 features: attention; `log1p(attention * transformed_value_norm)`; and `asinh(attention * transformed_value_projection)` across the 12 heads.
- For head `h` and source `j`, the transformed value is `W_O[h] V[h,j]`. The projection direction is the normalized classifier-weight difference between the predicted class and the runner-up class, chosen from the frozen model's logits. Record the implementation and its numerical normalization rule in the machine protocol. This quantity is a class-conditioned proxy, not causal attribution through the complete ViT.
- Labels are `1[predicted_class != true_class]`. True class, correctness, corruption identity, severity, source index and split identity are never inference features. These metadata remain available to the evaluator for grouping and reporting.

## 3. Fixed contestants and training

| Arm | Input and operation | Width / depth |
|---|---|---|
| `full_graph` | Full features; edge-gated mean MPNN on original incidence | 64 / 2 |
| `full_rewired` | Same architecture and values; constrained rewired incidence | 64 / 2 |
| `full_set` | Full node and edge multisets; no endpoints; explicit CLS access | 84 |
| `full_endpoint` | Full endpoint-local edge records plus an all-node branch; no iterative neighborhood updates | 47 |
| `raw_graph` | Attention-only features; same edge-gated MPNN family | 64 / 2 |
| `raw_set` | Attention-only node and edge multisets; no endpoints; explicit CLS access | 93 |
| `logit` | All 100 classifier logits; MLP `100 -> 64 -> 32 -> 1` | Fixed |

The internal detectors do not receive a concatenated 100-logit vector. Class conditioning in the full feature bundle uses the frozen prediction and runner-up as described above. There is no newly trained fusion gate in this experiment. Include analytic MSP and entropy as references without counting them as trained fits.

The fixed widths are chosen for parameter matching, not selected using performance. Before full training, verify the trainable parameter counts and that each set control is within 5% of its corresponding MPNN. Do not adjust a width after examining its validation performance. If implementation differs from the intended capacity, resolve the discrepancy as a documented readiness correction before the training matrix is frozen. Report pooling and architectural differences; equal counts do not imply equal expressive capacity.

- Seeds: **1, 2, 7, 17, 27**, for every trained arm: **35 planned fits**. Report every seed; do not substitute seeds with better results.
- AdamW; learning rate 0.002; weight decay 0.0001; dropout 0.15.
- Batch size 24 for graph and set models, 256 for the logit MLP. Preserve the selected batch size in checkpoints and evaluation configuration. A required memory correction is applied consistently to the affected comparison, documented before its full runs, and never justified by test scores.
- Train with binary cross-entropy on error logits, using `pos_weight = training_correct_count / training_error_count`, computed once from training records only. Any fitted input normalization also uses training records only and is saved with the model. No label-dependent resampling is introduced.
- Maximum 60 epochs; checkpoint selection by validation AUROC; patience 8; `min_delta = 0`; no minimum epoch requirement. An exactly tied validation AUROC keeps the earlier best checkpoint. All arms use the same rule.
- Preserve optimizer, epoch, random states, data-order state and best-checkpoint state for faithful resume. Restarting is not a new opportunity to select a seed or configuration. Record training and validation history, runtime, resources and any failed attempt.

### Constrained rewiring

For edges `(u -> v, a)` and `(s -> t, b)`, propose `(u -> t, a)` and `(s -> v, b)`. Reject self-loops, duplicate edges and ineffective swaps. This preserves every node's directed in- and out-degree, edge count, node features, edge-feature vectors and their original source associations. CLS identity and degrees are preserved, but its neighbors may change. Incoming weighted strengths, destination coherence and paths need not be preserved.

Use a deterministic, cached null for each record, with an explicit independent rewiring random stream. The same null for a record is used throughout training, validation and test inference; no new online augmentation is introduced. Null generation must not depend on labels or performance. Target 2 accepted swaps per edge, with at most 20 attempted swaps per edge, using a compiled CPU implementation on Slurm.

Before full training, inspect training/validation graph diagnostics. Require mean changed-edge fraction at least 0.8 among graphs with at least 20 edges for an informative null. Persist attempted/accepted swaps, graph-wise changed fractions and quantiles, the fraction of weakly changed graphs, and CLS-neighbor changes. Retain all graphs, including nonrewirable graphs. If the requirement fails, mark this control non-diagnostic and obtain a documented protocol decision; do not silently relax constraints or remove unfavorable graphs. This bounded construction does not claim uniform sampling or proven Markov-chain mixing.

## 4. Evaluation and uncertainty

### Scores, thresholds and ties

Larger scores always mean greater likelihood of classifier error. Use the saved detector's native failure logit for AUROC and threshold decisions, avoiding probability saturation creating artificial ranking ties. MSP's error score is `1 - max(softmax(logits))`; entropy uses its usual increasing-uncertainty direction. Save display probabilities separately if needed. AUROC gives a tied positive-negative pair credit 0.5.

For each arm and seed, sort failure scores of **correct predictions in the complete validation mixture** ascending. With `n` correct validation records, set the threshold to the score at one-based rank `ceil(0.95*n)`. Flag a record iff `score > threshold`. Thus ties are handled conservatively and the empirical validation false-alarm rate is at most approximately 5%; do not randomize ties or tune the threshold again on a test condition. If there are no correct validation records, the threshold is undefined and the evaluation cannot satisfy this protocol.

Use the same frozen per-model threshold for the full test mixture, all nine conditions and the confidence slice. Report the achieved validation false-alarm rate and test error recall, false-alarm rate, rejection fraction, retained coverage and error rate among retained records. A zero denominator produces `NA`, not zero. If all records are rejected, retained risk is undefined and coverage is zero. For ordering-dependent coverage summaries, break score ties by immutable record-key order, identically across reruns, and state that convention.

### Primary estimand

For each seed, calculate test AUROC of `full_graph` minus test AUROC of `full_set`. The primary estimate is the arithmetic mean of these five paired differences. It is **not** AUROC of seed-averaged scores and is not an ensemble result.

Generate **2,000 paired bootstrap draws** with random seed `20260911`, sampling the 800 test source photographs with replacement. Every selected source contributes all nine versions with its bootstrap multiplicity. The same source draw is used for every arm and seed. Calculate the five seed-wise AUROC differences in each draw and average them; use the 2.5th and 97.5th empirical percentiles for the primary 95% interval. Record the percentile interpolation convention in the statistical implementation. This interval reflects source-sampling uncertainty conditional on the five fitted seeds; also show seed-wise results and variation.

AUROC requires at least one correct and one incorrect classifier prediction in the evaluated data. Report `NA` otherwise. Record undefined bootstrap draws rather than redrawing until results become defined. If any primary draw is undefined, report the support problem and withhold a confirmatory interval; do not manufacture finite values.

The predeclared practically meaningful margin is **0.005 AUROC**. A lower interval bound above 0.005 supports an advantage of at least that magnitude. An upper bound below 0.005 rules out that size of benefit for the tested comparison. An interval spanning the margin leaves the practical question unresolved. Failure to reject a zero difference is not equivalence. Report a smaller positive effect as small, not as satisfying the practical margin.

### Prespecified secondary analyses

- Full MPNN versus endpoint set, and original versus rewired MPNN, with paired source-group uncertainty. These support architectural interpretation and do not supersede the primary comparison.
- Feature-bundle interaction: `(AUROC(full_graph) - AUROC(full_set)) - (AUROC(raw_graph) - AUROC(raw_set))`, paired by source and seed. A negative interaction indicates a larger graph-versus-set advantage in the attention-only regime. Treat this as a prespecified secondary finding, not a second confirmatory superiority test.
- Each of the nine fixed conditions: source and record counts, error prevalence, AUROC when defined, and fixed-threshold quantities. These are descriptive diagnostics, with no selection of favorable corruption conditions. Pooled discrimination can partly reflect different condition difficulty.
- Confidence slice: include **all** test records with classifier MSP at least 0.9, whether correct or wrong. For inferential interpretation require at least 200 distinct source photographs containing a confident error and at least 200 containing a confident correct prediction; the two source sets may overlap. Otherwise report descriptive results and insufficient support. Do not lower the confidence cutoff or search classes to increase significance. This floor does not guarantee power to detect a 0.005 effect.
- Full-logit MLP, MSP and entropy contextualize whether internal information helps relative to the classifier output. No affirmative multiplicity-corrected superiority claim is made from secondary intervals. Clearly label all secondary uncertainty as such; do not promote a favorable secondary result to the original primary outcome.

## 5. Test blindness and independent checks

Feature extraction may materialize the test cache before training is complete because the frozen transformation is fixed. Before the approved freeze manifest, allowed test-cache checks are limited to opaque record IDs, source/split correspondence, checksums, completeness, shapes, dtypes and required finite-value integrity. Do not display or interpret test classifier accuracy, error counts, confidence distributions, score distributions, AUROC or other aggregates. Do not inspect test examples to choose a hypothesis. Training/validation may be inspected for readiness and debugging.

The freeze manifest binds the protocol and implementation hashes, immutable cohort/cache hashes, complete arm/seed matrix, selected checkpoints, train-fitted preprocessing, validation thresholds and secondary definitions. The test evaluator must refuse to run without it or when any bound artifact has changed. The completed matrix is frozen before one final scoring pass. No model/threshold changes follow test results. A byte-identical rerun solely to recover an interrupted artifact is recorded as recovery, not a new opportunity to tune.

All numerical checks run on Slurm:

1. Source groups are disjoint; all versions and contestant inputs align; labels derive solely from the frozen classifier.
2. Feature values, predictions, losses and gradients are finite; edge direction, self-edge exclusion and the no-cap threshold rule match the specification.
3. Node/edge-set outputs are invariant to record order within numerical tolerance; endpoint controls retain the intended endpoint information; graph batching does not mix examples.
4. Rewiring preserves its stated invariants, including all node/edge feature values. Empty/nonrewirable graphs and zero-edge aggregation have defined, finite behavior.
5. Save/load and resume preserve the prescribed computation. Evaluate a fixed training/validation audit batch before and after restore, in evaluation mode, on the same device/dtype and with the same batch layout. Require elementwise failure-score agreement with **atol 1e-5 and rtol 1e-5**, finite outputs, and unchanged ordering for pairs separated by more than the sum of their allowed numerical errors. Record near-tie order changes instead of requiring bitwise equality of floating scatter reductions. When thresholds are available, decisions safely outside the tolerance neighborhood must be unchanged; report any near-threshold differences.
6. Also check score agreement for a fixed alternative batch partition. A tolerance failure is a real technical failure to diagnose, not permission to silently widen tolerances or bypass the check. Deterministic numerical tolerance is not a license to change preprocessing, checkpoint weights or feature identities.

The scientist reviews readiness before full launch and reviews the full results after evaluation. Technical failures and omitted/non-diagnostic controls limit the claim explicitly. They are not negative scientific evidence about GNNs.

## 6. Execution budget, artifacts and closeout

- **24 hours of active team work**: accumulated team-active wall-clock time, counting overlapping agents once. Idle cluster waiting does not consume this active-work budget. This is neither 24 summed agent-hours nor an automatic 24/48-hour GPU-training cutoff.
- At most **8 concurrent GPUs**. All feature extraction, training, tests, bootstrap/statistics and heavy storage stay on Slurm. The MacBook is for writing and orchestration only. No automatic move to a local or paid GPU.
- At six active hours, hold a documented readiness/budget review using measured extraction, I/O, memory and training speed. Admit work only if its complete comparison, uncertainty analysis and packaging fit the remaining active budget with at least 25% projected runtime headroom where runtime is estimated. Do not spend the budget searching for a positive model.
- Allow at most two focused technical repairs per stage and no repeated repair attempt for the same unchanged failure. Log failed attempts and resource use. A failed rewiring control does not erase valid real-graph/set results, but its mechanistic conclusion remains unavailable.
- The core is the four full-information arms plus the logit baseline, each with five seeds. If measured resources force reduction, the entire two-arm attention-only experiment is the first removable scientific block. Decide and record this before any test metrics, based on feasibility rather than result direction; do not silently drop individual unfavorable seeds or controls. A reduced matrix needs an amended freeze manifest and an explicitly narrower final claim.
- Slurm dependencies, durable checkpoints and result saving continue through laptop closure. New scientific choices wait for an active authorized agent. A running job, a successful exit code and a completed scientific comparison are distinct states in the run ledger.
- Canonical cache metadata: `cache/manifest.json` and immutable `cohort.json`. Per-fit artifacts: `<arm>/seedN/config.json`, `history.json`, `best.safetensors`, resumable `latest.pt`, and `complete.json`. Completion records bind hashes and checkpoint-selection evidence; they must not imply missing evaluations passed.
- Save checkpoint artifacts to the authorized Hugging Face destination with immutable revision/hash references. Preserve code, protocol, commands, configuration, compact results and reproduction links for the GitHub documentation. Do not upload private chat/transcription context as experimental data. Publication and Git actions remain the orchestrator's responsibility.
- Final deliverables: concise Hebrew findings, complete seed-level table, primary/secondary uncertainty with limitations, all nine condition diagnostics, fixed-threshold results, failures and omissions, code/environment/data/checkpoint provenance, and exact reproduction commands. A clear null or inconclusive result is an acceptable outcome; packaging the existing work is preferable to an incomplete positive story.

## Source of the hypotheses

The dated WhatsApp source and September 10–11 meeting transcripts are preserved in the parent project's `Context/` directory. Noya's August 23–24 messages motivate learned matched baselines; her September 7 messages separate attention-only structure from additional internal information. Ishi's September 6 report establishes that the earlier controls targeted TransformerConv and that the edge-gated method is a useful selected architecture, without proving graph necessity. The new cohort and five-seed matrix must not be represented as a numerical reproduction of their 75,000-record study.

Pinned prior report: [Polygraph graph-necessity, architecture and last-four study](https://github.com/noyahoch/Polygraph/blob/495ea05c0a7f844d69ca8b7c76c5f10c940f11bb/docs/results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md).
