# Frozen follow-up: score complementarity and LogitDynamics decomposition

Authorized by Omri on September 21, 2026, before new results. This document
implements the approved two-experiment plan and its seven coding-review
requirements. All numerical execution, including tests and CPU statistics,
belongs in Slurm allocations. The Mac is for editing and coordination only.

## Scientific scope and immutable parent

Question 1: does the fitted graph score add error-ranking utility to the fitted
LogitDynamics score, beyond adding the matched set score? Question 2: what does
the fixed LD readout gain from intermediate auxiliary class projections and
from the explicit dynamics block? Both studies are exploratory development-data
reuse. Neither is an untouched-test, topology-necessity or causal-depth claim.

Preserve the completed parent experiment:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500`.
Its scientific source is `release-400f1fbbc105a370`; original source-manifest
SHA-256 is `d78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3`.
Parent `evaluation/scores.npz` SHA-256 is
`8b17c6a55c26beb0dc9ac54c574849b61d93c8feedd576677f5e96c20f6dff75`.
Original model/source backup is private `omrifahn/polygraph-experiments`,
revision `86605781c0528e286483e305072f7787393da860`, prefix
`logit_dynamics_20260919_114500/snapshot_910575`.

No new ViT, GNN or auxiliary-head training, no cross-fitting, additional model
arms, architecture search, score-transformation search or original-output
replacement. Write all follow-up code and outputs in a separate
`complementarity_20260921` namespace. Do not add files to the frozen LD package.

## Cohorts, labels and pre-result freeze

Reuse seeds 7, 17, 27. Error label one means the frozen classifier prediction
is wrong. A source photograph is `image_id`; `source_id` denotes the input
condition family, not the photograph. Condition is `(source_id, severity)`.
All nine views follow their photograph. Require exact metadata alignment,
unique record IDs, complete nine-view groups and the original error labels.

Fusion uses only the parent's 800 development photographs / 7,200 records.
Sort photographs by ascending SHA-256 hexadecimal digest of the UTF-8 string
`polygraph-complementarity-20260921:v1:{image_id}`, ordinary decimal ID;
break a digest tie by numeric ID. First 400 form fusion_fit, remaining 400
fusion_assessment. No label-based balancing or reselection. All seeds share
this partition. Old meta400 is LD probe-training data and must not be reused
as a supposedly held-out common stacking pool.

Decomposition retains LD probe_train800, probe_val400 and all original
dev_eval800. This assessment cohort differs from fusion_assessment400.
Always use separate tables and compare methods only within the same cohort.

Prepare immutable protocol/source/configuration/split/draw manifests before
fitting or reading new effects. Both study definitions are frozen together.
The final evaluation gate requires complete original fusion fits and all nine
decomposition fits; operational inability to finish a study is reported as
incomplete, never as a scientific null or a reason to choose different arms.

## Fusion models

D is the saved LD raw error logit. G and S are the saved G_mean/S_mean values,
the means of four within-seed sigmoid detector scores; do not substitute mean
raw logits or logit-transform those averages. Standardize these supplied
float64 scores only using fusion_fit photographs.

Fit DG, DS and DDprime for each seed. Dprime pairs are fixed cyclically:
7->17, 17->27, 27->7. These overlap and are not independent replications.
Every combiner is StandardScaler followed by logistic regression with
intercept, L2, C=1, lbfgs, max_iter=1000, tol=1e-8, class_weight=None,
warm_start=False. Use installed-version-compatible L2 syntax without changing
the objective. No tuning/validation selection of combiners. Rank by raw
decision score, not its sigmoid. D stays unchanged. Report standalone D/G/S
on fusion_assessment as context, not extra trained arms.

## LD feature cache and decomposition

Materialize FP32 85-feature matrices from the original cached FP16 CLS states,
original final logits and frozen seed-specific heads in the pinned CUDA path.
Roles per seed are probe_train7200, probe_val3600, dev_eval7200 records.
Persist role metadata, complete semantic feature names, inputs/source hashes
and cache hashes. The feature order is the parent's 12 auxiliary blocks,
original-classifier block and seven dynamics values.

Hard compatibility gate, before any A/B/C fit:

1. Verify names and order, not only shape. Columns 66:72 are depth12 auxiliary
   predicted-final-class logit and competitor ranks1..5; 72:78 are the original
   classifier's corresponding block; 78:85 are exactly top1_switch_rate,
   topk_weighted_jaccard, unique_topk_count, top1_mode_frequency, top1_entropy,
   top1_unique_count, top1_commitment_depth.
2. Predetermine semantic-check records: smallest record_id in each of the nine
   conditions in each role, plus record15321 when present (known prior numerical
   edge case). For every seed compare cached blocks against underlying head and
   original classifier outputs using a separately written reference calculation.
   Check original-final predicted-class selection, exclusion from competitors,
   stable lower-class-index tie handling, top-five identities and all dynamics.
3. Load original D weights and original training scalers and replay all three
   seeds' full validation and development scores FROM THE NEWLY SAVED CACHE,
   with original batch512/precision. Require exact metadata and finite values;
   compare scores at original atol=rtol=1e-4, recording maxima and violations.
   Old checks may substitute only if they bind this exact cache and path.
4. Check training-only normalization against the original normalizers. Selected
   column normalizers must represent the same training rows and ddof=0; zero
   scale maps to one. Never fit them on validation/development.

Feature slices (zero-based, upper-exclusive): A=72:78 (6), B=66:78 (12),
C=0:78 (78), D=0:85 (85). A is not the old full-logit MLP. Reuse completed D;
fit only A/B/C in seeds7/17/27, nine small linear readouts. Preserve the LD
initialization/sampler conventions, FP32 CUDA computation, AdamW lr0.001,
weight_decay0.01, betas0.9/0.999, eps1e-8, batch256, weighted BCE with negative/
positive count from probe_train only, exactly100epochs, no early stopping.
Select maximum validation AP, earliest epoch on exact tie. Save all histories,
selected weights and validation scores. No auxiliary-head updates.

Checkpoint each epoch with model, optimizer, sampler/RNG, best state and history;
test resumed next update against uninterrupted execution and selected-model
save/reload. Final development predictions follow the all-fit freeze gate.

## Bootstrap, point estimates and inference

Point estimates come from the original nine fitted combiners and nine fitted
decomposition probes, not averages over bootstrap refits. Compute each seed's
AUROC/AP, then arithmetic mean metrics and mean within-seed contrasts; do not
pool predictions across seeds. AP means sklearn average_precision_score.

Generate exactly2,000 draws with fixed IDs0..1999, before effects are examined.
Use numpy SeedSequence(20260921).spawn(3), assigned in order to fusion fit400,
fusion assessment400 and decomposition assessment800. Draw each pool's size
with replacement from its ascending numeric image_id list. Store integer
photo multiplicities. Share corresponding draws across every arm and seed;
do not resample seeds. The two assessment cohorts have independent streams.

Fusion: each draw independently resamples fit and assessment photographs and
refits all nine scalers/combiners. Apply the same unnormalized integer row
weights to scaler fitting and logistic fitting; each photo count repeats on
all nine rows and total fit weight is3600. Zero-weight rows contribute nothing.
Recompute score orderings after every refit. Evaluate using assessment weights.
This includes combiner-fit and assessment sampling uncertainty conditional on
the base models and fixed partition.

Decomposition: use fixed fitted A/B/C and original D predictions, and sample
the800 assessment photographs; do not refit these probes inside bootstrap.
Intervals condition on all fitted models. Neither interval corrects prior
development exposure or covers future base-model training variability.

Primary family: DG-D, DG-DS, C-B. Use NumPy linear percentile interpolation
at EXACT user endpoints [0.0083333333,0.9916666667], a nominal95% Bonferroni
family with98-1/3% marginal intervals. About17draws occupy each tail; a barely
positive endpoint is not numerically decisive. Keep draw count fixed at2000.
Secondary contrasts DG-DDprime, B-A, D-C use descriptive95% intervals
[0.025,0.975], without extra family-controlled superiority claims.

Store each draw by ID with statuses and convergence warnings. No replacement
draws, discarded failures, fewer-seed averages, tolerance relaxation or objective
changes. Continue unaffected calculations but withhold any contrast interval
unless all its required2000draws are valid. Missing weighted class, nonfinite
values or convergence warning mark a fit unsuccessful. A failed point fit
blocks the affected study's completion. Numerical failure is not a null result.

## Weighted/duplicated correctness and diagnostics

Before accepting weighted bootstrap, compare fixed draws IDs0,1,7,17,27 against
literal duplication of each photograph's nine rows, with identical objectives.
Scaler mean/scale tolerance is atol=rtol=1e-12; raw decision score tolerance is
atol=rtol=1e-6. Save coefficients as diagnostics, but compare predictions as the
fit-equivalence gate. Include fixtures with zero weights, identical-feature
ties and separated score groups; compare weighted/duplicated AUROC/AP at
absolute1e-12 on those fixtures. Preserve near-tie ranking/metric differences
on real fitted scores for diagnosis rather than silently ignoring them.
All tests run in Slurm. Check save/reload and fixed-ID bootstrap continuation.

Predetermined descriptive diagnostics on fusion_assessment for standalone D/G/S
and the three combinations: per-condition AUROC/AP with all nine conditions,
record counts, correct/error counts and NA where undefined. Within-photo rank
accuracy includes only photographs with both correct and incorrect views;
compare every incorrect-correct pair, award one for higher error score, half
for a tie, zero otherwise; average within each photo and then equally across
eligible photos. Report eligible counts, per-seed values and seed means.

## Resources, execution and preservation

One sole Slurm operator. New scope ceiling:7200allocated GPU-seconds including
preflight/failures, at most3concurrent GPUs; CPU statistical jobs have at most
14400cumulative allocated wall-seconds. Queue time is not compute time. Measure
throughput before committing remaining reservations. First50fixed bootstrap
draws are retained as part of2000; report timing, not effects, at the timing
gate. Use one numerical-library thread per bootstrap worker and no more workers
than allocated CPUs. Durable jobs/checkpoints must outlive the Mac/VPN session.
At most two focused mechanical fixes per stage; no scientific retuning. Report
predicted cap overrun before enlarging budget. Preserve interrupted attempts.

Before submitting, reconcile current jobs and exact source/config manifests;
job ID alone is insufficient. Never duplicate an existing run. No paid hardware,
Overleaf edit, push or merge. Local small single-purpose commits are authorized.
Preserve pre-existing untracked results outside this namespace.

Private HF backup in omrifahn/polygraph-experiments must include every model,
normalizer, prediction, selected/resume state, history, failure, draw manifest,
protocol, executable source snapshot, environment inventory and source/split
bindings. Keep previous snapshots unchanged. Declare backup complete only after
remote availability and checksums are verified at an immutable revision.

Report all planned results and limitations. C-B is usefulness of intermediate
projections under this readout recipe. D-C includes added class-identity signals,
not merely summaries already present in C. DDprime is not compute-matched.
Nonsignificance is not equivalence. Additional graph-score utility is not
proof that topology or repeated message passing is necessary.
