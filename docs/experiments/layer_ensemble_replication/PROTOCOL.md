# Fixed seed-17/27 layer-stack replication

This is a new development-data replication, not an amendment that makes the
September 14 seed-7 run on time. The user selected seeds 17 and 27 before these
replication outcomes, and allowed up to eight concurrent training GPUs on the
existing Slurm resources. Implementation is authorized; scientific submission
still requires explicit authorization of a new run identity, future cutoffs and
resource reservation.

## Source and historical status

The baseline is the [September 14 pilot plan](../../../pilots/layer_ensemble_20260914/PLAN.md)
and [detailed protocol](../september14/PROTOCOL.md). The supplied pasted-text
attachment described a different topology experiment. The user explicitly
confirmed these repository documents as the new replication's baseline; this
does not reconstruct or alter historical approval evidence.

The historical implementation contains recovery cutoffs distinct from the
initial prospective documentation. Keep the actual executed release and
execution manifest when explaining chronology. Never change a historical
timestamp, cutoff, failed attempt, freeze, or late-diagnostic label.

The new manifest binds the original role map, cache identity, and seed-7 source
artifacts, including its explicit `complete_late_diagnostic` marker and formally
incomplete status. Descriptive use of seed 7 is not a new on-time result.
Historical artifacts must be read using their archived source when numerical
restoration is needed; current code hashes must not be substituted for the
executed release.

## Fixed information and training budget

Keep the frozen ViT, the existing attention graph cache and all feature
construction unchanged. Layers 3, 6, 9 and 12 correspond to `block2`, `block5`,
`block8` and `block11`. Each detector uses the same final hidden representation.

| Role | Source photographs | Records | Permitted use |
| --- | ---: | ---: | --- |
| base_train | 1,600 | 14,400 | Detector parameters and training-only weighting |
| checkpoint | 400 | 3,600 | Checkpoint selection |
| meta | 400 | 3,600 | Logistic heads and their standardizers |
| dev_eval | 800 | 7,200 | Frozen evaluation |

Copy the original role-map bytes, not a seed-dependent regenerated split.
All nine variants of each photograph remain in one role. No source images,
feature extraction, label changes or additional conditions are introduced.

Exactly eight fresh fits: both seeds 17 and 27, each with all four arms.
Preserve the edge-gated mean GNN, width 64, two message-passing layers, dropout
0.15, AdamW learning rate 0.002 and weight decay 0.0001, batch 24, and exactly
20 completed epochs. Selection is strictly greatest checkpoint-role AUROC,
retaining the earliest tie. Preserve the original sampler, class weighting,
precision, RNG and resumable optimizer state. No seed-7 warm starts, replacement
seeds, early-stopped prefixes, subset ensembles, or outcome-dependent retries.

## Two heads for each training seed

Freeze all four selected checkpoints before producing that seed's 3,600-row
meta matrix. Columns are raw error logits in the fixed order
`block2, block5, block8, block11`.

Fit exactly two heads independently for each seed:

- `stack`: all four raw logits.
- `last_only`: the `block11` raw logit.

Each fits its own meta-only `StandardScaler`, then L2 logistic regression with
`lbfgs`, `C=1.0`, `max_iter=1000`, `tol=1e-6`, `class_weight=None`, and
**`random_state=7`**. The logistic random state does not follow the detector
training seed. Save coefficients, intercepts, scaling, iterations, warnings and
JSON reconstruction checks at `atol=rtol=1e-12`. Failed convergence prevents
completion; it does not authorize a changed recipe. Preserve negative fitted
last-only coefficients.

Both seeds' base and head freezes must precede either new dev-evaluation
readout. A checksummed joint head-freeze barrier binds both pipelines. The
predictor and prediction loader enforce this barrier, not merely the scheduler.

## Primary and descriptive reporting

For seed `s`, define:

`delta_s = AUROC(stack_s) - AUROC(last_only_s)`.

The primary replication estimate is `(delta_17 + delta_27) / 2`.
Report each seed's two AUROCs and difference, plus mean and sample standard
deviation (`ddof=1`) across the two replication seeds. Sample SD is not a
standard error or confidence interval.

Report 7, 17 and 27 together only as a separate descriptive summary, retaining
seed 7's late label and verifying record/label/cache/role compatibility first.
No seed is chosen by performance. Do not concatenate repeated seed-image rows
as independent observations or average predictions into a different ensemble.

Keep pooled AUROC on all 7,200 dev-evaluation records. Registered descriptive
scores are each layer's raw logit and the equal arithmetic mean of the four
**sigmoid-transformed logits**, not mean raw logit or sigmoid of mean logit.
No new subgroup search, condition-macro primary or practical-benefit threshold
is introduced.

### Uncertainty

Use 2,000 paired source-image bootstrap draws, RNG seed `20260914`. Resample
the 800 `image_id` photographs, never corruption-family `source_id` or individual
rows. All nine variants inherit their photograph's multiplicity. Share each
draw across methods and seeds, compute each seed's AUROC difference, then
average the differences within the draw.

Use half credit for score ties and 95% percentile intervals with NumPy's
linear method. Record one-class draws without redrawing and withhold affected
intervals. Save the source multiplicities, per-seed draws and mean-effect draws.

These intervals are conditional on fitted models and heads. They do not
estimate population-level training-seed uncertainty or remove prior exposure
to development data. Two new training seeds do not create an independent test.

## Isolation, failure and authorization

The new root contains an immutable `replication.json`, the unchanged
`role_map.json`, and separate `seed17`/`seed27` directories. Each seed has its
own execution, runs, base/head freezes, predictions and evaluation. The shared
cache is linked read-only by identity; the historical source is never modified.

The Slurm [operating guide](../../../pilots/layer_ensemble_20260914/slurm/REPLICATION.md)
defines dry-run planning, explicit submission authorization, resource caps,
durable job IDs, dependencies and reconciliation of ambiguous submissions.
No laptop watcher is essential. All numerical tests, training, prediction,
evaluation, statistics and heavy checksums execute inside Slurm allocations.
Synthetic implementation tests do not authorize scientific jobs.

The new base, prediction and evaluation cutoffs must be explicit,
timezone-aware, strictly ordered and future at protocol creation. A missed
cutoff means incomplete, even if a later diagnostic succeeds. No automatic
GPU upgrade, paid resource, batch-size change, new seed or architecture search.

Completion requires all eight 20-epoch fits, four converged heads, the joint
freeze, both aligned dev-evaluation exports and the full bound report inventory.
Individual successful files or Slurm exit codes alone are insufficient.

## Limits, test access and archival

The comparison mixes multiple-layer information, four detectors versus one,
and generic ensemble benefits. It cannot establish GNN necessity, topology
necessity, or message-passing causality, and does not support language-model
hallucination claims.

No original-test access is allowed in this workflow. The
[September 10 catalog](../../models/SEPTEMBER10_CATALOG.md) records a separate
historical test evaluation; do not call the project's test globally untouched.
Any future test decision requires frozen models/analysis and a cohort-exposure
audit, with separate authorization.

Private HF archival is separately gated on server authentication, private
visibility, actual quota, checksums, immutable revisions and restoration
verification. Follow the [archival procedure](../../models/README.md); never
present the old partial snapshot as complete. No caches/checkpoints on the Mac
or in Git, and no credentials or chat exports in an upload. Keep all Git commits
local; no push or merge.
