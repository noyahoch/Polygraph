# September 14 overnight learned ensemble — approved core

This version replaces the prospective union-layer and multi-seed expansion. The active namespace is `pilots/layer_ensemble_20260914`; the scope identifier is `layer_ensemble_20260914_core_seed7`. Old September 10 results and September 13/14 diagnostic artifacts remain historical evidence, not results of this experiment.

## Question and fixed scope

Does a learned combination of four separately trained layer-specific error detectors outperform a learned last-layer-only control? The target is a frozen CIFAR-100 ViT's classification error (`pred != label`), not corruption identity or language-model hallucination.

Exactly four fresh GPU fits: `block2`, `block5`, `block8`, `block11` (human layers 3, 6, 9, 12), all at seed 7. Each uses the previously reviewed edge-gated mean GNN, width 64, two message-passing layers, dropout 0.15, AdamW with learning rate 0.002 and weight decay 0.0001, batch 24 and **exactly 20 completed epochs**. The best checkpoint is selected by strictly greatest checkpoint-role AUROC, keeping the earliest tie. This matches update count and data exposure, not GPU/FLOP cost or guaranteed convergence. No union arms or other seeds launch automatically.

The frozen ViT, eager attention extraction, selected layer's per-head attention above 0.02, key-to-query edge direction, final hidden representation, and shared feature cache remain unchanged. The cache contains only the existing 3,200 development photographs and all nine fixed clean/corruption views per photograph. No cached detector checkpoint trained on the old 2,400/800 split may be reused as a new fit.

## Separate data roles

The original immutable cohort record order determines first-photo occurrence, preserving the original SHA-based sample order. Split the original 2,400 training photographs into the first 1,600 for `base_train`, the next 400 for `checkpoint`, and the last 400 for `meta`. The original 800 development validation photographs become `dev_eval`.

| Role | Source photographs | Records | Allowed use |
|---|---:|---:|---|
| base_train | 1,600 | 14,400 | Base detector parameters, preprocessing and class weights |
| checkpoint | 400 | 3,600 | Base checkpoint selection only |
| meta | 400 | 3,600 | Both logistic heads and their standardizers |
| dev_eval | 800 | 7,200 | Final frozen comparison only |

All nine variants stay with their source photograph; no role overlaps another. `role_map.json`, `execution.json`, the feature manifest and all selected base artifacts are checksum-bound. The original held-out 800 photographs remain closed. Existing benchmark/development reuse means this is not a wholly independent new test set.

## Learned combination and fair meta-stage control

After all four base fits complete and their selected checkpoints are frozen, generate one meta prediction matrix with columns in the exact order `block2, block5, block8, block11`. Entries are **raw detector error logits**.

Train exactly two CPU heads:

1. `stack`: all four raw logits.
2. `last_only`: the `block11` raw logit, using the same meta records and learning recipe.

Each head independently fits `StandardScaler` on meta records only, then `LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=1000, tol=1e-6, class_weight=None, random_state=7)`. There is no search, cross-validation, coefficient constraint, temperature fitting or outcome-dependent retry. Preserve all coefficients, intercepts, scaler means/scales/variances, iteration counts and convergence warnings. Any convergence failure prevents a completed comparison. Both heads are frozen before dev-evaluation prediction generation. JSON parameter reconstruction is checked against sklearn's decision function with absolute and relative tolerance `1e-12` on the meta inputs.

A last-only coefficient can be negative. Preserve it and report the raw last-layer score separately; do not assume calibration must preserve ranking or change its sign after evaluation.

## Evaluation and uncertainty

One primary contrast: **AUROC(learned stack) − AUROC(learned last-only control)** on all 7,200 `dev_eval` records. Higher scores mean a predicted ViT error.

Predeclared secondary/descriptive scores: the equal arithmetic mean of the four **sigmoid-transformed** raw logits and every single layer's raw logit. No subgroup search, coefficient, checkpoint or architecture is selected from `dev_eval`.

Use 2,000 paired bootstrap draws, RNG seed `20260914`, sampling the 800 **image_id** source photographs with replacement. All nine variants inherit the photograph's multiplicity, and the same multiplicities are used for both primary scores. `source_id` names corruption family and is never the bootstrap grouping key. Weighted AUROC gives half credit to score ties. The 95% percentile interval uses NumPy's linear method; if a draw has only one outcome class, record it without redrawing and withhold the interval.

Uncertainty is conditional on this one training seed and the fitted meta heads. It does not establish stability across training seeds. Four detector models versus one also changes model/inference compute and can benefit from generic averaging; the comparison does not isolate layer information from all ensemble effects or establish GNN/topology necessity. Negative results and unresolved intervals are valid outcomes.

## Execution, deadlines and completion

All extraction, fitting, prediction, verification, bootstrap and model saving run in Slurm. The Mac is used only for source authoring and coordination. Preserve the existing shared cache; fresh model runs use a separate namespace. Total authorized reservation is at most 50 GPU-hours, including the prior diagnostic/capture allocation caps; at most eight concurrent GPUs, with only four base contenders in this scope.

All times are Israel time: base fits must finish by **September 14, 23:00**; meta/head/dev prediction stages by **September 15, 04:00**; computations and durable report by **07:00**, leaving the **08:00** user-summary buffer. Midnight is a target, not a guaranteed scientific completion time. Ops implements the exact job caps and a remote failure guardian.

Successful completion requires four fresh, valid 20-epoch bases, both converged frozen heads, aligned final predictions and the paired report. Missing or invalid required artifacts produce an explicit incomplete status; never replace the ensemble with a subset or compare arbitrary wall-time prefixes as equally trained models. Preserve checkpoints and partial status. No automatic union runs, seed expansion, paid compute or evaluation-driven hyperparameter changes.

Runtime outputs include `base_freeze.json`, `heads_freeze.json`, `heads/*.json`, `predictions/{meta,dev_eval}.{npz,json}`, and `evaluation/{report.json,REPORT.md,bootstrap.json,bootstrap_source_counts.npz,scores.npz,complete.json}`. The source release and workflow are archived with private Hugging Face artifacts; heavy shared features and original images remain on Slurm.

## Review status before runtime preflight

The experiment engineer independently reviewed `combine.py`/`evaluate.py` for role isolation, fixed raw-logit columns, both meta heads, parameter serialization and photograph-paired AUROC. The scientific reviewer independently reviewed the engineer's role-map, fixed-20 trainer, prediction/freezing code and bounded smoke check; these received static PASS. Ops separately reviewed the learned-head/evaluation contract. The reviewer also checked Ops' source-bound workflow, fixed resource caps, duplicate-submission protection, all-four completion gates, and 23:00/04:00/07:00 cutoff semantics. The guardian permits CPU evaluation after timely verified predictions and publishes success only after the final evaluation completion manifest and all artifact hashes exist.

These are **static review verdicts, not numerical validation results**. The allocated Slurm smoke must still verify weighted AUROC against sklearn with ties and integer replication, JSON head scores/probabilities, source grouping, disjoint role loading, all-four batch/hidden identity, sampler resume, deterministic checkpoint restoration with unchanged tolerances, and restoration of the ordinary production backend. Record its actual job ID and result separately; passing it does not establish detector quality or full-cohort throughput.
