# Final fixed20 detector benchmark

This is the approved comprehensive comparison, not a new untouched-test claim.
The user approved the plan, including undertraining diagnostics and a secondary
100-logit linear probe, and explicitly requested implementation/execution on
September 16. A measured resource reservation and future cutoffs still precede
the full scientific launch. Implementation progress is not a completed result.

## Amendment 2026-09-16: three training seeds

Recorded on 2026-09-16, before any benchmark result existed. By user decision,
the training seeds were reduced from five (1, 2, 7, 17, 27) to three
(**7, 17, 27**). Seeds 7/17/27 were chosen because compatible G fits already
exist for exactly those seeds; the choice is not performance-based, and no
results were observed. Methods, the six primary contrasts, bootstrap settings
and the ddof=1 seed SD (now over n=3) are unchanged.

| Count | Before | Amended |
| --- | ---: | ---: |
| Training seeds | 5 | 3 |
| Neural fits (G/H/S/O) | 65 (20/20/20/5) | 39 (12/12/12/3) |
| Imported G fits | 12 | 12 |
| New neural fits | 53 | 27 (G 0, H 12, S 12, O 3) |
| Linear L fits | 1 | 1 |
| Meta heads (stack + last-only per G/H/S seed) | 30 (6 imported, 24 new) | 18 (6 imported, 12 new) |
| Reported methods | 25 | 25 |

## Fixed question and data

Can the existing graph-based ensembles outperform strong hidden-state,
same-feature non-message-passing and output-only baselines under a shared
training budget?

Every detector predicts whether the **frozen ViT is wrong**, not the image class
or corruption type. Retain the exact source-photo roles:

| Role | Photographs | Records | Use |
| --- | ---: | ---: | --- |
| base_train | 1,600 | 14,400 | Parameters and training-only preprocessing |
| checkpoint | 400 | 3,600 | Checkpoint selection |
| meta | 400 | 3,600 | Logistic heads and their standardizers |
| dev_eval | 800 | 7,200 | Complete frozen comparison |

All nine variants stay with their photograph. Training seeds are **7, 17, 27**
(amended 2026-09-16; see above); seeds do not change data membership. The development evaluation has
already been examined and is not an independent new test. No original-test
scoring is authorized.

## Models and reuse

| Family | Inputs | Architecture |
| --- | --- | --- |
| G, four layers | Selected attention graph plus final H12 token features | Existing edge-gated mean GNN, width 64, two message-passing layers |
| H, four layers | All 197 hidden tokens of that layer plus coordinate/CLS flags, no attention | Existing HiddenTokenSetModel, width 96 |
| S, four layers | Exactly G's node values and edge-value multiset, without incidence processing | Existing M5NodeEdgeSetModel, width 84 |
| O | All 100 frozen classifier logits | Existing 100-64-32-1 MLP |
| L, secondary | All 100 frozen classifier logits | One regularized logistic-regression probe |

Human layers **3, 6, 9, 12** map to graph blocks **2, 5, 8, 11** and hidden-state
indices **3, 6, 9, 12**. G/S use the same final-layer hidden values at every
graph layer. H therefore tests representations alone; G-versus-H is not an
identical-information message-passing isolation experiment.

G/H/S trainable parameter targets are respectively **130434 / 129986 / 131126**,
with O at **8577**. Verify the counts before launch without performance-driven
width selection. G/H/S capacity is closely matched; O/L are small practical
baselines.

The matrix contains **39 neural fits**. Twelve compatible G fits from the
completed seed-7 and seed-17/27 studies are imported read-only, so G needs no
new fit, leaving **27 new neural fits** (H 12, S 12, O 3), plus one L fit. Do not reuse the September 10 fits:
their training roles differ.

Preserve every imported scope, checksum, timestamp, checkpoint choice and
status. Seed 7 remains a late diagnostic from a formally incomplete original
protocol. Reusing its evidence does not retroactively make it on time.

## Training and fixed-budget diagnostics

Every neural base uses exactly **20 completed epochs**, AdamW learning rate
0.002, weight decay 0.0001, dropout 0.15, batch 24 and FP32. Class weights come
only from base_train. Select strictly greatest checkpoint-role AUROC and retain
the earliest tie. O's logit scaler is fitted only on base_train.

Save selected weights, optimizer/RNG/sampler resume state, all epoch histories,
checkpoint predictions and checksum completion records. Missing fits cannot be
replaced by a smaller ensemble or a shorter wall-time prefix.

For all 39 neural fits, including imports, report:

- Selected epoch and whether it equals 20.
- Actual checkpoint AUROC at 15 and 20 and the signed difference, **not the
  running-best AUROC**.
- Training loss at 15 and 20, absolute/percentage decrease, and OLS slope over
  epochs 16-20. A zero epoch-15 loss makes the percentage explicitly undefined.
- Epoch-20 selection counts and denominators by family/layer and family.

Predeclare a descriptive budget-sensitivity warning when the selected epoch
is 20, checkpoint AUROC increased from 15 to 20, and training loss decreased.
Show the raw quantities whether the warning is present or not. No warning is
proof of undertraining; absence is not proof of convergence.

Bind diagnostics before new development readouts. Legitimate warnings do not
fail an otherwise complete fit and never trigger extensions, retries or
architecture changes. Describe affected comparisons as **under the fixed20
budget**, not intrinsic inferiority.

## Ensembles and linear baseline

For each G/H/S family and seed, report four raw detectors, the equal mean of
their sigmoid scores, a learned four-score stack, and a learned last-only
control. Meta heads use StandardScaler fitted on meta only, then L2/lbfgs
logistic regression with C=1, max_iter=1000, tol=1e-6, class_weight=None and
random_state=7. Preserve signs and convergence warnings; JSON reconstruction
must agree at atol=rtol=1e-12.

There are **18 meta heads** (a stack and a last-only head for each of nine
G/H/S family/seed groups): six imported (the three G groups) and 12 new. Stacking/averaging combine
model scores within a seed; mean AUROC across seeds is not another ensemble.

L uses base_train-only standardization, all 100 raw logits, fixed L2/lbfgs
logistic regression with C=1, max_iter=1000, tol=1e-6 and random_state=7. Its
class-weight ratio matches neural BCE: negative weight 1, positive weight
n_negative/n_positive. Fit and freeze once on Slurm CPU, with convergence,
feature-order and serialization checks. Do not substitute the older two-feature
confidence/margin `output_lr`. L has no neural epoch history or training-seed SD.

Together with O, L, MSP and entropy, the benchmark reports **25 methods**.
MSP and entropy are fixed classifier-output references, not learned fits.

## Primary statistics

Six contrasts are fixed:

| Graph recipe | Comparator |
| --- | --- |
| G stack | S stack |
| G mean | S mean |
| G stack | H stack |
| G mean | H mean |
| G stack | O |
| G mean | O |

Compute each within-seed AUROC difference, then average over all three seeds.
Use **10,000 paired image_id bootstrap draws**, RNG seed **20260914**, sharing
photograph multiplicities across methods and seeds. Retain all nine views,
half-credit score ties and NumPy linear percentiles. Do not pool duplicated
seed-image rows or average predictions across seeds.

Show ordinary 95% intervals and Bonferroni-adjusted percentile intervals for
the six-comparison primary family: tail quantiles 0.05/(2*6) and
1-0.05/(2*6). Family coverage is nominal/approximate, not an exact finite-sample
guarantee. Record undefined draws without redrawing; withhold affected intervals.

L remains secondary and does not alter the six-comparison adjustment. Report
per-seed results and sample SD (ddof=1); L/MSP/entropy have no training-seed SD.
Intervals are conditional on fitted seeds and do not correct prior
development-data exposure. Report negative/inconclusive outcomes without
choosing a winning comparator.

Secondary outputs include average precision, tie-aware risk-coverage/AURC,
the fixed nine-condition descriptive breakdown, undertraining diagnostics,
parameters and measured resource cost. No evaluation-label threshold tuning.

## Execution and preservation

Use the new [source package](../../../pilots/final_comparison_20260916/) and
separate scientific/control namespaces. Preserve the old attention cache; add
a separately bound H3/H6/H9 sidecar and verify frozen ViT logits and H12 parity.

Global gates require all bases, all heads, L and diagnostics before new
dev_eval exports. Require a complete artifact inventory, not just successful
scheduler exits.

All numerical tests, feature extraction, training, prediction, restoration,
statistics and heavy file operations execute on Slurm. At most eight existing
GPU-class allocations are permitted, subject to the separately approved
resource budget. No stronger GPU, paid compute, automatic scientific retry,
extra seed or automatic deadline extension.

One operator owns durable submissions and reconciliation. Server-side
dependencies and a source-bound, write-once guardian survive laptop closure.
The laptop is safe to close for uninterrupted execution only after the complete
workflow is handed off and verified running/pending under that guardian.

No universal GNN-necessity, message-passing-only causal or language-model claim
is supported. S removes incidence/association and message passing together;
an endpoint-preserving causal control is not added automatically.

Keep heavy artifacts on Slurm. HF backup requires separate authentication,
quota and verification gates. All code/documentation commits remain local;
no push or merge.
