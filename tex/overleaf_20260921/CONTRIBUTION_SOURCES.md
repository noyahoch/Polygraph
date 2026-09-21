# Sources and scope of the contribution update

This revision adds completed follow-ups to the proposal imported in commit
`0304212`. It uses preserved reports, protocols and the model implementations;
no experiments, metrics or confidence intervals were recomputed. Scientific
claims in the original literature discussion were outside this review.

## Source-to-claim map

Paths below are relative to this folder. Line ranges describe the source
versions inspected during this revision, not the manuscript's changing layout.

| Added claim or definition | Supporting source and relevant location |
|---|---|
| Earlier group work already used internal/output fusion and rich set controls | [September 5 team report](../../docs/results/POLYGRAPH_TOPOLOGY_DEPTH_LAST4_REPORT_2026-09-05.md), executive summary; set controls at lines 64–75; conditional gate at 114–122. No new historical results were inferred. |
| September 11 rich graph−set +0.006233, 95% interval [0.002618, 0.009836]; attention-only −0.003686, [−0.008349, 0.001045], five seeds | [Completed results](../../docs/experiments/september10/RESULTS.md), lines 3–15; [protocol](../../docs/experiments/september10/PROTOCOL.md), feature regime and budget; [scientific review](../../docs/experiments/september10/FINAL_SCIENTIFIC_REVIEW.md), pinned native-artifact hashes. The directory name precedes the completion date. These are rich36 results, not raw12 ensemble replications. |
| Earlier learned four-layer stack improves on the specified learned single-layer control | [Layer replication results](../../docs/experiments/layer_ensemble_replication/RESULTS_20260916.md), lines 15–33 and 87–91; [protocol](../../docs/experiments/layer_ensemble_replication/PROTOCOL.md), lines 61–72. Primary replication used seeds 17/27. No matched four-model final-layer ensemble was evaluated. |
| Actual layers, H12 inputs, 784 node / 12 edge features and per-head thresholding | [Model constructors](../../pilots/final_comparison_20260916/models.py), lines 11–29; [graph loading](../../pilots/layer_screen_20260913/data.py), lines 14–32; [cache extraction](../../pilots/layer_screen_20260913/extract.py), lines 29–45. Values at or below 0.02 are zero-filled; these are not the original proposal's fully retained head channels. |
| G message passing, S endpoint removal, retained node/edge values and different pooling | [Shared model definitions](../../polygraph/training/models.py), lines 174–182, 204–230 and 278–352; [set dataset treatment](../../pilots/final_comparison_20260916/data.py), lines 335–341. S does not preserve edge incidence; it retains positions, CLS/H12/diagonal-attention values and edge counts. This is not a message-passing-only intervention. |
| Fixed 20-epoch G/S protocol, imported G fits, within-seed mean sigmoid scores | [Final-comparison protocol](../../docs/experiments/final_comparison_20260916/PROTOCOL.md), lines 37–83 and 108–119; [evaluation implementation](../../pilots/final_comparison_20260916/evaluate.py), lines 87–104. Reuse is not another independent training replication. |
| LD architecture, frozen ViT-B, twelve class heads, 85 features and 1,200/800/400 training roles | [LD protocol](../../docs/experiments/logit_dynamics_20260919/PROTOCOL.md), source allocation and optimization sections; [training](../../pilots/logit_dynamics_20260919/train.py), lines 25–32 and 111–219; [features](../../pilots/logit_dynamics_20260919/features.py). The earlier local comparator is a [CLS-sequence GRU](../../polygraph/training/baselines.py), lines 148–163. |
| Standalone 800-photo table and ordinary 95% G−D interval [−0.015874, −0.002361] | [September 19 report](../../docs/experiments/logit_dynamics_20260919/results/REPORT.md), lines 3–22; [machine-readable result](../../docs/experiments/logit_dynamics_20260919/results/report.json), `method_summary`, `primary`, `per_seed_metrics`. All three seed pairs favor D. |
| Fusion400 and decomposition800 tables; all six reported contrast intervals | [September 21 report](../../docs/experiments/complementarity_20260921/results/report/REPORT.md), tables and paired contrasts; [machine-readable result](../../docs/experiments/complementarity_20260921/results/statistics/results.json), `points.cohorts` and `contrasts`. Values are original-fit seed means. |
| 400/400 fusion split, fixed combinations, A/B/C/D definitions, 2,000 photo draws, exact adjusted quantiles and conditional uncertainty | [Frozen follow-up protocol](../../docs/experiments/complementarity_20260921/PROTOCOL.md), cohort, fusion, decomposition and bootstrap sections; [independent cross-review](../../docs/experiments/complementarity_20260921/reviews/independent_scientific_cross_review.md). Old meta400 trained LD and cannot be used as a fresh shared stacking pool. |

The September 21 native results belong to campaign
`52a9cad66bebfd276ea536d2389a8d5acc04309ed8ae36c0636e94def6671588`.
Their [preservation record](../../docs/experiments/complementarity_20260921/PRESERVATION.md)
identifies the verified immutable private snapshot and its source manifest.
The inspected method files match that frozen release, rather than an unrelated
similarly named implementation.

## Missing requested uncertainty

The completed September 21 results, frozen protocol and executable contrast
inventory contain exactly B−A, C−B, D−C, DG−D, DG−DD′ and DG−DS.
There is **no preserved DD′−D interval**. The manuscript therefore reports that
ordering descriptively, without a superiority inference or a newly derived CI.
DG−DD′ does have the descriptive 95% interval [0.001942, 0.008673]; B−A has
[0.025273, 0.037483]. Their secondary status is retained.

## Integration edits and remaining issues

The only replacement of original prose is:

> “Only the detector is trained…” → “In this proposed graph-detector setup, only the detector is trained…”

This narrowly prevents the original weighted-BCE statement from appearing to
describe LD's auxiliary multiclass heads or the unweighted logistic fusion.
All other manuscript changes are additions: the TL;DR sentence, the research
question extension, the bridge before the unchanged proposed architecture,
and the completed-follow-up section with two tables. The title, authors,
document class, preamble, comments, bibliography and original plans are intact.

The resulting document is **7 pages**, versus 4 for the imported proposal;
it has not been forced into a five-page limit by deleting colleagues' material
or changing fonts, margins or spacing. A later general manuscript edit would
be needed if five pages is a strict final-submission limit.

Unrelated pre-existing issues were left untouched: the two adjacent fallback
ViT-fine-tuning descriptions overlap, and the ACM template includes publication
metadata/footer conventions that are not tailored to a course submission.
The original unseen-corruption plan remains a proposal; the added results do
not claim to have validated it. Original literature/priority claims were not
independently audited or expanded.
