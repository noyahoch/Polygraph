# Scientific results review — September 21, 2026

The completed report is consistent with the frozen comparisons and their stated inferential scope. No blocking scientific/report discrepancy was found in the reviewed artifacts. The defensible result is that DG improves over D in this fitted development-data analysis, while an advantage of DG over matched DS is not established. C improves over B under the fixed readout recipe. These statements require the qualifications below.

This is a read-only post-run artifact review by the agent that implemented the fusion/statistics modules. It is not an independent numerical recomputation, external replication, fresh-environment replay, or review by an implementationally independent analyst. No numerical calculations, tests, remote connections, submissions, or source changes were performed during this review.

## Evidence reviewed

Artifact base: `/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/complementarity-20260921/ops/final_compact/`.

Reviewed `report/REPORT.md`, `report/report.json`, `statistics/results.json`, `statistics/complete.json`, `statistics/weighted_duplication.json`, and `statistics/draw_manifest.json`, against `/Users/omrifahn/Desktop/University/Master - Courses/Graphs ML/Final_Project/September_2026/Polygraph-September-10th/docs/experiments/complementarity_20260921/PROTOCOL.md`.

The two JSON result files are byte-identical by a read-only file comparison. The Markdown point estimates, contrasts and diagnostic entries agree with the inspected JSON values at their displayed precision. Campaign identity is `52a9cad66bebfd276ea536d2389a8d5acc04309ed8ae36c0636e94def6671588` throughout the inspected statistical receipts. The review did not independently verify remote backup availability or re-read every raw bootstrap draw; those are separate operational/artifact checks.

## All planned contrasts

Values below are copied from `statistics/results.json`, not recalculated. Each estimate is the mean original-fit within-seed AUROC difference. Primary intervals use the exact frozen decimal quantiles `[0.0083333333, 0.9916666667]` with linear interpolation, forming the nominal 95% Bonferroni family across the three primary contrasts. They should not be labeled ordinary unadjusted 95% intervals. Secondary intervals use `[0.025, 0.975]` and remain descriptive.

| Contrast | Cohort | Status | Original estimate | Saved interval |
|---|---|---|---:|---|
| DG−D | Fusion 400 | Primary | 0.007202711734107618 | [0.003415267292678116, 0.011330381073906493] |
| DG−DS | Fusion 400 | Primary | 0.00004321892631631297 | [−0.0013110156684055017, 0.001507827537605504] |
| C−B | Ablation 800 | Primary | 0.004135042273705129 | [0.0004276025553280066, 0.00791111064495215] |
| DG−DDprime | Fusion 400 | Secondary | 0.005213192444462537 | [0.0019416303866205964, 0.008672887373150674] |
| B−A | Ablation 800 | Secondary | 0.03123296949070749 | [0.025272578912304913, 0.03748251502156918] |
| D−C | Ablation 800 | Secondary | −0.0008997871045821615 | [−0.0017501087917153797, −0.00004847893264230776] |

All six entries report `valid_draws=2000`, empty invalid-ID lists, and no withheld interval. `statistics/complete.json` records `all_2000_draws_processed=true` and `all_primary_intervals_available=true`. The results record `failed_fit_draw_ids=[]`. These are completed-run receipt statements, not a new recomputation of the 2,000 draws by this reviewer.

The saved per-seed contrasts corroborate the direction qualifications. In seed order 7, 17, 27:

| Contrast | Seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| DG−D | 0.007410355737638485 | 0.008459561571759089 | 0.005738217892925279 |
| DG−DS | −0.00033899090808997556 | 0.001101358281518916 | −0.0006327105944800016 |
| C−B | 0.005054512965662505 | 0.0015372851961527045 | 0.0058133286593001765 |
| DG−DDprime | 0.006287991609250643 | 0.00419283946897242 | 0.005158746255164548 |
| B−A | 0.03085764129815538 | 0.03284187197526767 | 0.02999939519869943 |
| D−C | −0.0006381234400349989 | −0.001867469185284354 | −0.00019376868842713169 |

DG−DS changes sign across seeds; it does not support a graph-specific gain beyond matched set fusion. Its interval containing zero does not establish equivalence. DG−D and C−B are positive in all three saved seed comparisons, but three fixed seeds are not resampled independent population replications. C−B has a relatively small positive lower endpoint and is subject to the frozen finite-bootstrap-tail qualification. D−C is negative for all three seeds and its descriptive interval is narrowly below zero; this does not justify a family-controlled or general claim that the dynamics block is harmful or unnecessary.

## Cohorts and original-fit point estimates

The fusion table uses 400 assessment photographs, 3,600 records and 1,108 errors. The other 400 development photographs fit the nine original combiners. The ablation table uses all 800 original development photographs, 7,200 records and 2,271 errors. These tables must remain separate; their D values differ because they use different assessment cohorts, not because D was refit. The saved point-estimate source explicitly says original fitted models, never the bootstrap-refit mean.

| Fusion method, assessment 400 | Saved mean AUROC | Saved mean AP |
|---|---:|---:|
| D | 0.9001261316598191 | 0.8000409644269256 |
| G | 0.8956085466271854 | 0.7889325562396348 |
| S | 0.8949537436765157 | 0.7821838305016539 |
| DG | 0.9073288433939268 | 0.8126916281169922 |
| DS | 0.9072856244676105 | 0.8128921944738474 |
| DDprime | 0.9021156509494643 | 0.8042946799061784 |

| Ablation method, assessment 800 | Saved mean AUROC | Saved mean AP |
|---|---:|---:|
| A | 0.8635976827206422 | 0.7143825711396872 |
| B | 0.8948306522113497 | 0.7897825342145407 |
| C | 0.8989656944850549 | 0.8024697137380584 |
| D | 0.8980659073804728 | 0.7990389046451846 |

The report preserves all seed-specific AUROC/AP values and seed standard deviations. The primary inference concerns AUROC contrasts; AP point estimates do not acquire simultaneous confidence guarantees from the AUROC family. The saved ablation D mean AUROC matches the previously reported original D mean.

## Verification and diagnostics

The draw manifest retains IDs 0–1999, entropy 20260921, ascending numeric photo order, and spawn keys 0/1/2 for fusion fit400, fusion assessment400 and ablation assessment800. It preserves the exact approved primary and secondary quantiles. Fusion uncertainty is labeled as fit-photo and assessment-photo sampling with fixed constituent models/partition. Ablation uncertainty is labeled assessment-photo sampling conditional on fitted readouts/heads. Seeds are explicitly not resampled.

The weighted-duplication receipt reports `complete=true` and `passed=true`, includes all fixed draw IDs 0, 1, 7, 17, 27 across all three seeds and all three combiners, and records successful synthetic fixtures. Every inspected same-score AUROC/AP equivalence entry is defined, passed, and has recorded absolute difference 0 at absolute tolerance 1e−12, relative tolerance 0. Every inspected actual-comparison ranking diagnostic records zero changed average-rank rows and zero changed stable sort positions. Raw decision differences are retained rather than claimed bitwise equal; for example, draw27/seed17/DDprime records `7.105427357601002e-15`. The recipe’s scaler and decision tolerances remain 1e−12 and 1e−6 respectively. These receipts provide evidence for the specified equivalence checks, not a claim that all fitting paths are generally bitwise equivalent.

The diagnostic table includes all nine `(source_id,severity)` conditions, each with 400 records and explicit error/correct counts. Every condition has both outcomes; the clean condition has only 29 errors. Condition diagnostics are descriptive, with no separate inferential family. Improvement is not uniform across every metric/condition: the clean-condition DG AUROC/AP entries are below D, and condition `(3,3)` has higher DG AUROC but lower DG AP. Do not describe a universal condition-level improvement.

Within-photo ranking is explicitly defined using every incorrect/correct pair, half credit for exact ties, equal averaging across eligible photographs, and separate per-seed outputs. All methods/seeds report 295 eligible photographs out of 400. The saved seed means are D `0.8623006905210295`, G `0.8687561653663348`, S `0.8659375840731772`, DG `0.8696220069948883`, DS `0.8706125011209757`, and DDprime `0.8636162675993185`. These are descriptive values for a different ranking question from pooled assessment AUROC; they do not supply an additional significance claim.

## Defensible interpretation and limits

- DG−D supports additional ranking utility from combining the fitted graph score with D in this particular fixed-recipe analysis. DG−DS does not establish utility beyond adding the matched set score. Neither result proves a necessary role for graph topology or repeated message passing.
- C−B supports usefulness of intermediate auxiliary projections under this trained readout recipe. It is not a causal depth result, a necessity claim, or a guarantee for other heads/readouts/training seeds.
- B−A and DG−DDprime are positive descriptive comparisons. A is the six-feature original-classifier readout, not the older full-logit MLP. DDprime uses overlapping cyclic partners and is not compute-matched; its three entries must not be called independent replications.
- D−C includes added class-identity signals in the dynamics block. Its small negative descriptive result should be reported accurately without promoting it to a broad rejection of dynamics or a post-hoc family-controlled finding.
- Both studies reuse previously exposed development data. The intervals condition on the frozen constituent models, and the ablation intervals additionally hold the new readouts fixed. They do not correct prior development exposure, cover future base-model training variability, or turn this into an untouched-test result.
- The nominal Bonferroni family uses bootstrap percentile intervals, not exact finite-sample coverage. With the fixed 2,000 draws and roughly 17 draws per primary tail, small endpoint distances warrant restraint. No additional experiments or changed settings are requested by this review.

No missing evidence in this bounded report audit blocks a qualified summary. Remote backup/checksum verification, raw prediction reconstruction and full per-draw numerical recomputation are outside this review’s evidence claim.
