# Polygraph Evidence-Flow Research: Interim Report — 2026-09-03

All numbers below use the rebuilt 75,000-record dataset. Historical metrics are not mixed into these comparisons.

## Executive summary

The best confirmed standalone internal model so far is **M5: evidence-flow edges plus full final-layer token hidden states**.

| Method | Seed 7 | Seed 1 | Seed 2 | Mean ± std |
|---|---:|---:|---:|---:|
| MSP | 0.86510 | — | — | 0.86510 |
| Raw-logit MLP | 0.88010 | 0.87659 | 0.88009 | **0.87893 ± 0.00166** |
| M5 hidden evidence-flow | 0.88625 | 0.88600 | 0.88496 | **0.88574 ± 0.00056** |
| Exploratory output + M5 gate | 0.89392 | — | — | 0.89392 |
| Strict raw-logit MLP | 0.87809 | 0.88006 | 0.88009 | **0.87941 ± 0.00093** |
| Strict output + M5 gate | 0.89112 | 0.89321 | 0.88952 | **0.89128 ± 0.00151** |

M5 improves over the seed-matched raw-logit MLP in all three seeds:

- Seed 7: +0.00615
- Seed 1: +0.00942
- Seed 2: +0.00487
- Mean improvement: **+0.00681**

Standalone M5 is a reproducible but small improvement over the strongest output baseline. It does not meet the pre-registered ≥0.01 threshold for a “confirmed improvement over the output baseline.” The strictly trained conditional gate does: it improves its matched output expert by **+0.01186 mean AUROC**, has wholly positive base-image bootstrap intervals in all three seeds, and does not collapse on the weather holdout.

M5 improves MSP by **+0.02064** mean AUROC, consistent across all three seeds. The final scientific result is stronger and more specific: internal evidence is complementary to the full output distribution, and a leakage-controlled conditional gate converts it into a **confirmed main-plan improvement over the full-logit output baseline**.

## Evidence-flow ablation

| Model | Representation | Test AUROC | Δ vs M0 |
|---|---|---:|---:|
| M0 | Raw attention | 0.83477 | — |
| M1 | Attention + message magnitude | 0.83685 | +0.00209 |
| M2 | Attention + class-direction proxy | 0.86602 | +0.03125 |
| M3 | Magnitude + class direction | 0.86545 | +0.03068 |
| M4 | M3 + compact class-conditioned nodes | 0.87725 | +0.04249 |
| M5 | M3 + full hidden token states | **0.88574 ± 0.00056** | +0.05097 |

Current interpretation:

- Message magnitude alone contributes little.
- Predicted-versus-runner-up class direction contributes substantial information.
- Combining magnitude with class direction does not improve over class direction alone.
- Compact class-conditioned node evidence contributes another large gain.
- Full token hidden states add approximately **+0.00849** over compact evidence on the available seed-7 comparison.
- M5 exceeds the full-logit output detector consistently, but by only about 0.007 AUROC on average.

## Results by degradation severity

“Per degradation” here means pooled by degradation severity, not corruption type. M5 and raw-logit MLP values are three-seed means. The other learned ablations are seed 7.

| Severity | MSP | Raw-logit MLP | M0 raw graph | M2 class direction | M4 compact | M5 hidden |
|---|---:|---:|---:|---:|---:|---:|
| Clean / 0 | 0.92058 | 0.91180 | 0.82838 | 0.88533 | **0.92420** | 0.90199 |
| Severity 1 | 0.89276 | 0.89652 | 0.85518 | 0.88588 | 0.89896 | **0.89899** |
| Severity 2 | 0.89036 | 0.89730 | 0.85263 | 0.88115 | 0.89798 | **0.90321** |
| Severity 3 | 0.87270 | 0.87888 | 0.83931 | 0.87173 | 0.88181 | **0.88630** |
| Severity 4 | 0.85367 | 0.87318 | 0.83272 | 0.86008 | 0.86807 | **0.88411** |
| Severity 5 | 0.81841 | 0.85467 | 0.80551 | 0.83831 | 0.84372 | **0.86079** |
| All severities | 0.86510 | 0.87893 | 0.83477 | 0.86602 | 0.87725 | **0.88574** |

M5’s advantage is strongest at degradation severities 2–5. The clean slice contains only 176 records and has visibly higher seed variability, so it should not drive conclusions.

## Output-only findings

| Output method | Main AUROC |
|---|---:|
| MSP | 0.86510 |
| Entropy | 0.86517 |
| Margin | 0.86367 |
| pNormSoftmax | 0.86406 |
| p-normalized max logit | 0.86396 |
| Max logit | 0.86239 |
| Energy | 0.86120 |
| Sorted-centered logit LR | 0.86341 |
| Sorted-centered logit MLP | 0.86712 mean |
| Raw-logit LR | 0.78322 |
| Raw-logit MLP | **0.87893 mean** |

The nonlinear raw-logit model substantially exceeds its sorted-centered counterpart. Therefore, the gain is not explained solely by the class-invariant shape of the probability distribution; class-specific logit patterns or calibration differences appear to matter.

The grouped bootstrap for raw-logit MLP versus MSP is positive for every seed. For example, seed 7 has mean delta +0.01501 with 95% interval **[+0.00857, +0.02183]**.

## TCP result

CLS TCP is weaker than the raw-logit MLP:

- Main seeds 7/1/2: 0.86892 / 0.86780 / 0.87253
- Mean: approximately 0.86975
- Weather results are inconsistent and average below the weather raw-logit MLP.

TCP is therefore not currently a finalist.

## Current-plan CLS baselines

The final-CLS MLP completed for all three seeds before the first CLS-suite process failed during sequence-model scoring:

- Seed 7 / 1 / 2: **0.87972 / 0.88869 / 0.88394**
- Mean ± standard deviation: **0.88412 ± 0.00366**
- Severity-pooled means, clean through severity 5: **0.90608 / 0.89856 / 0.90064 / 0.88469 / 0.87762 / 0.86423**

The CLS MLP is 0.00519 above the raw-logit MLP mean and 0.00162 below M5. The first CLS-sequence evaluation materialized the entire split on GPU and failed with CUDA OOM while an unrelated process occupied 15.2 GB. The scorer was changed to bounded 1,024-record batches; completed CLS MLP outputs were detected and not retrained.

The resumed CLS-sequence GRU then completed all seeds:

- Seed 7 / 1 / 2: **0.88291 / 0.88386 / 0.88495**
- Mean ± standard deviation: **0.88390 ± 0.00083**
- Severity-pooled means, clean through severity 5: **0.91120 / 0.89455 / 0.89657 / 0.88479 / 0.88089 / 0.86527**

CLS sequence and final CLS MLP are effectively tied in mean AUROC; the sequence model is much less seed-variable. M5 remains 0.00183 above CLS sequence and 0.00162 above final CLS MLP—differences too small to establish a graph-specific benefit.

## Conditional combination

The exploratory mixture gate selected using a group-disjoint 70/30 subdivision of validation achieved:

- Raw-logit MLP seed 7: 0.88010
- M5 seed 7: 0.88625
- Output + M5 gate: **0.89392**
- Gain over output-only seed 7: **+0.01382**
- Gain over MSP: **+0.02883**

This result is promising but exploratory. The detector used the original validation split for early stopping, so strict meta-validation confirmation is required before treating 0.89392 as a clean final result.

G2, using raw-logit MLP plus CLS sequence, selected the mixture gate on the group-disjoint meta-validation subset and achieved **0.88996** test AUROC. This is +0.00986 over its seed-7 output expert but 0.00396 below the output+M5 exploratory gate. Mean gate weight was 0.5200 (correct 0.5138, incorrect 0.5263); these descriptive differences are not causal. G2 is exploratory for the same validation-reuse reason.

G3 added the raw-attention graph to the raw-logit and CLS-sequence scores. Meta-validation selected the logistic combiner, which achieved **0.88899** test AUROC: +0.00889 over seed-7 output alone, but slightly below G2 and 0.00494 below output+M5. Raw graph therefore does not add useful conditional information beyond the stronger CLS representation in this exploratory comparison.

G4 supplied raw-logit MLP, M5, CLS TCP, and graph-statistics MLP scores. Meta-validation selected logistic regression (0.90089 validation AUROC) and the fixed model achieved **0.89387** test AUROC. This is effectively tied with but 0.00005 below the simpler G1 output+M5 gate (0.89392). Adding TCP and hand-designed graph statistics therefore yields no measurable benefit; the strict candidate remains the simpler two-expert mixture gate.

Strict confirmation uses a deterministic 50/50 split of original validation groups: main `base_val` has 2,991 records from 498 photographs and `meta_val` has 3,009 records from 499 disjoint photographs. The raw-logit expert was retrained with original train plus base_val-only early stopping. Seeds 7 / 1 / 2 achieved **0.87809 / 0.88006 / 0.88009**, or **0.87941 ± 0.00093**, on the untouched main test. These strict output scores and checkpoints are now fixed; strict M5 retraining and gate fitting remain.

Strict M5 seed 7 selected epoch 4 on base_val, stopped at epoch 12, and achieved **0.88512** on the untouched test (+0.00702 over strict output seed 7). The fixed mixture-gate architecture was then fit only on meta_val using five base-image GroupKFold models; their averaged test prediction achieved **0.89112**. This is +0.01302 over strict output seed 7 and +0.02602 over MSP. The gate's out-of-fold meta-validation AUROC was 0.87325. Mean test gate weight was 0.4685 (correct 0.4623, incorrect 0.4748, confidence ≥0.9: 0.4660).

| Severity | Strict output s7 | Strict M5 s7 | Strict gate s7 |
|---|---:|---:|---:|
| Clean / 0 | 0.89928 | 0.90780 | **0.91606** |
| Severity 1 | 0.89482 | 0.89728 | **0.90398** |
| Severity 2 | 0.89643 | 0.89984 | **0.90796** |
| Severity 3 | 0.88182 | 0.88882 | **0.89543** |
| Severity 4 | 0.87289 | 0.88253 | **0.88795** |
| Severity 5 | 0.85113 | 0.86304 | **0.86754** |

This seed-7 strict result cleared the triage threshold and triggered the completed multi-seed confirmation below.

Strict seed 1 is also complete. M5 achieved **0.88507** versus its strict output expert's **0.88006** (+0.00501). The fixed five-fold group gate achieved **0.89321**, a gain of **+0.01315** over strict output and +0.02811 over MSP. The gate improvement is directionally consistent across the first two strict seeds; seed 2 is still required.

Strict seed 2 completed with M5 at **0.88320**, strict output at **0.88009**, and the fixed gate at **0.88952**. Across seeds 7 / 1 / 2, the strict gate is **0.89112 / 0.89321 / 0.88952**, or **0.89128 ± 0.00151**. The seed-matched improvements over strict output are +0.01302 / +0.01315 / +0.00942, with mean **+0.01186** and no negative seed. Strict M5 averages **0.88446 ± 0.00089**, only +0.00505 over strict output. Grouped bootstrap and weather confirmation are the remaining criteria before final classification.

Three-seed strict gate severity means (clean/0 through severity 5) are **0.91878 / 0.90630 / 0.90911 / 0.89196 / 0.88792 / 0.86754**; the corresponding strict-output means are **0.90974 / 0.89673 / 0.89767 / 0.88036 / 0.87430 / 0.85367**. The gain is not confined to one degradation severity.

The 2,000-repetition base-photograph bootstrap is positive for the strict gate in every seed. Gate-minus-strict-output intervals are seed 7 **+0.01301 [+0.01006, +0.01604]**, seed 1 **+0.01315 [+0.01014, +0.01609]**, and seed 2 **+0.00938 [+0.00742, +0.01146]**; all 2,000 deltas are positive in every seed. Gate-minus-MSP intervals are also wholly positive. In contrast, strict M5 alone has deltas +0.00702 / +0.00502 / +0.00307, and the latter two intervals cross zero. The strict gate now satisfies every main-plan confirmation criterion; only the required weather non-collapse criterion remains.

## Weather-family holdout (completed non-graph references)

The current rebuilt weather plan has 14,232 test records; 4,306 belong to the held-out weather-plus-extra sources. Absolute values are not compared to the main plan because the test sets differ.

| Method | Seeds | All-test AUROC | Held-out weather + extras AUROC |
|---|---|---:|---:|
| MSP | deterministic | 0.86252 | **0.88441** |
| Raw-logit MLP | 7 / 1 / 2 | 0.87069 ± 0.00064 | 0.88389 ± 0.00144 |
| Final-CLS MLP | 7 / 1 / 2 | **0.87507 ± 0.00176** | 0.88405 ± 0.00276 |
| CLS-sequence GRU | 7 / 1 / 2 | 0.87201 ± 0.00110 | 0.87438 ± 0.00414 |

The final-CLS MLP is currently the strongest completed method on weather all-test, but none of the learned references improves the held-out-family slice over MSP beyond noise. The CLS sequence model loses about 0.0100 there. The M5 graph finalist and final conditional model still require weather evaluation.

The first M5 weather attempt was deliberately terminated after one completed epoch because an unrelated GPU job occupied 15.6 GB and 89% utilization: the epoch took roughly 10 minutes versus about 2–3 minutes under normal conditions, projecting beyond the per-run cutoff. `state_seed7.pt` was preserved for exact resumption; no result is inferred from this interrupted attempt.

The strict weather split has 2,496 base-validation records from 496 photographs and 2,362 meta-validation records from 496 disjoint photographs. Retraining the raw-logit expert with base_val-only selection gave seed 7 / 1 / 2 all-test AUROCs of **0.86745 / 0.86830 / 0.87138**, or **0.86904 ± 0.00169**. Its held-out weather-plus-extras mean is **0.88162 ± 0.00283**, slightly below MSP's 0.88441. These are the clean strict reference scores for the weather gate.

Severity-pooled mean AUROCs (clean/0 through severity 5) are:

| Method | Clean / 0 | Sev. 1 | Sev. 2 | Sev. 3 | Sev. 4 | Sev. 5 |
|---|---:|---:|---:|---:|---:|---:|
| MSP | **0.89537** | **0.89732** | 0.88001 | 0.87306 | 0.84349 | 0.82336 |
| Raw-logit MLP | 0.88041 | 0.89185 | 0.88454 | 0.87473 | 0.85721 | 0.85518 |
| Final-CLS MLP | **0.89564** | 0.89303 | **0.88841** | 0.87658 | 0.86045 | 0.86411 |
| CLS sequence | 0.85964 | 0.88450 | 0.88227 | 0.87470 | 0.85982 | **0.86599** |

## Sanity and code audit

- Rebuilt-store M0 is 0.83477 versus historical 0.8417, a difference of −0.00693. This is within the predefined sanity tolerance.
- All original and new tests pass: 39 original tests, 6 Deep-Sets tests, 7 control tests, 7 sidecar tests, and 5 research-evaluation tests. The added mutation check confirms that true-class-derived TCP targets do not affect or enter graph inference.
- A real metadata bug was found: PyG automatically incremented `store_index` during batching because the field name contains “index.” It did not affect model training, labels, or reported AUROCs, but it invalidated saved alignment IDs. The batching behavior is fixed and mutation-tested. Legacy score vectors were reconciled only where plan hash and four independent metadata arrays matched exactly row-for-row.

## Grouped bootstrap and complementarity

All intervals below use 2,000 paired resamples of base-photograph groups. The first bootstrap command failed after computation because it referenced the exploratory gate under an obsolete directory name; no partial output was accepted. The corrected command reran all resamples and wrote `runs/research_20260830/final/bootstrap_internal_current.json`.

| Comparison | Seed | Mean AUROC delta | 95% interval | P(delta > 0) |
|---|---:|---:|---:|---:|
| M5 − raw-logit MLP | 7 | +0.00627 | [+0.00028, +0.01234] | 0.9785 |
| M5 − raw-logit MLP | 1 | +0.00935 | [+0.00400, +0.01472] | 0.9995 |
| M5 − raw-logit MLP | 2 | +0.00497 | [−0.00139, +0.01141] | 0.9345 |
| M5 − MSP | 7 | +0.02110 | [+0.01340, +0.02907] | 1.0000 |
| M5 − MSP | 1 | +0.02092 | [+0.01441, +0.02723] | 1.0000 |
| M5 − MSP | 2 | +0.01994 | [+0.01277, +0.02706] | 1.0000 |
| Exploratory G1 gate − raw-logit MLP | 20260830 | +0.01386 | [+0.01028, +0.01731] | 1.0000 |
| Exploratory G1 gate − MSP | 20260830 | +0.02885 | [+0.02247, +0.03524] | 1.0000 |

M5's seed-matched delta is positive in all seeds, but one seed's group interval crosses zero and the mean effect is below +0.01. It is therefore a reproducible **possible small improvement**, not a confirmed improvement over the output baseline. M5 recovers 33.5–38.9% of errors missed by its seed-matched raw-logit detector at a 50% flag budget; its Spearman correlation with raw-logit scores is 0.879–0.897. Against MSP, M5 recovers 41.0–47.8% of MSP-blind errors and has lower rank correlation (0.814–0.851).

M5 does not separate from simpler hidden-representation baselines under grouped uncertainty. Against final-CLS MLP, its seed-matched bootstrap mean deltas are **+0.00648, −0.00271, and +0.00102**, and every 95% interval crosses zero. Against CLS sequence they are **+0.00336, +0.00205, and −0.00007**, again with every interval crossing zero. This is strong evidence that the best standalone internal result is primarily a hidden-representation result, not a demonstrated graph/evidence-flow advantage.

The exploratory G1 gate's interval is clearly positive, but this does not repair its validation reuse. Strict confirmation remains necessary.

## Matched structure controls (in progress)

### C2: NodeEdgeSet

C2 receives the same base node features and attention edge values as M0, but processes the node and edge multisets independently and never receives edge incidence. It has 15,457 parameters versus M0's 16,066. Seed 7 selected epoch 18 using validation AUROC and stopped at epoch 26.

| Slice | C2 NodeEdgeSet AUROC | M0 raw graph AUROC | Delta C2 − M0 |
|---|---:|---:|---:|
| All | **0.84556** | 0.83477 | +0.01079 |
| Clean / 0 | **0.88559** | 0.82838 | +0.05721 |
| Severity 1 | **0.87374** | 0.85518 | +0.01856 |
| Severity 2 | **0.86198** | 0.85263 | +0.00936 |
| Severity 3 | **0.84755** | 0.83931 | +0.00824 |
| Severity 4 | **0.84137** | 0.83272 | +0.00865 |
| Severity 5 | **0.81106** | 0.80551 | +0.00555 |
| Unseen extra sources | **0.85396** | 0.84162 | +0.01234 |

C2 is better than M0 overall and at every severity. Therefore the raw M0 comparison does **not** support the claim that connectivity adds information beyond its matched node/edge value multisets. Endpoint-local and rewiring controls are still required before deciding whether connectivity becomes useful for the stronger evidence-conditioned representations.

### C3: EndpointSet

C3 sees `[source node, target node, attention edge]` records and pools them without iterative neighborhood aggregation. It has 9,217 parameters. Seed 7 selected epoch 22 and stopped at epoch 30.

| Slice | C3 EndpointSet AUROC | M0 raw graph AUROC | Delta C3 − M0 |
|---|---:|---:|---:|
| All | **0.83754** | 0.83477 | +0.00278 |
| Clean / 0 | **0.85176** | 0.82838 | +0.02337 |
| Severity 1 | **0.85870** | 0.85518 | +0.00352 |
| Severity 2 | **0.85353** | 0.85263 | +0.00090 |
| Severity 3 | **0.83948** | 0.83931 | +0.00017 |
| Severity 4 | **0.83682** | 0.83272 | +0.00410 |
| Severity 5 | **0.80821** | 0.80551 | +0.00270 |
| Unseen extra sources | 0.83974 | **0.84162** | −0.00188 |

C3 essentially matches M0 overall and at severities 2–5. Together, C2 and C3 show that M0's performance does not require iterative message passing or intact higher-order topology. The endpoint-local control is not fully structure-blind, but it is a direct negative result for a strong raw-attention message-passing claim.

### C1: EdgeSet

C1 receives attention edge values only and discards nodes and endpoints. Its width was capacity-matched at 64, giving 13,313 parameters. Seed 7 selected epoch 17 and stopped at epoch 25.

| Slice | C1 EdgeSet AUROC | M0 raw graph AUROC | Delta C1 − M0 |
|---|---:|---:|---:|
| All | **0.84736** | 0.83477 | +0.01259 |
| Clean / 0 | **0.87552** | 0.82838 | +0.04713 |
| Severity 1 | **0.87395** | 0.85518 | +0.01877 |
| Severity 2 | **0.86686** | 0.85263 | +0.01423 |
| Severity 3 | **0.85198** | 0.83931 | +0.01267 |
| Severity 4 | **0.84120** | 0.83272 | +0.00848 |
| Severity 5 | **0.81229** | 0.80551 | +0.00678 |
| Unseen extra sources | **0.85642** | 0.84162 | +0.01480 |

The edge-value multiset alone outperforms M0 overall and at every degradation severity. The historical flat-attention result was therefore not representative of a capacity-matched, full-threshold edge-set baseline. For raw attention, the evidence now favors distributional edge-value statistics over graph connectivity.

### C5/C6: Rewiring diagnostics

`target_permute` was run with the fixed M0 training protocol. It reached epoch 29 in 91.98 minutes and was aborted at the pre-registered 90-minute per-run limit before early stopping. Its best validation AUROC was 0.80395 at epoch 27, versus M0's 0.82186. The resumable state is preserved, but no test result is reported because the configuration was not fixed by completed early stopping.

The measured cost was approximately 3.17 minutes per epoch, projecting a 60-epoch upper bound near 190 minutes. `shuffle_attr` performs the same deterministic per-record permutation and has the same data-path cost, so it was skipped rather than knowingly launching another run beyond the limit. This is a runtime skip, not a negative test result. The incomplete target-permutation trajectory is directionally consistent with connectivity mattering, but it cannot overturn the completed C1–C3 controls or support a final claim.

## Compact class-conditioned node evidence

### D1: TransformerConv with compact evidence and raw attention

D1 appends four deployment-available token features—predicted-class logit lens, runner-up logit lens, their margin, and hidden norm—to the base node features. Ground-truth class is not an input. The model has 16,962 parameters; seed 7 selected epoch 7 and stopped at epoch 15.

| Slice | D1 compact + attention | M0 raw graph | Delta D1 − M0 |
|---|---:|---:|---:|
| All | **0.86938** | 0.83477 | +0.03461 |
| Clean / 0 | **0.91955** | 0.82838 | +0.09117 |
| Severity 1 | **0.89891** | 0.85518 | +0.04372 |
| Severity 2 | **0.89441** | 0.85263 | +0.04178 |
| Severity 3 | **0.87400** | 0.83931 | +0.03469 |
| Severity 4 | **0.85717** | 0.83272 | +0.02445 |
| Severity 5 | **0.82545** | 0.80551 | +0.01993 |
| Unseen extra sources | **0.87812** | 0.84162 | +0.03650 |

D1 improves raw graph M0 by **+0.03461** and MSP by **+0.00428**, but trails the three-seed raw-logit MLP mean by **0.00955**. Comparing fixed seed-7 models, M4's evidence-flow edges improve over D1 by **+0.00787**, suggesting that class-conditioned edge flow adds information beyond compact class-conditioned node evidence. D2 and D3 will determine whether D1's gain requires message passing.

### D2: NodeEdgeSet with compact evidence and raw attention

D2 receives exactly the compact node features and raw attention edge values used by D1 but discards all edge incidence. Its width-48 model has 15,649 parameters; seed 7 selected epoch 11, stopped at epoch 19, and trained for 45.0 minutes (the GPU was shared with an unrelated process).

| Slice | D2 compact NodeEdgeSet | D1 compact graph | Delta D2 − D1 |
|---|---:|---:|---:|
| All | **0.87360** | 0.86938 | +0.00422 |
| Clean / 0 | **0.92058** | 0.91955 | +0.00103 |
| Severity 1 | 0.89875 | **0.89891** | −0.00016 |
| Severity 2 | **0.89577** | 0.89441 | +0.00136 |
| Severity 3 | **0.87978** | 0.87400 | +0.00578 |
| Severity 4 | **0.86407** | 0.85717 | +0.00690 |
| Severity 5 | **0.83496** | 0.82545 | +0.00952 |
| Unseen extra sources | **0.88229** | 0.87812 | +0.00417 |

D2 slightly but consistently outperforms D1 overall and at severities 2–5. Thus the large gain from compact class-conditioned nodes does **not** require graph connectivity; independent node and edge distributions are sufficient and perform better in this comparison. D2 remains 0.00532 below the three-seed raw-logit MLP mean.

### D3: EndpointSet with compact evidence and raw attention

D3 represents every attention edge as a local `[source node, target node, edge value]` record, pools records permutation-invariantly, and performs no neighborhood aggregation or path reasoning. It has 9,601 parameters; seed 7 selected epoch 11, stopped at epoch 19, and trained for 36.6 minutes.

| Slice | D3 compact EndpointSet | D1 compact graph | Delta D3 − D1 |
|---|---:|---:|---:|
| All | **0.87588** | 0.86938 | +0.00649 |
| Clean / 0 | 0.91671 | **0.91955** | −0.00284 |
| Severity 1 | **0.90181** | 0.89891 | +0.00290 |
| Severity 2 | **0.89876** | 0.89441 | +0.00435 |
| Severity 3 | **0.88215** | 0.87400 | +0.00815 |
| Severity 4 | **0.86559** | 0.85717 | +0.00842 |
| Severity 5 | **0.83637** | 0.82545 | +0.01093 |
| Unseen extra sources | **0.88377** | 0.87812 | +0.00565 |

D3 is the best of the compact/raw-attention models and trails the raw-logit MLP mean by only **0.00305**. It also beats the iterative TransformerConv D1 overall and at every corrupted severity. The compact-evidence benefit is therefore endpoint-local or distributional; this ablation provides no evidence that higher-order message passing is responsible for it. D3 is 0.00986 below M5, so the remaining question is whether M5's advantage arises from full hidden semantics, evidence-flow edge values, or their interaction.

## Graph-level attention statistics

The final-layer graph-statistics extraction completed rather than being skipped. It processed all 75,000 records sequentially in **356.5 seconds**, used no ViT forward pass or GPU, and produced a 47 MB aligned float32 sidecar. Its manifest validates the store-key hash, all 38 shard counts, model ID, record count, layer 11, and the 161-feature schema. Predictive LR, HGB, and MLP results follow below as they complete.

All planned graph-statistics predictors then completed in 122.6 seconds. Hyperparameters were fixed and the MLP checkpoints were selected using validation AUROC only.

| Model | Seeds | Main AUROC | AUPRC | AURC |
|---|---|---:|---:|---:|
| Graph statistics LR | deterministic | 0.82751 | 0.80433 | 0.24982 |
| Graph statistics HGB | deterministic | 0.84811 | 0.82141 | 0.23203 |
| Graph statistics MLP | 7 / 1 / 2 | **0.85254 ± 0.00116** | 0.82467 mean | 0.23074 mean |

| Severity | LR | HGB | MLP mean |
|---|---:|---:|---:|
| Clean / 0 | 0.83639 | 0.86493 | **0.88120** |
| Severity 1 | 0.84547 | **0.87055** | 0.86912 |
| Severity 2 | 0.84577 | 0.86625 | **0.87096** |
| Severity 3 | 0.82383 | 0.84748 | **0.85199** |
| Severity 4 | 0.82920 | 0.84770 | **0.85417** |
| Severity 5 | 0.80246 | 0.81818 | **0.82274** |

Cheap global summaries contain reproducible error signal, but the best graph-statistics model remains below MSP by 0.01255 and below the raw-logit MLP by 0.02638. These features are not competitive standalone; their remaining value is as an optional low-dimensional gate context, evaluated without feeding degradation identity or severity.

On the weather plan, graph-statistics LR, HGB, and the three-seed MLP obtain all-test AUROCs of **0.82795**, **0.84405**, and **0.84805 ± 0.00144**, respectively. The MLP's held-out weather-plus-extras AUROC is **0.86008**, versus MSP's 0.88441. The negative standalone conclusion therefore generalizes to the weather holdout, and G4 already showed that adding graph statistics to the main exploratory gate does not help.

## Architecture ablation: explicit SimpleMPNN

F0 replaced TransformerConv's learned secondary attention with the pre-registered explicit mean-aggregation MPNN while keeping base nodes and raw-attention edges fixed. The 17,250-parameter run was aborted at the per-run cutoff after **5,542.9 seconds (92.4 minutes)**. It had reached epoch 40 without early stopping; best validation AUROC was only approximately 0.8267, while M0's completed TransformerConv best validation AUROC was 0.8219. The resumable state is retained, but there is no selected checkpoint or test result, so this is not evidence that SimpleMPNN improves test performance.

F1 (SimpleMPNN with evidence-flow edges) and F2 (SimpleMPNN with compact nodes plus evidence-flow) were skipped. F0's measured runtime already exceeded the 90-minute limit, and their 36-dimensional edge records increase rather than reduce the dominant per-edge MLP workload. Consequently, whether secondary learned GNN attention helps remains unresolved under this study's compute policy.

The optional T1 final-four-layer temporal graph was also skipped on measured runtime grounds. It uses the same explicit MPNN as F0 over four layer copies plus temporal identity edges; because the one-layer F0 already exceeded 90 minutes without selecting a checkpoint, T1's optimistic projection is far beyond the allowed per-run limit. No temporal result is inferred.

## Full-hidden raw-attention control

H0 isolates hidden semantics from the evidence-flow edge representation. It uses the same final-layer TransformerConv and full 768-dimensional token hidden states as M5, but supplies only raw 12-head attention weights on edges. The 188,098-parameter seed-7 model selected epoch 5, stopped at epoch 13, and achieved **0.87537 AUROC** (AUPRC 0.85361, AURC 0.21956).

M5 at the same seed achieves 0.88625, a **+0.01088** improvement over H0. H0 is also 0.00188 below M4 (compact node evidence plus evidence-flow edges). Therefore full hidden states alone do not explain M5: evidence-flow edges add a meaningful one-seed gain when node semantics are held fixed. Conversely, M5 exceeds M4 by +0.00900 at seed 7, so richer hidden semantics also add information when evidence-flow edges are held fixed. These two controlled contrasts support an interaction of hidden representation and class-conditioned edge flow; they do not establish a causal decomposition.

H0 AUROC by clean/0 through severity 5 is **0.88598 / 0.88542 / 0.89002 / 0.87742 / 0.87242 / 0.85718**. M5's advantage is concentrated in corrupted data and is particularly visible at severities 4 and 5.

## Optional graph TCP multitask ablation

The pre-registered trigger for a one-seed TCP auxiliary loss was met, so M5 was retrained with a second head regressing the true-class-probability logit during training only. The true class and TCP target are absent from inference inputs. The 191,235-parameter model selected epoch 7 and stopped at epoch 15.

| Inference score | Main AUROC |
|---|---:|
| Single-task M5 error head, seed 7 | **0.88625** |
| Multitask error head | 0.88342 |
| Negative predicted-TCP head | 0.85869 |
| Fixed average of validation-standardized error and negative-TCP heads | 0.87139 |

The multitask error head loses 0.00283 versus single-task M5, while the TCP head and fixed average are substantially weaker. TCP supervision therefore does not improve this graph representation; per protocol, no confirmation seeds were launched.

The optional CLS-trajectory TCP pilot was also run because final-CLS TCP was within 0.01 AUROC of the then-current output baseline and the measured cost was low. A 348,993-parameter GRU over all 12 CLS states selected epoch 1 and completed in 209.2 seconds with 534 MB peak GPU allocation. It achieved **0.86474 AUROC**, versus final-CLS TCP seed 7 at 0.86892, ordinary CLS-sequence error classification at 0.88291, and MSP at 0.86510. Because the seed-7 pilot did not improve its TCP comparator or clear a finalist threshold, seeds 1/2 and a weather repeat were skipped.

## What is established so far

- MSP is not the output-only ceiling.
- Raw attention and message magnitude are weak standalone error signals.
- Predicted-class-conditioned value direction is substantially more useful than raw attention.
- Compact class evidence nearly matches the full-logit detector.
- Full hidden token states plus evidence flow provide a consistent, small improvement over the full-logit detector across three seeds.
- The conditional gate may convert this complementarity into a larger gain, but it is not yet strictly confirmed.

## Literature grounding

The required papers were checked at the method and experiment level. Their claims and this repository's tests must not be conflated:

- [Kobayashi et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.574/) show that attention-weight interpretation omits the norm of the transformed value and analyze `||A f(x)||` in BERT and NMT. This motivates M1's transformed-message magnitude. Our failure-prediction result—M1 improves M0 by only 0.00209—does not dispute their interpretability/alignment findings; it says that norm adds little for this detector and dataset.
- [Abnar and Zuidema (ACL 2020)](https://aclanthology.org/2020.acl-main.385/) define attention rollout and maximum-flow-style attention flow over a layerwise DAG, explicitly accounting for residual connections, and validate against blank-out and input-gradient importance. Our final-layer evidence-flow features are neither rollout nor their flow algorithm. The skipped temporal experiment means their cross-layer hypothesis is not directly tested here.
- [Chefer, Gur, and Wolf (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html) propagate class-specific Deep Taylor relevance through attention, residual, and normalization operations and evaluate explanation quality. Our predicted-versus-runner-up projection is a lightweight class-direction proxy; it is not Chefer relevance and is not a causal attribution.
- [Corbière et al. (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html) introduce TCP as a training target because the true-class probability is unavailable at inference, and learn it from internal representations across classification and segmentation experiments. Our `cls_tcp` follows that separation of training target from inference feature; it is weaker than current CLS/output baselines here.
- [Cattelan and Silva (UAI 2024)](https://proceedings.mlr.press/v244/cattelan24a.html) evaluate logit-only post-hoc confidence methods over 84 pretrained ImageNet classifiers and find p-norm logit normalization broadly effective, including under shift. We borrowed the validation-selected p-norm grid. It did not improve this ViT, whereas a nonlinear raw-logit MLP did, demonstrating that the best output baseline is model/data dependent.
- [Frasca et al. (ICLR 2026)](https://openreview.net/pdf?id=4twbqwV4br) represent LLM computational traces as attention/activation attributed graphs and report CHARM gains and cross-dataset transfer for hallucination detection. Polygraph tests the analogous design principle on frozen-ViT image error prediction. The completed matched controls show that CHARM's topology conclusion does not automatically transfer: raw-attention sets beat the GNN here.
- [Beigelman and Freiman (CVPR Workshops 2026)](https://openaccess.thecvf.com/content/CVPR2026W/HOW/papers/Beigelman_LogitDynamics_Reliable_ViT_Error_Detection_from_Layerwise_Logit_Trajectories_CVPRW_2026_paper.pdf) train lightweight intermediate class heads on a frozen ViT and probe predicted/top-competitor trajectories and ranking instability, reporting useful AUCPR and transfer. Our CLS-sequence GRU uses stored CLS states and is not a LogitDynamics reproduction because it lacks those supervised per-layer heads and explicit top-K dynamics.

## Work still required

- Final report assembly and final commit.

The rewiring, graph-statistics, SimpleMPNN, and temporal decisions are already fixed and reported above; they will not be relaunched.

## Strict weather confirmation — seed 7

The first strict weather replicate is complete. M5 was retrained using only the strict `base_val` half for early stopping (best epoch 4; stopped at epoch 12), then the already-fixed mixture gate was fit by five-fold base-image GroupKFold on the disjoint `meta_val` half. The untouched 14,232-record weather test was evaluated once.

| Method | All-test AUROC | Delta vs matched output |
|---|---:|---:|
| Strict raw-logit MLP, seed 7 | 0.86745 | — |
| Strict M5, seed 7 | 0.86754 | +0.00009 |
| Strict output + M5 gate, seed 7 | **0.87673** | **+0.00928** |

The gate's out-of-fold meta-validation AUROC was 0.87663. Its test AUPRC was 0.85761 and AURC was 0.21845. The result is encouraging but remains a single weather seed; seeds 1 and 2 are required before the holdout conclusion is fixed. The internal M5 detector alone ties the output expert on weather, while conditional combination recovers a larger improvement.

On the 4,306 held-out weather-plus-extra records, seed-7 AUROCs are 0.87989 for strict output, 0.87480 for M5, and **0.88581 for the gate**. Severity-pooled gate AUROCs for clean/0 through severity 5 are **0.86653 / 0.89518 / 0.88866 / 0.87883 / 0.86971 / 0.86293**. Thus this first weather replicate does not collapse on the unseen-family slice, although the standalone internal expert is weaker than output there.

Seed 1 also completed. Strict output, M5, and the fixed gate scored **0.86830 / 0.87105 / 0.87790** on all weather-test records, so the gate gains +0.00960 over its matched output expert. On held-out weather plus extras they scored **0.87935 / 0.88111 / 0.88817**. Gate AUROC by clean/0 through severity 5 is **0.87630 / 0.89656 / 0.89136 / 0.87942 / 0.86691 / 0.86716**. The all-test and held-out improvements have now replicated across the first two seeds; seed 2 remains.

Seed 2 completed at **0.87138 / 0.87350 / 0.88070** for strict output / M5 / gate. The three-seed weather summary is:

| Method | Seed 7 | Seed 1 | Seed 2 | Mean ± std | Held-out weather + extras mean ± std |
|---|---:|---:|---:|---:|---:|
| Strict raw-logit MLP | 0.86745 | 0.86830 | 0.87138 | 0.86904 ± 0.00169 | 0.88162 ± 0.00283 |
| Strict M5 | 0.86754 | 0.87105 | 0.87350 | 0.87070 ± 0.00245 | 0.87898 ± 0.00296 |
| Strict output + M5 gate | **0.87673** | **0.87790** | **0.88070** | **0.87844 ± 0.00167** | **0.88835 ± 0.00215** |

The gate improves its seed-matched output expert by +0.00928 / +0.00960 / +0.00933 (mean **+0.00940**) on all weather-test records. On the genuinely held-out weather-plus-extra slice it gains **+0.00674** over output and +0.00395 over MSP. It therefore does not collapse under the required family shift. Three-seed mean gate AUROC by clean/0 through severity 5 is **0.87310 / 0.89729 / 0.89070 / 0.88007 / 0.86934 / 0.86622**.

Combining this non-collapse result with the main-plan three-seed mean gain of +0.01186 and wholly positive base-image bootstrap intervals, the strict output+M5 mixture gate now meets the study's criteria for a **confirmed improvement over the strongest output-only baseline on the main benchmark**. The smaller weather gain is supportive generalization evidence, not a claim that its mean weather delta itself exceeds the separate 0.01 effect-size threshold.
