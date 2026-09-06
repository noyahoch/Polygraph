# Polygraph: Graph Necessity, Detector Architecture, and Last-Four-Layer Study

Completion date: 2026-09-06. This is the completed follow-up to the evidence-flow study.

## Executive answer

The best output-only reference remains the strict raw-100-logit MLP at **0.87941 ± 0.00093** main-test AUROC. The best standalone internal detector is the newly selected two-layer edge-gated mean evidence-flow MPNN at **0.88832 ± 0.00230**, a seed-matched mean gain of **+0.00891** over output alone. That is a reproducible possible small improvement, but it does not meet the predeclared +0.01 threshold for a confirmed standalone win. The validation-selected conditional output+edge-gated mixture reaches **0.89391 ± 0.00037** (seeds 1/2/7: 0.89388/0.89438/0.89346), gaining **+0.01449** over output and **+0.02881** over MSP (0.86510). Its group-bootstrap interval is above zero in every seed. On the different weather test it reaches **0.88199 ± 0.00169**, versus output 0.86904 and MSP 0.86252; on unseen weather plus extras it reaches **0.89119 ± 0.00199**, versus output 0.88162 and MSP 0.88441. Therefore the final result is a **confirmed improvement over the strongest output baseline**, driven by conditional combination of full-output evidence with hidden, class-conditioned internal evidence. It is not evidence that raw attention topology alone is necessary.

## 1. Scope and starting point

The frozen ViT and immutable rebuilt graph store contain 75,000 records. The current store—not historical numbers—is used for every direct comparison. Current MSP is 0.86510 on main, while the historical store gave about 0.8695. Current raw-attention M0 is 0.83477, 0.00693 below the historical 0.8417 and within the prespecified sanity tolerance. The rebuilt data therefore passed the anchor check, but old and new metrics must not be mixed.

The preceding study established that the full output vector matters: the strict raw-logit MLP used here obtains 0.87941. It also established M5: final-layer full hidden tokens plus attention, transformed message magnitude, and a predicted-versus-runner-up class-direction proxy. This follow-up asks whether explicit graph connectivity/message passing is necessary, whether another GNN operator improves M5, and whether blocks 8–11 add useful dynamics.

## 2. Leakage controls and data access

- ViT weights remained frozen. No corruption name/family, severity, source, true class, or correctness was supplied as an inference feature.
- Main strict detector plan: 52,000 train, 2,991 base-validation, 17,000 test. Main meta-validation: 3,009 records. The validation parts contain disjoint base photographs (498 and 499 groups).
- Weather strict detector plan: 36,238 train, 2,496 base-validation, 14,232 test; meta-validation has 2,362 records. Its validation parts contain 496 disjoint photograph groups each.
- Model family, width, depth, and last-four promotion were selected on base-validation only. The selection manifest was frozen before new test evaluation and records `test_metrics_consulted: false`.
- Gates were selected by mean out-of-fold meta-validation AUROC only. Test was scored after the gate choice was fixed.
- Every bootstrap resampled complete base-photograph groups.

Frozen selection-manifest SHA-256: `12fe759fe54ee2289ed82ad1256dc7a209b42984a3164e202e08011c4ff0c3b4`.

## 3. Environment and tests

Ubuntu Linux; NVIDIA RTX A5000 24 GiB; driver 535.161.08; PyTorch 2.5.1+cu121; PyG 2.7.0; Transformers 5.16.1; Python 3.11. The final run passed **75 tests** and `compileall`. Tests cover checkpoint compatibility, exact node/edge input matching, set/record-order invariance, deterministic rewiring, edge direction, sidecar alignment, last-four identity, group-disjoint splits, group bootstrap, test-blind selection, and ordered graph sequences.

## 4. Complete output baseline reference

| Method | Main AUROC | Weather AUROC | Setting |
|---|---:|---:|---|
| MSP | 0.86510 | 0.86252 | none |
| Margin | 0.86367 | 0.86173 | none |
| Entropy | 0.86517 | 0.86158 | none |
| Energy | 0.86120 | 0.85715 | analytic direction |
| Max logit | 0.86239 | 0.85876 | analytic direction |
| p-normalized maximum | 0.86396 | 0.86085 | p=2 |
| pNormSoftmax | 0.86406 | 0.86062 | p=2, T=4 |
| Raw-logit LR | 0.78322 | 0.78430 | class weighted |
| Sorted-centered LR | 0.86341 | 0.85974 | class weighted |
| Raw-logit MLP, ordinary plan | 0.87893 ± 0.00166 | 0.87069 ± 0.00064 | seeds 7/1/2 |
| Sorted-centered MLP | 0.86712 ± 0.00042 | 0.86166 ± 0.00154 | seeds 7/1/2 |
| **Raw-logit MLP, strict reference** | **0.87941 ± 0.00093** | **0.86904 ± 0.00169** | seeds 1/2/7 |

The nonlinear raw-logit model exceeds its sorted-centered analogue by about 0.012, whereas raw-logit linear regression fails. Useful output signal therefore includes nonlinear class-specific calibration/logit patterns, not only distribution shape. MSP is not the output-only ceiling.

## 5. TCP and earlier representation ablations

Final-CLS TCP regression did not beat the full-output detector. Graph TCP multitask scored 0.88342 on its error head versus 0.88625 for matching single-task M5; its TCP head scored 0.85869 and fixed combined head 0.87139. TCP was a training target only.

| ID | Representation | Main AUROC |
|---|---|---:|
| M0 | raw attention | 0.83477 |
| M1 | attention + transformed-message magnitude | 0.83685 |
| M2 | attention + predicted-vs-runner-up direction | 0.86602 |
| M3 | magnitude + direction | 0.86545 |
| M4 | M3 + compact class-conditioned nodes | 0.87725 |
| M5 | M3 + full hidden nodes | 0.88446 ± 0.00089 |

Magnitude adds only +0.00209 over M0. Class direction adds +0.03125. Combining magnitude with direction does not improve direction alone. Full hidden semantics add about +0.009 over compact evidence at seed 7. The class-direction feature is a proxy, not causal attribution: residuals, normalization, and the block MLP intervene.

## 6. Matched graph-necessity controls

All controls receive the same 784-dimensional per-token input available to M5. S1 also receives the full 36-dimensional edge-feature multiset but no incidence. S2 sees endpoint-local records plus an independent node branch, but has no iterative neighborhood aggregation.

| Model | Structural access | Params | Main AUROC (1/2/7) | Mean ± std |
|---|---|---:|---|---:|
| S0 HiddenTokenSet | hidden-token multiset; no edges | 75,138 | —/—/0.88384 | 0.88384 |
| S1 NodeEdgeSet h128 | separate node/edge multisets | 253,570 | 0.88236/0.88562/0.88219 | 0.88339 ± 0.00158 |
| S2 EndpointSet h96 | endpoint-local records | 313,058 | 0.88164/0.88168/0.88063 | 0.88132 ± 0.00049 |
| M5 TransformerConv | iterative attributed graph | 191,170 | 0.88507/0.88320/0.88512 | 0.88446 ± 0.00089 |

M5 exceeds S2 by only +0.00315 mean. Per-seed group-bootstrap M5−S2 intervals all cross zero. S0 and S1 are within 0.0011 of M5. Thus graph message passing is **not shown to be necessary** once full hidden-token information is available. The strict output+control gates score 0.88981 for S1 and 0.88894 for S2, below the prior output+M5 gate at 0.89128.

## 7. Rewiring and edge alignment

| Diagnostic, seed 7 | Validation AUROC | Test AUROC |
|---|---:|---:|
| M5 unperturbed | 0.87503 | 0.88512 |
| Train/evaluate target-permuted | 0.87415 | 0.88287 |
| Train/evaluate attribute-shuffled | 0.87644 | 0.87755 |
| Fixed M5, target permutation at inference | — | 0.88126 |
| Fixed M5, attribute shuffle at inference | — | 0.87920 |

Target permutation changes fixed-checkpoint M5 by −0.00386; edge-attribute shuffle changes it by −0.00591. Retraining recovers some loss. Connectivity/alignment affect the predictor modestly, but do not support a large or necessary higher-order-topology contribution.

## 8. GNN architecture and detector depth

| Family, fixed width/depth | Base-validation AUROC, seed 7 |
|---|---:|
| residual TransformerConv | 0.87626 |
| GINE | 0.87531 |
| GATv2 | 0.87585 |
| **edge-gated mean MPNN** | **0.88490** |

The edge-gated mean model uses an edge-conditioned sigmoid gate and target-wise mean aggregation, without TransformerConv's neighbor-normalized secondary attention. Its 131,970 parameters are fewer than M5's 191,170. Depth screening gave 0.87921/0.88490/0.87888 at 1/2/4 message-passing layers; two layers were fixed. A second-family depth grid was not launched because it was lower-value than the remaining representation experiments.

## 9. Last-four ViT blocks

The block-8–11 sidecar was captured for all 75,000 records in 471.5 seconds with zero prediction disagreements. On 512 validation records, block 11 reproduced the existing final hidden sidecar exactly (maximum absolute difference 0).

| Model, seed 7 | Information | Base-val | Test AUROC |
|---|---|---:|---:|
| L0 token trajectory | blocks 8–11, no graph | 0.86809 | 0.87706 |
| L1 union graph | attributed union over blocks 8–11 | 0.87212 | 0.88203 |
| Single-layer edge-gated | block 11 hidden/evidence flow | **0.88490** | **0.88942** |

L1 loses −0.00739 test AUROC to the single-layer model; paired group-bootstrap 95% interval [−0.01323,−0.00166]. L0 is worse. L1-SET was stopped after partial epoch 1 base-validation AUROC 0.85626 because the user explicitly requested skipping it once L1 was worse; it has no test result. L2 ordered graph sequence and L3 token-persistent graph were consequently not launched. The exact four-layer union-graph-versus-union-set necessity comparison remains unresolved.

## 10. Main benchmark: all degradations

| Method | AUROC | AUPRC | AURC ↓ | risk@.5 ↓ | risk@.8 ↓ | risk@.9 ↓ | confidence≥.9 AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSP | 0.86510 | 0.83451 | 0.22180 | 0.21353 | 0.40794 | 0.45536 | — |
| Strict output MLP | 0.87941 | 0.85834 | 0.21578 | 0.19933 | 0.40022 | 0.45248 | 0.85927 |
| M5 TransformerConv | 0.88446 | 0.86307 | 0.21212 | 0.19749 | 0.39975 | 0.45261 | 0.86854 |
| Edge-gated evidence flow | 0.88832 | 0.86815 | 0.21047 | 0.19267 | 0.39841 | 0.45179 | 0.87265 |
| **Output + edge-gated gate** | **0.89391** | **0.87484** | **0.20711** | **0.18725** | **0.39647** | **0.45109** | **0.87932** |

Standalone seeds are 0.88512/0.89043/0.88942 for 1/2/7. Gate seeds are 0.89388/0.89438/0.89346. Seed-matched gate-minus-output deltas are +0.01382/+0.01428/+0.01537. The gate was selected by mean OOF meta-validation AUROC (0.87815), ahead of M5 (0.87599), S1 (0.87498), S2 (0.87495), and L1 (0.87406).

## 11. Main benchmark by degradation severity

These are pooled by **severity**, not corruption type.

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.89276 | 0.89036 | 0.87270 | 0.85367 | 0.81841 |
| Strict output MLP | 0.89673 | 0.89767 | 0.88036 | 0.87430 | 0.85367 |
| M5 | 0.89881 | 0.90125 | 0.88367 | 0.88269 | 0.86138 |
| Edge-gated evidence flow | 0.89779 | 0.90473 | 0.89053 | 0.88573 | 0.86728 |
| **Output + edge-gated gate** | **0.90653** | **0.91148** | **0.89569** | **0.89065** | **0.87073** |

The final gate is strongest at every severity. Its gain is not produced by one severity bucket, although absolute detection becomes harder at severity 5.

## 12. Weather-family holdout

Main and weather absolute AUROCs are not directly comparable because the test sets differ.

| Method | All test AUROC | Unseen weather+extras | confidence≥.9 |
|---|---:|---:|---:|
| MSP | 0.86252 | 0.88441 | — |
| Strict output MLP | 0.86904 ± 0.00169 | 0.88162 ± 0.00283 | 0.83966 |
| M5 | 0.87070 ± 0.00245 | 0.87898 ± 0.00296 | 0.84385 |
| Edge-gated evidence flow | 0.87636 ± 0.00209 | 0.88403 ± 0.00299 | 0.84812 |
| **Output + edge-gated gate** | **0.88199 ± 0.00169** | **0.89119 ± 0.00199** | **0.85659** |

Weather standalone seeds: 0.87907/0.87602/0.87398; gate: 0.88391/0.88226/0.87980 (1/2/7). The gate improves all-test output by +0.01295 and unseen weather+extras by +0.00957; it exceeds MSP on the unseen slice by +0.00678.

### Weather AUROC by degradation severity

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.89732 | 0.88001 | 0.87306 | 0.84349 | 0.82336 |
| Strict output MLP | 0.89083 | 0.88290 | 0.87217 | 0.85507 | 0.85408 |
| M5 | 0.88745 | 0.88212 | 0.87104 | 0.86427 | 0.85913 |
| Edge-gated, seed 7 | 0.89123 | 0.88204 | 0.87473 | 0.86566 | 0.86538 |
| **Output + edge-gated, seed 7** | **0.89876** | **0.88946** | **0.88165** | **0.87032** | **0.86869** |

Full per-seed severity records are in `severity_metrics.json`.

## 13. Complementarity

At a 50% flag budget, standalone edge-gated evidence flow recovers **45.4%** of MSP-blind and **37.4%** of output-MLP-blind errors. Its mean Spearman correlations are 0.829 with MSP and 0.884 with output. The gate recovers 40.6%/19.8% and correlates 0.870/0.973. The gate stays close to the strong output ranking while making a smaller set of useful internal corrections. Exact caught/missed/overlap counts are in `model_comparison.csv`.

On weather, standalone recovers 43.9%/35.1% of MSP/output blind errors; the gate recovers 40.9%/24.3%, with output correlation 0.952.

## 14. Statistical uncertainty

All intervals use 2,000 paired base-photo-group bootstrap repetitions.

- Edge-gated minus output, seeds 1/2/7: +0.00515 [−0.00111,+0.01126], +0.01032 [+0.00493,+0.01589], +0.01136 [+0.00552,+0.01714]. Direction is positive in all seeds, but mean observed delta is +0.00891 and one interval crosses zero.
- Edge-gated minus M5: +0.00014 [−0.00547,+0.00579], +0.00725 [+0.00222,+0.01280], +0.00439 [−0.00111,+0.01017].
- Final gate minus output: +0.01386 [+0.01060,+0.01706], +0.01427 [+0.01154,+0.01706], +0.01538 [+0.01273,+0.01807]. Every bootstrap delta is positive.
- Final gate minus MSP is approximately +0.02880/+0.02930/+0.02840; all intervals are wholly positive.
- L1 minus single-layer edge-gated: −0.00735 [−0.01323,−0.00166].

Holm adjustment over predeclared primary comparisons preserves the decisive output/MSP and final-gate results. The standalone architecture effect does not meet the +0.01 mean convention.

## 15. Negative, failed, and skipped experiments

- Raw attention and message magnitude alone remain weak.
- S0/S1/S2 nearly match M5; a large necessity claim for explicit connectivity is unsupported.
- Rewiring and attribute shuffling hurt only modestly.
- Residual TransformerConv, GINE, and GATv2 lost to edge-gated mean on validation.
- One and four detector layers lost to two; L0 and L1 lost to the selected single-layer detector.
- L1-SET: user-directed skip after L1 underperformed; partial epoch 1 retained, no test.
- L2/L3: not launched after L0/L1 losses and the request to stop unnecessary four-layer controls.
- Second-family depth screen: skipped to avoid a low-information architecture grid.
- Inference profiling: skipped because the user explicitly said timing measurement was unnecessary and prioritized completion. Recorded training runtimes and parameters remain available.
- Recoverable failed attempts: two S1 h128 OOM kills, interrupted low-batch L1 superseded by completed batch-96, and weather S2 timeout superseded by its completed resume. All remain in the ledger.
- Temporal graph: not run because final-four L1 already lost and temporal expansion was lower priority. No all-layer value/message extraction or new dense attention was run.

## 16. What explains the best result?

Nonlinear full-output calibration is a strong baseline. Hidden token semantics contain nearly all standalone M5 signal even without edges (S0). Predicted-class-conditioned evidence flow is useful in earlier controlled ablations, while message magnitude alone is not. A simple edge-conditioned mean architecture extracts somewhat more signal than TransformerConv, and a conditional gate converts that complementary signal into a consistent improvement over output alone.

The result is representation- and combination-specific, not proof that raw attention connectivity or high-order paths are causal. The defensible summary is: **full hidden/internal evidence is complementary to the complete logit vector; an edge-gated evidence-flow expert is a useful carrier, but explicit graph message passing is not established as necessary.**

## 17. Scientific claims

### Supported

- MSP is not the output ceiling.
- Internal hidden/class-conditioned evidence adds reproducible information beyond all 100 logits when used by the fixed gate.
- The gate is a confirmed improvement over output on main and does not collapse on unseen weather plus extras.
- Edge-gated mean outperforms tested TransformerConv/GINE/GATv2 variants under fixed validation selection.
- Tested last-four L0/L1 forms do not improve the selected final-block detector.

### Not supported

- Raw attention topology is necessary or superior to matched controls.
- M5 materially beats full-information set/endpoint controls.
- Message magnitude is a major standalone signal.
- Four-layer dynamics improve error prediction in tested forms.
- The class-direction proxy is causal attribution or Chefer relevance.

### Still uncertain

- Whether completed L1-SET would match L1; it was explicitly skipped.
- Whether an efficient token-persistent temporal representation helps.
- Transfer to a different frozen ViT and independent corruption universe.

Methodological context: [Kobayashi et al.](https://aclanthology.org/2020.emnlp-main.574/) motivate transformed-value norms; [Abnar and Zuidema](https://aclanthology.org/2020.acl-main.385/) study cross-layer attention flow; [Chefer et al.](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html) propagate class relevance through operations omitted by our proxy; [Corbière et al.](https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html) motivate TCP; [Cattelan and Silva](https://proceedings.mlr.press/v244/cattelan24a.html) motivate p-norm controls; and [Frasca et al.](https://openreview.net/pdf?id=4twbqwV4br) motivate message passing over attributed attention graphs. These supply design principles, not evidence for Polygraph's conclusions.

## 18. Recommended next steps

1. Distill the edge-gated internal score into cheap token-set gate context. Storage <1 GB; medium runtime. Stop if strict AUROC loses >0.005 or output-blind recovery drops materially.
2. Repeat the fixed two-expert protocol on one independently trained ViT. Storage/runtime large. Stop broad extraction if a grouped pilot finds no internal complementarity.
3. Only if topology remains central, complete one efficient matched token-persistent graph versus set control. Storage moderate; runtime medium/high. Stop topology work if the set is within grouped uncertainty.

## 19. Reproduction commands

```bash
.venv/bin/python scripts/run_topology_depth_last4.py --stage controls --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage rewiring --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage architectures --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage depth --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage last4_capture --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage multilayer --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage confirm --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage evaluate --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/run_topology_depth_last4.py --stage gates --run-root runs/topology_depth_last4_20260905 --resume
.venv/bin/python scripts/finalize_topology_depth_last4.py
.venv/bin/python -m pytest -q tests/test_polygraph.py tests/test_deepsets.py tests/test_research_controls.py tests/test_sidecars.py tests/test_research_eval.py tests/test_topology_depth_last4.py
```

Every actual invocation, retry, exit status, and runtime is in `ledger.jsonl` and `experiment_table.csv`.

## 20. Artifact map

- Selection: `runs/topology_depth_last4_20260905/final/selection_manifest.json`
- Summary: `runs/topology_depth_last4_20260905/final/summary.json`
- Experiment/model tables: `runs/topology_depth_last4_20260905/final/{experiment_table.csv,model_comparison.csv}`
- Severity 1–5: `runs/topology_depth_last4_20260905/final/severity_metrics.json`
- Bootstrap/Holm: `runs/topology_depth_last4_20260905/final/paired_bootstrap.json`
- Best scores: `runs/topology_depth_last4_20260905/final/best_scores_{main,weather}.npz`
- Models/scores: `runs/topology_depth_last4_20260905/{controls,architectures,multilayer,gates}/`
- Last-four sidecar: `data/graph_dataset/sidecars/hidden_last4_missing/manifest.json`
- Rewiring cache: `data/graph_dataset/sidecars/rewire_target_l11/manifest.json`
- Earlier evidence-flow artifacts: `runs/research_20260830/`

Large stores, sidecars, checkpoints, arrays, and logs remain gitignored.

## 21. Final git state

Research branch: `research/topology-depth-last4-20260905`. Follow-up began at `eb1da9b`. Focused implementation commits preceding this report: `1aba7bb`, `55dbd4c`, `00710d7`, `d948b46`, `9a20bfe`, `cf60989`. The final report/finalizer commit and exact status are in the handoff because a commit cannot include its own hash.
