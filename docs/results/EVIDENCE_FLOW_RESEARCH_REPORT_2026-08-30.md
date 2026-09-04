# POLYGRAPH: Frozen-ViT Evidence-Flow Error Prediction — Final Research Report

Completed 2026-09-04. The filename retains the study's pre-registered 2026-08-30 identifier. This report supersedes the earlier missing-artifact blocker report and uses only the rebuilt 75,000-record store for direct comparisons.

## EXECUTIVE ANSWER

Yes: a leakage-controlled conditional mixture gate **confirmedly improves over the strongest full-output baseline on the main plan**. The strongest output-only method is a raw 100-logit MLP: 0.88010/0.87659/0.88009 AUROC (mean **0.87893 ± 0.00166**) under the ordinary plan, and 0.87809/0.88006/0.88009 (**0.87941 ± 0.00093**) when retrained on the strict base-validation split. The best standalone internal method is M5—full final-layer token hidden states with attention, message magnitude, and predicted-vs-runner-up direction—at 0.88625/0.88600/0.88496 (**0.88574 ± 0.00056**). The strictly confirmed output+M5 gate reaches 0.89112/0.89321/0.88952 (**0.89128 ± 0.00151**), a seed-matched mean gain of **+0.01186** over strict output and +0.02618 over MSP (0.86510). Every main group-bootstrap interval is above zero. On the different weather test, the strict gate scores 0.87673/0.87790/0.88070 (**0.87844 ± 0.00167**) versus output 0.86904 mean; on unseen weather plus extras it scores **0.88835** versus output 0.88162 and MSP 0.88441. The gain is not output-only or raw-topology-specific: it comes from conditionally combining a strong full-logit expert with complementary hidden/evidence-flow information.

## 1. RESEARCH QUESTION

The original hypothesis was that directed connectivity in a frozen ViT's attention graph predicts classifier errors beyond a flat multiset of attention values. The stronger hypothesis was that attention weights omit what is sent: transformed-value magnitude, predicted-class direction, hidden semantics, and layer dynamics may be more informative, and a conditional model may learn when output confidence should not be trusted. The ViT remained frozen throughout. No corruption identity, family, severity, source, correctness, or true class was used as an inference feature; true class was used only for TCP training targets.

## 2. STARTING POINT

Historical anchors were MSP 0.8695, raw graph 0.8417, hidden fusion 0.8758, CLS sequence 0.8759, top-100 graph 0.8125, and top-100 flat attention MLP 0.7421; historical weather values were MSP 0.8885 unseen, fusion 0.8738, CLS sequence 0.8731, and graph 0.8380. They are provenance checks, not direct comparators.

On the rebuilt store, MSP is **0.86510**, 0.00440 below the historical main value. M0 raw graph is **0.83477**, 0.00693 below the historical raw graph and inside the predeclared 0.01–0.015 sanity tolerance. H0 full-hidden/raw-attention is 0.87537, close to the historical fusion anchor. Current CLS sequence is 0.88390 mean. These differences make it essential not to mix historical and rebuilt test metrics.

## 3. ENVIRONMENT

- Ubuntu 20.04.6 LTS, Linux 5.15; Intel i7-13700K, 31 GiB RAM.
- NVIDIA RTX A5000, 24,564 MiB VRAM; driver 535.161.08. `nvidia-smi` reports CUDA compatibility 12.2. `nvcc` was absent and was not needed.
- Python 3.11 environment at `.venv`; PyTorch 2.5.1+cu121, torch CUDA runtime 12.1, torch-geometric 2.7.0, transformers 5.16.1, scikit-learn 1.9.0, NumPy 2.4.6.
- Installation used `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`, followed by `torch-geometric>=2.5,<2.8`, repository requirements excluding torch packages, pytest, pandas, scipy, matplotlib, and psutil.
- CUDA matrix multiplication, PyG TransformerConv forward/backward, and frozen-ViT smoke tests passed. Starting repository commit: `8036fd902dca7e2b37a681fd8fa73113603f1676` on `main`. The final research branch and commits are in Section 24.

The full ledger attributes about 16.7 GPU-hours to completed neural-model commands across the complete multi-day study, excluding some CPU-only work. This exceeded the initial eight-hour target because the user subsequently authorized completing all remaining experiments and strict confirmation; every >90-minute candidate was still aborted or skipped according to the per-run policy.

## 4. DATA AND SPLITS

The store contains 75,000 records from `edumunozsala/vit_base-224-in21k-ft-cifar100`, threshold tau 0.02, 197 tokens, 12 heads, and 12 layers. Main splits contain 52,000/6,000/17,000 train/validation/test records and hold out gaussian blur, saturate, spatter, and speckle noise. Weather splits contain 36,284/4,646/14,232 records and additionally hold out brightness, fog, frost, and snow; 4,306 weather-test records are unseen weather-plus-extra sources. Test sets are exactly balanced between 8,500/8,500 and 7,116/7,116 correct/error records. All split overlap checks are by base photograph and pass.

Aligned sidecars are complete for 100 logits, final-layer hidden states, compact four-dimensional class evidence, final-block value/message statistics, and 161 graph statistics. Each contains 75,000 records, shard counts, model ID, store-key SHA-256 `3c5d2c98055553fb2e2a316411d314ddd4db8c9c1f6ce1e74b8870cc2539d96a`, schema, dtype, and provenance. Strict output/M5/gate score registries align exactly for every seed: main 17,000-record store-index digest starts `caed4d`; weather 14,232-record digest starts `c6d4a9`.

## 5. CODE AUDIT FINDINGS

The historical flat attention control received only edge values, while the GNN also received coordinates, CLS identity, layer position, self-attention diagonals, and incidence. Matched NodeEdgeSet and EndpointSet controls were added. The prior multi-layer SequenceConcat model processes layer graphs independently and concatenates readouts; it is not a temporal graph. Sidecars are keyed by immutable store order and validated at load time. `store_index` required a PyG batching override because ordinary `Data` treats fields containing `index` as incrementable; new aligned scores are protected by tests. Evaluation was made bounded and seed-selectable, and old checkpoints still load through defaulted configuration fields.

## 6. EXPERIMENT LEDGER

The exhaustive 181-record ledger, including environment failures, completed extraction, failed retries, aborts, and skips, is exported as `runs/research_20260830/final/experiment_table.csv`. Scientific runs are summarized here.

| IDs | Configuration | Seeds | Status / outcome |
|---|---|---:|---|
| O1–O11 | output-only suite, main + weather | deterministic or 7/1/2 | completed |
| CLS/TCP | CLS MLP, CLS GRU, final-CLS TCP | 7/1/2 | completed both plans |
| C1–C3 | EdgeSet, NodeEdgeSet, EndpointSet | 7 | completed |
| C5 | deterministic target rewiring | 7 | aborted at 91.98 min before selection |
| C6 | deterministic attribute shuffle | 7 | skipped from matched >90-min cost |
| D1–D3 | compact evidence graph/set controls | 7 | completed |
| M0–M4 | attention/message/evidence ablation | 7 | completed |
| M5 | evidence flow + hidden | 7/1/2 | completed |
| H0 | hidden + raw attention | 7 | completed |
| F0 | SimpleMPNN + attention | 7 | aborted at 92.4 min |
| F1/F2/T1 | heavier SimpleMPNN/temporal variants | 7 | skipped from measured projection |
| G1–G4 | exploratory combiners | fixed seed | completed |
| strict M5/gate | group-disjoint confirmation | 7/1/2 | completed main + weather |
| graph TCP multitask | M5 plus 0.2 TCP loss | 7 | completed; failed promotion |
| CLS-sequence TCP | 12-layer GRU TCP pilot | 7 | completed; failed promotion |

## 7. OUTPUT BASELINES

| Method | Main AUROC | Weather AUROC | Validation-selected setting |
|---|---:|---:|---|
| MSP | 0.86510 | 0.86252 | none |
| Margin | 0.86367 | 0.86173 | none |
| Entropy | 0.86517 | 0.86158 | none |
| Energy | 0.86120 | 0.85715 | analytic error direction |
| Max logit | 0.86239 | 0.85876 | analytic error direction |
| p-normalized maximum | 0.86396 | 0.86085 | p=2 both plans |
| pNormSoftmax | 0.86406 | 0.86062 | p=2, T=4 both plans |
| Raw-logit LR | 0.78322 | 0.78430 | class weighted |
| Sorted-centered LR | 0.86341 | 0.85974 | class weighted |
| Raw-logit MLP | **0.87893 ± 0.00166** | **0.87069 ± 0.00064** | seeds 7/1/2 |
| Sorted-centered MLP | 0.86712 ± 0.00042 | 0.86166 ± 0.00154 | seeds 7/1/2 |

MSP is not the output ceiling. The 0.01181 main gap between raw and sorted-centered MLPs indicates that class-specific logit patterns/calibration contribute beyond distribution shape. The failure of raw-logit LR and success of its MLP show that this information is nonlinear. Seed-7 raw-logit MLP minus MSP has grouped-bootstrap mean +0.01501, 95% CI [+0.00857,+0.02183].

## 8. STRUCTURE CONTROLS

| Model | Inputs / structural access | Params | AUROC |
|---|---|---:|---:|
| C1 EdgeSet | edge-value multiset only | 13,313 | **0.84736** |
| C2 NodeEdgeSet | independent node and edge multisets | 15,457 | 0.84556 |
| C3 EndpointSet | endpoint-local records, no message passing | 9,217 | 0.83754 |
| M0 TransformerConv | full raw-attention graph | 16,066 | 0.83477 |

All completed set controls match or beat M0. Therefore raw-attention connectivity is not shown to add information beyond node/edge content. C5 target permutation had worse best validation AUROC (0.80395 versus M0 0.82186) but exceeded the runtime limit before test selection; it is suggestive only. C6 was not launched because it shares the measured expensive permutation path. The strongest valid claim is negative: the earlier topology claim does not survive matched controls.

## 9. GRAPH-STATISTICS RESULTS

The aligned 161-feature extractor processed all records in 356.5 seconds without ViT inference. Main AUROCs are LR 0.82751, HGB 0.84811, and MLP 0.85372/0.85294/0.85097 (**0.85254 ± 0.00116**). Weather MLP is 0.84805 ± 0.00144 and 0.86008 on held-out weather plus extras. Global attention statistics contain signal but are below MSP and the output MLP; adding them in G4 did not improve the simpler G1 gate.

## 10. COMPACT CLASS-EVIDENCE RESULTS

| Model | Main AUROC |
|---|---:|
| M0 raw graph | 0.83477 |
| D1 compact evidence TransformerConv | 0.86938 |
| D2 compact NodeEdgeSet | 0.87360 |
| D3 compact EndpointSet | **0.87588** |
| M4 compact evidence + evidence-flow graph | 0.87725 |
| H0 full hidden + raw attention | 0.87537 |
| M5 full hidden + evidence flow | 0.88625 seed 7 |

Predicted-class/runner-up token evidence helps substantially (+0.03461 for D1 versus M0), but D2/D3 beat D1. Its benefit is endpoint-local/distributional, not evidence for higher-order topology. Full hidden semantics add +0.00900 to M4 at seed 7. Conversely, evidence-flow edges add +0.01088 to H0 when hidden nodes are held fixed.

## 11. MESSAGE-FLOW RESULTS

| ID | Edge/node representation | AUROC | Delta vs M0 |
|---|---|---:|---:|
| M0 | attention A | 0.83477 | — |
| M1 | A + log1p(A·transformed-value norm) | 0.83685 | +0.00209 |
| M2 | A + asinh(A·class-direction proxy) | 0.86602 | +0.03125 |
| M3 | A + magnitude + direction | 0.86545 | +0.03068 |
| M4 | M3 + compact node evidence | 0.87725 | +0.04249 |
| M5 | M3 + full hidden nodes | **0.88574 ± 0.00056** | +0.05097 |

Message magnitude alone does not materially help. Predicted-versus-runner-up direction is the decisive edge-level signal. Magnitude and direction are not complementary in M3. The class-direction quantity is a lightweight projection of transformed values toward the final classifier competition, not exact causal contribution: residuals, normalization, and the block MLP intervene.

## 12. ARCHITECTURE RESULTS

F0 SimpleMPNN, which removes TransformerConv's learned secondary attention, reached epoch 40 in 92.4 minutes without early stopping and was aborted; best validation AUROC was about 0.8267. F1/F2 were projected slower because their edge vectors are larger. No selected test result exists, so secondary learned attention remains unresolved. T1 four-layer temporal MPNN was skipped: the one-layer architecture already exceeded the 90-minute limit, making four graph copies clearly unaffordable. Existing CLS-sequence results do not test temporal graph connectivity.

## 13. CONDITIONAL COMBINATION

Exploratory G1 output+M5 gate scored 0.89392; G2 output+CLS-sequence gate 0.88996; G3 output+raw graph+CLS sequence selected logistic at 0.88899; G4 output+M5+TCP+graph statistics selected logistic at 0.89387. Thus extra weak inputs did not improve the two-expert gate.

Strict confirmation split original validation by base photograph into base_val/meta_val (main 2,991/3,009 records from 498/499 groups; weather 2,496/2,362 from 496/496). Detectors used base_val only. The fixed mixture gate was fit with five-fold GroupKFold only on meta_val and test was evaluated once. Mean gate weight was higher on errors than correct predictions, but this is descriptive, not causal.

## 14. MAIN BENCHMARK

Three-seed means are shown for strict finalists.

| Method | AUROC | AUPRC | AURC ↓ | risk@.5 ↓ | risk@.8 ↓ | risk@.9 ↓ | confident AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSP | 0.86510 | 0.83451 | 0.22180 | 0.21353 | 0.40794 | 0.45536 | — |
| Strict raw-logit MLP | 0.87941 | 0.85834 | 0.21578 | 0.19933 | 0.40022 | 0.45248 | 0.85927 |
| Strict M5 | 0.88446 | 0.86307 | 0.21212 | 0.19749 | 0.39975 | 0.45261 | 0.86854 |
| Strict gate | **0.89128** | **0.87197** | **0.20854** | **0.18922** | **0.39752** | **0.45133** | **0.87610** |

Strict gate AUROCs are 0.89112/0.89321/0.88952. Strict output is 0.87809/0.88006/0.88009. The seed-matched deltas are +0.01302/+0.01315/+0.00942.

Per degradation severity (pooled by severity, not corruption type):

| Method | Clean/0 | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.92058 | 0.89276 | 0.89036 | 0.87270 | 0.85367 | 0.81841 |
| Strict output | 0.90974 | 0.89673 | 0.89767 | 0.88036 | 0.87430 | 0.85367 |
| Strict M5 | 0.90418 | 0.89881 | 0.90125 | 0.88367 | 0.88269 | 0.86138 |
| Strict gate | **0.91878** | **0.90630** | **0.90911** | **0.89196** | **0.88792** | **0.86754** |

## 15. WEATHER-FAMILY HOLDOUT

Main and weather absolute AUROCs are not directly comparable because the test sets differ.

| Method | All AUROC | Held-out weather+extras AUROC | Confident AUROC |
|---|---:|---:|---:|
| MSP | 0.86252 | 0.88441 | — |
| Strict output | 0.86904 ± 0.00169 | 0.88162 ± 0.00283 | 0.83966 |
| Strict M5 | 0.87070 ± 0.00245 | 0.87898 ± 0.00296 | 0.84385 |
| Strict gate | **0.87844 ± 0.00167** | **0.88835 ± 0.00215** | **0.85370** |
| Existing CLS sequence | 0.87201 ± 0.00110 | 0.87438 ± 0.00414 | — |

Weather gate seeds are 0.87673/0.87790/0.88070. All-test gain over matched output is +0.00940; held-out-family gain is +0.00674. Weather group-bootstrap intervals for gate minus output are [+0.00463,+0.01382], [+0.00588,+0.01363], and [+0.00617,+0.01250].

| Method | Clean/0 | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.89537 | 0.89732 | 0.88001 | 0.87306 | 0.84349 | 0.82336 |
| Strict output | 0.87507 | 0.89083 | 0.88290 | 0.87217 | 0.85507 | 0.85408 |
| Strict M5 | 0.85734 | 0.88745 | 0.88212 | 0.87104 | 0.86427 | 0.85913 |
| Strict gate | 0.87310 | **0.89729** | **0.89070** | **0.88007** | **0.86934** | **0.86622** |

## 16. COMPLEMENTARITY

At a 50% flag budget on main, strict M5 catches 6,821 errors on average, recovers 45.9% of MSP blind errors and 36.7% of output blind errors, and has Spearman correlations 0.827/0.886 with MSP/output. The gate catches 6,892 errors, recovers 41.4% of MSP blind errors and 19.0% of output blind errors, with correlations 0.868/0.976. The gate deliberately stays close to the stronger output ranking while using M5 to correct a smaller, valuable subset.

On weather, M5 recovers 43.9%/32.8% of MSP/output blind errors; the gate recovers 41.1%/21.9%. The gate catches about 5,707 of 7,116 errors at 50% budget versus 5,637 for M5. Complete per-seed caught/missed/overlap counts and correlations are in `summary.json` and `model_comparison.csv`.

## 17. STATISTICAL UNCERTAINTY

All bootstrap resampling uses complete base-photograph groups, 2,000 repetitions. Main gate-minus-output mean deltas and 95% intervals are +0.01301 [+0.01006,+0.01604], +0.01315 [+0.01014,+0.01609], and +0.00938 [+0.00742,+0.01146]; every sampled delta is positive. Gate-minus-MSP intervals are also wholly positive. Strict M5-minus-output is smaller: +0.00702 [+0.00122,+0.01282], +0.00502 [−0.00081,+0.01136], +0.00307 [−0.00265,+0.00867]. Standalone M5 is therefore a possible small improvement, while the strict gate satisfies the defined confirmed-improvement criteria.

Weather gate-minus-output intervals are wholly positive in all seeds, but the mean effect is +0.00940, just below the study's separate 0.01 effect-size convention. This supports non-collapse without overstating the weather effect.

## 18. NEGATIVE RESULTS

- Raw attention M0 is below MSP; magnitude M1 adds only +0.00209.
- M3 does not beat direction-only M2; magnitude and direction are not complementary here.
- EdgeSet and NodeEdgeSet beat raw graph M0; matched controls invalidate the strong raw-topology claim.
- Compact EndpointSet beats compact TransformerConv D1.
- Graph statistics, p-normalized logits, energy, max logit, raw-logit LR, TCP, and CLS-sequence TCP are not competitive.
- Graph TCP multitask: error head 0.88342 versus single-task M5 0.88625; TCP head 0.85869; fixed average 0.87139.
- Adding raw graph, TCP, and graph statistics to exploratory combiners does not beat the two-expert output+M5 gate.
- SimpleMPNN did not reach a selected checkpoint within the runtime limit.

Runtime skips and recoverable failures are explicit. C5 target rewiring and F0 SimpleMPNN were aborted at 5,518.8 and 5,542.9 seconds without test selection. C6 attribute shuffle was skipped because it uses C5's identical per-record permutation path; F1/F2 were skipped because their larger edge records cannot be cheaper than F0; T1 was skipped because it expands the already-over-budget F0 graph fourfold, and T2 was consequently not triggered. Current-store C4 top-100 was not reproduced because the continuation protocol prioritized M0-matched controls and prohibited rerunning established experiments; 0.8125 remains historical only. CLS-sequence TCP seeds 1/2 and weather were skipped after seed 7 scored 0.86474; M5-TCP confirmation was skipped after its seed-7 error head lost to M5. The first non-strict weather M5 process was terminated when an unrelated job occupied 15.6 GB of VRAM and made its projection exceed the cutoff; it was superseded by the stricter three-seed weather run. Initial missing-store skips and failed high-batch extraction attempts were superseded by the successfully rebuilt, validated store and resumable sidecars. Exact commands, timings, exit codes, and reasons for every instance remain in the 181-row ledger.

## 19. WHAT ACTUALLY EXPLAINS THE BEST RESULT

First, the nonlinear full output distribution is important: raw-logit MLP beats MSP by about 0.014 and sorted-centered MLP by about 0.012, implicating nonlinear, class-specific logit calibration. Second, internal semantics are complementary: M5 consistently exceeds output alone and recovers about 37% of output-blind errors. Third, predicted-class direction matters far more than magnitude (M2 versus M1), and evidence-flow edges improve a hidden-node graph by +0.01088 (M5 versus H0). Raw connectivity itself is not the explanation because matched set controls beat M0 and compact endpoint records beat D1. The best result is therefore a conditional combination of full-output evidence with hidden, predicted-class-conditioned internal evidence—not raw attention topology alone.

## 20. SCIENTIFIC CLAIMS WE CAN AND CANNOT MAKE

### SUPPORTED

- MSP is not the output-only ceiling on this rebuilt dataset.
- Predicted-class-conditioned internal representation contains reproducible complementary information beyond all 100 logits.
- A fixed, group-disjoint mixture gate converts that information into a confirmed main-plan gain and does not collapse on unseen weather families.
- Message magnitude alone is weak; predicted-vs-runner-up direction is useful.
- Full hidden semantics and evidence-flow edges each contribute in controlled seed-7 ablations.

### NOT SUPPORTED

- Raw attention connectivity outperforms matched value controls.
- Message magnitude and class direction are complementary.
- TCP supervision improves the final graph.
- Current SimpleMPNN or temporal graphs outperform TransformerConv.
- The class-direction proxy is exact causal attribution or Chefer relevance.

### STILL UNCERTAIN

- Whether higher-order topology helps once endpoint controls receive full hidden and evidence-flow features.
- Whether explicit MPNN or true temporal graphs help under a more efficient sparse implementation.
- How much of M5's internal gain transfers beyond this ViT and corruption universe.

Methodological grounding: [Kobayashi et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.574/) motivate transformed-value norms; [Abnar and Zuidema (ACL 2020)](https://aclanthology.org/2020.acl-main.385/) establish cross-layer attention rollout/flow, not tested by our final-layer proxy; [Chefer et al. (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html) propagate class-specific relevance through operations our proxy omits; [Corbière et al. (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html) motivate TCP as a training-only target; [Cattelan and Silva (UAI 2024)](https://proceedings.mlr.press/v244/cattelan24a.html) motivate p-norm confidence baselines; [Frasca et al. (ICLR 2026)](https://openreview.net/pdf?id=4twbqwV4br) motivate message passing on attributed attention graphs; and [Beigelman and Freiman (CVPRW 2026)](https://openaccess.thecvf.com/content/CVPR2026W/HOW/papers/Beigelman_LogitDynamics_Reliable_ViT_Error_Detection_from_Layerwise_Logit_Trajectories_CVPRW_2026_paper.pdf) motivate layerwise ViT dynamics. Borrowing a design principle is not evidence that their claims automatically transfer to Polygraph.

## 21. RECOMMENDED NEXT STEP

1. Train a matched full-hidden/evidence-flow EndpointSet. Reason: it directly isolates iterative topology at M5's information level. Storage: none. Runtime: medium, likely 45–70 minutes. Stop topology work if it matches M5 within grouped uncertainty.
2. Distill M5 into compact gate context. Reason: the gate is the actual win, while M5 inference over dense stored graphs is expensive. Storage: under 1 GB for graph embeddings. Runtime: medium. Stop if strict AUROC loses more than 0.005 or output-blind recovery drops materially.
3. Repeat the fixed strict gate on one independently trained ViT architecture. Reason: this tests model-level transfer rather than more capacity. Storage/runtime: large because new aligned logits/internal features are required. Stop broad extraction if a 10% grouped pilot shows no internal complementarity.

## 22. REPRODUCTION COMMANDS

From the repository root with `.venv/bin/python`:

```bash
.venv/bin/python scripts/inventory_polygraph.py --store-dir data/graph_dataset/store --output runs/research_20260830/inventory/inventory.json
.venv/bin/python scripts/verify_current_anchors.py
.venv/bin/python scripts/evaluate_output_baselines.py --plan data/graph_dataset/split_plan_main.json --store-dir data/graph_dataset/store --logits-dir data/graph_dataset/sidecars/logits --out-dir runs/research_20260830/output/main
.venv/bin/python -m polygraph.training train --plan data/graph_dataset/split_plan_main.json --out-dir runs/research_20260830/message_flow/M5_evidence_hidden --layers last --architecture transformerconv --node-features hidden --edge-features evidence_flow --message-stats-dir data/graph_dataset/sidecars/message_stats_l11 --batch-size 64 --seeds 7 1 2 --epochs 60 --patience 8 --min-delta 0.002 --epochs-per-process 0
.venv/bin/python -m polygraph.training evaluate --run-dir runs/research_20260830/message_flow/M5_evidence_hidden --plan data/graph_dataset/split_plan_main.json --no-baselines
.venv/bin/python scripts/create_strict_meta_plans.py --help
.venv/bin/python scripts/evaluate_strict_output.py --help
.venv/bin/python scripts/evaluate_strict_gate.py --help
.venv/bin/python scripts/finalize_research_outputs.py
```

Exact commands for every completed, failed, aborted, and skipped process—including M0–M4, controls, graph statistics, strict seeds, weather, TCP, and tests—are preserved verbatim in `experiment_table.csv`; this avoids silently simplifying retry/resume commands.

## 23. ARTIFACT MAP

- Graph store: `data/graph_dataset/store/manifest.json`
- Sidecar manifests: `data/graph_dataset/sidecars/{logits,compact_evidence_l12,message_stats_l11,graph_stats_l11}/manifest.json`; hidden: `data/graph_dataset/hidden12/manifest.json`
- M0: `runs/research_20260830/message_flow/M0_attention/`
- M1–M5: `runs/research_20260830/message_flow/`
- Controls: `runs/research_20260830/controls/`
- Compact/H0/TCP: `runs/research_20260830/node_evidence/`
- Strict main/weather output, M5, and gate: `runs/research_20260830/combiners/strict/`
- Final summaries: `runs/research_20260830/final/{summary.json,experiment_table.csv,model_comparison.csv,bootstrap_deltas.json,best_scores_main.npz,best_scores_weather.npz,current_state.json}`
- Logs and ledger: `runs/research_20260830/logs/`, `runs/research_20260830/ledger.jsonl`
- Human-readable interim audit trail: `docs/results/EVIDENCE_FLOW_INTERIM_REPORT_2026-09-03.md`
- This report: `docs/results/EVIDENCE_FLOW_RESEARCH_REPORT_2026-08-30.md`

Large datasets, sidecars, checkpoints, scores, and logs remain gitignored.

## 24. FINAL GIT STATE

Research branch: `research/evidence-flow-20260830`. Starting commit: `8036fd902dca7e2b37a681fd8fa73113603f1676`. Key final implementation commits include `5f9e28b` (strict combiner confirmation), `6b82fcd` and `238d01e` (TCP multitask and auxiliary scoring), `f77c1ce` (seed-selective evaluation), `4148c47` (CLS-trajectory TCP), and `3cf656b` (final score registry/statistics). The report commit and exact final `git status --short` are recorded in the final handoff because a commit cannot contain its own hash.
