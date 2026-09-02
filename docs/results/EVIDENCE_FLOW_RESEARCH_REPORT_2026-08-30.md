# POLYGRAPH Evidence-Flow Research Audit — 2026-08-30

## EXECUTIVE ANSWER

Nothing in this run reliably beat MSP, because the graph store, split plans, checkpoints, and aligned
score files were not present and the protocol explicitly forbids regenerating the 1,010,000-image scan
or 198 GB store. Therefore no new scientific experiment was run and no new test score was produced.
The best result available in the version-controlled historical report is `cls_seq`: main AUROC 0.8759
versus MSP 0.8695, delta +0.0064, seed 7 only (standard deviation unavailable). This is at most a
**possible improvement**, not a confirmed win. On the different weather-plan test, historical
`cls_seq` scores 0.8689 versus MSP 0.8588 overall, but 0.8731 versus 0.8885 on unseen weather plus
extras. The small main gain is representation/layer-dynamics based, not demonstrated to be graph
specific, output-only, evidence-flow based, or due to a conditional combination.

## 1. RESEARCH QUESTION

The original hypothesis is that the directed connectivity induced by ViT attention contains error
information beyond the multiset of attention values. The stronger evidence-flow hypothesis is that
attention weights alone omit the magnitude and class direction of the transformed values being sent,
and that error prediction may improve by representing those messages, predicted-class-conditioned
token evidence, hidden states, layer dynamics, or a conditional gate that learns when output confidence
is unreliable. The classifier must remain frozen. This run cannot adjudicate the stronger hypothesis.

## 2. STARTING POINT

The anchors below were found in `docs/TEAM_REPORT.md` and
`docs/results/complementarity.md`; they were not recomputed because the corresponding NPZ files and
checkpoints are absent.

| plan/slice | MSP | graph | fusion | CLS sequence | top-100 graph | flat attention MLP |
|---|---:|---:|---:|---:|---:|---:|
| main, all (n=17,000) | 0.8695 | 0.8417 | 0.8758 | 0.8759 | 0.8125 | 0.7421 |
| weather, unseen weather+extras | 0.8885 | 0.8380 | 0.8738 | 0.8731 | — | — |

The report also gives weather-plan **all-test** values of MSP 0.8588, graph 0.8190, fusion 0.8657,
and CLS sequence 0.8689. These are not inconsistent with the row above: they are different slices of
the same 14,450-record weather test. All requested anchor literals were found within 0.003. Alignment,
sample count, and metric calculations could not be independently verified.

## 3. ENVIRONMENT

| item | value |
|---|---|
| OS | Ubuntu 20.04.6 LTS; Linux 5.15.0-139-generic |
| CPU / memory | Intel i7-13700K; 31 GiB RAM |
| GPU / VRAM | NVIDIA RTX A5000; 24,564 MiB |
| driver | 535.161.08 |
| `nvidia-smi` CUDA compatibility | 12.2 (driver maximum, not toolkit installation) |
| `nvcc` | `/usr/bin/nvcc`, CUDA toolkit 10.1.243; unused |
| Python | repository-local CPython 3.11.16 installed by `uv` |
| PyTorch / CUDA runtime | 2.5.1+cu121 / CUDA 12.1 |
| PyG | 2.7.0 |
| transformers | 5.16.1 |
| starting checkout | `8036fd902dca7e2b37a681fd8fa73113603f1676` on `main` |
| audited package base | `9a9bf6bc344449a6a9f2fdc83c539a0cba4ffa25` (`origin/full_pipeline`) |
| ending code commit before report | `e947772e220f374e3c5d036382f56ca76f569678` |

The initial checkout did not contain the package; the already-fetched `origin/full_pipeline` lineage
did. The research branch was advanced to that lineage before package work. There were no pre-existing
tracked or staged changes; the captured diffs are empty.

Exact environment commands:

```bash
uv python install 3.11
uv venv --python 3.11 --clear .venv
uv pip install --python .venv/bin/python pip setuptools wheel
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install 'torch-geometric>=2.5,<2.8' \
  -r runs/research_20260830/env/requirements_without_torch.txt \
  pytest pandas scipy matplotlib psutil
```

CUDA matrix multiplication, PyG `TransformerConv` forward/backward, and a frozen ViT pass on two
synthetic PIL images all passed. The ViT yielded 12 attention layers of shape `[B,12,197,197]` and a
CLS trajectory of shape `[B,12,768]`.

## 4. DATA AND SPLITS

No `data/` directory exists. Searches under the parent repositories (depth 5) and `/home/yishail`
(depth 6) found no `graph_dataset/store/manifest.json`. At inventory time 356 GiB was free.

Historical, unverified store identity: 75,000 selected records, 38 shards, approximately 198 GB,
`tau=0.02`, model `edumunozsala/vit_base-224-in21k-ft-cifar100`, 197 tokens, 12 layers, and stored CLS
embeddings. Historical main split: 52,000/6,000/17,000. Historical weather split:
36,284/4,646/14,450. The reports state group disjointness and balanced cells, but this run could not
rerun overlap, balance, key coverage, model-ID, hash, or shard-offset checks. No sidecars are present or
were created.

## 5. CODE AUDIT FINDINGS

The previous `EdgeSetModel` consumes only `edge_attr`; it omits node coordinates, CLS identity, layer
position, and per-head diagonals supplied to the GNN. It is therefore not a complete same-information
control. The existing `SequenceConcatModel` applies one shared GNN to disconnected per-layer graphs,
mean-pools each layer, concatenates layer summaries, and has no same-token temporal edges. Its failure
would not refute temporal graph structure.

Hidden shards are selected by graph shard and offset, and record count is asserted. However, the loader
does not validate store-key hash, requested hidden layer, model ID, complete shard-count vector, or
atomic-manifest provenance. `Data` objects do not expose `store_index`. These are concrete extension
requirements, not findings demonstrated by experiments.

No package changes were made after the missing-store stop condition, so the omissions above were not
fixed and new-checkpoint compatibility could not be tested. Existing checkpoint round-trip behavior did
pass the inherited test suite. The audit found no new demonstrated package bug; it found insufficient
sidecar alignment guards and an incomplete scientific control.

## 6. EXPERIMENT LEDGER

Completed or failed operational tasks:

| ID | configuration | runtime | status | result |
|---|---|---:|---|---|
| `env_venv_attempt` | system Python 3.8 `venv` | <1 s | failed | missing `ensurepip`; sudo forbidden |
| `env_install_torch` | torch 2.5.1+cu121 | 241 s | completed | install log |
| `env_install_dependencies` | PyG and requirements | 21 s | completed | install log |
| `smoke_torch_gpu` | 4096² CUDA matmul | <2 s | completed | GPU smoke log |
| `smoke_pyg_gpu` | TransformerConv F/B | <2 s | completed | PyG smoke log |
| `smoke_frozen_vit` | two synthetic images | 32 s | completed | ViT smoke log |
| `tests_existing_polygraph` | 38 tests | 7 s | completed | test log |
| `tests_existing_deepsets` | 6 tests | 1 s | completed | test log |
| `compile_existing` | package/tests/scripts | <1 s | completed | compile log |
| `suite_preflight` | required artifact check | 0 s | failed | four required files absent |
| `anchor_report_verification` | version-controlled reports | <1 s | completed | historical-only JSON |
| `suite_report_handoff` | continue after blocker | 0 s | completed | this report |

No trained configuration has a parameter count or best epoch in this run. The complete ledger,
including timestamps, commands, commits, status, reasons, and paths, is in
`runs/research_20260830/final/experiment_table.csv`.

## 7. OUTPUT BASELINES

| method | main AUROC | status / selection |
|---|---:|---|
| MSP | 0.8695 | historical report |
| margin | 0.8681 | historical report |
| entropy | — | skipped: logits/store/plans absent |
| energy | 0.8653 | historical report |
| max logit | — | skipped: logits/store/plans absent |
| p-normalized logit | — | skipped; no validation selection possible |
| pNormSoftmax | — | skipped; no validation selection possible |
| raw-logit LR | — | skipped |
| sorted-centered-logit LR | — | skipped |
| raw-logit MLP | — | skipped |
| sorted-centered-logit MLP | — | skipped |
| CLS TCP | — | skipped |

The old report's claim that MSP is the output ceiling is supported only against margin, energy, and a
two-feature output LR/MLP. It is not established against full-logit or p-normalized baselines.

## 8. STRUCTURE CONTROLS

| control | main AUROC | status |
|---|---:|---|
| edge set | — | skipped |
| node-plus-edge set | — | skipped |
| endpoint-record set | — | skipped |
| top-100 graph | 0.8125 | historical seed 7 |
| target-permuted graph | — | skipped |
| edge-attribute-shuffled graph | — | skipped |
| full graph | 0.8417 | historical seed 7 |

The historical top-100 graph exceeds the flat top-100 attention MLP by 0.0704, but the GNN also sees
node information. Thus the strongest valid claim is narrower: the configured GNN representation is
more predictive than flattening its edge values alone. A connectivity-specific claim remains unproven
until C2/C3 and rewiring controls run.

## 9. GRAPH-STATISTICS RESULTS

Extraction and LR/MLP/HGB evaluation were skipped because the graph store is absent. Whether graph
summaries contain signal or improve a gate remains unknown.

## 10. COMPACT CLASS-EVIDENCE RESULTS

Compact-evidence extraction and N1–N3 were skipped because logits, hidden12, and the store are absent.
Historical raw graph, fusion, and CLS results alone do not isolate class conditioning. Explicit predicted
class versus runner-up conditioning remains untested.

## 11. MESSAGE-FLOW RESULTS

M0–M5 were skipped. Raw attention `A`, message magnitude `A||W_hV_j||`, and the signed predicted-versus-
runner-up class-direction proxy were not compared. The proposed proxy would not be an exact causal
logit contribution: residual paths, layer normalization, and the block MLP intervene.

## 12. ARCHITECTURE RESULTS

SimpleMPNN and TemporalMPNN were not implemented or run after the stop condition. Consequently there is
no TransformerConv-versus-explicit-message-passing comparison and no temporal result.

## 13. CONDITIONAL COMBINATION

All G1–G4 logistic, residual, and mixture-gate fits were skipped because aligned train/validation/test
scores and base-image groups are absent. No gate behavior can be reported, and strict confirmation was
not triggered.

## 14. MAIN BENCHMARK

| method | seeds | AUROC mean ± std | delta vs MSP | interpretation |
|---|---|---:|---:|---|
| MSP | deterministic historical | 0.8695 | — | reference |
| graph | 7 historical | 0.8417 ± N/A | −0.0278 | worse |
| fusion | 7 historical | 0.8758 ± N/A | +0.0063 | possible only |
| CLS sequence | 7 historical | 0.8759 ± N/A | +0.0064 | possible only |

No seed 1/2 values, new AUPRC/AURC/risk values, or new slices exist. Historical all-test AUPRC is
0.8416 MSP, 0.8152 graph, 0.8513 fusion, and 0.8500 CLS sequence; historical risk@0.5 is 0.2088,
0.2405, 0.2061, and 0.2022 respectively. These were not reproduced.

## 15. WEATHER-FAMILY HOLDOUT

| method | all AUROC | unseen weather+extras AUROC | unseen delta vs MSP |
|---|---:|---:|---:|
| MSP | 0.8588 | 0.8885 | — |
| graph | 0.8190 | 0.8380 | −0.0505 |
| fusion | 0.8657 | 0.8738 | −0.0147 |
| CLS sequence | 0.8689 | 0.8731 | −0.0154 |

These historical values use a different test set from the main plan. They show that the small main-plan
representation gain did not extend to the central unseen-family slice.

## 16. COMPLEMENTARITY

Historical main-test Spearman correlation with MSP is approximately 0.79 for graph and 0.83 for fusion.
At a 50% flag budget, graph catches 818 errors MSP misses and fusion catches 847; graph loses 1,087
errors MSP catches. Graph flags 46.1% and fusion 47.7% of MSP-blind errors at their own 50% budgets.
Historical confident-prediction AUROC is MSP 0.8423, graph 0.8186, fusion 0.8614, and CLS sequence
0.8598. This supports complementarity for the historical seed, not superiority or causality.

## 17. STATISTICAL UNCERTAINTY

Group bootstrap was impossible without aligned score vectors and base-image group identifiers. The
required 2,000-repetition result is explicitly marked unavailable rather than replaced by invalid row-wise
resampling. Seed variation is unavailable. All +0.006-range historical differences are below the 0.02
single-seed scientific-win threshold and fail the three-seed confirmation criteria.

## 18. NEGATIVE RESULTS

Historical negative results are raw graph below MSP, CHARM-lite 0.8233 below raw graph 0.8417,
top-100 graph below the full graph, flat attention MLP 0.7421, output LR below MSP, energy below MSP,
Mahalanobis 0.6623, and increased GNN capacity failing to improve validation performance. Historical
CLS/fusion advantages reverse on unseen weather plus extras. TCP, matched controls, rewiring, graph
statistics, compact evidence, message variants, SimpleMPNN, conditional combiners, and temporal graphs
were not negative results—they were skipped and remain unknown.

## 19. WHAT ACTUALLY EXPLAINS THE BEST RESULT

The available ablations most strongly associate the historical best result with hidden representation
and layerwise dynamics: CLS sequence is best on main, and fusion is similar. Attention connectivity is
complementary but has lower overall AUROC. Output evidence beyond MSP is incompletely tested. Message
magnitude, explicit class conditioning, and conditional gating have no result here, so they cannot
explain the best observed score.

## 20. SCIENTIFIC CLAIMS WE CAN AND CANNOT MAKE

### Supported

- The exact frozen ViT/PyTorch/PyG path works on the available CUDA host.
- Historical version-controlled tables contain all requested anchors.
- The existing edge-set control omits node information, and the existing multi-layer graph has no
  temporal identity edges.
- Historical internal detectors rank errors differently from MSP and recover some MSP-blind errors.

### Not supported

- No method is a confirmed improvement over MSP.
- The full output vector has not been shown to be bounded by MSP in this repository state.
- The historical top-100 comparison does not isolate graph connectivity from node features.
- The class-direction proxy is not Chefer relevance and would not be causal attribution.

### Still uncertain

- Whether matched topology controls preserve the GNN advantage.
- Whether transformed-value/message magnitude or predicted-class conditioning improves prediction.
- Whether an explicit MPNN or conditional gate turns complementarity into robust selective ranking.
- Whether any candidate survives three seeds, group bootstrap, and unseen-weather confirmation.

## 21. RECOMMENDED NEXT STEP

1. Restore or mount the immutable store, plans, hidden12, checkpoints, and score NPZ files. Information:
   enables exact alignment and all remaining work. Storage: existing ~219 GB, no duplication. Runtime:
   minutes to validate. Stop if manifests/model ID/key hashes do not match.
2. Run full-logit extraction and p-normalized/output baselines first. Information: tests whether MSP is
   genuinely the output ceiling. Storage: roughly 15 MB float16 logits plus metadata for 75k records.
   Runtime: short GPU extraction and CPU fits. Stop output-only direction if fixed validation-selected
   methods fail to improve MSP by 0.005 at seed 7.
3. Then run C2/C3/rewiring and final-layer message statistics. Information: separates connectivity from
   matched node/edge content and tests evidence flow. Storage: compact sidecars, well below 10 GB.
   Runtime: medium. Stop if no message candidate improves raw graph by 0.015 or reaches the triage gate.

## 22. REPRODUCTION COMMANDS

Completed tasks can be reproduced with:

```bash
.venv/bin/python scripts/inventory_polygraph.py \
  --root "$PWD" --store-dir data/graph_dataset/store \
  --output runs/research_20260830/inventory/inventory.json
.venv/bin/python scripts/verify_current_anchors.py
.venv/bin/python tests/test_polygraph.py
.venv/bin/python tests/test_deepsets.py
.venv/bin/python -m compileall -q polygraph tests scripts
.venv/bin/python scripts/run_research_suite.py \
  --stage all \
  --main-plan data/graph_dataset/split_plan_main.json \
  --weather-plan data/graph_dataset/split_plan_weather.json \
  --store-dir data/graph_dataset/store \
  --run-root runs/research_20260830 \
  --budget-gpu-hours 8 --timeout-minutes 120 --resume
```

The last command is not a dry run. In this artifact state it records the required skips and proceeds to
reporting. No command can reproduce a scientific experiment until the missing immutable assets return.

## 23. ARTIFACT MAP

- Report: `docs/results/EVIDENCE_FLOW_RESEARCH_REPORT_2026-08-30.md`
- Environment: `runs/research_20260830/env/system.txt`, `pip_freeze.txt`, and three smoke logs
- Inventory: `runs/research_20260830/inventory/inventory.json`, `files.txt`, `resolved_paths.env`
- Anchor audit: `runs/research_20260830/final/current_anchor_verification.json`
- Ledger: `runs/research_20260830/ledger.jsonl` and `final/experiment_table.csv`
- Summary: `runs/research_20260830/final/summary.json`
- Score placeholders: `final/best_scores_main.npz`, `final/best_scores_weather.npz` (zero records,
  explicitly marked unavailable; not scientific score artifacts)
- Bootstrap status: `final/bootstrap_deltas.json` (unavailable with reason)
- Test/install/suite logs: `runs/research_20260830/logs/`
- Sidecar manifests, checkpoints, plots: none created

Literature basis (method and experiment sections inspected):

- Kobayashi et al. show that attention-weight interpretation omits transformed value-vector norms;
  their BERT/NMT analyses motivate message magnitude, not error prediction
  ([EMNLP 2020](https://aclanthology.org/2020.emnlp-main.574/)).
- Abnar and Zuidema add residual-aware attention rollout and maximum-flow views and compare them with
  blank-out/input-gradient importance; this motivates cross-layer connectivity, with acknowledged
  simplifying assumptions ([ACL 2020](https://aclanthology.org/2020.acl-main.385/)).
- Chefer et al. propagate class-specific relevance using Deep Taylor/gradient information and evaluate
  explanation quality; that method is categorically different from the proposed forward-only class
  direction proxy ([CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)).
- Corbière et al. define TCP using the true class as a training target, learn it because the true class is
  unavailable at inference, and evaluate across several vision tasks; this supplies the ConfidNet design
  principle ([NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html)).
- Cattelan and Silva evaluate post-hoc logit confidence across 84 ImageNet models and distribution shift,
  selecting logit p-normalization on held-out data; this motivates O6/O7
  ([UAI 2024](https://proceedings.mlr.press/v244/cattelan24a.html)).
- Frasca et al. construct attention graphs with activations/diagonals and message passing for LLM
  hallucination detection, including cross-dataset studies; CHARM-lite in this repository lacks their
  activation half ([arXiv:2509.24770](https://arxiv.org/abs/2509.24770)).
- Beigelman and Freiman train frozen-ViT intermediate heads, form layerwise predicted-class/competitor
  logits plus top-k stability features, and test in- and cross-dataset error prediction; this motivates
  dynamics baselines but does not establish this repository's historical result
  ([arXiv:2604.10643](https://arxiv.org/abs/2604.10643)).

## 24. GIT STATE

Focused commits in this study:

- `d30d3d668b178eb21af5189e0ff43c04178f8524` — reproducible environment/artifact inventory
- `e947772e220f374e3c5d036382f56ca76f569678` — artifact-aware anchors and suite orchestration
- `02b93e085a6faab654f5a57147e3b97a82a0479d` — blocker report and historical anchor analysis
- `2feba149a8c01cfa01a14c220ab3181592d979ab` — complete blocked experiment/test register

Final `git status --short` after committing the report metadata was empty. Generated `data/`, `runs/`,
caches, checkpoints, tensors, and downloaded papers remain gitignored and uncommitted.

## APPENDIX A — COMPLETE SKIP REGISTER

All entries below have the same precise reason: missing
`data/graph_dataset/store/{manifest.json,store_keys.json}` and missing main/weather split plans. Those
assets are non-regenerable within the protocol because rescanning/extracting is forbidden.

- Controls: `C1_edge_set`, `C2_node_edge_set`, `C3_endpoint_set`, `C4_top100_graph`,
  `C5_target_permute`, `C6_shuffle_attr`.
- Logits/output: `logits_full_store`, `O1_msp`, `O2_margin`, `O3_entropy`, `O4_energy`,
  `O5_max_logit`, `O6_p_normalized`, `O7_pnorm_softmax`, `O8_raw_logit_lr`,
  `O9_sorted_logit_lr`, `O10_raw_logit_mlp`, `O11_sorted_logit_mlp`.
- TCP: `cls_tcp_main`, `cls_tcp_weather`, `cls_seq_tcp_optional`.
- Graph statistics: `graph_stats_extract`, `graph_stats_lr`, `graph_stats_mlp`, `graph_stats_hgb`.
- Compact evidence: `compact_evidence_extract`, `N1_compact_graph`,
  `N2_compact_node_edge_set`, `N3_compact_endpoint_set`.
- Message flow: `message_stats_l11_extract`, `M0_attention`, `M1_attention_message`,
  `M2_attention_decision`, `M3_evidence_flow`, `M4_evidence_compact`, `M5_evidence_hidden`,
  `graph_tcp_multitask_optional`.
- Explicit MPNN: `SMP0_attention`, `SMP1_best_flow`, `SMP2_compact_best_flow`.
- Combiners: `G1_logistic`, `G1_residual`, `G1_gate`, `G2_logistic`, `G2_residual`, `G2_gate`,
  `G3_logistic`, `G3_residual`, `G3_gate`, `G4_logistic`, `G4_residual`, `G4_gate`,
  `strict_confirmation`.
- Temporal: `T1_temporal_final4`, `T2_temporal_compact`.

There were no aborted experiments. The suite did not consume scientific GPU budget.

The requested research-only test modules `test_research_controls.py`, `test_sidecars.py`, and
`test_research_eval.py` were also skipped because their corresponding extension was not implemented
after the missing-store stop condition. The inherited 44 tests and full compilation passed.
