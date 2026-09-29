# Handoff: ViT Attention-Graph Failure Prediction (CIFAR-100)

Last updated: 2026-08-16 (session 2). This document is self-contained. You should not need prior conversation context.

---

## 1. Project Goal

Freeze a CIFAR-100 ViT. Convert its self-attention into per-image graphs. Train **only** a small GNN on top to predict whether the ViT misclassified the image.

Target label:

```
y_err(x) = 1  if argmax_c f_c(x) != y(x)
           0  otherwise
```

The ViT is never fine-tuned. The research question is whether **attention topology** carries failure signal beyond what the softmax confidence already tells you.

**The bar to beat is the MSP baseline (max softmax probability), which scores AUROC 0.9101 on our test set. No graph model has beaten it, and Stage 5 now shows the graph adds nothing measurable on top of it either.** Best graph-only model so far is 0.7898.

**Headline result from session 2:** graph-only stays well below MSP, and an MSP+graph combiner lands at 0.9112 ± 0.0046 vs MSP's 0.9101 — a +0.001 difference, i.e. noise. The one durable positive: on the high-confidence subset (`conf ≥ 0.95`) the graph scores ~0.71 AUROC across 3 seeds, so it is not merely a confidence proxy. See Sections 5.5-5.7.

**Headline change from session 3: the project was rebuilt around CIFAR-100-C at full
scale, and the proposal's design is now implementable end to end.** The 1,700-row cache
was never the real limit -- the positive class was: only 852 of the 10,000 clean test
images are misclassified, and the POC split consumed 850 of them. The clean train split
does not help (measured: 99.45% accuracy, 274 errors, confidence inflated by fine-tuning).
The full grid is now scanned -- **1,010,000 records, 239,921 ViT errors (23.8%)** across
clean data plus 19 corruptions x 5 severities -- and a 75,000-record plan is built
(52k train / 6k val / 17k test, 50/50, group-disjoint by base image, stratified per
(corruption, severity) cell, the 4 extra CIFAR-C corruptions held out to test only).
Graph extraction returned to the proposal's threshold rule `max_h A > tau` at tau=0.02
(bit-identical to the old POC rule; ~7,600 edges/layer), stored sorted so every stricter
tau and every top-K view -- including the old top-100 -- derive from one extraction. CLS
embeddings are stored per layer for the representation baselines. Training now has a real
validation split, early stopping, multi-seed runs, and evaluation slices with trained
baselines (output-statistics logistic; CLS-embedding MLP) so the graph cannot win merely
by being trained. Historical intermediate results from the interim top-100 dataset
(MSP 0.8675 on its balanced test, decaying 0.8999 -> 0.8194 with severity) remain in
`data/graph_dataset/graphs/`, readable via [legacy_dataset.py](../legacy/legacy_dataset.py).
Everything current is documented in the root [README.md](../README.md); the pipeline is
`python3 -m polygraph.data` / `python3 -m polygraph.training`.

---

## 2. Environment (read this first — several footguns)

| Item | Value |
|---|---|
| Python | `.venv/bin/python` (Python 3.9) |
| Torch | 2.8.0, MPS available (Apple Silicon) |
| Working dir | `/Users/yishail/Library/CloudStorage/OneDrive-Mobileye/Documents/graph_learning_project` |

Footguns:

- **`python` is not on PATH. `python3` resolves to system Python which has NO torch.** Always invoke `.venv/bin/python` explicitly.
- **`rg` (ripgrep) is not installed** in the shell. Use `grep`.
- **HuggingFace was blocked by a corporate proxy** (`Tunnel connection failed: 403 Forbidden`). **As of session 2 this is no longer true** — `huggingface.co` returned HTTP 200 and the ViT downloaded and re-extracted graphs fine. Re-verify before assuming either state; the proxy may flip back.
  - The offline path still works regardless: `prepare()` in [lightweight_attention_experiments.py](../legacy/lightweight_attention_experiments.py) skips loading the ViT entirely when both the graph cache and scan cache exist on disk.
  - You only need network if you delete the caches or change `--edges-per-layer` / `--seed` (which forces graph re-extraction).
- **If you run commands in a restricted/sandboxed shell, ViT loading fails with `PermissionError ... /Users/yishail/.cache/huggingface/...`.** This is a *filesystem* permission problem, not a network one — the error message misleadingly suggests a stale lock file. Run unsandboxed (or point `HF_HOME` somewhere writable).
- **`choose_device()` returns `mps` if available, else `cpu`, and a restricted shell hides MPS.** The same config scored graph AUROC **0.7688 on cpu vs 0.7776 on mps** — identical seed and settings. Backend alone moves AUROC by ~0.01, so **always confirm the `Using device:` line matches across runs you intend to compare.**

---

## 3. Data and Splits

- Dataset: CIFAR-100 **test split**, read from local parquet at `data/hf_cifar100/cifar100/test-00000-of-00001.parquet`.
- Frozen model: `edumunozsala/vit_base-224-in21k-ft-cifar100`.
- Full-split ViT scan (cached at `reports_vit_scan_test/vit_scan_with_indices.json`):
  - 10,000 images, 9,148 correct, 852 wrong, **accuracy 91.48%**, mean confidence 0.9573.

Detector split (balanced 50/50, cached at `reports_lightweight_top100/experiment_splits_train1500_test200_seed7.json`):

| Split | Correct | Wrong | Total |
|---|---:|---:|---:|
| Train | 750 | 750 | 1500 |
| Test | 100 | 100 | 200 |

There is **no validation split** — this was a deliberate POC simplification. Consequence: no early stopping, no principled model selection. Every result below is a fixed-30-epoch, single-seed number. **Treat small differences as noise.**

> Note: this test set is byte-identical to the one used in the older `poc_gnn_vit_cifar100.py` runs (same `balanced_train_test_indices(correct, wrong, 750, 100, seed=7)` call). Confirmed by MSP scoring exactly 0.9101 in both. So old and new numbers are directly comparable.

**Caveat carried forward from the original POC: all detector train and test examples come from the CIFAR-100 test split. These are feasibility results, not clean generalization benchmarks.**

---

## 4. Graph Construction (current)

For each ViT layer `l` with attention `A^(l,h)` (12 layers, 12 heads, 197 tokens = 1 CLS + 196 patches):

1. Score each token pair: `score[i,j] = max_h A[h,i,j]`
2. Zero the diagonal (no self-edges)
3. Keep the **top 100 (i,j) pairs globally for that layer** — this is the `top_global` rule
4. Edge direction `j -> i` (information flows from key `j` to query `i`)
5. Edge feature = raw per-head vector `[A[h=1,i,j], ..., A[h=12,i,j]]` (12-dim)

Node features (**minimal only** — ViT hidden states are NOT used anywhere in the current runs):
- normalized patch row, normalized patch col
- CLS indicator (column index 2 — used to locate the CLS node after PyG batching)
- normalized layer coordinate
- per-head diagonal attention values

This replaced the **old** rule (`max_h A > tau=0.02` plus top-8 per query), which produced ~5,652 edges/layer and ~92,000 edges for a packed 12-layer graph. The new rule gives exactly 100 edges/layer, 1,200 for all 12.

Cache: `reports_lightweight_top100/cache/top_global_100_seed7_graphs.pt` (~406 MB, 1,700 rows, 12 layers each).

---

## 5. Results So Far

### Baseline: MSP (max softmax probability)

Computed directly from cached ViT confidences on the exact test split.

| Split | AUROC | AUPRC |
|---|---:|---:|
| Train (1500) | 0.9058 | 0.8920 |
| **Test (200)** | **0.9101** | **0.8994** |

Why MSP is not trivially perfect — the ViT is often **confidently wrong**:

| Confidence ≥ | Wrong images | Correct images |
|---:|---:|---:|
| 0.99 | 3 / 100 | 55 / 100 |
| 0.95 | **33 / 100** | 93 / 100 |
| 0.90 | 44 / 100 | 96 / 100 |
| 0.80 | 59 / 100 | 97 / 100 |

Median confidence: 0.9909 on correct, 0.8778 on wrong. Mean: 0.9752 vs 0.7974.

### Stage 1 — Sequence pipeline debug (PASSED)

Purpose: the previous agent's all-layer model scored exactly 0.5000 AUROC (pure chance), which looked like a bug. These checks establish it wasn't.

| Check | Test AUROC | Test AUPRC | Test Acc | Notes |
|---|---:|---:|---:|---|
| Tiny 32-example 12-layer overfit | 1.0000 | 1.0000 | 0.9375 | train loss 0.0005 → can memorize |
| Final layer, normal path | 0.7479 | 0.7392 | 0.690 | reference |
| Final layer via sequence len=1 | 0.7478 | 0.7339 | 0.675 | **matches reference → wrapper is correct** |
| Duplicate-final 12-position sequence | 0.7725 | 0.7620 | 0.700 | multi-position batching/reshape works |

**Conclusion: the sequence implementation is sound.** The old chance result was caused primarily by edge-count explosion suffocating optimization, not by a code bug.

### Stage 2 — Per-layer sweep (mean pooling, one model per layer)

| Layer (1-based) | Test AUROC | Test AUPRC | Test Acc |
|---:|---:|---:|---:|
| 1 | 0.5128 | 0.5117 | 0.520 |
| 2 | 0.5959 | 0.5726 | 0.565 |
| 3 | 0.4696 | 0.4781 | 0.500 |
| 4 | 0.5377 | 0.5061 | 0.505 |
| 5 | 0.4963 | 0.5168 | 0.525 |
| 6 | 0.6230 | 0.6260 | 0.580 |
| 7 | 0.5069 | 0.5294 | 0.520 |
| 8 | 0.6607 | 0.6367 | 0.620 |
| 9 | 0.5440 | 0.5378 | 0.550 |
| 10 | 0.5497 | 0.5868 | 0.530 |
| 11 | 0.6195 | 0.6093 | 0.605 |
| **12** | **0.7476** | **0.7387** | **0.675** |

Signal is concentrated in the final layer. Most early/mid layers are near chance. Note the sweep is not monotonic (layer 8 > layers 9, 10), so the ordering of the middle layers is probably not stable across seeds.

> The generated `stage2_layer_sweep_report.md` says "Best validation layer: 1 one-based". **Ignore that line** — it is a leftover from when validation existed and reads a column that is now all `None`. Rank by test AUROC.

### Stage 3 — Readout / pooling comparison (layer 12 only)

This was the biggest modeling win.

| Readout | Test AUROC | Test AUPRC | Test Acc | Train AUROC | Params |
|---|---:|---:|---:|---:|---:|
| mean | 0.7252 | 0.7198 | 0.650 | 0.7841 | 13,953 |
| CLS only | 0.6534 | 0.6069 | 0.555 | 0.6945 | 13,953 |
| CLS + mean | 0.7560 | 0.7284 | 0.650 | 0.7863 | 14,977 |
| CLS + mean + max | 0.7367 | 0.7262 | 0.650 | 0.7841 | 16,001 |
| **CLS + gated** | **0.7898** | **0.7696** | **0.720** | 0.8053 | 16,066 |

"CLS" = the **post-GNN** embedding of the CLS node (not the raw ViT embedding). "Gated" = learned per-node attention scores, graph-wise softmax, weighted sum. Final vector is `[h_CLS ; gated_pool]`.

CLS alone is weak, mean alone is mediocre, but the combination is clearly best.

### Stage 4 — Multi-layer fusion (OVERFITS — negative result)

Architecture: shared GNN across selected layers, per-layer CLS+gated readout, learned layer-attention gate, plus a mandatory final-layer skip connection `[fused ; r_final]`.

| Model | Layers | Train AUROC | Test AUROC | Test AUPRC | Test Acc |
|---|---|---:|---:|---:|---:|
| Layer 12 only, CLS+gated | 12 | 0.8053 | **0.7898** | **0.7696** | **0.720** |
| Final-four fusion | 9-12 | 0.8636 | 0.7603 | 0.7577 | 0.690 |
| Final-two fusion | 11-12 | 0.8705 | 0.7366 | 0.7398 | 0.660 |

Perfectly monotonic: **more layers → better train fit → worse test.** Classic overfitting. The final-layer skip prevented collapse but did not prevent degradation. Adding even just layer 11 hurt.

### Stage 5 — Complementarity vs MSP (RUN in session 2)

Layer 12, `cls_gated`, 100 edges, 30 epochs, seed 7, mps. Outputs in `reports_edge_ablation_100/`.

| Detector | Test AUROC | Test AUPRC |
|---|---:|---:|
| MSP only | 0.9101 | 0.8994 |
| Graph only | 0.7776 | 0.7525 |
| MSP + graph (combiner) | 0.9146 | 0.9093 |

Correlation between graph score and MSP uncertainty: **0.387** (low — the graph is not just re-reading confidence).

By confidence slice:

| Conf ≥ | Images | Wrong | Correct | MSP AUROC | Graph AUROC | Combined |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 200 | 100 | 100 | 0.9101 | 0.7776 | 0.9146 |
| 0.8 | 156 | 59 | 97 | 0.8821 | 0.7565 | 0.8875 |
| 0.9 | 140 | 44 | 96 | 0.8509 | 0.7495 | 0.8570 |
| 0.95 | 126 | 33 | 93 | 0.8355 | 0.7208 | 0.8386 |
| 0.99 | 58 | 3 | 55 | 0.7758 | — | — |

> **Ignore the `conf ≥ 0.99` row.** It contains only **3 wrong images**; its AUROC swings between 0.28 and 0.51 across seeds and is pure noise. The report still prints it.

**Interpretation, against the decision rule this document originally laid out:**

- Combined (0.9146) is *nominally* above MSP (0.9101), but see Section 5.7 — across 3 seeds the gap is **+0.0011 ± 0.005**. **This is noise. The graph does not add usable signal on top of confidence.**
- It is *not* the "graph is redundant" case either: the standardized combiner coefficients are **0.85 (graph) vs 1.85 (MSP)** — the combiner genuinely uses the graph, it just does not convert into test AUROC.
- **The third criterion is met.** On `conf ≥ 0.95` (33 wrong / 93 correct) the graph scores **0.72 AUROC**, well above chance, precisely where MSP decays from 0.9101 to 0.8355. This is the only part of the project's premise that survived.

#### Bug found and fixed in `run_stage5()`

The first execution reported **combined 0.8557 — below MSP alone**, which is structurally impossible for an honest combiner (it can always fall back to its best single input). Cause: the combiner was `LogisticRegression()` on raw `[graph_logit, 1 - confidence]` with **no feature scaling**. `msp_uncertainty` spans ~[0, 0.2] (median 0.009) while `graph_logit` spans ~[-1.2, 1.5], so under the default L2 penalty (`C=1.0`) the MSP feature was effectively crushed — it contributed ~0.04 of score range vs the graph's ~1.0, and the combiner ranked mostly by the *weaker* feature.

Fix (already applied): feed MSP as **log-odds** (monotone, so MSP-alone AUROC is unchanged at 0.9101) and wrap the model in a `StandardScaler` pipeline. Combined then moved 0.8557 → 0.9146. **If you add features to this combiner, keep them standardized.**

> Unrelated red herring: sklearn prints `RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul` on every run. The feature matrices contain no inf/NaN — these are spurious FP-flag warnings from the Apple Accelerate BLAS backend. Ignore them.

### Stage 5.6 — Edge-count ablation (RESOLVES Section 6)

Layer 12, `cls_gated`, 30 epochs, seed 7, all on mps. One run per edge count; outputs in `reports_edge_ablation_{50,100,200,500}/`.

| Edges/layer | Graph AUROC | Graph AUPRC | Combined | Graph @ conf≥0.95 | Corr w/ MSP |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.7368 | 0.7286 | 0.9047 | 0.6947 | 0.339 |
| 100 | 0.7776 | 0.7525 | 0.9146 | 0.7208 | 0.387 |
| 200 | 0.7820 | 0.7742 | 0.9100 | 0.7299 | 0.423 |
| 500 | 0.7815 | 0.7865 | 0.9036 | 0.7276 | 0.435 |

**Graph AUROC saturates at ~100 edges.** 50 → 100 gains +0.041; 100 → 200 gains +0.004; 200 → 500 is flat (−0.0005). AUPRC keeps creeping up slightly, and correlation with MSP rises with density (denser graphs encode more of what confidence already knows).

**Conclusion: sparsification was NOT the cause of the 0.8286 → 0.7476 drop.** Going 5× denser buys essentially nothing. The remaining explanations for the old 0.8286 are epoch count (5 vs 30 — the old run may simply have overfit less), single-seed noise (~±0.02, see 5.7), or the node-feature difference. **Do not spend more time on edge density.**

### Stage 5.7 — Multi-seed verification

100 edges, layer 12, `cls_gated`, mps. The split file is fixed (`--split-file` default), so the test set — and therefore MSP = 0.9101 — is identical across seeds; only graph extraction order, init and batching change.

| Seed | Graph AUROC | Combined | Graph @ conf≥0.9 | Graph @ conf≥0.95 |
|---|---:|---:|---:|---:|
| 7 | 0.7776 | 0.9146 | 0.7495 | 0.7208 |
| 1 | 0.7405 | 0.9131 | 0.6946 | 0.6742 |
| 2 | 0.7513 | 0.9060 | 0.7356 | 0.7237 |
| **mean ± sd** | **0.7565 ± 0.0191** | **0.9112 ± 0.0046** | 0.7266 ± 0.0287 | **0.7062 ± 0.0278** |

Takeaways:
- **Single-seed graph AUROC carries ~±0.02 noise.** Every single-seed comparison in Sections 5.2-5.4 under ~0.04 should be treated as unresolved. In particular the Stage 3 readout ranking and the Stage 2 mid-layer ordering are not established.
- **Combined vs MSP: +0.0011 ± 0.0046. No complementarity gain.**
- **Graph @ conf≥0.95 is above chance in all 3 seeds (min 0.674).** This is the one robust positive finding.

---

## 6. RESOLVED: sparsification was not the problem

> **Status: closed in session 2 by the ablation in Section 5.6. Do not re-run this.**

The original concern: the older `poc_gnn_vit_cifar100.py` run on the **same test set**, same minimal node features, same TransformerConv/hidden-32 architecture, same mean pooling, but with the **old dense edge rule** (~5,652 edges) and only 5 epochs, scored:

```
last layer, minimal features, dense edges:  AUROC 0.8286
last layer, minimal features, top-100:      AUROC 0.7476   (Stage 2)
last layer, CLS+gated, top-100:             AUROC 0.7898   (Stage 3)
```

The hypothesis was that cutting to 100 edges cost ~0.08 AUROC. **The ablation refutes this.** Sweeping 50/100/200/500 edges at matched 30 epochs shows graph AUROC plateaus at ~0.78 from 100 edges onward; 500 edges is no better than 200. Edge density is not the lever.

What remains unexplained about the old 0.8286: most likely 5-epoch training overfitting less than 30-epoch, plus single-seed noise of ~±0.02 (Section 5.7). Not worth chasing on its own.

Still untested with the new pipeline: the old dense run with **`minimal+repr` node features** (adding ViT hidden states) scored **0.8681**, the best graph number ever recorded on this project. Since edge count is now ruled out, **node features are the leading candidate for the gap.**

---

## 7. DONE: Stage 5 complementarity test

> **Status: written, debugged, and executed in session 2. Results in Section 5.5. A real bug was found and fixed — read the bug note there before trusting any earlier Stage 5 output.**

Rerun with:

```bash
cd /Users/yishail/Library/CloudStorage/OneDrive-Mobileye/Documents/graph_learning_project
.venv/bin/python lightweight_attention_experiments.py --stage stage5 \
  --edges-per-layer 100 --epochs 30 --gnn-batch-size 64 \
  --output-dir reports_lightweight_top100 --log-every 30
```

It (1) retrains layer 12 `cls_gated` (~20s cached, ~2min if it must re-extract), (2) dumps `stage5_test_predictions.csv`, (3) fits a train-only logistic combiner on standardized `[graph_logit, logit(msp_uncertainty)]`, (4) reports MSP vs graph vs combined, (5) slices by confidence, (6) reports graph/MSP score correlation.

**Bottom line: the graph does not beat or meaningfully augment MSP overall. It does retain above-chance signal on confident failures.**

---

## 7.5. Why the old 0.8286 / 0.8681 beat everything current

Exact config of the old best runs, from `reports_balanced_750_100_transformer/poc_results.json`. Architecture, dropout, lr, weight decay, seed, split and MSP baseline are all **identical** to the current pipeline. What differs:

| Setting | Old (0.8286 minimal / 0.8681 minimal+repr) | Current (0.78-0.79) |
|---|---|---|
| Edge rule | `tau=0.02` + top-8 per query (~5,652 edges) | top-100 global |
| Epochs | **5** | 30 |
| GNN batch size | **4** | 64 |
| Validation | **`val_fraction` 0.3 (~1,050 train / 450 val)** | none (all 1,500 train) |
| Model selection | **best-val-loss checkpoint restored** | final-epoch weights |
| Readout | mean pooling | cls_gated |

Two confounds that matter more than edge density (which Section 6 already ruled out):

1. **The old run did early model selection.** `poc_gnn_vit_cifar100.py` tracks best val loss and restores that checkpoint before scoring. Combined with only 5 epochs, that is a strong regularizer the current pipeline lacks entirely — and Stage 4 established that the failure mode here is variance. The old run beat current numbers while training on ~30% *fewer* examples.
2. **Batch 4 vs 64** at the same LR: ~1,875 small noisy updates vs ~720 large-batch ones. A different optimization regime, not a like-for-like comparison.

`minimal+repr` is `x_repr = torch.cat([x_min, hidden.float()], dim=1)` — minimal features plus the **full 768-dim ViT hidden state** per token (~785-dim nodes vs ~17 today).

**Cheapest decisive next test (no cache rebuild needed):** re-run current layer-12 `cls_gated` at `--epochs 5 --gnn-batch-size 4`. That separates "optimization regime / model selection" from "node features" before anyone invests in extending the cache to carry hidden states.

---

## 8. Suggested Priority for the Next Agent

Priorities 1, 2 and 4 from the previous list are **done** (Sections 5.5-5.7). What is left, in order:

1. **Match the old training regime before rebuilding any cache.** Re-run current layer-12 `cls_gated` at `--epochs 5 --gnn-batch-size 4` (Section 7.5). No cache rebuild, ~1 min. The old 0.8286 used 5 epochs, batch 4, and a best-val-loss checkpoint on 30% fewer training examples. If that alone recovers ~0.83, the current gap is an optimization/regularization artifact, not a graph-content problem — and that changes what experiment 2 is worth.
2. **Re-test `minimal+repr` node features** (ViT hidden states, `x_min` + 768-dim hidden per token) with the `cls_gated` readout. Edge density is ruled out (Section 6), so node features are the leading content-side explanation for the old best-ever 0.8681. Note `x_repr` is built in the old script but the new `LayerGraphDataset` only stores `x` — you must extend the cache to carry hidden states. Budget for a cache rebuild.
2. **Focus the framing on confident failures, not on beating MSP overall.** Sections 5.5/5.7 show graph-vs-MSP head-to-head is a dead end (combined = MSP + 0.001). The live result is ~0.71 AUROC on `conf ≥ 0.95`. Consider training and evaluating *only* on the high-confidence subset, where MSP is structurally weak, rather than on the balanced 50/50 split.
3. **Use ≥3 seeds for every future claim.** Single-seed noise is ~±0.02 AUROC (Section 5.7), and cpu-vs-mps alone moves it ~0.01. Several existing single-seed conclusions (Stage 3 readout ranking, Stage 2 mid-layer order) are not actually established.
4. **Get a validation split** before any further architecture search. There is still no early stopping or principled model selection; Stage 4 showed the failure mode is variance, so tuning on test is especially dangerous here.
5. Only after the above, revisit fusion — and if you do, add regularization (higher dropout / weight decay / smaller hidden dim).

**Honest overall assessment:** after Stages 1-5, attention topology alone does not predict ViT failure as well as the softmax confidence already does, and it does not add to it. The project's remaining justification rests on the confident-failure subset (item 2) and on untested node features (item 1). If both come up empty, the negative result is the finding.

---

## 9. File Map

Code:
- [lightweight_attention_experiments.py](../legacy/lightweight_attention_experiments.py) — **the current runner.** Stages 1-5.
- [poc_gnn_vit_cifar100.py](../legacy/poc_gnn_vit_cifar100.py) — original POC. Still imported by the new runner for data loading, ViT scan, splits, and model loading helpers. Do not delete.
- [mlp_flat_attention_baseline.py](../legacy/mlp_flat_attention_baseline.py) — pre-existing baseline, not used in recent work.

Results added in session 2:
- `reports_edge_ablation_{50,100,200,500}/` — edge-count ablation (Section 5.6). `reports_edge_ablation_100/` is also the canonical seed-7 Stage 5 result (Section 5.5).
- `reports_seed_{1,2}/` — multi-seed replication (Section 5.7).
- `reports_lightweight_top100/stage5_*` — the first Stage 5 run. **Its `combined_auroc` of 0.8557 is from the pre-fix buggy combiner; superseded by `reports_edge_ablation_100/`.**

Caches (delete only if you want to re-extract; requires network):
- `reports_vit_scan_test/vit_scan_with_indices.json` — ViT correctness over all 10k images.
- `reports_lightweight_top100/cache/top_global_{50,100,200,500}_seed7_graphs.pt` plus `top_global_100_seed{1,2}_graphs.pt` — sparse graphs, 1,700 rows each. **~3 GB total after session 2.** All are regenerable; delete the non-100/seed7 ones first if you need disk.
- `reports_lightweight_top100/experiment_splits_train1500_test200_seed7.json` — frozen split indices. Loaded regardless of `--seed`, so the test set stays fixed across seeds.

Results (current pipeline, all in `reports_lightweight_top100/`):
- `stage1_sequence_debug_report.md`, `stage1_sanity_results.json`
- `stage2_layer_sweep.csv`, `stage2_layer_sweep_report.md`
- `stage3_pooling_comparison.csv`, `stage3_pooling_report.md`
- `stage4_layer_fusion_report.md` (final-four), `stage4_final_2_fusion_report.md` (final-two)
- `experiment_config.json`

Results (older POC, different sparsification — `reports_balanced_750_100_*`):
- `reports_balanced_750_100_transformer/` — dense last-layer: 0.8286 minimal, 0.8681 minimal+repr
- `reports_balanced_750_100_all_layers_transformer_minimal/` — one huge graph: 0.5452
- `reports_balanced_750_100_layer_sequence_minimal/` — old sequence: 0.5000

---

## 10. Known Cosmetic Bugs (harmless, but don't be confused)

- Stage 4 run names are hardcoded to `stage4_final_four_cls_gated_layer_fusion` even when `--fusion-layer-count 2`. The report filename and the `selected_layers_one_based` column are correct (`11-12`), only the internal `name` field is wrong.
- `stage2_layer_sweep_report.md` prints "Best validation layer" and a `Val AUROC` column of all `None`. Validation was removed; the line is meaningless.
- `train_model()` still accepts a `val_dataset` argument and reports a `val` block. When passed `[]` it is skipped and `final_val` is `{}`. Early stopping and best-checkpoint restore are inert without validation — models are always the final-epoch weights.
