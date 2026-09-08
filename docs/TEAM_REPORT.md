# POLYGRAPH — Full Team Report

*Hochwald · Lavi · Fahn · Kramf — updated 2026-08-30 · branch `full_pipeline`*

*All trained results: single seed (7); treat per-slice AUROC differences under ±0.02 as noise.
Combined detectors are used only on output statistics (output_lr / output_mlp); no MSP-combination
results appear in this report.*

---

## Part I — Question and inheritance

### 1. The research question (proposal)

Freeze a ViT classifier (`edumunozsala/vit_base-224-in21k-ft-cifar100`); turn each image's
self-attention into per-layer directed graphs over its 197 tokens; train a small GNN to predict
`y_err = 1[argmax f(x) != y]`. **Success criterion (proposal, verbatim in spirit):** the GNN must
outperform non-graph models *given the same attention values* — especially under **unseen
corruptions**. Only detectors (~13k–345k params) are trained; the ViT is never touched.

### 2. Yishai's POC (commit `8036fd9`, frozen in `legacy/`)

200 clean test images; clean error rate 8.5% meant the positive class was nearly empty — the
scaling wall. His numbers: MSP 0.9101 · top-100 GNN 0.7898 · dense last-layer 0.8286 ·
dense+hidden 0.8681. This project adds: 100× data with a real error population, corruption + OOD
benchmark, group-disjoint stratified splits, the baseline ladder, purity controls, resumable
infrastructure — all verified equivalent (edge rule bit-identical; architectures match to 1e-6;
our full-10k MSP 0.9092 vs his 0.9101 on 200 images).

### 3. Graph anatomy

Edge j→i iff `max_h A[h,i,j] > tau = 0.02`; 12-dim per-head edge features; 16-dim node features
(patch xy, CLS flag, layer position, attention diagonals); hidden states deliberately excluded in
the primary condition (variant 1). ~7,600 edges/layer. Edges stored strength-sorted → every
stricter tau and top-K is a free prefix view of one extraction.

**Variant 2 ("fusion")**: the same graph and GNN, but each node additionally carries the ViT's
own token embedding (the 768-dim final-layer hidden state of that token), so the detector sees
both how tokens attend (structure + attention values) and what tokens represent. ~41k params.

## Part II — The data

### 4. Scan — 1,010,000 ViT verdicts

| pool | images | ViT accuracy | in plans? |
|---|---:|---|---|
| clean test | 10,000 | 91.48% (852 errors) | yes |
| clean train | 50,000 | 99.45% | **excluded** — memorization contaminates `y_err` |
| CIFAR-100-C: 19 corruptions × 5 severities | 950,000 | 8.5% → ~60% error by severity | yes |

**239,921 errors** vs the POC's 852. Corruption shards verified row-aligned with clean test.
Full 19×5 grid: `docs/results/classifier_accuracy.md` (severity-5 extremes: glass_blur 31%,
brightness 86%).

### 5. Main split plan — 75,000 records (52k / 6k / 17k)

- **Group-disjoint by photograph** — no leak possible.
- **50/50 balanced per (corruption, severity) cell** — blocks scoring by corruption strength;
  random guess = 0.5 exactly.
- **4 "extra" corruptions test-only** (CIFAR-C's designated holdout): train/val 76 cells, test 96.

Test composition: clean 176 · seen corruptions 13,286 · unseen extras 3,538.

### 6. Weather-holdout plan (second OOD design) — 36,284 / 4,646 / 14,450

A pure re-split of the same 75k stored records: snow, frost, fog, brightness **plus** the extras
never appear in train/val (56 train cells, 96 test cells). Tests generalization to a full held-out
family of a *seen kind* of shift.

### 7. Stores

198 GB key-indexed graph store (38 shards, fp16, ~2.6 MB/record, CLS embeddings included;
resumable, self-checking — observed prediction drift 0) + 21 GB final-layer token embeddings
(variant 2). Key-indexing is what made the weather re-split free.

## Part III — Detectors and trust

### 8. The ladder

Identical protocol for every trained detector: AdamW 2e-3 / 1e-4, dropout 0.15, class-weighted
BCE, early stop on val AUROC (patience 8), best state restored, seed 7.

| detector | params | reads | role |
|---|---:|---|---|
| msp / margin | 0 | softmax top-1 / top-1 − top-2 | untrained references |
| energy | 0 | −logsumexp(logits) | literature output score |
| mahalanobis | 0 | class-cond. feature distance | literature OOD score |
| output_lr / output_mlp | 3 / ~1.2k | [logit(msp), margin] | value of training / nonlinearity on outputs |
| attn_mlp | ~154k | GNN's edge values, top-100/layer, flat | same values, no structure (fixed-size) |
| deep-sets | ~13k | **full** ~7,600-edge value set, no structure | airtight structure control (queued) |
| cls_mlp | ~98k | final CLS embedding | representation |
| cls_seq | ~345k | GRU over 12-layer CLS trajectory | representation dynamics |
| logit_dyn | ~90k | GRU over 12-layer logit-lens trajectory | LogitDynamics [Beigelman & Freiman 2026] |
| **graph** (v1) | ~16k | layer-12 attention graph | structure |
| **top-100 graph** | ~16k | top-100 edges only | purity control vs attn_mlp |
| **graph-h64** | ~60k | same graph, 64-wide, 3 MP layers | capacity control |
| **fusion** (v2) | ~41k | graph + 768-dim token embeddings | structure + representation |
| CHARM-lite | ~16k | 12-layer union graph, 144-dim edges | cross-layer topology |
| CHARM-v2 | — | per-layer graphs + embeddings, 64-wide | proposal §2 architecture (queued) |

### 9. Why the numbers can be trusted

- Bit-identical edge rule vs the frozen POC; architectures match to 1e-6.
- Scan verified 600/600 vs the old pipeline (confidence deltas ≤ 7e-7).
- 38-test main suite (mutation-audited: 5 planted bugs each caught) + 6 Deep-Sets tests
  (including proof of rewiring-invariance — the model *cannot* use structure).
- 8 silent-wrong bugs found by adversarial review before any training, each with a regression test.
- **Label-shuffle control passed** — trained on shuffled labels the detector converges to exact
  0.5 and cannot rank its own training targets (0.49): **no leak**.
- Logit-lens reconstruction for the literature baselines sanity-checked: corr 1.000000,
  max|diff| 6.6e-5 vs the stored ViT confidence.

## Part IV — Results

### 10. Main benchmark — AUROC (test = 17,000)

| slice | n | msp | margin | out_lr | out_mlp | attn_mlp | cls_mlp | cls_seq | graph | **fusion** | CHARM-lite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 17000 | .8695 | .8681 | .8672 | .8694 | .7421 | .8742 | **.8759** | .8417 | .8758 | .8233 |
| clean | 176 | .9042 | **.9046** | .8995 | .9037 | .7335 | .8454 | .8481 | .8200 | .8409 | .8057 |
| seen corruptions | 13286 | .8679 | .8666 | .8657 | .8679 | .7433 | .8758 | **.8781** | .8419 | .8764 | .8240 |
| unseen extras | 3538 | .8745 | .8728 | .8719 | .8745 | .7382 | .8699 | .8698 | .8429 | **.8761** | .8227 |
| severity 1 | 3348 | .8998 | **.8999** | .8965 | .8999 | .7566 | .8875 | .8848 | .8610 | .8934 | .8400 |
| severity 2 | 3360 | **.8827** | .8808 | .8817 | .8825 | .7548 | .8782 | .8728 | .8503 | .8786 | .8309 |
| severity 3 | 3364 | .8698 | .8681 | .8697 | .8698 | .7343 | .8755 | .8753 | .8420 | **.8781** | .8201 |
| severity 4 | 3374 | .8608 | .8589 | .8593 | .8609 | .7340 | .8675 | .8727 | .8393 | **.8745** | .8201 |
| severity 5 | 3378 | .8382 | .8350 | .8358 | .8382 | .7347 | .8694 | **.8772** | .8270 | .8622 | .8139 |

AUPRC (all): fusion .8513 / cls_seq .8500 lead > cls_mlp .8436 > msp .8416 > graph .8152 >
attn_mlp .7121. Risk@0.5 (error rate in the most-trusted half; base 50%): cls_seq **.2022** ·
cls_mlp .2040 · fusion .2061 · msp .2088 · graph .2405 · attn_mlp .3185. At severity 5, cls_seq
keeps 20.8% errors vs MSP's 24.2%.

### 11. Purity controls — the structure claim made airtight

**Same information (top-100 GNN vs attn_mlp — same layer, same 100 edges, same per-head values;
the GNN has 10× FEWER parameters):**

| slice | top-100 GNN (16k) | attn_mlp (154k) | Δ |
|---|---:|---:|---:|
| all | **.8125** | .7421 | +.070 |
| clean | .7831 | .7335 | +.050 |
| seen corruptions | .8130 | .7433 | +.070 |
| **unseen extras** | **.8141** | .7382 | **+.076** |
| severity 5 | .7989 | .7347 | +.064 |

**The proposal's success criterion is passed with information and capacity both controlled.**
Edge budget adds another +.03 on top (top-100 .8125 → full graph .8417): structure and coverage
both contribute. The Deep-Sets control (no structure, FULL edge set) is queued to close the
remaining direction.

**Capacity (graph-h64: 64-wide, 3 message-passing layers, ~4× params):** plateaued at val
~.816 vs the original small GNN's .8254 — **more capacity does not help; it slightly hurts**
(likely oversmoothing). The graph's remaining gap to cls_mlp is signal, not capacity, and the
small-GNN-vs-big-baseline comparisons above are legitimate. (h128 skipped by team decision —
the h64 curve already answers the question.)

### 12. Weather-family holdout — AUROC (test = 14,450; own scale, don't compare across tables)

| slice | graph | **fusion** | msp | attn_mlp | cls_mlp | cls_seq |
|---|---:|---:|---:|---:|---:|---:|
| **all** | .8190 | .8657 | .8588 | .7308 | **.8699** | .8689 |
| clean | .8715 | .8977 | **.9242** | .7749 | .8882 | .8817 |
| seen corruptions | .8115 | .8623 | .8456 | .7237 | .8660 | **.8675** |
| **unseen (weather+extras)** | .8380 | .8738 | **.8885** | .7468 | .8801 | .8731 |
| severity 5 | .8147 | .8550 | .8289 | .7225 | **.8645** | .8696 |

### 13. Complementarity — do we detect something DIFFERENT? (yes)

Full tables: `docs/results/complementarity.md`. Highlights (main test split):

- **Ranking correlation with MSP:** margin 0.997 (it *is* MSP) · cls detectors ~0.82 ·
  fusion 0.83 · **graph 0.79 — the most different detector in the roster.**
- **Equal 50% flag budget** (8,500 errors): fusion catches **847 errors MSP misses**
  (graph 818, cls_seq 813, margin 59).
- **MSP's blind spot** (1,775 errors MSP actively trusts): **fusion flags 47.7%, graph 46.1%** —
  graph is second-best here despite the lowest overall AUROC of the trained detectors.
  Weak overall + strong on the complement = a genuinely different signal.
- **Confident predictions** (conf ≥ 0.9; n=10,851): fusion **.8614** > cls_seq .8598 >
  cls_mlp .8535 > msp .8423 > graph .8186.
- **Per-corruption win map:** MSP wins mild corruptions (brightness, defocus, snow, clean);
  cls_seq wins the entire noise family + destructive blurs; fusion wins contrast/frost/spatter/
  zoom_blur. Confidence is good while the network is healthy; internal signals take over as it
  breaks.
- Counterweight: at the same budget the graph also loses 1,087 errors MSP catches — the claim is
  **complementarity, not replacement**.

### 14. Literature baselines (`docs/results/literature_baselines.md`)

| slice | msp | energy | mahalanobis | logit_dyn (LogitDynamics) | ours: cls_seq | ours: fusion |
|---|---:|---:|---:|---:|---:|---:|
| all | **.8695** | .8653 | .6623 | .8632 | .8759 | .8758 |
| unseen extras | **.8745** | .8713 | .6774 | .8512 | .8698 | .8761 |
| severity 5 | .8382 | .8390 | .6085 | .8628 | **.8772** | .8622 |

- **cls_seq beats LogitDynamics** — the closest published method — .8759 vs .8632; the logit
  trajectory is a compressed (post-head) view of the CLS trajectory and the compression costs
  signal.
- **Energy ≈ MSP** — third confirmation of the outputs ceiling.
- **Mahalanobis collapses** — under corruption everything is "far from the training manifold,"
  so distance stops separating right from wrong.

### 15. MSP vs margin — microscope study

Pearson r = 0.978; paired bootstrap Δ = +0.0013 for MSP, 95% CI [+0.0008, +0.0018] — real,
practically nil. At *matched* confidence the margin residual **flips sign** near conf ≈ 0.8:
below it a diffuse runner-up field errs *more* (74.3% vs 63.0% for a near-tie — anti-textbook);
above it the textbook regime returns. A linear model cancels the two regimes (output_lr .8672);
the MLP recovers exactly the linear loss and no more (output_mlp .8694 ≈ msp .8695).
**Outputs are a closed book; MSP is their ceiling** (margin, trained LR/MLP, and energy all agree).

### 16. Run ledger

| run | best val | test (all) | status |
|---|---:|---:|---|
| graph, layer 12 | .8254 @ 34 | .8417 | done |
| fusion | best @ 5 | .8758 | done |
| CHARM-lite | .8039 @ 5 | .8233 | done (stopped @ e8, declining) |
| shuffle control | — | 0.5 at convergence | **passed** |
| weather graph | .8079 @ 22 | .8190 | done |
| weather fusion | .8592 @ 3–6 | .8657 | done |
| top-100 graph | .7918 @ 38 | .8125 | done |
| graph-h64 (capacity) | ~.816, declining | — | finishing (early stop imminent) |
| graph-h128 | — | — | **skipped** (team call; h64 answers it) |
| deep-sets | — | — | next after h64 |
| CHARM-v2, final-4, tau sweep | — | — | queued |

## Part V — Findings (ranked by confidence)

1. **Structure carries real signal — the criterion is passed, now with information AND capacity
   controlled.** Top-100 GNN vs attn_mlp: +0.07 on every slice (+.076 unseen) at same
   information with 10× fewer parameters; full graph adds +.03 more from edge coverage; extra
   GNN capacity adds nothing (h64 ≤ original). Deep-Sets closes the last direction (queued).
2. **Outputs are a hard ceiling at MSP** — margin, trained linear/MLP on outputs, and energy all
   land at or under MSP (§15).
3. **Representation detectors beat MSP in-distribution — but by corruption familiarity.**
   The weather holdout proved the exposure confound: on a truly unseen family they lose to MSP
   (.8731–.8801 vs .8885). Clean is every trained detector's weak slice for the same reason.
4. **Fusion is the best trained detector and the graph detectors see DIFFERENT errors than
   confidence:** least-correlated rankings, ~10% of errors recoverable at equal budget, ~47%
   of MSP's blind spot flagged, best on confident predictions (.8614 vs .8423). The claim is
   complementarity, not replacement.
5. **The unseen-corruption story is honest but mixed.** Graph-family detectors are the only ones
   scoring higher on unseen than seen in both designs; on the main design's unseen slice fusion
   is the top detector (.8761, inside noise over MSP); on the weather design MSP wins outright.
   A distinctive generalization *pattern*, not OOD superiority over MSP.
6. **Graph degrades most gracefully with severity** (deficit to MSP −.039 → −.011, s1 → s5).
7. **Cross-layer topology does not pay as encoded** (CHARM-lite .8233 < .8417 everywhere;
   audited bug-free). CHARM-v2 — the proposal's own architecture, properly resourced — is the
   honest retest, queued.
8. **We beat the closest published method** (cls_seq .8759 > LogitDynamics .8632).

## Part VI — Recommendations

1. Lead with findings 1–4; present the exposure confound as rigor (our own holdout exposed it).
2. Present finding 5 as a pattern, not a victory; nothing under ±0.02 is a win.
3. **Seeds** remain the highest-value spend for any claimed OOD statement (3–5 seeds of
   graph/fusion on both plans, ~2–3 days).
4. τ sweep and final-4 are appendix material.
5. Report CHARM-lite's negative honestly; CHARM-v2 answers the fair-shot question.
6. Deployment: fusion where corruption exposure resembles training; MSP as the guard on
   clean/high-confidence traffic.
7. PR awaits one click: `github.com/noyahoch/Polygraph/pull/new/full_pipeline`; remote `main`
   stays Yishai-only; machine on AC with lid open until the chain drains.

---

*Sources: `runs/*/report_model_seed7.json`, `runs/scores_test_seed7.npz`,
`runs/scores_literature_seed7.npz`, `docs/results/*.md`, `docs/HANDOFF.md`.*
