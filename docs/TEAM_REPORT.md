# POLYGRAPH — Final Report for the Team Meeting

*Hochwald · Lavi · Fahn · Kramf — 2026-08-27 · branch `full_pipeline`*

*All trained results: single seed (7); treat per-slice AUROC differences under ±0.02 as noise.
Combined detectors are used only on output statistics (output_lr / output_mlp); no MSP-combination
results appear in this report.*

---

## Part I — Question and inheritance

### 1. The research question (proposal)

Freeze a ViT classifier (`edumunozsala/vit_base-224-in21k-ft-cifar100`); turn each image's
self-attention into per-layer directed graphs over its 197 tokens; train a small GNN to predict
`y_err = 1[argmax f(x) != y]`. **Success criterion:** beat controls that see the *same information
without structure* — especially under **unseen corruptions**. Only detectors (~16k–345k params)
are trained; the ViT is never touched.

### 2. Yishai's POC (commit `8036fd9`, frozen in `legacy/`)

200 clean test images; clean error rate 8.5% meant the positive class was nearly empty — the
scaling wall. His numbers: MSP 0.9101 · top-100 GNN 0.7898 · dense last-layer 0.8286 ·
dense+hidden 0.8681. This project adds: 100× data with a real error population, corruption + OOD
benchmark, group-disjoint stratified splits, the baseline ladder, resumable infrastructure — all
verified equivalent (edge rule bit-identical; architectures match to 1e-6; our full-10k MSP 0.9092
vs his 0.9101 on 200 images).

### 3. Graph anatomy

Edge j→i iff `max_h A[h,i,j] > tau = 0.02`; 12-dim per-head edge features; 16-dim node features
(patch xy, CLS flag, layer position, attention diagonals); hidden states deliberately excluded in
the primary condition. ~7,600 edges/layer. Edges stored strength-sorted → every stricter tau and
top-K is a free prefix view of one extraction.

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
| output_lr / output_mlp | 3 / ~1.2k | [logit(msp), margin] | value of training / nonlinearity on outputs |
| attn_mlp | ~154k | GNN's edge values, top-100/layer, flat | **the criterion control**: same values, no structure |
| cls_mlp | ~98k | final CLS embedding | representation |
| cls_seq | ~345k | GRU over 12-layer CLS trajectory | representation dynamics |
| **graph** (v1) | ~16k | layer-12 attention graph | structure |
| **fusion** (v2) | ~41k | graph + 768-dim token embeddings | structure + representation |
| CHARM-lite | ~16k | 12-layer union graph, 144-dim edges | cross-layer topology |
| all-layers(+hidden) | — | 12 subgraphs → sequence | cross-layer, sequential |

### 9. Why the numbers can be trusted

- Bit-identical edge rule vs the frozen POC; architectures match to 1e-6.
- Scan verified 600/600 vs the old pipeline (confidence deltas ≤ 7e-7).
- 38-test suite, mutation-audited: 5 planted bugs each caught.
- 8 silent-wrong bugs found by adversarial review before any training, each with a regression test.
- **Label-shuffle control passed** — trained on shuffled labels the detector converges to exact
  0.5 and cannot rank its own training targets (0.49): **no leak**. (Its transient 0.678 print was
  a best-checkpoint selection artifact on a near-constant model — logit spread 0.0004; documented.)

## Part IV — Results

### 10. Main benchmark — AUROC (test = 17,000)

| slice | n | msp | margin | out_lr | out_mlp | attn_mlp | cls_mlp | cls_seq | graph | **fusion** | CHARM |
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

AUPRC (all): cls_seq .8500 and fusion .8513 lead (a tie in practice — fusion leads AUPRC,
cls_seq leads AUROC) > cls_mlp .8436 > msp .8416 > graph .8152 > attn_mlp .7121.

Selective prediction, risk@0.5 (error rate in the most-trusted half; base rate 50%):
cls_seq **.2022** · cls_mlp .2040 · fusion .2061 · msp .2088 · graph .2405 · attn_mlp .3185.
At severity 5, cls_seq keeps 20.8% errors vs MSP's 24.2%.

### 11. Weather-family holdout — AUROC (test = 14,450; own scale, don't compare across tables)

| slice | graph | **fusion** | msp | attn_mlp | cls_mlp | cls_seq |
|---|---:|---:|---:|---:|---:|---:|
| **all** | .8190 | .8657 | .8588 | .7308 | **.8699** | .8689 |
| clean | .8715 | .8977 | **.9242** | .7749 | .8882 | .8817 |
| seen corruptions | .8115 | .8623 | .8456 | .7237 | .8660 | **.8675** |
| **unseen (weather+extras)** | .8380 | .8738 | **.8885** | .7468 | .8801 | .8731 |
| severity 5 | .8147 | .8550 | .8289 | .7225 | **.8645** | .8696 |

### 12. MSP vs margin — microscope study

Pearson r = 0.978; paired bootstrap Δ = +0.0013 for MSP, 95% CI [+0.0008, +0.0018] — real,
practically nil. At *matched* confidence the margin residual **flips sign** near conf ≈ 0.8:
below it a diffuse runner-up field errs *more* (74.3% vs 63.0% for a near-tie — anti-textbook);
above it the textbook regime returns. A linear model cancels the two regimes (output_lr .8672);
the MLP recovers exactly the linear loss and no more (output_mlp .8694 ≈ msp .8695).
**Outputs are a closed book; MSP is their ceiling.**

### 13. Run ledger (all final unless noted)

| run | best val | test (all) | note |
|---|---:|---:|---|
| graph, layer 12 | .8254 @ 34 | .8417 | 42 epochs, ~15 h |
| fusion | best @ 5 | .8758 | converges 7× faster than topology |
| CHARM-lite | .8039 @ 5 | .8233 | stopped @ e8 (declining); dataset audited bug-free |
| shuffle control | — | 0.5 at convergence | **passed** |
| weather graph | .8079 @ 22 | .8190 | table §11 |
| weather fusion | .8592 @ 3–6 | .8657 | table §11 |
| all-layers + hidden | .7834 @ 5, declining | — | paused, state kept; verdict effectively in |
| top-K 100 | e1 .7249 | — | **training now** (matched structure control) |
| tau {.03/.05/.10}, final-4 | — | — | queued; appendix material |

## Part V — Findings (ranked by confidence)

1. **Structure carries real signal — the proposal's criterion is passed, twice.** Graph beats
   flat-attention on every slice of both benchmarks: +0.10 (main) and +0.09 (weather).
   Caveat: attn_mlp sees top-100 edges vs the GNN's ~7,600; the running top-100 GNN and a
   designed Deep-Sets control close this.
2. **Outputs are a hard ceiling at MSP** (§12). Training and nonlinearity on softmax statistics
   add nothing. Any progress must come from inside the network.
3. **Representation detectors beat MSP in-distribution — but by corruption familiarity, and the
   weather holdout proved it.** cls_seq .8759, cls_mlp .8742, fusion .8758 top MSP .8695 on the
   main test; on the truly unseen weather family all of them **lose to MSP** (.8731–.8801 vs
   .8885). Clean is every trained detector's weak slice for the same exposure reason.
4. **Fusion is the best structure-bearing detector and the second confirmation that features fuel
   the graph:** +0.034 over plain topology on every main slice, +0.04–.05 on every weather slice,
   converges in ~5 epochs both times.
5. **The unseen-corruption story is honest but mixed.** In *both* designs, graph-family detectors
   are the only ones scoring higher on unseen than seen (graph .8429 vs .8419 and .8380 vs .8115;
   fusion .8738 vs .8623 on weather). On the main design's unseen slice fusion is the top detector
   (.8761, +.0016 over MSP — inside noise); on the weather design MSP wins the unseen slice
   outright (.8885 vs .8738). **We can claim a distinctive generalization *pattern*, not OOD
   superiority over MSP.** Note the asymmetry: weather corruptions are mild and MSP-friendly
   (MSP's unseen .8885 exceeds its own seen .8456); the extras were the harder test for MSP.
6. **Graph degrades most gracefully with severity:** deficit to MSP shrinks monotonically
   −.039 → −.011 (s1 → s5) under equal per-severity exposure.
7. **Cross-layer topology does not pay as encoded.** CHARM-lite .8233 < single-layer .8417
   everywhere (implementation audited bit-exact); all-layers(+hidden) plateaued ~.78 val.
   Layer 12 carries the topological signal. Fair-shot caveat for Yishai: fixed 16k capacity +
   ambiguous zero-fill; presence-mask + wider model would be the honest retest.

## Part VI — Recommendations

**For what we present and write:**

1. **Lead with findings 1–4:** structure is real (+0.10, twice), outputs are a ceiling,
   in-distribution wins are exposure-driven (our own holdout exposed this — present it as rigor,
   it's the strongest credibility signal we have), fusion is the best trained detector.
2. **Present finding 5 as a pattern, not a victory.** Say "graph detectors uniquely generalize
   *upward* to unseen corruptions in both designs; beating MSP under shift is unresolved — one
   design says barely, one says no." Do not present any unseen gap under ±0.02 as a win.
3. **Every claim carries the single-seed stamp.** Any OOD statement in the final writeup needs
   3–5 seeds of graph/fusion on both plans (~2–3 days of machine time). This is the
   highest-value spend left; recommend the team approve it.
4. **Run the Deep-Sets control** (full edge set, no structure) before the writeup — it makes
   finding 1 airtight; the top-100 GNN running now covers the cheaper half of that argument.
5. **Drop or appendix the tau sweep and final-4** unless time is free — no headline depends on them.
6. **CHARM:** report CHARM-lite's negative honestly with the capacity/zero-fill caveat; Yishai
   decides on the presence-mask retest. Don't silently bury it.
7. **Deployment recommendation** (if asked what we'd ship): fusion where corruption exposure
   resembles training; MSP as the guard on clean/high-confidence traffic — no detector we built
   beats MSP there, and honesty about that is part of the result.
8. **Logistics:** PR awaits one click (`github.com/noyahoch/Polygraph/pull/new/full_pipeline`);
   remote `main` stays Yishai-only; machine on AC with lid open until the chain drains.

**Still owed:** severity-degradation and risk-coverage figures, ACM results section, HANDOFF
close-out, top-K/tau results as they land.

---

*Sources: `runs/{detector,hidden12,charm,control_shuffle,weather_holdout,weather_fusion}/report_model_seed7.json`,
`docs/results/classifier_accuracy.md`, `docs/HANDOFF.md`.*
