# Polygraph — Consolidated State

*Updated 2026-09-07 · branch `full_pipeline`. Single source of truth for the team's P2/P3
branch decision. Detailed numbers live in `docs/TEAM_REPORT.md`, `docs/results/*.md`.
Steps 0–1 (pre-registration §8), 3, 4 are DONE. Step 2 (routing test) is IN PROGRESS —
building the within-group testbed; see §6 and the live status in §9.*

> **Caveat on every CI below:** bootstrap over test samples at a **single training seed** —
> captures test-set sampling variance, not seed-to-seed training variance (~±0.02 AUROC).
> Multi-seed deferred until a headline is chosen (G5).

---

## 1. Verdict in a paragraph

The corruption benchmark and its purity controls are solid and honest: **attention structure
beats the same information without structure** (+0.07–0.10, confirmed from both the
same-information and full-edge-set directions, capacity-controlled). But the exploitation
numbers (G1) show the **pure graph is near-redundant given representations** — its marginal
contribution over MSP+representation is +0.0016 [+0.0008, +0.0024], statistically real and
practically nil. The one live path to a *structure* win is a task where failure is caused by
**routing, not state** (a shortcut/backdoor). The confident-error finding is real and
cross-domain but is a **representation** result, and the LLM version of it was just published
by others. Three viable paths remain (§5), and the branch between them is a team decision on
the numbers below.

## 2. What is solid (corruption benchmark)

- 1.01M-verdict scan, 75k stratified group-disjoint plan, 198GB key-indexed graph store +
  21GB hidden states. Leak-free (label-shuffle → 0.51 over 30 seeds). Mutation-audited
  (38+11+11 tests).
- **Structure > same-information baselines:** top-100 GNN beats attn_mlp by +0.07 on every
  slice (same layer, same edges, 10× fewer params); full graph 0.8417 vs Deep-Sets (full
  edge set, no structure) 0.8127. Capacity doesn't help (h64 0.835 < original 0.842).
- Representation detectors (cls_seq 0.876, fusion 0.876) beat MSP (0.870) in-distribution
  but lose to it on held-out corruption families (exposure confound, proven on the weather
  holdout).
- Literature: our cls_seq beats LogitDynamics (0.876 vs 0.863); energy confirms the outputs
  ceiling; Mahalanobis collapses.

## 3. The honest ceiling (G1 exploitation — `docs/results/exploitation_g1.md`)

| question | number | reading |
|---|---|---|
| graph marginal over msp+cls_seq (corruptions) | **+0.0016 [+0.0008, +0.0024]** | near-redundant |
| fusion marginal over msp+cls_seq | +0.0066 [+0.0052, +0.0080] | the gain is hidden states, not topology |
| POPE combined (output+probe) vs output | **+0.0243 [+0.003, +0.048]** | internal signal has exploitable value... |
| POPE confident-slice, probe vs output | +0.152 [+0.027, +0.279] | ...concentrated where output is blind |

## 4. The confident-error law, refined (cross-domain, but a representation result)

Internal signals beat output confidence **specifically on confident errors**, in both
domains, scaling with how decoupled confidence is from correctness:
- corruption (fusion vs MSP, confident slice): +0.019 [+0.012, +0.026]
- POPE (probe vs output, confident slice): +0.152 [+0.027, +0.279]

**Refinement from the y_hall slice (corrects our starting intuition):** the probe's edge is
NOT on hallucinations. On absent-object hallucinations the OUTPUT wins (0.878 vs 0.695) — the
model is *less* confident when it hallucinates. The probe wins on **missed present objects**
(0.802 vs 0.689) and on the confident-error slice generally. The signal is "confident errors
the model has no output-level doubt about," not "hallucinations."

**The heart of P1 (Step 4, category-holdout shift test):** the probe's BROAD advantage is largely familiarity — it collapses −0.115 under object-category shift, the same exposure confound that sank representation detectors on the corruption weather holdout. But the CONFIDENT-error law SURVIVES category shift (+0.19 [+0.044,+0.331] on held-out categories). So the real, shift-robust internal-signal advantage lives ONLY in the confident slice — that sharpened claim, not the broad one, is P1's headline.

**Collision (G4, first-pass search — full lit gate still mandatory):** "Wrong With
Conviction" (2026) publishes this law for LLMs with theory; several 2026 VLM-probe papers
crowd the space. Our confident-error observation must be positioned as *confirming and
extending*, not discovering. What remains ours: the **regime comparison** (label-internal vs
label-external, headroom as explanatory variable), the **structure-vs-state** question under
purity controls, and the **headroom protocol** itself.

## 5. The three paths and the decision tree

- **P1 — cross-regime confident-error study.** Backbone: the same law across a regime where
  outputs are a structural ceiling (corruptions) and where they are not (VLM). Downgraded
  from headline to component post-collision. Conditions met: G1 exploitation positive. Needs:
  G5 statistics, lit-gate positioning.
- **P2 — the routing paper (the graph's remaining shot).** Hinges on structure beating a
  probe where failure is routing-caused. Synthetic 95%-spurious failed by design (§6);
  now testing at 100% correlation (backdoor) and prepared for natural benchmarks
  (Waterbirds/CelebA). Protocol fixed per G3 (within-group error-vs-correct only; graph gets
  its τ/layer sweep before any negative counts).
- **P3 — the adjudication/measurement paper.** "Which internal signal (output/state/
  structure) pays where (regime, confidence slice, seen/unseen)?" We own the instrument
  (store + purity-controlled ladder + controls + headroom protocol). Available regardless of
  how the routing test lands; no collision can take it.

**Decision tree:** routing test (G3-fixed) → graph beats probe: **P2 headline**, P1 inside,
P3 as framing. → graph loses under swept config + clean control: **P3 headline** (the loss is
a finding), P1 inside. Either branch is a paper.

## 6. Experiment status

| item | status |
|---|---|
| Corruption benchmark + purity controls + complementarity + literature | DONE, committed, pushed |
| G1 exploitation numbers (corruptions + POPE) | DONE (`exploitation_g1.md`) |
| POPE probe pilot (n=3000) + y_hall slice | DONE |
| POPE within-distribution generalization (group-disjoint by image) | DONE — confident-error law holds on unseen images: **+0.199 [+0.129,+0.266]** confident, +0.004 flat overall |
| POPE **shift** guard (category holdout) | DONE (`step4_category_holdout.md`) — probe's OVERALL edge is familiarity (collapses −0.115 under category shift), but the CONFIDENT-error law SURVIVES (+0.19 [+0.044,+0.331] on held-out categories) |
| Backdoor models A / B′ (100% ASR, fixed target, distinct triggers) | DONE — for the INPUT-LEVEL trigger-detection transfer test (`backdoor_testbed.md`) |
| Rotating within-group testbed (amendments A1, A2) | **IN PROGRESS** — see §9. Fixed target collapses "hijacked≡class T" (probe reads it trivially); rotating y→y+1 forces a visual-vs-prediction mismatch. Gate is a usable resisted population (≥500, spread), NOT balanced ASR |
| Step 2 ladder (graph-vs-probe within-group) + localization + transfer | NOT YET RUN — gated on the resisted-count check now running |
| Synthetic 95%-spurious | ABANDONED (design flaw, not fundamental — see exploitation doc §"Correction") |
| Waterbirds + CelebA-blond routing pilots | PREPARED, NOT launched (`pilots/waterbirds/PLAN.md`) — gated on the backdoor result |
| Polygraph tail (CHARM-v2, final-4, τ sweep) | PAUSED (appendix material) |
| G5 multi-seed (headline slices only) | DEFERRED until headline chosen |
| G4 full literature gate | MANDATORY before any writing; seeded by 4 papers in TEAM_REPORT |

## 7. Status against the Routing Matrix phase plan

1. **Steps 0–1 (pre-registration §8):** DONE, frozen before any routing result.
2. **Step 3 (unseen-slice lookup):** DONE — clean negative (`step3_unseen.md`).
3. **Step 4 (POPE category-holdout):** DONE — shift guard (`step4_category_holdout.md`).
4. **Step 2 (backdoor routing test):** IN PROGRESS — testbed calibration (§9). Once the
   resisted-count gate passes: extraction → ladder (output/guard/probe/swept-graph/fusion,
   within-group hijacked-vs-resisted) → difficulty-confound control → cross-trigger transfer,
   all into `routing_backdoor.md` with §8 criteria marked pass/fail.
5. **Team meeting after Step 2**, on this doc; team makes the P2-variant/P3 call on a matrix
   declared before its numbers existed.

---

## 8. PRE-REGISTRATION — The Routing Matrix (frozen 2026-09-03, before any routing result)

**Standing rule: no criterion, threshold, or task list below may be edited after the
corresponding result exists. Every task in the matrix is reported, win or lose.**

### 8.1 The claim under test

> Attention topology helps failure detection **when the failure is caused by routing** (the
> model attended to the wrong evidence), and is near-redundant given representations **when
> the failure is caused by state** (the model attended correctly but concluded wrongly).

A boundary claim: a 2-column matrix; both columns required.

### 8.2 The task matrix (fixed now; all cells reported)

| Task | Failure cause | Status |
|---|---|---|
| CIFAR-100-C corruptions | state | DONE — graph marginal +0.0016 (negative cell, held) |
| POPE / LLaVA | state (per y_hall refinement) | DONE — probe win; shift axis pending (Step 4) |
| Backdoor triggers (synthetic routing, GT mask) | routing | testbed validated; test = Step 2 |
| Waterbirds | routing (natural) | prepared; gated on Step 2 |
| CelebA-blond | routing (natural, standard pairing) | prep alongside Waterbirds |

No task added to or removed from this matrix after Step 2's numbers exist.

### 8.3 Success criteria (fixed now)

`probe` = best hidden-state detector under the standard ladder protocol. `graph` = the GNN
after its fixed τ/layer sweep (τ ∈ {0.02, 0.1, 0.3} × layer ∈ {6, 9, 12}, selection on val
only — the sweep is the graph's fair configuration, not tuning-to-result).

- **Ranking win:** graph AUROC > probe AUROC on the routing task, outside the bootstrap CI,
  on the within-group comparison (errors-vs-correct among triggered/conflicting inputs only).
- **Attribution win:** graph AUROC within CI of probe (no detection cost) AND the graph's
  explanation localizes the causal evidence: pointing-game hit-rate ≥ 0.8 on true-positive
  detections vs the GT mask, AND beats both localization baselines (raw last-layer CLS
  attention / attention rollout, and the probe's input-gradient saliency). All three evaluated
  identically.
- **Generalization win (supporting, not sufficient alone):** detector trained on trigger A
  transfers to trigger B (different position + pattern) with AUROC drop < 0.05, probe's larger.
- **Clean negative:** none of the above under the swept configuration and the within-group
  control → routing column closed, P3 becomes the headline.

**Decision tree.** Ranking win → P2 as "structure detects routing failures". Attribution win
only → P2 as "structure adds verifiable attribution at zero detection cost" (a score says
*don't trust this*; a graph says *don't trust it because it used the sticker, here it is*).
Neither → P3, where this whole matrix, negatives included, is the study.

---

## 9. LIVE STATUS — the routing testbed (Step 2), as of 2026-09-07

**Goal:** the one test that decides whether the graph ever beats a probe — does structure win
when failure is caused by ROUTING (attention hijacked by a trigger), not state? The backdoor
is the easiest possible routing task (attention hijack by construction, with a ground-truth
mask), so it is the right first testbed. If the graph can't win here, natural benchmarks are
moot; if it can, Waterbirds/CelebA follow.

**The construction has been hard — an honest record.** Building a *usable* within-group pool
(triggered inputs that are both hijacked=error and resisted=correct) took many fine-tune
cycles:
- Fixed-target backdoor → "hijacked ≡ predicted class T", trivially readable from the final
  state (empty tie). **Amendment A1:** switch to a rotating all-to-all map (y→y+1), so
  detection requires a visual-evidence-vs-prediction mismatch — the real state-vs-structure
  fight.
- Rotating is much harder to train: poison 2%→ASR 0.06, 6%→0.74, 15%→0.89. At 0.89 the pool
  was degenerate (1340 hijacked, 4 resisted).
- **Amendment A2 (the key correction):** ASR≈0.5 was a self-imposed constraint, not the frozen
  criterion. The criterion needs a *usable* resisted population (≥500, spread across classes),
  NOT balance — imbalance is exactly what the weighted-BCE/subsample/AUPRC protocol already
  handles (as it did for corruptions, whose error rate was never 50%). Plus a pre-registered
  difficulty-confound control: resisted images are the "easy" ones, so a detector could win by
  reading image difficulty; control = contrast each detector's triggered vs clean
  errors-vs-correct AUROC; the routing claim predicts the graph's edge appears only in the
  triggered pool.

**Running now:** the 6% model (ASR ~0.74) re-training with the corrected gate so it saves,
then a fast resisted-counter checks ≥500 resisted across ≥20 classes. That count is the
go/no-go for the ladder.

**Early mechanical signal (promising):** the trigger DOUBLES CLS attention concentration
(0.131 vs 0.052 clean) — the hijack is visible in the raw attention, so the untrained guard
likely has real signal.

**Contingency:** if the resisted-count gate keeps failing, a lighter-touch fine-tune (fewer
epochs preserves the resisted population) is next; if the synthetic route stays impractical,
pivot the routing test to Waterbirds (naturally balanced minority/majority groups). Either way
the corruption + POPE results (P1, P3) already stand independently.
