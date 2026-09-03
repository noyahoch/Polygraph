# Polygraph — Consolidated State

*Updated 2026-09-02 · branch `full_pipeline`. Single source of truth for the team's P2/P3
branch decision. Detailed numbers live in `docs/TEAM_REPORT.md`, `docs/results/*.md`.*

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
| POPE transfer slice | DONE — popular↔adversarial invalid (76% shared questions, leak caught); honest group-disjoint-by-image test confirms confident-error law at higher power: **+0.199 [+0.129,+0.266]** confident, +0.004 flat overall |
| Track B backdoor (100% BadNets) — synthetic routing testbed | **VALIDATED** (clean 0.9145, ASR 0.9995 — `docs/results/backdoor_testbed.md`) |
| Synthetic 95%-spurious | ABANDONED (design flaw, not fundamental — see §3 of exploitation doc) |
| Waterbirds routing pilot | PREPARED, NOT launched (`pilots/waterbirds/PLAN.md`) — team call |
| Polygraph tail (CHARM-v2, final-4, τ sweep) | PAUSED (appendix material) |
| G5 multi-seed (confident slices only) | DEFERRED until headline chosen |
| G4 full literature gate | MANDATORY before any writing; seeded by 4 papers in TEAM_REPORT |

## 7. Immediate next actions (on the numbers, for the team)

1. When the backdoor model validates (clean-acc drop <2pts, ASR ≥95%), run the **G3-fixed
   graph-vs-probe routing test** on it — the first real test of whether structure beats a
   probe when failure is routing-caused.
2. Read the POPE transfer slice: if the confident-error law does not survive
   popular↔adversarial transfer, it may itself be disguised familiarity.
3. Team decides P2 vs P3 emphasis from the routing-test result; then G5 multi-seed on the
   chosen headline slices, then the mandatory lit gate, then writing.
