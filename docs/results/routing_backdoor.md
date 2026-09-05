# Step 2 — Backdoor routing test (IN PROGRESS)

Pre-registered success criteria (from CONSOLIDATED_STATE §8.3), quoted verbatim, pass/fail
marked when numbers exist:
- [ ] Ranking win: graph AUROC > probe AUROC, outside bootstrap CI, within-group.
- [ ] Attribution win: graph within CI of probe AND pointing-game >= 0.8 vs GT mask, beating
      rollout and probe-gradient saliency.
- [ ] Generalization win (supporting): train trigger A -> test trigger B, AUROC drop < 0.05,
      probe's drop larger.
- [ ] Clean negative -> routing column closed, P3 headline.

## Amendment log (changes to the TESTBED, each dated and made BEFORE its result existed)

**A1 (2026-09-03, before any within-group number — commit pending).** The frozen criteria
define the *comparison*, not the *model*. Two testbed degeneracies were found after freezing
and are fixed here, both decided before any within-group AUROC was computed:
1. *ASR ~99.95% empties the comparison group* — with near-perfect attack, essentially all
   triggered inputs are errors, leaving no "correct" triggered inputs to rank against. Fix:
   use an INTERMEDIATE-ASR model (target ~0.5-0.85) so the triggered pool contains both
   hijacked (error) and resisted (correct) inputs — the within-group variation the criterion
   needs.
2. *A fixed target collapses "hijacked" onto "predicted class T"* — which sits trivially in
   the final hidden state, so probe and graph both saturate ~0.99 (an empty tie, not a
   no-detection-cost result). Fix: ROTATING all-to-all mapping (y -> y+1 mod 100). Now
   "hijacked" is not any specific class; the detector must find a visual-evidence-vs-prediction
   mismatch — the legitimate state-vs-structure competition.

Models A/B (100% ASR, fixed target) are retained ONLY for the input-level trigger-detection
transfer test. The within-group routing detector uses intermediate-ASR rotating models C/D
(two triggers, for within-group cross-trigger transfer).

The "too easy" guard (untrained max per-patch CLS-attention share) is kept; if it separates
hijacked-vs-resisted, that mechanically supports the routing story even if it shrinks the
learned GNN delta — reported, not buried.
