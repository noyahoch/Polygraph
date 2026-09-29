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

**A2 (2026-09-07, before any within-group number — commit pending).** The ASR≈0.5 target
was a self-imposed constraint, NOT the frozen criterion. The criterion needs a USABLE
correct-under-trigger (resisted) population, not a balanced one. Corrected testbed gate for
the rotating within-group model:
- clean-acc drop < 2 pts (unchanged), AND
- **>= 500 resisted samples (prefer >= 1000), spread across classes** — inspect per-class ASR
  under the rotating y->y+1 map; if all resisted concentrate in a few classes, that is a
  problem to see BEFORE the ladder.
- ASR anywhere in ~0.5-0.9 is acceptable; ASR is not itself a gate.
This is distinct from the 100%-ASR models A/B', whose gate (ASR>=0.95) serves the
input-level trigger-detection transfer test, a different comparison.

Class imbalance is handled by the standard protocol (weighted BCE, train balanced by
subsampling, AUPRC reported alongside AUROC, natural-distribution metrics also reported) —
the same way corruption error rates (never 50%) were handled.

**Pre-registered secondary analysis (difficulty confound), declared now as a prediction.**
Resisted images are systematically the "easy" ones (strong object evidence beat the trigger),
so a detector could separate hijacked-vs-resisted by reading IMAGE DIFFICULTY (a state signal)
rather than routing. Control: for every detector, contrast its within-TRIGGERED
errors-vs-correct AUROC with its errors-vs-correct AUROC on the SAME model's CLEAN pool. The
routing claim predicts the graph's advantage appears specifically in the triggered pool and is
NOT reproduced on the clean pool. Declared before any number so it is a prediction, not an
excuse.
