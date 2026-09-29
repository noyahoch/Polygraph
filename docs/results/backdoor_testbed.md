# Backdoor routing testbed — VALIDATED (Track B)

Standard BadNets on the CIFAR-100 ViT: 16px checkerboard trigger (bottom-right) →
target class 0, 2% poison rate, 100% trigger→target correlation, 2 epochs.

| metric | value | gate |
|---|---:|---|
| clean accuracy | 0.9145 | within 2pts of 0.9148 baseline — PASS |
| attack success rate | 0.9995 | >= 0.95 — PASS |

The 100% correlation is what the 95% spurious construction lacked: the shortcut now dominates
the true cue in training and is adopted almost perfectly. This model makes CONFIDENT errors
caused by attention routing (trigger patch → CLS hijack → target class), with the true object
still present — the clean "routing, not state" failure mode.

## The routing test this enables (G3-fixed protocol, NOT yet run — team's P2 call)

- PRIMARY comparison: errors-vs-correct WITHIN the triggered-input pool only (never
  triggered-vs-clean, which would let a detector win by spotting the anomalous input — OOD
  through the back door).
- Detectors: output (msp/energy), probe (CLS hidden), graph, fusion.
- Graph gets its free sweep before any negative counts: tau in {0.02,0.1,0.3}, layer in {6,9,12}.
- Generalization: cross-trigger transfer (train on one position/pattern, test on another) —
  a detector that transfers is reading "attention hijack topology", not "this patch".
- Decision: graph beats probe on within-triggered error ranking → P2; loses under swept
  config → P3 (the loss is a finding).
