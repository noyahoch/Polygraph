# Waterbirds routing pilot — PREPARED, NOT LAUNCHED

Team decision required before running (P2/P3 branch). This file is config + protocol only.

## Why Waterbirds
Canonical natural spurious correlation: waterbird/landbird x water/land background. The
background is a genuinely EASIER cue than fine-grained bird ID, so a model adopts it as a
shortcut WITHOUT any manufactured manipulation — exactly the reliability-dominance condition
the synthetic 95% patch failed to meet. Worst-group (waterbird-on-land, landbird-on-water)
errors are confident and routing-caused (attended to background, not bird). Group label is
external to the output distribution -> high headroom by construction.

## Data
- Source: Waterbirds (CUB-200 birds composited on Places backgrounds), the Sagawa et al.
  (2020) GroupDRO release. ~4,795 train / 1,199 val / 5,794 test, each with (y, group).
- Download: standard `waterbird_complete95_forest2water2` tarball. NOT fetched here.

## Fine-tune (matches Polygraph protocol)
- Backbone: google/vit-base-patch16-224-in21k (ImageNet-pretrained, NOT the CIFAR checkpoint
  — Waterbirds is a 2-class bird task), fine-tune all params, AdamW 2e-5, class-weighted BCE
  is n/a (2-class CE), early stop on val worst-group acc, seed 7.
- Target behavior gate: overall test acc high AND worst-group acc materially lower (the
  spurious gap). If no gap, the model didn't take the shortcut — stop.

## Routing test (G3-FIXED protocol — the whole point)
1. PRIMARY comparison: errors-vs-correct WITHIN the worst-group (minority) test pool only.
   Comparing minority-vs-majority would let a detector win by spotting the rare group
   (OOD-through-the-backdoor). Within-group only.
2. Detectors: output (msp/energy), probe (CLS hidden), graph, fusion — Polygraph ladder.
3. GRAPH GETS ITS FREE SWEEP before any negative counts: tau in {0.02,0.1,0.3}, layer in
   {6,9,12} (strength-sorted store makes these prefix views; all layers stored).
4. Headline number: does the graph beat the probe on within-minority error ranking?
   -> yes: P2 (routing paper). -> no, under swept config + clean control: P3 (adjudication).

## Extraction
Reuse polygraph.data.graphs + a lean per-image capture (like pilots/vlm_pope/run.py):
logits, CLS hidden, thresholded attention graph. No 198GB store needed at pilot scale.
