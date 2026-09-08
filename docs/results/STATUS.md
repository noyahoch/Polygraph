# Status — 2026-08-23 18:00

## Completed
- Scan: 1,010,000 records (full CIFAR-100-C grid), 239,921 ViT errors; verified vs POC ground truth.
- Plan: 75k records (52/6/17k), 50/50, group-disjoint, cell-stratified, extras test-only, clean_train excluded.
- Store: 198 GB, tau=0.02 (~7,600 edges/layer), CLS embeddings, key-indexed, drift-checked.
- Package: polygraph/ (data + training), 36 mutation-audited tests, models bit-identical to POC.
- Classifier accuracy table (19x5): docs/results/classifier_accuracy.md.

## Results so far (17k test)
- MSP 0.8695 | margin 0.8681 | output_lr 0.8672 (training on outputs adds nothing).
- Slices: clean 0.9042, seen 0.8679, unseen-extras 0.8745, severity 0.8998 -> 0.8382.
- Graph (seed 7, live): best val AUROC 0.8234 @ epoch 28 — project-best; ~0.046 under MSP.

## Running (automatic)
Seed 7 early stop -> evaluation (attn_mlp, cls_mlp, cls_seq + combiner; full 8x10 table)
-> mega-chain: shuffle control, top-K=100 anchor, weather-family holdout (full eval),
tau {0.05, 0.03, 0.10}, final-4 / all-12 layers.

## Next (decisions)
1. 3-seed reruns of any claimed number (noise ±0.02).
2. Variant-2 node features (token embeddings) — decide after cls_mlp result.
3. CHARM-style all-layer model (optional extension).
4. Far-OOD (optional per proposal).
5. Writeup: severity figure, risk-coverage curves, ACM template results section.

## Ops notes
Training is segmented (8 epochs/process) against Metal's kernel-cache leak; fully resumable.
Machine must stay on AC with lid open for the queue to progress.
