# Results explorer: existing experiments, not new model runs

Open [the offline visual report](index.html). It embeds its small plotted data,
requires no server or internet connection, and has JSON/CSV download controls.
The overview is followed by optional, explicitly post-hoc exploratory panels.

## What is comparable

- **Current layer study:** same 800 development photographs and 7,200 records,
  seeds 17/27 as the primary replication and seed 7 as a late diagnostic.
  The comparator is the same final-layer GNN with the same type of logistic
  meta learner, not a graph-free baseline.
- **Historical September 10 study:** existing results on its separate
  800-photo test group, five seeds, including full-logit, confidence and
  non-message-passing controls. Only previously computed summary tables are
  read; historical test predictions are neither downloaded nor rescored.

The panels deliberately do not create a cross-experiment AUROC leaderboard.
Different data roles, training protocols and evaluation photographs prevent
interpreting their absolute scores as a direct improvement.

## What is new in this visualization

Original metrics and confidence intervals are copied from completed Slurm
reports. The current saved-score AUROCs are checked against those reports,
using the existing tie-aware evaluator; the saved heads must reconstruct the
scores. No models, thresholds, layers or seeds are selected from these plots.

The following are **post-hoc descriptive EDA**, requested after completion:

- Spearman rank correlations between layer scores.
- ROC curves and correct/error score-rank distributions.
- All nine fixed condition slices, including negative differences.
- Existing checkpoint-role learning curves and selected checkpoint markers.
- Already fitted, meta-standardized logistic coefficients.

No additional bootstrap intervals, significance tests or inference runs were
performed. Condition slices share photographs and have no new confidence
intervals. Pooled AUROC includes cross-condition ranking pairs; it is not
the average of condition-specific AUROCs. Standardized coefficients with
correlated inputs are not causal importance. Score rank is not calibrated risk.

Every seed was trained for 20 epochs; a selected checkpoint can be earlier.
Checkpoint curves are not development-evaluation or test curves and do not
authorize further tuning.

## Lightweight local bundle

The result-only collector downloaded **63 files, 2,241,655 bytes** (about
2.24 MB): three 7,200-row development score archives, small reports, histories,
configurations, head parameters and provenance manifests, plus the historical
summary/CSV/bootstrap metadata. Each score archive is approximately 293 KiB.

No feature caches, source images, detector weights or optimizer/RNG state were
downloaded. Those remain on Slurm. The collector has a 1 MiB per-file and
8 MiB total preflight limit and an explicit filename allowlist. Downloads are
verified against the original hash-bound completion records.

The user's September 16 request explicitly permits fast local processing of
these small results. The first complete EDA build took **0.144 seconds** on
the Mac, using NumPy only. This exception does not move scientific training,
feature extraction, model inference or heavy storage to the laptop.

## Reproduce the result-only view

Run from the project root, with a writable local output directory. The SSH
wrapper uses the existing user's connection configuration; no credentials
belong in result bundles or commits.

```bash
python3 -B -m pilots.results_20260916.collect \
  --ssh-wrapper /absolute/path/to/work/slurm-ops/ssh-c003.sh \
  --out /absolute/path/to/results-explorer/raw

python3 -B -m pilots.results_20260916.data \
  --raw /absolute/path/to/results-explorer/raw \
  --out /absolute/path/to/results-explorer

python3 -B -m pilots.results_20260916.render \
  --data /absolute/path/to/results-explorer/results.json \
  --out /absolute/path/to/results-explorer/index.html

python3 -B -m unittest \
  pilots.results_20260916.test_data pilots.results_20260916.test_render
```

Keep the raw bundle outside Git; the small, self-contained HTML snapshot can
be versioned with documentation. No web fonts, chart CDNs, analytics or
third-party requests are needed.

The 25 results-only tests pass, including tie handling, undefined slices,
source-image grouping, input checksum/size gates, and safe offline JSON
embedding. Browser checks covered seed switching, condition/ROC controls,
responsive layout, and the generated JSON/CSV download payloads, with no
external requests or JavaScript errors. The integrated browser did not expose
a native download event, so payloads were inspected directly; the standalone
JSON and CSV files are also available in the local bundle.

Authoritative sources: [September 10 results](../september10/RESULTS.md),
[completed layer replication](../layer_ensemble_replication/RESULTS_20260916.md),
and the [replication protocol](../layer_ensemble_replication/PROTOCOL.md).
Seed 7's original late/incomplete status remains unchanged.
