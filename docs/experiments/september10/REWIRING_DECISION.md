# Pre-test decision on insufficient rewiring — 2026-09-11

**Decision authority: root. Status: explicitly approved before production test scoring.**
Retain the registered seven arms and five seeds, **all 35 fixed fits**, including
`full_rewired`. Admit structurally valid rewiring with insufficient mixing only under
this recorded decision. The quality result remains **FAIL / non-diagnostic**; this
decision does not turn it into a passing null.

No production detector test scores or test-performance aggregates were visible when
this decision was made. The production training/scoring workflow had stopped before
training at the diagnostic quality gate. Diagnostic feature extraction is distinct from
opening production test results. The reported mixing evidence below uses training and
validation graphs only.

## Evidence and source receipts

Operations reported the following from Slurm preflight job **877598**, using the
270-record diagnostic cache:

| Development diagnostic | Recorded result |
| --- | --- |
| Eligible training/validation graphs, at least 20 edges | 180 |
| Mean changed-edge fraction | 0.5441644076, below the registered 0.80 requirement |
| Fraction below 0.80 changed edges | 0.76111 |
| Median changed-edge fraction | 0.56213 |
| 90th percentile changed-edge fraction | 0.87779 |
| Graphs not reaching the target of 2 accepted swaps per edge within 20 proposals per edge | 113 of 180 |
| Construction and per-shard structural integrity | Passed; graphs retained |

The original Slurm receipt is
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910/preflight/877598/diagnostic_cache/rewire/manifest.json`.
Its diagnostic-cache parent contains the corresponding capture and validation evidence.
Operations reported `construction_complete=true`, per-shard structural `passed=true`,
and overall `complete=false` due to the quality gate. The original receipt must remain
unchanged; later admission must reference it rather than rewrite it as a success.
These values were supplied by Operations; this decision's author did not run numerical
analysis or inspect production test results on the Mac.

## Basis in the registered protocol

[PROTOCOL.md](PROTOCOL.md), constrained-rewiring section (line 72 in the registered
version), requires failed mixing to be marked non-diagnostic and referred for a
documented protocol decision. The execution-budget section (line 124) states that a
failed rewiring control does not erase valid graph-versus-set results while its
mechanistic conclusion remains unavailable. This is that explicit decision.

Referenced protocol SHA-256:
`fcdbc850550b3d31ff98f556b1d3584cec6f5150901830df75a5b8e00cc1b5db`.

## Fixed implementation and admission conditions

- Preserve the exact directed double-edge-swap construction, fixed random stream,
  2 accepted swaps per edge target, 20 proposals per edge limit, and 0.80 quality
  threshold. Do not tune these quantities or search for a more favorable null.
- Retain every graph, including weakly changed and nonrewirable graphs. No source,
  condition, graph, arm or seed is selectively removed. All 35 fits remain fixed.
- Structural and data-integrity failures remain fatal: directed degrees, edge counts,
  no self-loops/duplicates, unchanged node/edge values and original source associations,
  record alignment, cache/checkpoint provenance, and test blindness must still pass.
  This decision admits insufficient mixing only.
- Run the already-planned full training/validation quality diagnostics before the
  evaluation freeze and bind their complete values and status to the frozen artifacts.
  The 180-graph diagnostic does not certify the full development cohort. Preserve the
  actual full-cohort pass/fail result without tuning or suppressing it, whatever it is.
  Test diagnostics or test performance cannot determine quality admission.
- Bind this exact decision and its SHA-256 into the admission and evaluation provenance.
  Keep construction integrity, quality adequacy and explicit admission as distinct
  facts; an admitted low-quality control must never be labeled a passing diagnostic null.
- Ordinary readiness, numerical correctness, resource admission, full-matrix completion,
  validation selection/calibration and immutable test-freeze requirements still apply.

## Permitted interpretation

The primary `full_graph - full_set` comparison and endpoint control remain registered
with their existing limitations. Insufficient mixing weakens the intended mechanistic
comparison; it is not evidence that GNNs fail or succeed.

Report `full_graph - full_rewired`, its seed-level values and uncertainty **descriptively**
for the realized partial perturbation, with the insufficient-control status clearly
visible. Do not use this contrast, in either direction, to claim topology was removed,
graph structure is necessary or unnecessary, or the original and rewired models are
equivalent. Rewiring also alters destination/attribute alignment and paths, so it cannot
isolate higher-order topology alone. The full-cohort diagnostic report does not create
a later opportunity to promote this arm to a confirmatory mechanistic result.

No scientific model, feature, split, training hyperparameter, seed, threshold, or primary
estimand is changed by this decision.
