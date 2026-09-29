# Statistics / operations boundary handoff — 2026-09-11

Status: cross-module source review completed; numerical checks remain pending. No local numerical execution or SSH occurred. This reviewer authored the evaluator, so this is a review of the core/operations interfaces, not independent approval of the statistical implementation.

## Record identity and blinding

- `image_id` is the original CIFAR photograph/base row. `source_id` identifies the corruption family. Bootstrap groups must use **image_id**.
- Cohort construction assigns a photograph to one split before creating its nine views. Extraction uses the same original row for clean and corrupted presentations. Dataset metadata and PyG batching preserve those IDs. Core validation checks canonical identities, source-disjoint splits, error labels `y=(pred!=label)`, and all 36,000 records in the production cache.
- The 270-record diagnostic contains 90 records/10 whole photographs in each split. Its disposable training benchmark uses train/validation only and publishes timing, not predictive metrics. A diagnostic cache cannot be frozen for final test scoring.
- Rewiring diagnostics/gates use train/validation only. Test rewiring records expose opaque identities and integrity checks, not changed-fraction summaries.
- Raw cache metadata necessarily contains test outcomes. Blinding is enforced by the reviewed workflow and inference interfaces, not by operating-system access restrictions. Production must use `evaluate score`, which checks the complete frozen matrix, code and calibration artifacts. Lower-level `CachedDataset`/`predict_split` checks do not independently enforce every matrix/reference condition.

## Exact Slurm commands

`PY`, `CACHE`, `RUNS`, `FREEZE`, and `RESULTS` denote the pinned Slurm Python and reviewed absolute paths. All commands run from the immutable source release.

```bash
# Each of seven arms × seeds 1,2,7,17,27. Resume is implicit.
"$PY" -m pilots.topology_20260910.train --cache "$CACHE" --run-root "$RUNS" --arm "$ARM" --seed "$SEED" --device cuda

# CPU; after successful completion of all 35 fits.
"$PY" -m pilots.topology_20260910.evaluate freeze --cache "$CACHE" --run-root "$RUNS" --out "$FREEZE"

# 35 independent GPU array items; one GPU each, concurrency at most eight.
"$PY" -m pilots.topology_20260910.evaluate score --cache "$CACHE" --freeze "$FREEZE" --out "$RESULTS" --arm "$ARM" --seed "$SEED" --device cuda

# CPU; after successful completion of the entire scoring array.
"$PY" -m pilots.topology_20260910.evaluate analyze --cache "$CACHE" --freeze "$FREEZE" --out "$RESULTS"
```

The training CLI has **no `--resume` or `--max-epochs` flag**. Analysis already writes the Hebrew report. Optional deterministic regeneration uses `evaluate report --results RESULTS`.

## Required dependencies and outputs

Before freeze: completed non-diagnostic cache; `cache/validation.json` with `passed=true`; schema-2 completed rewiring manifest with a passed development gate; and all 35 run directories containing config, history, selected safetensors, resumable latest state, validation NPZ/JSON, and complete markers.

Freeze emits its JSON and same-stem `.validation_references.npz`/`.validation_references.json` siblings. Keep them together and unchanged. Each scoring item emits `predictions/ARM__seedN.npz` and a matching `.receipt.json`, binding the prediction to the freeze hash. CPU analysis refuses missing or inconsistent receipts.

Final outputs: `summary.json`, `results.csv`, `bootstrap.json`, `bootstrap_draws.npz`, `report.he.md`, `evaluation_state.json`, `evaluation_complete.json`, all model predictions/receipts, and MSP/entropy NPZ predictions. Final publication needs the freeze, results root and cache metadata root. Administrative backup/failure jobs use appropriate terminal dependencies and must not block training or imply scientific success.

The default is the complete 35-fit matrix. The 25-fit primary-only reduction requires the existing explicit pre-test root approval document; it is never inferred from missing jobs or results.
