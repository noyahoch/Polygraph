# Core implementation handoff

Source-only implementation; no imports, numerical checks, models, datasets or GPU work ran on the Mac.
The isolated worktree is on `September-10th`, based on `9af8890405297c0f37dbaa54f6be29a84cb0c37e`.
Historical `runs/` are excluded by sparse checkout. The original Polygraph checkout is preserved.

## Slurm entry points

```bash
python -m pilots.topology_20260910.protocol --out-dir "$CACHE"
python -m pilots.topology_20260910.extract prepare --data-root "$DATA"
python -m pilots.topology_20260910.validate --device cuda --out "$RUN/model_preflight.json"
python -m pilots.topology_20260910.extract capture --data-root "$DATA" --cache "$SMOKE_CACHE" --device cuda --batch-size 18 --max-records 270 --resume
python -m pilots.topology_20260910.rewire --cache "$SMOKE_CACHE" --workers 2
python -m pilots.topology_20260910.validate --cache "$SMOKE_CACHE" --device cuda --require-rewire
python -m pilots.topology_20260910.extract capture --data-root "$DATA" --cache "$CACHE" --protocol "$CACHE/protocol.json" --device cuda --batch-size 32 --resume
python -m pilots.topology_20260910.rewire --cache "$CACHE" --workers 4
python -m pilots.topology_20260910.validate --cache "$CACHE" --device cuda --require-rewire
```

All numerical entry points require `SLURM_JOB_ID`. Extraction requires CUDA without fallback.
The independent operations downloader owns fetching/checking the official archive; `extract prepare` only validates existing files.
Expected inputs: `DATA/cifar-100-python/test` and `DATA/CIFAR-100-C/{gaussian_noise,motion_blur,fog,jpeg_compression,labels}.npy`.
Dependencies beyond the team's package: `safetensors`, `numba`; normal Torch/PyG/Transformers/Pillow/NumPy dependencies remain necessary.
Run compile/import and numerical tests on Slurm before any production extraction.

## Ownership and contracts

Engineering owns `protocol.py`, `data.py`, `extract.py`, `models.py`, `rewire.py`, `validate.py` and `__init__.py`.
The training engineer owns `train.py`; other workers own evaluation, publishing and Slurm scripts.

`CachedDataset(cache, split, arm, freeze=None)` exposes callable `labels()`, `logits()`, `metadata()`, `shard_blocks()`.
`labels()` and PyG `data.y` are classifier-error labels, **not CIFAR classes**.
Metadata: `label` is true CIFAR class, `pred` is predicted class, `y=int(pred!=label)`; other fields are record_id, image_id, source_id, severity, split_id, confidence and margin.
Class-conditioning uses predicted/runner-up classifier weights only, never true labels.

`build_model(arm)` returns the unchanged Ishi graph/set class (returns logits, embedding), or an output-only Sequential MLP.
Arms: full_graph, full_rewired, full_set, full_endpoint, raw_graph, raw_set, logit.
Seeds: 1, 2, 7, 17, 27. Graph/set batch24; logits batch256.
No top-K cap: final-layer full threshold `max_head_attention > .02`, 197 nodes.

Cache completion: `cache/manifest.json` with complete=true, shard hashes, protocol/cohort/index identities.
Rewire completion: `cache/rewire/manifest.json` with complete=true and source manifest hash.
Rewiring performs 2E accepted swaps in at most20E proposals; no loops/duplicates, exact in/out degrees and source-attached edge values. Every nonempty graph must change at least80% of edge targets. Failure leaves complete=false and must not silently discard cases or loosen rules.
Validation completion: `cache/validation.json` with passed=true. Diagnostic caches remain explicitly diagnostic and cannot be used for final test scoring.

## Freeze contract

`data.check_freeze(cache, freeze_path)` checks frozen=true, matching protocol_sha256/cohort_sha256/cache_manifest_sha256, rejects diagnostic caches, then verifies all entries in `runs`.
`run_root` resolves relative to the freeze file's parent unless absolute.
`runs` maps `ARM/seedN` to config_sha256, best_sha256, validation_npz_sha256, validation_json_sha256 and complete_sha256, corresponding to config.json, best.safetensors, validation.npz, validation.json and complete.json.
Thresholds and any learned scaler must live in hash-bound config/validation artifacts. Training's predictor additionally checks the requested run is included.

## Outstanding validation

No success claims until Slurm executes the model-count/gradient/save-load/set-invariance/direction tests, small legacy-extraction comparison, constrained-rewire gate and cache checks. The extraction comparison uses pinned model/processor plus existing FrozenClassifier methods, and numerical tolerance1e-5 rather than bitwise CUDA equality.
No classifier predictions, performance metrics or held-out scores were computed during local authoring.
