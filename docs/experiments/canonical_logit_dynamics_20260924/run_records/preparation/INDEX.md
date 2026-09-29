# September 24 preparation receipts

This is an immutable evidence snapshot of preparation, download and environment
repair attempts. It is **not** an instruction to submit or resume jobs. Consult
`../../SESSION.md` and reconcile current server receipts through the sole Slurm
operator before acting. Job IDs must also match user, account, name and time.

Original local receipt root:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924`

Remote campaign root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`

Every copied file is retained byte-for-byte. [COPY_MANIFEST.json](COPY_MANIFEST.json)
records its original absolute local path, size and SHA-256. Directory names
`reconciled_0914`, `reconciled_0922`, and `reconciled_0924` are the operator's
snapshot labels; receipt timestamps inside the files are authoritative.
No metrics were recomputed to prepare this snapshot.

## What happened

| Attempt | Recorded outcome | Evidence in this snapshot |
|---|---|---|
| `924758`, initial incoming-bundle validation | Failed during metric import; preserve the partial artifact checks | Already preserved in [input_validation](../input_validation/validation_report_attempt1.json); not duplicated here |
| `924767`, validation repair | Passed all 18 recorded checks using the preserved sklearn/scipy overlay | [Passed validation report](../input_validation/validation_report_repair1.json) and its adjacent submission receipt |
| `924773`, `fetch` | Failed when the default HF Xet cache hit home quota | [Failure receipt](reconciled_0922/ops/stage_receipts/fetch_924773.json), [stderr](reconciled_0922/logs/fetch_924773.err), bootstrap v1 config/source manifest |
| `924774`, `fetch_r1` | Completed with HF/Xet scratch paths changed; scientific inputs unchanged | [Completion receipt](reconciled_0922/ops/stage_receipts/fetch_r1_924774.json), [dataset manifest](reconciled_0914/data/dataset_manifest.json), bootstrap v2 config/source manifest |
| `924782`, `prepare` | Failed before campaign preparation because pandas metadata was missing | [Failure receipt](reconciled_0922/ops/stage_receipts/prepare_924782.json) and [stderr](reconciled_0922/logs/prepare_924782.err) |
| `924783`, initial GPU `preflight` | Original submission preserved; science v2's repair note records cancellation after its preparation dependency failed | [Original submission](reconciled_0922/ops/submissions/preflight.json); no cancellation accounting receipt is included in this snapshot |
| `924784`, isolated parquet dependency repair | Completed; actual canonical row decoding and fast-processor checks passed | [Dependency manifest](reconciled_0922/dependencies/parquet_overlay_v1/complete.json), [runtime verification](reconciled_0922/dependencies/parquet_overlay_v1/runtime_verification.json), [installer output](reconciled_0922/logs/parquet_deps_924784.out) |
| `924785`, `prepare_r1` | Submitted under science v2; this snapshot contains submission evidence only | [Submission](reconciled_0924/ops/submissions/prepare_r1.json) |
| `924786`, `preflight_r1` | Submitted after `924785`; this snapshot contains submission evidence only | [Submission](reconciled_0924/ops/submissions/preflight_r1.json) |

Do not interpret a submission receipt as completion. Later preflight, extraction,
training or evaluation results are outside this preparation snapshot unless a
subsequent, separately named evidence directory is added.

## Exact configurations and source identities

- [config_bootstrap_v1.json](config_bootstrap_v1.json) binds
  [bootstrap-source-v1/source_manifest.json](releases/bootstrap-source-v1/source_manifest.json).
- [config_bootstrap_v2.json](config_bootstrap_v2.json) binds
  [bootstrap-source-v2/source_manifest.json](releases/bootstrap-source-v2/source_manifest.json).
- [config_science_v1.json](config_science_v1.json) binds
  [science-source-v1/source_manifest.json](releases/science-source-v1/source_manifest.json).
- [config_science_v2.json](config_science_v2.json) binds
  [science-source-v2/source_manifest.json](releases/science-source-v2/source_manifest.json).

The configurations preserve exact commands, resource requests, stage dependency
names, environment paths and expected source hashes. The source manifests list
file identities for the corresponding remote releases. Full executable trees
are kept in those remote release directories and the project source history;
this folder intentionally contains manifests rather than duplicate source trees.

The science configurations include a proposed full DAG for budget accounting.
Their presence does **not** mean the full DAG was submitted: science v2 explicitly
limits the then-approved phase to preparation and preflight pending measured
parity and feasibility review. Preserve original intents and failed attempts.

## Source data provenance

[dataset_revisions.json](reconciled_0914/data/dataset_revisions.json) and
[dataset_manifest.json](reconciled_0914/data/dataset_manifest.json) preserve the
resolved download revisions and parquet file checksums:

- `uoft-cs/cifar100`: `aadb3af77e9048adbea6b47c21a81e47dd092ae5`.
- `WNJXYK/TTA-CIFAR-100-C`: `a12f0bcc1da33fa26d8c76ce8c1fb32e6f913bea`.

These identify the actual September 24 downloads, not an inferred historical
revision of Yishai's data. The raw parquet files, source images and CLS caches
remain on Slurm and are not copied here.

## Isolated environment repair

[install_parquet_overlay.py](install_parquet_overlay.py) is the executed repair
script, also preserved at its operator snapshot path. It verifies downloaded
wheel hashes against version-specific PyPI metadata and installs into a new
campaign-owned overlay without modifying the old experiment environment.
The small [Slurm script](reconciled_0922/ops/parquet_dependencies.slurm), intent,
submission and stdout/stderr records are retained alongside it.

Pinned additions: pandas `2.3.3`, pyarrow `21.0.0`, python-dateutil
`2.9.0.post0`, pytz `2025.2`, tzdata `2025.2`, and six `1.17.0`.
The recorded runtime preserved Python `3.12.13` and NumPy `2.5.2`; actual imported
module paths and remaining package versions are in the runtime verification.
It decoded canonical clean-test row zero as RGB with label 49 and produced a
finite float32 processor tensor of shape `[1, 3, 224, 224]`. This is an environment
readiness check, not a passed ViT/LD parity or training result.

No credentials, model weights, raw datasets, installed wheel contents or full
source releases are included in this folder. Source-file hashes were checked
while copying; no extraction, fitting, inference or statistical calculation ran
on the Mac for this preservation work.
