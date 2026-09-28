# Completed bounded GPU preflight — job 924803

**The fixed 496-record panel passed. This is not an all-71,991-row extraction,
completed LD training or final comparison.** The full-export gates and subsequent
scientific stages remain separate. This snapshot does not authorize submission;
consult `../../SESSION.md` and the current coordinating task.

Original local receipt root:
`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924/runtime_preflight_r2_complete`

Remote campaign root:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`

[COPY_MANIFEST.json](COPY_MANIFEST.json) binds each byte-identical copy to its
absolute original local path, SHA-256 and byte size. No experiment values were
recomputed for this preservation step. Tensors, raw images/parquet files and the
retrieval archive are excluded.

## Scope and outcome

[preflight/complete.json](preflight/complete.json) records 496 captured panel
rows and passing finite-gradient, save/load, exact-resume, changed-identity
rejection, semantic-feature-order, six preserved scientific feature/normalization
checks, and weighted-AUROC versus literal-duplication fixtures including tied
scores and unequal numbers of views. The immutable individual attempt receipt
is retained next to this completion marker.

[preflight_cls/manifest.json](preflight_cls/manifest.json) explicitly says
`mode: preflight`, `records: 496`, and `complete: true`. Its fixed parity rules
require exact true labels and classifier predicted classes, with confidence
and probability-margin absolute tolerance `2e-4` and relative tolerance zero.
The recorded maxima are:

- Confidence absolute difference: `1.0132789611816406e-05`.
- Top-one-minus-top-two probability-margin absolute difference:
  `1.913309097290039e-05`.

These are panel parity measurements, not LD detection scores. The four copied
JSON shard receipts preserve panel row identities, original scan metadata and
per-shard parity/checksums. They refer to the `.pt` tensors retained remotely;
this folder contains no tensor payloads. [index.json](preflight_cls/index.json)
preserves the captured panel order and offsets.

## Timing and hardware, exactly as recorded

[terminal_status.txt](terminal_status.txt) identifies job `924803`, user
`omrifahn`, account `gpu-students`, name `canonical0924-085200-preflight_r2`,
and terminal state `COMPLETED`. The cluster record shows `s-005`, September 24
09:35:52–09:42:38 Israel, elapsed **6 minutes 46 seconds**.

The capture manifest reports an NVIDIA GeForce RTX 2080 Ti, batch 32 and peak
allocated GPU bytes `882246656`. Its own timers report:

| Recorded timer | Seconds |
|---|---:|
| Decode | 11.557796041015536 |
| Inference | 17.41534813772887 |
| Model loading | 56.98954669293016 |
| Parity checks | 0.6330884066410363 |
| Complete extractor invocation | 254.09246916417032 |

These nested stage timers are not interchangeable with scheduler wall time.
Cold input verification, imports, model loading and allocation overhead must
not be multiplied by the full-row/panel-row ratio when estimating later work.
This snapshot makes no new throughput calculation or completion prediction.

## Frozen identities and provenance

- Config: [science v4](../preflight_recovery/config_science_v4.json), SHA-256
  `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
- Source release: `science-source-v2`; source manifest SHA-256
  `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`,
  preserved in [the preparation snapshot](../preparation/releases/science-source-v2/source_manifest.json).
- Campaign SHA-256:
  `f489a75f558d7588b946ebf5f5586e418b5473c156c6a5b88a4a071118a65268`.
- Capture manifest SHA-256:
  `600cd5dd00c7a5981926791fa0ea1013ebd51970e415867b8d97e1bf3bbe522e`.

The [stage completion receipt](ops/stage_receipts/preflight_r2_924803.json)
binds config, source release, dependency overlay and executed command; the
[submission receipt](ops/submissions/preflight_r2.json) binds scheduler identity
and the sealed successful preparation predecessor.

The copied [data provenance](preflight_cls/data_provenance.json),
[model provenance](preflight_cls/model_provenance.json), and
[processor configuration](preflight_cls/processor.json) retain actual source
revisions/file hashes, model/config/processor artifact hashes, class mappings
and preprocessing identity. Logs remain under `logs/`, including nonfatal
warnings. No tolerances or scientific settings were changed to obtain this pass.

The earlier failed preflight and superseded operational configurations remain
preserved in [preflight_recovery](../preflight_recovery/INDEX.md). Successful
panel checks do not erase that failed allocation or substitute for validation
on every row of the forthcoming full extraction.
