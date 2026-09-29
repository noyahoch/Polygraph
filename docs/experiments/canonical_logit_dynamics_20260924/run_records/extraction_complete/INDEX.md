# Full extraction completed — September 24, 2026

Job `924822` completed extraction and the frozen parity checks for **all 71,991 required rows**. This extends the earlier 496-row preflight to the full required extraction. The three seed-fit receipts in this snapshot are **initial running receipts only**; they do not establish training completion or model performance.

## Recorded completion and identity

- Scheduler: `COMPLETED`, user `omrifahn`, account `gpu-students`, name `canonical0924-085200-extract`, node `s-005`.
- Scheduler start/end: September 24, 2026, **09:52:21–10:19:06 Israel time**; recorded elapsed time `00:26:45`.
- Wrapper completion: `2026-09-24T07:19:06.789799+00:00`; status `complete`, extraction command return code `0`.
- Configuration SHA256: `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c` (`config_science_v4.json`).
- Source manifest SHA256: `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f` (`science-source-v2`).
- Campaign SHA256: `f489a75f558d7588b946ebf5f5586e418b5473c156c6a5b88a4a071118a65268`.

The scheduler, wrapper and internal extraction timers have different boundaries. Their original values are preserved; this index uses the scheduler elapsed time for the complete allocation.

## Recorded parity gate

`cls/manifest.json` is the authoritative full extraction manifest: `mode=full`, `complete=true`, `records=completed_records=71991`. It records exact original-prediction and label agreement. Both confidence and top-two softmax margin use the frozen maximum absolute tolerance `0.0002` with relative tolerance `0.0`.

The manifest records maximum absolute differences of `5.5730342864990234e-05` for confidence and `0.00011068582534790039` for margin. These are copied values from the completed Slurm receipt, not recomputed statistics. The manifest also binds record order, every tensor-shard checksum, model provenance, processor configuration and data provenance.

## Downstream state at capture

The scheduler snapshot was checked at `2026-09-24T07:19:15Z` (10:19:15 Israel). Seed fits `924823` (seed 1), `924824` (seed 2) and `924825` (seed 7) had started at 10:19:07 Israel and were `RUNNING` on `s-005`. Freeze, prediction, evaluation and backup jobs were still pending dependencies.

No training metrics were inspected or recomputed to create this preservation snapshot. It does not establish completion of the comparison, uncertainty estimates or private backup.

## Preserved originals

- `cls/manifest.json`: complete full extraction manifest and shard-checksum listing.
- `cls/model_provenance.json`, `cls/processor.json`, `cls/data_provenance.json`: model, processor and input provenance.
- `ops/stage_receipts/extract_924822.json`: successful extraction command, runtime identities and wrapper completion.
- `ops/stage_receipts/fit1_924823.json`, `fit2_924824.json`, `fit7_924825.json`: initial running receipts only.
- `scheduler_and_identity_status.txt`: terminal extraction status, initial downstream states and configuration/source/campaign hash snapshot.
- `logs/extract_924822.out` and `.err`: original extraction logs, including the retained warning.
- `COPY_MANIFEST.json`: exact local origins, byte sizes and SHA256 checksums of every copied evidence file.

Remote campaign: `/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`.

Archive copies, tensor payloads and per-record index arrays are excluded. Heavy artifacts remain in the remote campaign; their referenced checksums are preserved here. Future actions must reconcile full job identity and receipts, not job numbers alone.
