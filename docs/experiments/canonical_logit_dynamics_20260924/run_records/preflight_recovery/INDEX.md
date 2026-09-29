# Preflight recovery evidence — September 24

This snapshot continues [the preparation evidence](../preparation/INDEX.md)
without altering it. Read `../../SESSION.md` for current state, and reconcile
live server state through the sole Slurm operator before any action. These
files are historical evidence, not commands or new submission authority.

Original local root:
`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/canonical-bundle-20260924`

Remote campaign root:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`

[COPY_MANIFEST.json](COPY_MANIFEST.json) records exact source paths, byte counts
and SHA-256 for every byte-identical copied file. This documentation copies
existing results; it does not recompute experiment statistics.

## Successful preparation, job 924785

[prepare_r1_924785.json](runtime_prepare_r1/ops/stage_receipts/prepare_r1_924785.json)
records successful import checking and canonical campaign preparation under
science v2. Its wrapper elapsed time is 42.405660015996546 seconds. The
[import check](runtime_prepare_r1/ops/import_check.json) records all eight core
module imports, source syntax checks and the actual runtime package versions.
This is preparation success, not a completed numerical preflight.

[campaign.json](runtime_prepare_r1/campaign.json) seals source hashes, recipe,
input hashes, runtime, protocol, exact role-map identity and the preflight-panel
identity. [preflight_records.json](runtime_prepare_r1/preflight_records.json)
contains the compact frozen panel. The approximately 1.4 MB role-map array and
full record table remain on Slurm; their hashes are in the campaign and their
construction is fixed in the committed protocol/source. They were not copied
into this compact snapshot. No fitted model or prediction is implied by these
preparation manifests.

## Timed-out first GPU preflight, job 924786

[terminal_status.txt](runtime_preflight_r1_failed/terminal_status.txt) records
`TIMEOUT`, 10 minutes 8 seconds, on `s-005`, with user `omrifahn`, account
`gpu-students`, start `2026-09-24T09:23:32` and end `2026-09-24T09:33:40` in the
cluster's displayed local time. Batch and diagnostic step records are retained.

The [last in-process stage receipt](runtime_preflight_r1_failed/ops/stage_receipts/preflight_r1_924786.json)
still says `running` because termination prevented its final update. It must
not override the terminal scheduler accounting. Stdout/stderr are preserved
under `runtime_preflight_r1_failed/logs/`.

The [extension attempt](runtime_preflight_r1_failed/ops/amendments/preflight_r1_924786_timelimit20.txt)
records the actual request for a **20-minute** time limit and the scheduler's
`Access/permission denied` response. The running job therefore retained its
10-minute limit. Do not report this attempted extension as applied.

Static inspection found two full parquet-checksum passes before the extractor
creates its output directory. The operator observed shared-filesystem waiting;
the copied [I/O diagnosis](runtime_preflight_r1_failed/IO_DIAGNOSIS.md) preserves
the existing allocation's process observations and labels the timing estimate
as administrative I/O, not inference performance. This motivated a longer
operational allowance. No scientific source, feature
recipe or parity tolerance was changed for the retry. The snapshot does not
claim that classifier parity or LD fitting passed before the timeout.

## Superseded v3 and submitted v4

[config_science_v3.json](config_science_v3.json), SHA-256
`7d1b883cdf383b632dcf73bd5ebba559756b492aa5ebab53c6b25f0ab883ae6f`,
was a 30-minute retry candidate. The operator confirmed it was **never submitted**.

[config_science_v4.json](config_science_v4.json), SHA-256
`38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`,
requests the approved **45-minute** `preflight_r2`, preserving the same
`science-source-v2` manifest and sealed campaign. It carries 608 GPU-seconds
from the failed preflight as prior usage and rewires the eventual extraction
dependency to the replacement preflight. The config's full-stage entries are
reserved planning information, not evidence that those jobs were submitted.

[preflight_r2.json](runtime_preflight_r2/ops/submissions/preflight_r2.json) records
job **924803**, submitted `2026-09-24T06:35:51.715274+00:00` (09:35:51 Israel),
with one GPU, a 45-minute limit, account `gpu-students`, and job name
`canonical0924-085200-preflight_r2`. It uses the completed preparation receipt
as a sealed dependency rather than an expired scheduler dependency. The
[intent](runtime_preflight_r2/ops/submissions/preflight_r2.intent.json) is also
preserved. This snapshot records submission, **not completion**; later runtime
and parity results belong in a subsequent evidence snapshot.

[APPROVAL_CONTEXT.md](APPROVAL_CONTEXT.md) distinguishes the coordinating task's
explicit approval, conveyed in agent messages, from the copied scheduler
receipts. There was no separate downloaded approval file to present as one.

No credentials, raw parquet/images, weights, CLS tensors, archive payloads or
large numerical arrays are included. The prior preparation snapshot, SESSION,
PROTOCOL and scientific source files were not edited during this preservation.
