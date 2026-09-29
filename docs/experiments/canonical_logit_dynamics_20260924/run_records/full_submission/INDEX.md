# Full-phase submission evidence — September 24, 2026

This snapshot preserves the authorization and submission of the canonical LogitDynamics comparison. It is **not a completion receipt or a results report**. The initial queue snapshot, checked at `2026-09-24T06:52:52Z` (09:52:52 Israel), shows extraction running and all nine downstream jobs waiting on dependencies.

## Identity and gates

- Remote campaign: `/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200`.
- User/account: `omrifahn` / `gpu-students`; job-name prefix: `canonical0924-085200-`.
- All ten jobs were submitted at `2026-09-24T09:52:21` Israel time.
- Approved configuration: `config_science_v4.json`, SHA256 `38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`.
- Source release: `science-source-v2`, manifest SHA256 `7949aa99337f4f81e60827f64284c0e04da1456ab8ca78dbe532d0d13c74f50f`.
- The approval follows completed preflight job `924803`; see the separate `../preflight_passed/` snapshot for that 496-row panel. The full extraction must still pass parity for every required row before fitting can proceed.
- The approval permits at most three simultaneous GPUs, 43,200 cumulative GPU-seconds and 7,200 cumulative CPU-job wall-seconds. These are ceilings, not measured consumption.

The extraction submission has no scheduler dependency because the preflight had already completed; the executable gate remains bound to its sealed receipt. No scientific settings or tolerances were changed for this full-phase submission.

## Submitted chain and initial state

| Stage | Job | Scheduler prerequisites | Initial state |
|---|---:|---|---|
| Extraction | 924822 | Completed preflight validated by receipt | RUNNING on s-005 |
| Fit seed 1 | 924823 | 924822 | PENDING: Dependency |
| Fit seed 2 | 924824 | 924822 | PENDING: Dependency |
| Fit seed 7 | 924825 | 924822 | PENDING: Dependency |
| Freeze | 924826 | 924823, 924824, 924825 | PENDING: Dependency |
| Predict seed 1 | 924827 | 924826 | PENDING: Dependency |
| Predict seed 2 | 924828 | 924826 | PENDING: Dependency |
| Predict seed 7 | 924829 | 924826 | PENDING: Dependency |
| Evaluate | 924830 | 924827, 924828, 924829 | PENDING: Dependency |
| Private backup | 924831 | 924830 | PENDING: Dependency |

These are historical job identities. Reconcile user, account, job name, submission time, source hashes and completion receipts before any later operational action; a job number alone is insufficient.

## Preserved files

- `ops/full_phase_root_approval.json`: original recorded full-phase authorization, constraints and immutable source/configuration references.
- `ops/submissions/*.intent.json`: all ten original submission intents.
- `ops/submissions/*.json` without `.intent`: all ten original scheduler submission receipts.
- `initial_status.txt`: initial queue/accounting snapshot and remote configuration/source hash verification.
- `full_phase_submission_receipt.txt`: concise original stage-to-job/dependency mapping.
- `COPY_MANIFEST.json`: original absolute paths, byte sizes and SHA256 checksums for every copied evidence file.

The source archive `submissions.tar.gz` was excluded to avoid duplicating the same receipts. This snapshot contains no tensor cache, model weights, dataset payload, or claim that training, evaluation or backup completed.
