# Final Hugging Face restoration addendum review

**Independent static GO** for the bounded addendum below. This is approval to schedule
the verification, not evidence that the future restoration has already passed.

- `slurm/post_backup_verify.py`: `04a932dd00727ec160d504fea47e7c0e27f5ba558e277ecc6ba927de01996e97`
- `slurm/post_backup_verify.sbatch`: `8c9a55a72bb8ef3e3a918281f72c33483cbfb0c6c964894e3b402af4c5822622`
- Existing scientific workflow: `1bb59c6e0bffd7b170e8737a8954678fd9c725cff7816321ca8779d8b7d98914`.
- Existing scientific release: `81b4e83e8bc2a823`.

The separate job uses one GPU, two CPUs, 32 GB RAM and a 60-minute ceiling. Operations
confirmed an explicit scheduler dependency `afterok:879968`, with
`--kill-on-invalid-dep=yes`, and an immutable addendum directory containing the reviewed
manifest and exact two files. The current submitted phase-two workflow is not modified.

Before downloading, the wrapper requires the exact successful final-backup execution,
scientific completion, the complete 35-run freeze, and the matching authoritative HF
receipt. It downloads the receipt's immutable commit into a new job-specific server
directory and verifies the snapshot, all declared artifacts, final evaluation inventory,
freeze and complete allowlisted source hashes against the approved release.

Restoration executes in a fresh Python process. The wrapper clears inherited scientific
checkout paths from `PYTHONPATH`, preserves only the checksum-verified staged third-party
packages, puts the downloaded source first, and explicitly checks that the scientific
modules came from that source. It verifies the Python version, every recorded effective
dependency version and the corresponding requirements lock. This reuses the existing
pinned Slurm environment; it is not a fresh dependency installation or a demonstration of
independence from installed third-party/native libraries.

The representative `full_graph/seed1` audit uses the existing restored-checkpoint API and
compares 48 validation scores and threshold decisions, plus an alternate batch partition.
It does not fit a model, rescore detector test data, upload, or install dependencies. The
all-35 claim concerns artifact/source verification; numerical restoration in this job is
limited to that representative completed run. No real network-cut recovery is claimed.

Inputs are read without modifying the frozen runs, cache, downloaded scientific artifacts
or submitted workflow. Temporary bytecode/native caches are job-local; download metadata
and verification receipts use separate directories. Credentials are supplied through the
existing private token path and their values are not logged. The overall success receipt
is written only after the child audit succeeds, the downloaded artifacts verify again and
the authoritative publication receipt remains unchanged. Failure leaves an explicit
failed receipt rather than a completion claim.

No numerical work or Python execution was performed on the Mac for this review. Actual
success remains contingent on the future Slurm job's `restoration/JOB/verification.json`
reporting `status: passed`.
