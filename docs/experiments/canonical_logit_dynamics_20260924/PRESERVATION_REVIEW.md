# Final preservation review — September 24, 2026

Root reviewed the completed Slurm backup receipt, the exact released backup implementation and the compact downloaded manifests. This is a provenance/hash review, not a numerical recomputation or a clean-environment scientific replay.

## Remote verification evidence

- Job **924831**, `canonical0924-085200-backup`, user `omrifahn`, account `gpu-students`, completed at **10:25:27 Israel** after 98 seconds.
- Private repository: `omrifahn/polygraph-experiments`.
- New prefix: `canonical_ld_20260924_085200/snapshot_v1`.
- Immutable revision: `903edb4f409a8a1af7617a991d2f3462dce9fb6d`.
- Receipt status: `verified`; repository remained private; all 1,399 pre-existing repository files remained unchanged.
- Archive: 152,412,506 bytes compressed, 373,569,189 uncompressed, **2,361 files**.
- Archive SHA-256: `d9961fe12c22585ae0d72afcecdc5ccd0658163a7d17ac110405d98e84fc3958`.
- File-inventory SHA-256: `fca471481e47bd28ed085a0fdd5f01b4cebe3c5369db28be5491b6c22853021a`.
- Backup-manifest SHA-256: `6ebc3cfe529d61da7cc84b33e2c95e120d5d75d3a2b6ab8d769d80fad0db53c6`.

The archived executable `backup.py` matches the locally inspected implementation by SHA-256. That implementation verifies the immutable revision and repository privacy, preserves pre-existing blobs, downloads every uploaded snapshot file from that revision, checks its size/hash, then compares every downloaded archive member against the downloaded file inventory before setting `status: verified`. Its final success output, final publication receipt and terminal scheduler state agree. No credentials were displayed or copied into documentation.

Root independently matched the compact FILE_MANIFEST and BACKUP_MANIFEST bytes to the receipt; matched all **34 locally available final scientific files** (including the compact prediction binaries in the workspace, freeze, histories, configurations, normalization and evaluation artifacts) against archived member hashes and sizes; and matched the unchanged protocol to the recorded protocol checksum. This verifies correspondence to the remotely verified package without downloading the heavy archive onto the Mac.

The inventory contains the exact science release and source manifest, runtime and dependency verification, configuration v4, campaign and role map, original incoming benchmark evidence, model and optimizer/RNG continuation states, predictions, and the fixed bootstrap outputs. All numerical work and archive download/member verification ran on Slurm; local work was document and file-hash inspection only.

## Scope and limits

Raw image parquets and full CLS shards remain on Slurm, with checksummed source/cache manifests and reconstruction recipes in the archive. Restoration from recorded absolute paths requires the same layout or documented path remapping. Preservation does not demonstrate a fresh-install or raw-image end-to-end replay.

The archive deliberately captures its own backup job's log/receipt in progress. The separate final publication receipt provides the verified immutable revision and is preserved locally. The final human review and curated RESULTS.md were written after upload and are local Git artifacts, not members of the earlier immutable snapshot.

Original sources: workspace `work/canonical-bundle-20260924/final_results/publication/` and `final_results/terminal_status.txt`. Compact byte-identical copies and copy provenance are in [run_records/final_comparison](run_records/final_comparison/INDEX.md).

**Conclusion:** the scientific package is privately backed up and verified at the stated revision; no additional upload or scientific rerun is required for the authorized preservation scope.
