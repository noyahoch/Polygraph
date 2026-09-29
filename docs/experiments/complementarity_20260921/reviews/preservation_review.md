# September 21 complementarity: preservation review

**Verdict: ACCEPT the completed preservation on the available receipt and manifest evidence.** No required artifact category is missing for this follow-up. This review is independent of the operator's report writing, but it did not make a separate live Hub request or download the archive again. It checked local metadata, hashes and inventory only; no model, prediction, metric or bootstrap calculation was repeated.

Evidence root:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/complementarity-20260921/ops/`

The principal sources are `final_compact/publication/receipt.json`, `final_compact/publication/bundle/FILE_MANIFEST.json`, `final_compact/statistics/complete.json`, `final_compact/report/complete.json`, `final_compact/ops/parent_integrity_after_fits.json`, `CLOSING_RECEIPT.json` and `retrieved_report_checksums.json`. Pre-prediction fit/gate coverage was separately accepted in the adjacent `gate_review.md`.

## Immutable private backup

The publication receipt records successful verification of private model repository `omrifahn/polygraph-experiments`, immutable revision `769e8b48c9e33edaebac81e97405945197054c11`, prefix `complementarity_20260921_135802/snapshot_v1`. It records `status=verified`, `private=true` and `archive_members_verified=true` at 2026-09-21 11:22:42 UTC.

The receipt binds the archive to SHA-256 `b0022d597ffda2c336e7791526635d7ade601fe184460cecb762b1b64238448c`, size 48,354,170 bytes. It records 2,429 archived files. The local `FILE_MANIFEST.json` hashes to `5a62b0d97a95e35639fa318c5067ac05eb71c6eab7f324605cb27b8cfa12b290`, matching that receipt. Direct metadata inspection confirms the manifest contains 2,429 members.

Static inspection of the exact released backup implementation shows that its successful receipt follows: private-repository and immutable-revision checks; downloading every uploaded bundle file from the immutable revision; verifying byte counts and SHA-256; checking the downloaded archive's complete member inventory; and verifying each member's size and SHA-256 against the downloaded manifest. Thus the receipt reports actual remote-byte/member verification, not merely successful upload. This review relies on that executed receipt and source, rather than claiming its own fresh remote verification.

## Required inventory

The archived manifest includes:

- All 160 files in the exact release source manifest, with identical listed hashes, including the new package, operational entrypoints and reused parent implementation. The frozen protocol, operational configuration and release/source manifests are present.
- Actual per-job environment inventories, including package/import paths, versions, numerical-thread settings and CUDA workspace configuration. This preserves the environment description; it is not a bundled fresh-install environment or container.
- Campaign and source-photo split assignments, parent role maps, the fixed draw JSON/NPZ manifests, cache metadata, semantic evidence and all nine new 85-feature caches.
- All nine A/B/C readouts, each with selected native checkpoint, portable safetensors, resumable optimizer/RNG state, full history, train-only normalizer, validation predictions, config and completion receipt. Their completed 100-epoch coverage is documented in `gate_review.md`.
- The nine original DG/DS/DDprime combiner states and scalers in `fusion/models.json`, their fitting inputs and freeze receipt. The JSON inspected for the gate review is the same hash included in this archive.
- Frozen prediction matrices and their gate receipts for both studies, original D/head model artifacts and scalers, and bound parent D/G/S score archives.
- Exactly the fixed bootstrap draw filenames `statistics/draws/0000.json` through `1999.json`, with no missing or extra draw IDs. Every file named by the statistical completion receipt is present in the archive inventory with its exact expected hash. This includes draw states, fitted combiner/scaler states within them, fixed multiplicities, timing and weighted/literal verification evidence.
- English and Hebrew reports, the machine-readable results, preflight receipts, the global evaluation gate, job submissions, operational logs and preservation evidence.

`statistics/complete.json` states that all 2,000 draws were processed and all primary intervals are available. This confirms recorded completion status, not an independent numerical certification of those intervals. The exact feature/fit/freeze checks remain those reviewed previously; no missing-seed substitution is recorded.

## Local report and preservation chain

I independently hashed the nine compact report/statistical files listed in `retrieved_report_checksums.json`; all match its expected immutable-snapshot hashes. The statistical completion file hashes to `18726dcb309c8310e977636ab21d1fbfb775eb87f478a64fcc8a80e85e162e12`, exactly the value bound by `report/complete.json`. The report-completion file hashes to `e328586c4a7e9e295c891cf9c33b4d7da6e4ef08fc64c54a92a741f406d81fc7`. The JSON report and statistical results share SHA-256 `1e8d15e152127de7e007448c204b5b65f94c28c59308bdba6271ecdf615cd000`.

The parent-integrity receipt records unchanged hashes for the bound baseline/LD inputs, old selected heads/probes, original predictions/results and prior replay audits, including the previously failed CPU audit. It records 133 parent source files and no source failures. The backup receipt additionally records that all 1,391 pre-existing Hub files retained their blob identities. These are checks of the named parent inputs/source and repository files, not a claim of a filesystem-wide audit of every old experiment.

## Precise limitations

The original raw-image/CLS caches remain on Slurm, as declared; the new feature caches are archived. This snapshot contains parent G/S scores, not duplicate copies of every old G/S training checkpoint. The completed parent snapshots remain unchanged. Recomputing G/S from images still requires their parent assets and workflow.

Frozen manifests retain absolute server paths; restoration elsewhere requires reconstructing that layout or documenting a path mapping. Fresh-environment/raw-image end-to-end reproduction was not performed. The running backup job's own receipt/log is necessarily captured in an in-progress state inside its archive; the separate completed publication receipt supplies the final Hub revision and verification result.

Final scientific interpretation and independent result review belong to the root/statistical reviewers. There is no preservation defect here that warrants retraining, prediction repetition, new experiments or changing the frozen method.
