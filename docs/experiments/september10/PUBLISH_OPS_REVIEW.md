# Publication and Slurm interface review — 2026-09-11

Scope: independent source review of `publish.py`, `test_publish.py`, the Slurm stage/submit/preflight interfaces, and the restoration guide. No numerical code, imports, model execution, uploads, or tests ran on the Mac. This review does not approve the evaluator's numerical implementation, which this reviewer previously authored.

## Material findings and the authorized repair

1. **P1: final publication silently omitted evaluator artifacts.** The original final-file allowlist lacked `results.csv`, `bootstrap_draws.npz`, `report.he.md`, the evaluation completion/state records, and the per-model/baseline predictions. It also copied `freeze.json` without its hash-bound analytic-validation reference NPZ/JSON. The resulting Hub snapshot could be byte-verified while missing necessary scientific artifacts. The focused repair now requires the evaluator's complete artifact inventory, verifies its completion manifest and freeze identity, includes every model prediction/receipt plus MSP/entropy scores, and copies/verifies both validation-reference artifacts. A supplied final-results directory that is absent or incomplete fails publication. The final freeze must contain the full 35-fit matrix, or the explicitly approved 25-fit reduction; completed cache/rewiring/readiness provenance and any declared processor metadata are copied and checked.

2. **P1: source bundling required Git metadata absent from Slurm releases.** Operations confirmed that immutable production releases contain `source_manifest.json` and no `.git`. The original publisher unconditionally invoked Git and would fail every real backup, while mocks replaced that function and missed the incompatibility. The repair supports the release manifest directly, requires the registered base commit, verifies every declared file before copying, and checks copied source against the release hashes. Only allowlisted files are bundled. The bundle records the exact source, effective dependency versions, and source-manifest identity; it explicitly records that no Git audit patch is available. Git-backed checkouts retain the original optional audit-patch path. Generated model cards no longer imply that all snapshots contain a Git patch.

3. **Restore identity follow-through:** the publisher's validation restore check now passes the arm to the trainer's cache-identity helper, so `full_rewired` restores check the fixed rewiring-manifest hash as well as the common cache identity.

The reviewer implemented these changes under explicit root authorization, in `publish.py` and `test_publish.py`. They therefore require a separate independent re-review. This document is not an independent approval of the repair itself.

## Controls checked in the original implementation and preserved

- The existing destination must be private before uploading or downloading. The publisher does not create a public repository or change visibility. Download requires a full immutable 40-character commit and checks the resolved revision.
- Two publication commits separate artifact upload from cards linking to the exact artifact revision. `backup_complete` is written only after a pinned download verifies all manifest-listed files. Upload failure is recorded separately from scientific completion.
- A shared receipt-directory lock enforces one central publisher. Receipts/staging must be disjoint from the run tree. Run files, weights, thresholds, scalers, and freeze-bound completion files are read/copied, not rewritten.
- Explicit run/source/result allowlists, relative-path checks and symlink rejection exclude the original Context trees, arbitrary old checkpoint trees, environment variables and credential dumps. This is an upload-scope control, not a claim that arbitrary future source comments can never contain sensitive information.
- All 35 planned fits remain visible in intermediate snapshots; missing/in-progress states do not imply successful training. Completed-run hashes cannot silently change between publications. Final-result publication now has a stricter complete-matrix and complete-evaluation gate.
- A fresh-source restore uses the complete source overlay, not a patch applied twice. The numerical smoke checks fixed validation rows and an alternative batch partition against saved native scores using the prescribed tolerance. Actual clean-environment/model round trips remain to be executed on Slurm.

## Slurm interface findings

`submit.py` requires an exact workflow hash, engineering/science GO, and an explicitly approved stage list. It checks the declared seven-arm/five-seed matrix, permits at most one GPU per job, caps array concurrency at eight, and serializes GPU stages to avoid overlapping arrays. `stage.py` checks the workflow/source hashes, requires successful prerequisite receipts, records child exit status, and propagates failures. Preflight runs only synthetic checks and a separate diagnostic cache; it does not open production test metrics.

Operations confirmed during review that **no production workflow had yet been generated**. Consequently, this review cannot certify that its concrete 35 commands correspond to the declared fit list, that GPU scoring follows the final freeze, or that final backup/failure jobs have the correct terminal dependencies. Those checks must be performed on the generated immutable workflow before release GO. Operations is adding explicit administrative `afterany` dependencies; scientific stages should remain gated by successful prerequisites. Backup authentication/storage failure must not become a training dependency or a successful-science marker.

## Required evidence before publication readiness

- Slurm syntax checks and the updated publisher mocks, including a real manifest-backed bundle without Git, complete final-artifact coverage, missing-calibration/missing-model rejection, immutable revision checking, and no run-tree mutation.
- Independent re-review of this focused repair.
- A real private-Hub upload, immutable download and checksum verification, followed by validation-only native restore checks for the available required architectures, including the rewired null identity and logit scaler. Mock bytes are not numerical model-restore evidence.
- Review of the actual source-hash-bound production workflow and its final/failure publication dependencies. No unconditional production GO is issued here.

The restoration guide's prose about always including an audit diff also needs to be reconciled by its owner with the manifest-backed release path. Generated model cards and bundle metadata already state the distinction accurately.
