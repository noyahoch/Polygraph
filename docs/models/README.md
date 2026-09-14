# Private checkpoints and exact reuse

The authorized destination is the **private model repository**
[`omrifahn/polygraph-experiments`](https://huggingface.co/omrifahn/polygraph-experiments).
The [September 10 model catalog](SEPTEMBER10_CATALOG.md) links all 35 completed fits,
their exact configurations, final result artifacts and executed source at verified immutable
revisions. Final backup and the separate native restoration audit completed on September 11.
The audit verified all artifact/source/dependency bindings and numerical agreement for
`full_graph/seed1` on 48 validation rows, using freshly downloaded artifacts and source with
the existing pinned Slurm environment. The [catalog](SEPTEMBER10_CATALOG.md#interpretation-and-verified-restoration)
records the receipt and the limits of that verification; this page describes the procedure.

The collection contains all seven prespecified arms and seeds **1, 2, 7, 17, 27**.
Every snapshot lists all 35 planned fits, including those not started or incomplete.
Neither the repository front page nor its model cards selects one best seed. The
[completed September 10 readout](../experiments/september10/RESULTS.md) reports the
registered comparison and its limitations.

## Storage and credentials

Create the model repository privately through the user-authorized setup, then authenticate
on the Slurm host with a token restricted to writing that repository. Use the Hugging Face
credential store (`hf auth login`) or a securely injected `HF_TOKEN`. Do not put tokens in
shell command arguments, source, run configs, receipts, documentation or dependency dumps.
The publisher verifies that the existing repository is private before every upload. It
fails if access is missing or the repository is public; it never changes visibility.

Use the installed, recorded `huggingface_hub` version and `hf_xet` on Slurm. The API uses
[`HfApi.upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload), which
supports resumable, deduplicated uploads. `upload_large_folder` is deprecated in current
[Hub documentation](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder).
Ops must verify the actual installed API/version before the first publication.

Preflight records estimated snapshot bytes and the Hub's reported repository storage when
available. Remaining account quota is **unknown** unless the operator supplies a known
available-byte ceiling with `--quota-available-bytes`. Repository storage usage is not
remaining quota. Backend quota failures retain the Slurm artifacts and mark the backup
incomplete. There is no paid storage purchase, billing change, or automatic compute service.

## One centralized Slurm publisher

Run one CPU job using the shared receipt directory for this collection. The directory must
be outside the immutable run tree; `fcntl` prevents two publishers from using it at once.
All commands below execute in a Slurm allocation, including tests and checksum verification.
The MacBook performs source editing and orchestration only.

```bash
python -m pilots.topology_20260910.test_publish

python -m pilots.topology_20260910.publish \
  --run-root /absolute/slurm/experiment/runs \
  --repo-id omrifahn/polygraph-experiments \
  --receipt-dir /absolute/slurm/experiment/publisher-receipts \
  --code-root /absolute/slurm/project \
  --cache-metadata-root /absolute/slurm/experiment/cache \
  --once
```

For the central background job, omit `--once` and pass `--interval 3600 --max-cycles 48
--terminal-marker /absolute/slurm/experiment/workflow-terminal.json`. The marker means
that the workflow has ended, including a documented failed or stopped workflow; it does
not mean that the scientific experiment succeeded. The same central wait loop checks the
35 registered completion markers at most 30 seconds apart and wakes for each new fit.
Its pre-upload baseline preserves completions arriving during upload or terminal handling.
The configured 48 progress cycles allow at most 35 additional completion-event cycles and
one terminal catch-up, within the unchanged 48-hour Slurm wall-time ceiling. Separate
counts and snapshot identity are saved in the external `last_cycle.json` receipt. The
publisher takes its final snapshot and stops when the terminal completion set is covered.
It also stops at the bounded cycle limits or after two
consecutive failures. Rerun the same command after a diagnosed interruption; durable staging
and phase receipts resume the upload. A successful upload receipt is not a restore-smoke result.

Once the final matrix is frozen, add `--freeze /absolute/slurm/experiment/freeze.json`.
After `evaluation_complete.json` exists, also add `--results-root /absolute/slurm/experiment/results`.
Final result publication requires the freeze and the evaluator's complete checksummed
artifact inventory. The separate publisher never rewrites
the freeze file or any checkpoint, scaler, validation threshold or completion file.

## Exact upload scope

For each `ARM/seedN`, the publisher enumerates `config.json`, `history.json`,
`best.safetensors`, `latest.pt`, `validation.npz`, `validation.json`, and `complete.json`.
It also permits named attempt/resume/restore audit files, explicit scaler/threshold files,
and named final metric/prediction files when present; see the small explicit `RUN_FILES`
and `FINAL_FILES` allowlists in [publish.py](../../pilots/topology_20260910/publish.py).
The actual trainer stores fitted logit normalization in `config.json` and its frozen
threshold in `validation.json`; their hashes therefore bind these quantities.

Every individual file is copied to separate staging with SHA-256 and before/after identity
checks. Completed fits additionally require matching configuration, selected checkpoint,
validation arrays/JSON, history and resumable-state hashes. Validation preprocessing must
match the config. During training, `latest.pt` is the authoritative resume state; the
separate best/history files may lag an epoch boundary, and the card says `in_progress`.
Only `training_complete` fits are eligible for `load_run` inference and restore auditing.

The freeze mapping binds each `runs["ARM/seedN"]` entry through `config_sha256`,
`best_sha256`, `validation_npz_sha256`, `validation_json_sha256` and `complete_sha256`.
Previously completed bound artifacts cannot silently change on a later publication.
Cache metadata includes `manifest.json`, `cohort.json`, `protocol.json`, and any declared
`processor.json`. A frozen publication also includes the hash-bound cache validation
receipt and rewiring manifest, plus both analytic-validation reference files beside the
freeze. Final results include the complete evaluator manifest, tables, report, bootstrap
draws, and every required model/baseline prediction and model receipt. No image archives,
tensor caches or source photographs are uploaded.

The code bundle contains the exact allowlisted current Python source, new experiment
package, protocol/docs, and a dependency/version fingerprint, tied to base commit
[`9af8890405297c0f37dbaa54f6be29a84cb0c37e`](https://github.com/noyahoch/Polygraph/tree/9af8890405297c0f37dbaa54f6be29a84cb0c37e).
It includes the imported `polygraph` source. Restoration uses this complete source overlay
and does not require a laptop or a checkout containing historical checkpoints.

An immutable Slurm release supplies `source_manifest.json` with the full base commit and
file hashes. The publisher verifies every declared file, copies only allowlisted source,
and records the release-manifest hash and included file hashes in `bundle.json`; this path
requires no Git metadata and includes no Git audit patch. For a Git-backed checkout,
tracked and untracked allowlisted files form the same complete source overlay, with an
additional `working-tree.patch` audit diff. **Do not apply that optional patch to the
finished overlay again.** No pushed Git branch or new Git commit is implied.

There is no recursive upload of the project, parent `Context`, WhatsApp/transcriptions,
credentials, unrelated files, historical data, old run trees, or other checkpoints.
Symlinks and escaping paths are rejected. Dependency fingerprinting records names and
versions only; it does not dump environment variables or credential-bearing package URLs.

## Revisions and verification

Publication has two phases: first upload the checksummed artifacts, then upload model
cards with links pinned to that immutable artifact commit. `publication.json` records the
artifact commit, snapshot identity and documentation hashes. A second immutable revision
contains both artifacts and their final cards. The external `<snapshot-id>.json` and
`latest.json` receipts record that revision and report `backup_complete: true` **only after
downloading and verifying every manifest-listed byte**. Partial/interrupted snapshots are
never presented as a complete verified backup. Receipt files stay outside bound run files.

Copy the full 40-character `revision` from a successful receipt, not a moving branch name:

```bash
python -m pilots.topology_20260910.publish \
  --repo-id omrifahn/polygraph-experiments \
  --revision FULL_40_CHARACTER_COMMIT_FROM_RECEIPT \
  --download /absolute/slurm/restore/snapshot
```

For a fresh environment, the downloaded bundle is at
`/absolute/slurm/restore/snapshot/topology_20260910/code`. Provision a clean Slurm
environment using its `environment.json` and `requirements.lock.txt`, with the recorded
Python and matching torch/CUDA stack. Package versions alone are not a portable guarantee
that a different CUDA platform will work. Do not dump or copy authentication into that
environment fingerprint. Run Python from the complete `code/source` overlay, which
contains both `polygraph` and `pilots`; no unpublished Mac files are needed.

Run the validation-only audit from that source root using the same pinned cache on Slurm:

```bash
cd /absolute/slurm/restore/snapshot/topology_20260910/code/source
python -m pilots.topology_20260910.publish \
  --verify /absolute/slurm/restore/snapshot \
  --smoke-run full_graph/seed1 \
  --cache /absolute/slurm/experiment/cache \
  --device cuda \
  --smoke-output /absolute/slurm/restore/receipts/full_graph-seed1.json
```

Repeat for the required representative architectures and completed fits. This invokes
`load_run(run_dir, device) -> (model, config)`, verifies the exact implementation and
preprocessing, and compares up to 48 fixed validation-reference predictions. It preserves
the original batch layout (materializing at most one 256-row logit batch), also checks an
alternative partition, and enforces `atol=rtol=1e-5`, stable separated-pair ordering and
threshold decisions outside the tolerance neighborhood. The audit never reads the test
split. It writes a separate receipt and does not alter the downloaded checkpoint tree.

Full inference uses `predict_split(run_dir, cache, split, device, freeze=None)`, returning
aligned metadata and native error `score`/`logit` arrays. `split="test"` additionally
requires the original authorized freeze, unchanged bound files and matching cache identity.
Do not rewrite the freeze to point to different content or use restored copies to create
another test-informed model-selection opportunity.
