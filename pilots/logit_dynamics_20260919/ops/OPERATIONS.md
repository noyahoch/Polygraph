# LogitDynamics operator scripts

All scientific commands run in Slurm. Source editing, source packaging, metadata hashing, upload and submission are coordination operations. Run the orchestration helpers from the local checkout; never use them to run tensor work locally. The existing trusted SSH wrapper remains external and is never committed with credentials.

## Helpers

- `build_release.py --repository PATH --output PATH`: freeze the reviewed source into a new immutable source archive and JSON receipt. Contains source/protocol only.
- `make_config.py --phase preflight|validation|full --release-receipt PATH --root REMOTE_ROOT --output PATH`: prepare a phase configuration. Review caps and commands before upload. `full` is a template; actual approved configuration governs each run.
- `stage_release.py --receipt PATH --config PATH --ssh-wrapper PATH`: upload a new release and configuration with checksum verification. Refuses existing release/config paths.
- `submit.py --config PATH --phase PHASE --approved-config-sha256 HASH`: run on the login node to submit that frozen phase. Uses fsynced intent/receipt files; never retry an ambiguous intent without reconciling the scheduler.
- `stage_runner.py`: invoked only in a Slurm allocation. Verifies source/configuration, stages the checksum-bound existing package bundle to node-local scratch and runs commands.
- `status.py --root REMOTE_ROOT`: read-only scheduler/receipt summary, matching recorded user/name/date rather than job number alone.
- `validate_baseline.py --root REMOTE_ROOT`: Slurm CPU validation of historical score schema, method names, exact IDs/labels and 800 groups of nine views.
- `backup.py --root REMOTE_ROOT --release REMOTE_RELEASE`: Slurm CPU compact backup after completed evaluation. Uses the existing protected credential path without printing its contents; checks the destination is private. An upload failure does not trigger repeat training.

## September19 run

Remote root: `/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500`

Preflight `910039` passed in339 allocated seconds; immutable release `455dca0126e21333`. It checked11 tests, representative96-record exact parity and tiny real-data train/save/reload/optimizer resume. Final CPU validation `910046` passed13 science tests plus real 69-column historical score metadata checks. The first CPU attempt `910041` remains failed because its test selection incorrectly included a GPU-only test; only the selector changed for910046.

Full immutable source: `release-400f1fbbc105a370` (133 files). Archive SHA256 `07f81a1dcd5fe0c0e24e490f023de6c436218b88465be94d365073a3c0fcf2b2`. Scientific implementation commit `4310b9d`; protocol authorization commit `347dacd`. Final exact command arrays and dependencies are in `config_full_v2.json` (SHA256 `6f9db29db035974c6dc08fc249f694ccea1273abde8078de20a8d96d4fb94194`).

The root coordinator approved increasing only extraction's wall-time cap to270minutes. The 233.6-minute sample extrapolation repeats fixed startup overhead; it is a conservative unamortized projection, not a measured full-run ETA. Aggregate nominal GPU reservation including the completed15-minute preflight cap is675minutes (11.25GPUh); ceiling720minutes, max three concurrent GPUs. No scientific settings changed.

DAG: extraction `910058`; seed7/17/27 fits `910059/910060/910061`; all-three freeze `910062`; sequential three-seed prediction `910063`; CPU analysis `910064`; private compact backup `910065`. All dependent stages use `afterok` and invalid-dependency cancellation. No original held-out test is evaluated and no existing models are retrained.

Preserve all run roots, attempts, submission intents, receipts and source snapshots. On resumption, read remote state before any new submission. Keep monitoring lightweight manifests/logs, and route scientific code fixes through the engineering owner. Never silently change tolerances, settings or source for an already sealed campaign.
