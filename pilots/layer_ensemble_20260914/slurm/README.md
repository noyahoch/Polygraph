# September14 core Slurm workflow

User-approved scope: four single-layer detectors (`block2`, `block5`, `block8`, `block11`), seed7, exactly20 epochs, followed by a learned four-input logistic head and a matched last-layer-only head. Source-photo roles are1600 base training,400 checkpoint selection,400 meta training and800 development evaluation, with all nine variants kept together. Unions, extra seeds and the original held-out test are deferred.

All extraction, training, checks and statistics run on Slurm. The Mac is used only for source editing and coordination. There is no paid-resource fallback. Local commits are authorized; pushing is not.

The shared capture job892193 and its original source remain untouched. On September14 at13:15 Israel, obsolete pending jobs892205/892206/892207 were cancelled before they created any fit jobs. The new namespace references the shared feature cache without copying it.

## Accepted jobs

Submitted September14 at13:25:09 Israel:

- Guardian892491: independent CPU allocation, verified RUNNING on `rack-ai-01` at13:25:45.
- Preflight892492: one GPU, after successful capture892193.
- Dispatcher892493: CPU, after successful preflight892492.

The dispatcher submits all four fresh fits in parallel only after complete-cache and role/source-bound preflight checks. Actual fit IDs then appear in `manifests/dispatch.json` and the fsynced `manifests/submissions.tsv`. Postprocessing depends on all four successful fits and runs meta prediction, the two CPU heads, development prediction and CPU evaluation in one GPU allocation to avoid another GPU queue.

Immutable release: `92c616b82342449d` (106 files). Workflow SHA256:
`7556cdfe8a737f9505c0dc1ddd89f9d6a02011edc53a0f33ca99edb5e00fe74c`.
`launch.py` verifies every source hash before initial submission; `runner.py` repeats verification in each allocated stage. Never edit an active release or its workflow.

## Deadlines and accounting

All timestamps use Asia/Jerusalem. All four base fits must finish20 epochs and checkpoint audits by September14 at23:00. Development predictions must finish by September15 at04:00. Timely predictions may proceed to CPU statistics/reporting until07:00, targeting05:00. The independently running guardian preserves a complete report or an explicitly incomplete report; it never ranks truncated models as a completed comparison or creates a subset ensemble.

The ceiling is50 GPU-hours. Conservative nominal caps are: prior diagnostics265min, ongoing capture710min, preflight30min, four fits385/290/285/265min, and postprocessing300min (including a15min CPU-head interval while the GPU remains allocated). Total2530min=42h10, leaving7h50 unused allowance. Prior diagnostic allocations actually consumed4014sec; substituting that observed usage gives38.865 GPU-hours of finished actual usage plus ongoing/future reservations. These ceilings are not predicted actual runtimes. CPU jobs are excluded. Final accounting replaces terminal GPU job caps with their actual scheduler elapsed time.

## Artifacts and recovery

Remote root:
`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_ensemble_20260914`.

Roles and execution are frozen as `role_map.json` and `execution.json` during preflight; their hashes are not claimed before creation. Runs, optimizer/RNG checkpoints and histories stay in `runs/ARM/seed7`. `base_freeze.json`, meta predictions, both heads and `heads_freeze.json` precede development predictions. Evaluation produces `evaluation/complete.json` with artifact hashes.

The guardian writes `manifests/terminal.json`, `REPORT.md`, scheduler accounting and a compact reproducibility snapshot. Source, checkpoints, predictions and reports are backed up to the existing private Hugging Face repository `omrifahn/polygraph-experiments` under `layer_ensemble_20260914/`. Feature tensors, raw images and credentials are excluded. Upload failure has its own receipt; course-storage artifacts remain available.

Every submission has a durable intent before `sbatch`. An ambiguous intent must be reconciled with Slurm before resubmission; it is never treated as permission to duplicate a job. No automatic scientific retries, new seeds or unions are configured. The remote workflow survives laptop closure; an08:00 chat message is not guaranteed while the app is closed.
