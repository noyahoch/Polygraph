# Slurm operations for the September13 layer screen

All numerical work runs inside Slurm allocations. This directory is source code, not permission to submit a new study. Read `RESOURCE_STATUS_20260914.md` before attempting production. The original14-fit matrix was not admitted under the64GPU-hour policy. The user then approved a four-fit block11/union4 comparison, seeds7/17, with an80GPU-hour budget; its durable Slurm chain is submitted.

- `runner.py` verifies the immutable source manifest and workflow hash, prepares the pinned import bundle, executes the declared stage, and writes execution receipts. It is the unchanged source previously staged as `ops/main_runner.py`.
- `stage.sbatch` supplies the shared environment, job-local temporary paths and offline model cache. It calls the canonical `runner.py` path.
- `runtime.py` supplies allocated-job-only scheduler submission and durable JSON helpers. A per-stage fsynced intent precedes `sbatch`. An ambiguous intent requires reconciliation against Slurm; it never triggers an automatic duplicate. Recorded exact requests reuse the existing job ID.
- `merge_budget.py` merges the frozen five parent cases and thirteen compatible continuation cases, preserving failed/TIMEOUT history. It computes explicitly labeled resource scenarios and cannot submit jobs.
- `resource_options.py` reads that admission report and writes alternative resource projections. It does not change the matrix or authorize any option.
- `ramp.py` monitors the predeclared first two fits, requiring two completed epochs including validation, audit and checkpoint saves. It compares complete-cycle wall time to resource caps without selecting by model scores, then uses eight global dependency lanes if a future approved admission permits remaining fits.

All paths are supplied explicitly. The job environment must contain `OMRI_WORKFLOW_PATH` and its exact `OMRI_WORKFLOW_SHA256`; the workflow binds the immutable `code_root`, source hashes and declared stage commands. Pinned reusable environment/data/import bundle live in the September10 experiment. Dataset tensors and diagnostic states stay on Slurm. Do not put tokens, feature caches or raw datasets in Git.

Validation so far: the merge and options modules ran successfully in CPU jobs892167 and892178. The runtime submission-intent guard and ramp passed independent static review. The canonical runner is byte-identical to the successfully executed earlier runner; its canonical shell entry point and submission guard have not yet been exercised as production. No local numerical tests were run.

The active four-fit modules are `admit_four.py` (prior evidence/full-cache binding), `dispatch_four.py` (first two fits), `build_workflow_four.py` (source-only workflow authoring), and `finalize_four.py` (aligned four-fit validation summary plus partial/failure guards and private backup). They preserve the historical protocol/cache while recording the smaller active scope explicitly.
