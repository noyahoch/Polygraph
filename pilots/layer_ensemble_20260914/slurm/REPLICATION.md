# Separately authorized seed-17/27 replication

This is new orchestration, not an amendment to the September14 seed-7 release.
It does not submit anything by default. Implementation approval and permission
to use at most eight concurrent GPUs **are not scientific launch approval**.
A future launch needs its own explicit identity, future cutoffs, resource caps,
immutable source inventory and approved, expiring authorization JSON.

No training/evaluation on real data, extra seed, retry, benchmark, publication,
commit, push or upload is part of implementation validation. The original
held-out test remains closed. The seed-7 result remains a late diagnostic,
separate from the new primary seed-17/27 analysis.

## Static server-side graph

Exactly eight `sbatch` submissions declare eighteen bounded allocation tasks:

| Submission | Tasks | Resources | Dependency |
|---|---:|---|---|
| `guardian` | 1 | CPU, independent, fixed wall-time cap | none |
| `prepare` | 1 | one GPU: source validation plus both readiness checks | `after:guardian` |
| `fits` | 8 | one GPU/task, array throttle `max_concurrent_gpus` | `afterok:prepare` |
| `meta_heads` | 2 | one GPU/task, throttle `min(2, cap)` | `afterok:fits` |
| `joint_heads` | 1 | CPU | `afterok:meta_heads` |
| `dev_eval` | 2 | one GPU/task, throttle `min(2, cap)` | `afterok:joint_heads` |
| `evaluate` | 2 | CPU | `afterok:dev_eval` |
| `aggregate` | 1 | CPU | `afterok:evaluate` |

The fit-array order is seed17 × block2/5/8/11, then seed27 × block2/5/8/11.
Each fit is fresh, uses the existing fixed20-epoch recipe, and writes only
`ROOT/seedN/runs/ARM/seedN`. A dependency on an array's base Slurm ID requires
**every** array task to succeed. No dispatcher expands the graph.

Both complete four-base sets precede the meta/head array. Each meta task freezes
that seed's bases, exports only its meta role and fits both prescribed heads.
The singleton joint barrier calls:

```text
python -m pilots.layer_ensemble_20260914.replication freeze-heads --root ROOT
```

Both dev exports wait for that joint barrier, not just their own seed's head.
CPU per-seed evaluation follows the dev array; the final CPU stage calls
`aggregate_replications --root ROOT --out ROOT/evaluation
--late-seed7-root SOURCE_ROOT`. GPU allocations retain their reservation while
their meta task runs CPU head fitting; that time is **not** removed from billing.

Every stage uses `--no-requeue` and `--kill-on-invalid-dep=yes`. No automatic
scientific retry, subset ensemble, changed epoch budget or alternate matrix is
available.

Prepare first calls scientific `replication prepare`, then runs the existing
`smoke` readiness CLI sequentially for seed17 and seed27 in that same future GPU
allocation. Each check has an explicit `--max-seconds` bound and writes
`ROOT/seedN/manifests/readiness.json`. It reads only base/checkpoint roles;
temporary diagnostic steps are not scientific fits or meta/dev exports.
Both source/execution/role-bound, all-four-arm readiness reports must pass
before **any** of the eight fits. Their complete prepare allocation is charged
to GPU reservations. This is a future authorized prerequisite, not a current
experiment and not the CPU-only synthetic implementation validation.

## Namespaces and immutable inputs

Use disjoint, canonical absolute paths:

```text
SOURCE_ROOT/                     existing late seed-7 experiment; never modified
ROOT/                            new scientific root; empty until prepare
    replication.json
    role_map.json                byte-identical original role map
    joint_heads_freeze.json
    seed17/{execution.json,role_map.json,feature_cache,runs,...}
    seed27/{execution.json,role_map.json,feature_cache,runs,...}
    evaluation/
CONTROL_ROOT/                    separate sibling, never inside either experiment
    releases/RELEASE/
        source_manifest.json
        <exact inventoried source tree>
    workflow.json                frozen before the first sbatch
    authorization.json           frozen separately before the first sbatch
    logs/
    manifests/{launch.json,guardian.json,terminal.json,submissions.tsv,...}
    manifests/submission_intents/STAGE.json
    manifests/execution/
    job_work/                    job-owned runtime staging, normally cleaned on exit
```

The separate control namespace is essential: scientific `replication prepare`
requires an empty new root. It alone copies role-map bytes, binds historical
provenance, creates the new identity and links the original feature cache for
read-only use. No extraction, cache mutation or old-checkpoint copying occurs.

The release must contain exactly the source files in `source_manifest.json`
(apart from that manifest and bytecode caches). Its manifest has `release_id`
equal to its directory name and `source_sha256` equal to the workflow inventory.
The initial launcher and every allocated runner verify those hashes; the runner
also verifies that it is executing from that release. Never edit a staged active
release, workflow, authorization or historical artifact.

Runtime reuse follows the existing experiment patterns: `SOURCE_ROOT/env/bin/python`,
the hash-pinned import bundle, and the hash-pinned dependency-overlay manifest
and installed files. Verification/copying occurs inside Slurm. The existing
`atomic_json`, checksum and allocation helpers are reused. The seed-7
`submit_stage` helper is deliberately not reused: its scope, paths, budget and
stage wrapper are specific to that historical release.

## Pure planning API

`build_plan(config, now=aware_datetime)` and
`validate_plan(plan, authorization=None, now=aware_datetime)` are deterministic,
stdlib-only APIs. They do not inspect datasets, create directories, import ML,
query Slurm or submit jobs. `check_time=False` is only for reconstructing an
already frozen plan during allocated execution/reconciliation, not admission.

`config` requires exactly these fields; there are no default hardware, time or
budget claims:

| Field | Meaning |
|---|---|
| `run_id` | explicit new ID, 1–80 letters/digits/underscore/hyphen |
| `root`, `source_root`, `control_root` | disjoint canonical absolute paths |
| `code_root` | exactly `control_root/releases/RELEASE` |
| `python` | exactly `source_root/env/bin/python` |
| `source_sha256` | complete source-only release inventory |
| `environment` | `import_bundle_sha256`, `dependency_manifest_sha256` |
| `deadlines` | timezone-qualified ISO `base`, `predictions`, `evaluation` |
| `resources` | `account`, `gpu_partition`, `cpu_partition`, `gpu_constraint`, `gpu_cpus`, `cpu_cpus`, `gpu_memory_mb`, `cpu_memory_mb` |
| `caps` | positive integer wall minutes for all eight named submissions |
| `max_concurrent_gpus` | 1–8; fit and prediction/meta array throttles |
| `external_reserved_gpus` | concurrent reservations outside this workflow |
| `gpu_budget_minutes` | explicit total GPU-minute ceiling |
| `external_gpu_minutes` | explicit reservations charged outside this workflow |
| `submission_grace_minutes` | bounded static-launch commit window, 1–10 |
| `guardian_grace_minutes` | bounded reporting grace after evaluation cutoff, 1–30 |
| `readiness_max_seconds` | each of the two sequential readiness checks, 60–1500s |

Stage wall caps cannot exceed4320min. CPU requests are at most6/task, memory at
most32000MB/task; the guardian always requests1CPU. These are hard admission
bounds, not measurements or promises of queue availability.

```text
workflow_GPU_minutes = prepare + 8*fits + 2*meta_heads + 2*dev_eval
reserved_GPU_minutes = workflow_GPU_minutes + external_gpu_minutes
max_concurrent_gpus + external_reserved_gpus <= 8
reserved_GPU_minutes <= gpu_budget_minutes
```

CPU minute reservations also count every array task and all CPU cores. Fits may
be conservatively throttled below eight: admission counts `ceil(8/cap)` full fit
waves, and `ceil(2/cap)` full meta/prediction waves. Admission checks the complete
cap-based critical path against each cutoff and the authorization expiry.
These are conservative allocation ceilings, **not measured throughputs**.
Queue delay is unknown. A delayed stage is refused if its full cap no longer
fits before its cutoff. The guardian cap must cover the evaluation cutoff and
its reporting grace.
The prepare cap must exceed both readiness time limits combined, leaving time
for source validation and environment verification; the full cap is reserved.

For a future reviewed config, planning/validation are explicit and inert:

```bash
python -B -m pilots.layer_ensemble_20260914.slurm.replicate \
  --config future-config.json --out future-workflow.json

python -B -m pilots.layer_ensemble_20260914.slurm.replicate validate \
  --workflow future-workflow.json --sha256 REVIEWED_WORKFLOW_SHA256
```

Output includes the exact workflow checksum and `authorization_template`.
That template has `approved: false`, `authorized_at: null`, `expires_at: null`;
it is intentionally **not submission permission**. Workflow checksums bind the
exact pretty-printed, sorted UTF-8 JSON bytes, including the final newline.

## Explicit future submission authorization

Only after separate scientific approval, supply the completed authorization
with `approved: true`, `purpose: scientific_replication`, the exact workflow and
release hashes, run ID, roots, fixed matrix, deadlines and complete resource
claims from the template. It also binds zero retries and no original-test
access. `authorized_at` and `expires_at` must be timezone-aware; authorization
must currently be active, cover the full cap-based chain, and expire no later
than the evaluation cutoff.

```bash
# FUTURE SCIENTIFIC LAUNCH ONLY. Do not run as implementation validation.
python -B -m pilots.layer_ensemble_20260914.slurm.replicate validate \
  --workflow future-workflow.json --sha256 REVIEWED_WORKFLOW_SHA256 \
  --authorization separately-approved.json --submit
```

Providing approved JSON **without `--submit` still never submits**.
`--submit` cannot be combined with `--out`. Runtime/reconciliation modes cannot
submit. No shell-expanded scientific command or arbitrary task override is
accepted; the graph is reconstructed and compared with the frozen plan.

The initial operator records the independent guardian first. Prepare waits
inside its declared allocation for the bounded, durable **entire static-launch
commit** before touching scientific inputs. This avoids a fast-start/receipt
race and prevents partial submission from launching fits. The guardian watches
the commit window independently. After a committed launch, laptop shutdown
does not affect dependencies, stage deadlines or terminal reporting.

The guardian itself must run in its recorded singleton Slurm allocation. It
allows only the declared submission-grace interval for the initial `sbatch`
acknowledgement race. An ambiguous acknowledgement can be adopted only after
unique name/comment reconciliation proves the current allocation; no new job
is submitted. A different allocation or array task cannot observe, cancel or
finalize this workflow as its guardian.

## Disconnects, failures and terminal evidence

Each `sbatch` has a locked, fsynced intent **before** invocation, a unique
workflow/stage-bound job name and comment, and a durable actual ID afterward.
Repeated submission can reuse an identically recorded ID, never retry a fit.
An uncertain acknowledgement remains ambiguous even if the queue appears empty.
Reconciliation checks both `squeue` and `sacct`, with exact names/comments and
array-parent de-duplication. Zero matches, multiple matches, accounting lag or
query errors never justify another `sbatch`.

```bash
python -B -m pilots.layer_ensemble_20260914.slurm.replicate reconcile \
  --workflow CONTROL_ROOT/workflow.json --sha256 REVIEWED_WORKFLOW_SHA256 \
  --authorization CONTROL_ROOT/authorization.json \
  --authorization-sha256 ORIGINAL_AUTHORIZATION_SHA256
```

Reconciliation may record a uniquely recovered ID; it never submits. It remains
available after approval expiry. A failed static launch is terminal: reconcile
its IDs and preserve the failure, rather than expanding or retrying it.

The independent CPU guardian has a fixed Slurm wall cap, a finite cutoff/grace
window and bounded scheduler calls. It cancels only this workflow's bound
scientific IDs on failure, cutoff, expired authorization, uncommitted launch
or observation failure. Cancellation errors are retained explicitly. Each
scientific runner separately bounds its subprocess by allocation cap,
authorization expiry and scientific cutoff, forwards termination to its own
process group, and refuses work without a recent bound guardian heartbeat.
A guardian lost to uncatchable failure cannot claim completion; Slurm caps
and per-stage deadline checks still bound work.

Terminal commit uses a dedicated lock and is write-once. The original bound
guardian may read/repeat its existing result idempotently, without rewriting
bytes or repeating cancellation. An incomplete terminal receipt can never
be upgraded to complete under the same identity (nor can completion be
downgraded). Foreign or malformed terminal identities fail closed. Guardian
allocation failure/unknown liveness cannot be ignored to manufacture success.
Concurrent publishers wait at most35seconds for the first writer, then return
its bound committed result unchanged; an unavailable lock fails closed.
Synthetic tests exercise both identical and conflicting concurrent publishers,
asserting exactly one terminal write and one cancellation attempt.

`terminal.json` is successful only after **all required array elements** and
singleton scientific stages complete and the guardian verifies:

- the new replication identity and both-seed head-freeze bindings;
- both seed-specific, base/checkpoint-only readiness reports and their inputs;
- both per-seed complete evaluation inventories, their actual file hashes,
  current execution/roles/base/head/dev bindings, and prospective cutoffs;
- the complete aggregate inventory, its file hashes and bindings to both
  per-seed completion receipts, the replication manifest and joint freeze.
- matching report/bootstrap metadata, current new-code implementation bindings,
  and unchanged prebound historical provenance (without comparing seed7's
  deliberately historical implementation hashes to the new code).

The aggregate inventory is exactly `report.json`, `REPORT.md`, `bootstrap.json`,
`bootstrap_source_counts.npz`, `bootstrap_draws.npz`, and `scores.npz`; its inputs
must bind both new seed completions and the prebound, explicitly late seed7.

A lone `complete.json`, an omitted seed/file, an unbound aggregate, a changed
score archive, unknown scheduler state or a late result cannot report success.
The guardian only hashes existing artifacts in Slurm; it does not load arrays,
checkpoint tensors or recompute metrics. Historical seed-7 files are never
rewritten or relabeled as on-time.
Neither the sign of the effect nor whether its interval excludes zero can
affect completion. The allocated scientific stages perform the parent's full
manifest/joint-freeze validators; the stdlib guardian verifies their bound
receipts and source-provenance hashes without loading scientific arrays.

## Synthetic implementation validation

Authoring/short inspection is local. **All tests, statistics and large-file
hashing run in Slurm**, not on the Mac. The test module requires `SLURM_JOB_ID`,
uses synthetic files in a job-owned project directory and mocks scheduler
submissions, reconciliation and cancellation.

The unit selector for this file is:

```text
pilots.layer_ensemble_20260914.slurm.test_replicate
```

It covers inert defaults, the exact fit matrix, both readiness checks, the cross-seed both-head gate,
array/resource reservations and throttles, cutoff/expiry/authorization errors,
immutable source and intent checks, disconnect reconciliation, static-launch
commit gating, guardian allocation/acknowledgement races, immutable terminal
repeats and fail-closed terminal evidence. It never trains or evaluates
real data. Tests must use a separate verification namespace and the existing
CPU allocation/environment patterns. The sole permitted CPU validation job
must not be submitted until the parent supplies **READY and the exact combined
test selector**; preserve its durable intent and actual ID across disconnects.

For a failed bounded SSH connection, stop remote attempts and report that
GlobalProtect must reconnect to `vpn.tau.ac.il`. Do not request or exchange
credentials in chat. No paid-resource or alternative-host fallback is allowed.
