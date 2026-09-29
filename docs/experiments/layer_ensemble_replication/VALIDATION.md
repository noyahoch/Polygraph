# Implementation validation and evidence

This document records implementation tests, **not scientific replication
results**. No seed-17/27 detector was trained, no real meta/dev predictions were
generated, and no original-test evaluation or HF upload was performed.

The [protocol](PROTOCOL.md) and [operating guide](../../../pilots/layer_ensemble_20260914/slurm/REPLICATION.md)
remain the execution contract. Scientific submission requires separate explicit
authorization of an immutable workflow, future cutoffs and resource reservations.

## Slurm-only validation

All numerical tests and package compilation ran inside CPU-only Slurm
allocations using the existing recorded environment. The Mac performed source
authoring, coordination and short inspections only.

| Job / immutable revision | Outcome | Scope |
| --- | --- | --- |
| 897260 / v1 | Compilation passed; 74 tests ran with 18 fixture errors | Aggregate artifact fixtures attempted to write inside the immutable source checkout |
| 897260 / v2 | All 76 tests passed; compilation passed | Writable job-owned fixture root; corrected aggregate/guardian metadata checks |
| 897317 / v1, separate corrective namespace | All 86 tests passed; compilation passed | Includes ten guardian allocation/terminal-provenance regressions |
| 897383 / v1, subsequent concurrency follow-up | All 89 tests passed; compilation passed | Adds bounded terminal-lock waiting and three concurrent-publisher/timeout regressions |

The final suite ran in **153.759 seconds**, with test/compilation receipt
completion at **2026-09-16 01:16:02 Israel time**. The preceding 86-test suite
ran in 150.979 seconds and completed at 00:51:46. All three allocations requested
2 CPUs, 8000M memory and no GPUs. These observed test durations are not
scientific-run throughput estimates.

Source inventories were unchanged during validation:

- Initial namespace v1: `f366cb1cb7a7f5530be281cff9b6dd60a69f7513bf42b1119e5d4c9b39887f05`.
- Initial namespace v2: `5cc4e7f978400c712fecde970dbdd5b2ab8b33b4c98454e88dbaf061800d7415`.
- Corrective namespace v1: `9935b5ae0356efeae467efb612b32cff0261922466c214d57e7c1d656dea003a`.
- Concurrency follow-up v1: `c9a6da4ff60d844b22b9c64014aba578765e36e6c9f8da03a9dde43a9f616db9`.

The first failed run was retained, not overwritten or presented as a pass.
The earlier passing v2 did **not** close the guardian audit issue; the separate
corrective validation did.

### What the tests establish

- Fixed seed-17/27 identities and exact role-map/cache bindings.
- No cross-seed checkpoint, head or prediction substitution.
- Exactly 20 completed epochs and earliest greatest checkpoint-AUROC ties.
- Meta-only head fitting, preserved negative last-only coefficients, and
  serialization parity.
- Both seed pipelines frozen before either new dev-evaluation output.
- Correct source-image pairing, shared draws, sample SD and mean of seed-wise
  AUROC differences; no duplicated-row pooling or cross-seed prediction model.
- Strict completed-artifact inventories, provenance, timestamp gates and
  preservation of seed 7's separate late status.
- Default no-submit behavior, fixed scheduler matrix, explicit authorization,
  resource/dependency checks and ambiguous-submission handling.
- Guardian allocation identity, bounded initial acknowledgement handling,
  write-once terminal publication and fail-closed accounting behavior.

These are synthetic numerical and mocked scheduler tests. They do not establish
eight-GPU availability, concurrent NFS throughput, GPU numerical restoration of
the new seeds, scientific model quality, or recovery from every real cluster
failure. The separately authorized GPU readiness stage remains a prerequisite
to future scientific fits.

## Independent audit and correction

An independent read-only audit found no significant issue in the initial core
pipeline. Final integration review found a guardian bug: a new allocation could
restart the guardian and replace an earlier incomplete terminal receipt with a
complete one under the original workflow identity.

The correction binds the guardian to its recorded singleton allocation before
writes or cancellation, handles the initial acknowledgement race with bounded
waiting/reconciliation, and publishes terminal receipts under a lock without
rewriting an existing bound result. An existing incomplete result returns status
2, not success. Conflicting concurrent publication cannot replace history.

The reviewer independently confirmed that specific fix. Ten additional
regressions cover wrong allocation and array invocation, acknowledgement
race/timeout/lost receipt, immutable/idempotent terminal results, foreign
terminal identity, and failed original-guardian accounting.

A subsequent concurrency follow-up added a monotonic, bounded 35-second lock
wait. Identical or conflicting concurrent publishers return the first bound
terminal result without rewriting it or repeating cancellation; a lock timeout
fails closed. Three additional tests exercise identical publishers, a conflicting
attempt to upgrade incomplete to complete, and the timeout. The orchestrator
reviewed this focused delta and reconciled its later-arriving validation into
an additional local commit; the earlier snapshots and receipts remain intact.

## Persistent evidence

Initial validation root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/verification/replication_cpu_20260915_234834`

It retains both read-only releases, their result/log inventories, and
`complete.json`, including the failed v1 and successful v2 records. Slurm
897260 completed with exit `0:0` at 2026-09-16 00:23:42 Israel; accounting was
checked at 00:26:01. Only its completed job scratch directory was removed.

Corrective validation root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/verification/replication_guardian_fix_20260916_003446`

The receipt records job 897317, the exact three test selectors, source inventory,
unchanged-source check, and zero exit codes for unittest and compilation.
Its logs include `logs/v1.unittest.log` and `logs/v1.compileall.log`.
Neither namespace is a scientific experiment result or a replacement for the
historical September 14 run.

Concurrency follow-up root:

`/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/verification/replication_guardian_concurrency_20260916_010507`

It retains the unchanged source inventory, `v1.result.json`, `complete.json`
and the unittest/compilation logs. Job 897383 completed with exit `0:0` at
2026-09-16 01:18:02 Israel. The 01:19:44 scheduler reconciliation records all
three validation jobs as completed and an empty user queue. Only completed
scratch was cleaned; no scientific workload or prior release was changed.

### Repeating implementation tests

Inside an authorized CPU Slurm allocation, use the operating guide's pinned
environment/import/dependency setup. Set `POLYGRAPH_TEST_ARTIFACT_ROOT` to a
writable job-owned directory and `PYTHONPYCACHEPREFIX` to writable job scratch;
keep the source release immutable.

```bash
python -B -m unittest \
  pilots.layer_ensemble_20260914.test_replication \
  pilots.layer_ensemble_20260914.test_aggregate_replications \
  pilots.layer_ensemble_20260914.slurm.test_replicate
python -m compileall -q pilots/layer_ensemble_20260914
```

The validation used the recorded September 14 environment and dependency
overlay. The dependency manifest hash was
`66e9b9f841830177525f5baf7c806713fe23a25de704bdce8ac2c2fcd48fab6c`;
the existing import bundle hash was
`9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f`.
`CUDA_VISIBLE_DEVICES` was empty. No new dependencies or hardware upgrade were
needed. Existing Torch JIT and sklearn penalty deprecation warnings were visible;
they were not convergence failures or a reason to change the frozen recipe.

## Historical scientific result

The [Hebrew September 14 report](../september14/REPORT.md) now distinguishes the
verified late seed-7 result from the unchanged incomplete original protocol.
Its AUROC difference and interval come from the existing Slurm report, not from
these synthetic tests. Original artifacts and failed attempts remain preserved.

All implementation/documentation commits are local. No branch push, merge,
model publication, cache transfer to the Mac or new scientific result is
implied by this validation record.
