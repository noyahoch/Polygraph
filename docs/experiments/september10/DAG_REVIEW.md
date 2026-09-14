# Independent Slurm workflow review — 2026-09-11

Scope: bounded source review of the actual stage/argv builder, submission/execution
wrappers, administrative/backup scripts, and the publisher terminal-wake correction.
The reviewer did not author these changes and made no code edits. No numerical code,
SSH, scheduler submission, or Hub operation ran on the Mac.

**Final static template verdict: PASS.** Both identified phase-lifecycle gaps are
resolved in the reviewed source below. No remaining concrete dependency/serialization
blocker was found in this bounded review.

**Release boundary:** Operations supplied a concrete workflow builder, not yet an
instantiated production `workflow.json`. Allocation durations await measured preflight
evidence. Static template approval cannot authorize an unknown workflow hash or replace
the phase-specific engineering/science GO recorded by `submit.py`.

## Scientific and operational checks

- The builder enumerates exactly seven registered arms crossed with all five seeds for
  both training and scoring. Scientific stages use `afterok`: capture, rewiring,
  validation, the two-worker I/O check, training, full-matrix freeze, per-model scoring,
  and CPU analysis. The final scorer is explicitly `evaluate score --freeze`; CPU
  `evaluate analyze` requires successful GPU score receipts. No arm or seed is dropped
  automatically and no test metric chooses admission.
- Training additionally requires the external I/O admission receipt. The first phase
  may complete while the laptop is closed; proceeding to training remains a new recorded
  decision based on measured throughput and resource evidence.
- GPU tasks request one GPU, array concurrency is checked within 1–8, and previously
  submitted GPU stages are serialized with terminal dependencies. The planned diagnostic
  preflight is also an explicit capture prerequisite. CPU analysis/publication reserve
  no GPU.
- Submission is locked, keyed by workflow hash and stage, and saved atomically. Existing
  scheduler/accounting jobs with an unrecorded matching name stop resubmission for
  reconciliation. Failed scientific dependencies propagate through Slurm rather than
  triggering automatic configuration changes or candidate replacement.
- Every stage checks the submitted workflow SHA and source hashes under the immutable
  release, executes from that release, and persists running/completed/failed receipts.
  Heavy cache/run/result files and job state remain on Slurm. Credentials are obtained
  from the same user's persisted Hub store and do not enter command arguments.
- Publication is administrative: science does not depend on successful Hub access.
  One shared publication directory/lock serializes uploads. The final selector uses
  frozen final artifacts only when `evaluation_complete.json` exists; an invalid existing
  completion fails rather than being demoted to a later partial snapshot.

## Material phase-boundary findings

1. The first draft made the administrative summary depend on every scientific stage,
   although phase one submits only capture/rewire/validate/I/O. If preparation failed
   or training admission was withheld, later job IDs did not exist and `submit.py`
   could not schedule the summary or final backup. The failure path needed a summary
   depending only on actually submitted preparation jobs, with a phase-specific marker.
2. The initial dependency-free long-running publisher always required cache metadata.
   Starting it before that metadata existed could exhaust its two-failure bound before
   any model completed. Its launch needed to follow successful preparation/admission.

Root authorized Operations to resolve these two lifecycle issues without changing the
scientific chain. They are resolved by the stable correction reviewed below.

## Publisher wake correction

**Static PASS.** `wait_for_terminal_or_interval` uses a monotonic deadline and sleeps at
most 30 seconds between terminal-marker checks. The main loop retains the same publisher
lock across the wait and final snapshot, then exits. This creates no additional uploader.
The mock tests check normal interval preservation, terminal wake, final upload and lock
lifetime without real sleeps. This bound concerns an idle uploader's marker detection;
an upload already in progress still completes through its ordinary upload path.

- `publish.py`: `839a64d51a12cbc0b3304c1c6f2802c9ad468f674c205c8b7c0a76fc3a5c4a42`
- `test_publish.py`: `2ef22078849e67fa1117a00261d58bba86bfaf85993fe583a4a354833ee8e5cc`

The final backup must run after the central uploader terminates, so no later ordinary
partial snapshot can replace its final result status. Actual authentication and immutable
Hub upload/download remain execution gates.

## Phase-boundary correction review

**PASS.** The first phase now submits `preparation_summary` after any outcome of only
capture, rewiring, validation and the two-worker I/O job. It writes
`status/preparation.json`, distinguishing `ready_for_root_io_admission` from
`failed_or_incomplete`, and does not write the global terminal marker or submit training.
No model publisher runs in this phase under the root's decision; no trained checkpoint
yet exists, and the source/cache/administrative records remain on Slurm.

The central publisher is in phase two, after successful I/O, and requires both cache
validation and the root's I/O admission receipt. There is no dangling dependency on an
optional preparation backup. Phase-two science remains independent of publication
success. Its final administrative summary depends on all submitted scientific jobs with
`afterany`; final backup waits for both that summary and the central publisher to end.

Reviewed stable identities:

- `slurm/build_workflow.py`: `73d415576251b0391810d3c9915df05129b9a9e2da5c4a8cdd70f0bac513c21a`
- `slurm/administrative.py`: `6a5a68cb3203a620c65622e9fccac8db939585f6dda08a450e175e9658cab9d0`
- `slurm/submit.py`: `3e0518b6261fda857ab0d020234ee22cfd3dab1956821e13454cad641d08728d`
- `slurm/stage.py`: `ce9da27e66f89cc69594d8ab242af661d815f879a9ff4882a09be6c6cfc7ba33`
- `slurm/stage.sbatch`: `700ca1644b348d8a5382fe68dc7db53a2e1691198ac976e71a31e5b385fb3f98`
- `slurm/final_backup.py`: `912ef82199f93ac714c8b0997af00c4dc9f3106014e65e6d9eebf049872cd531`
- `slurm/benchmark.py`: `025b709b13be704a1a769960ce09dd530883dfb9704bb448cfb8785358182caa`

The actual instantiated workflow, measured durations, admitted concurrency, release hash,
Slurm source/mock checks and phase-specific GO remain required before submission. Any
later archive/unpack optimization to execution wrappers is outside this reviewed source
identity and needs its own bounded source check. This approval does not delay or replace
the already-running GPU diagnostics.

## Subsequent bounded import-wrapper check

Operations requested a separate check after measured filesystem/import delays and a home
quota failure. **Static PASS** for the source identities below: the import helper copies
the prebuilt archive sequentially to job scratch, verifies the pinned archive SHA and
package versions, rejects traversal/links/unexpected top-level entries, and extracts with
the safe data filter. Its Python path places the exact staged package trees before the
installed copies while `-m` still starts from the immutable experiment release. Native
libraries are not replaced. Explicit Hub/Torch/scratch cache destinations and a private
token-file path keep package/cache writes away from the failed home destination; token
values are not written into argv or execution receipts.

- `slurm/imports.py`: `85c0916877471a75fa684b1189f2806a39a3bb5969f5282ca5075a91fde6377d`
- `slurm/stage.py`: `a39b856895e7b6e2ba5252868556f91c76af983bb276d68b559272e9de8204ec`
- `slurm/stage.sbatch`: `e0cd0f2290c0b772e111833d7bd99e5607ba9741a9e9036db2eb6ba83d431e73`

Operations reported Slurm job **877593** passed actual staged module path/version/source
hash checks, four extraction-safety checks and both publisher terminal-wake mocks. The
reviewer did not rerun these locally. This narrow addendum does not approve later changes
to the scientific-admission logic or an uninstantiated workflow.

## Instantiated preparation workflow — GO

**Independent GO for preparation only**, after the bounded operational corrections.
Reviewed candidate: `workflow_candidate_71d6c05eb054a933.json`, immutable release
`71d6c05eb054a933`, workflow SHA-256
`e94c9033a697f1130820c2fbd1f4fe5d004cfea07c5e40165237f60e04b37855`.
This resolves the earlier uninstantiated-template limitation for the following stages
only: `capture`, `rewire`, `validate`, `io_one`, `io_two`, `preparation_summary`.

- Capture explicitly requires successful jobs 877765 and 877580 and their receipts,
  completed source data and the pinned environment. It uses the approved classifier
  extraction command, batch 18, shard 256, original production cache and resume behavior.
- The scientific `afterok` chain is capture → fixed rewiring plus explicit admission →
  validation → one-worker production-cache timing → paired two-worker timing. The
  preparation summary uses `afterany` on all five preparation stages and writes the
  phase-specific ready/failure record. No production fit, freeze or test score is launched
  by this phase.
- Resource ceilings are capture 360 minutes, rewiring/admission 120, validation 120,
  one-worker timing 20 and two-worker timing 20. These are ceilings, not runtime forecasts.
  GPU jobs use one GPU each; only the paired timing array has two concurrent tasks.
- Both disposable I/O configurations use 64 bounded training batches from production
  training records, record shard/cache-load evidence and startup time, and require at
  least three distinct training shards. Validation timing uses a fixed bounded subset;
  no model is saved or test data opened. The paired run uses a common-start barrier with
  a 300-second timeout, records the actual time window and refuses stale barriers.

The first candidate's shard rotation was not connected to a sampler, so both workers
would have read the same rows. The corrected benchmark passes the existing fixed-seed
`BlockShuffleSampler` to the loader. Worker zero remains comparable to the single-worker
baseline; worker one's block rotation now takes effect. Final benchmark SHA-256:
`4bc032b54fd52727f5bdf50228d934b66300afe903f5f6a698ce0a21b757b503`.
The final source manifest also includes the unchanged registered `PROTOCOL.md`, needed
by the eventual evaluation implementation identity. A direct text diff verified these
bounded changes and release-path updates against the reviewed first candidate.

Operations reported job **877765** passed 31 admission/training/evaluation checks,
8 core checks, admission and validation of the preserved diagnostic cache, and 17
publisher checks. Its original rewiring manifest SHA remained
`dddc39984b903a1b85533f52c53aec86269ce1c45054a08272bdf0af2760d906`.
Scientific source in the instantiated release matches that tested source; the new I/O
workload itself will produce the preparation evidence. The reviewer ran no numerical
checks on the Mac.

**The 35 production fits remain unadmitted.** Root must assess the full-cache quality and
one/two-worker resource evidence and record the separate I/O admission before phase two.
This GO is limited to the exact workflow hash and preparation-stage list above.

## Instantiated phase-two workflow and completion-triggered publication

**Independent static PASS** for phase-two candidate SHA-256
`e286e5b4293dbea98259f4ccd7222f5c1a73761e14fe09495abe27d4a93baff4`.
The earlier preparation-only restriction is superseded by the root's separate production
I/O admission, recorded at `manifests/io_admission.json`, SHA-256
`0a8b5ce03f31a221ce37f10dfd8d349dfa2941d7b6b94b4f0478b7afcbb38fc5`.
The final release must replace this candidate's old publisher source with the reviewed
pair below and record its resulting workflow hash before submission.

Operations reported all preparation jobs 877859–877864 completed successfully. In the
64-batch production-cache measurements, each worker read six training shards; the paired
workers shared no measured shards and overlapped for 10.323 seconds. Batch times were
0.15570 seconds for one worker and 0.14250/0.14483 seconds for the pair. These observations
support the root's resource admission; they do not establish measured eight-worker scaling.

- The first job is the registered `full_graph/seed1`, with a 360-minute ceiling. The
  remaining array contains exactly the other 34 registered arm/seed pairs, at most eight
  concurrent GPUs, and starts only after the first job succeeds and its completion marker
  exists. There is no extra pilot fit and no duplicate seed-one fit.
- The actual training command reloads the selected checkpoint, verifies the full
  validation score vector and the bounded restoration/threshold/batch-partition audit,
  writes the validation artifacts and hash-bound completion marker, and verifies those
  hashes before successful exit. An interrupted fit exits with status 75. Thus the
  dependency gates on the completed lifecycle, beyond merely reaching the training loop.
- Freeze depends on both training stages and invokes the existing complete-35 freeze.
  All 35 score commands use only `evaluate score` with that freeze. CPU analysis waits for
  the entire score array. Hyperparameters, seeds, selection rules and rewiring disposition
  remain unchanged; the 360-minute fit limits are resource ceilings.
- Publication is independent of scientific success. One central uploader starts after
  preparation and I/O admission. The final administrative summary uses `afterany` for all
  scientific stages; final backup waits for both the summary and central uploader. The
  existing selector publishes verified complete results when available, otherwise an
  explicitly partial collection, and rejects invalid final-completion evidence.
- Phase-one workflow/review records and submission history are preserved before the
  terminal phase migration. Existing locked submission, exact source/workflow checks,
  job deduplication and GPU-stage serialization remain unchanged.

The initial candidate omitted per-fit publication wakes. The separately reviewed bounded
repair observes only the 35 registered atomic completion markers, checks at most every
30 seconds while waiting, and retains the same publisher lock. A baseline captured before
upload ensures a completion during upload triggers another cycle. Terminal observation
precedes the final completion reread, which also catches a completion while writing the
cycle receipt. Unchanged completion sets do not repeatedly trigger uploads. The counters
allow 48 progress cycles, at most 35 completion cycles and one terminal catch-up within
the unchanged 48-hour Slurm wall limit. Upload verification, allowlists, privacy and
immutable artifact/source handling are unchanged.

- `publish.py`: `72437e42662aa0c2640e3789716de0642676f6b86886bb107e17d8e589ae46b0`
- `test_publish.py`: `78f0d8b268ebd035d8b1f1220354dfb900e411d1bd70a1148fd13bc2c00f9ba3`

The reviewer inspected the focused lifecycle regression tests but ran no numerical work
or Python tests on the Mac. Final release GO requires the focused Slurm result and a
mechanical comparison of the refreshed candidate/source hashes; no further scientific
review or repeated full test suite is required without new evidence.

### Final phase-two identity — GO

**Independent GO** for `workflow_phase2_81b4e83e8bc2a823.json`, SHA-256
`1bb59c6e0bffd7b170e8737a8954678fd9c725cff7816321ca8779d8b7d98914`,
using immutable release `81b4e83e8bc2a823`. A mechanical comparison found only the two
reviewed publisher source changes, their release-path references, and the added focused
test prerequisite for `train_first` and `publisher`; the 35-fit scientific DAG is unchanged.

Operations reported Slurm CPU job **879920** passed all **21 publisher tests**, exit 0,
completed at 16:04:47 Israel time. Its receipt binds source-manifest SHA-256
`553e3684bd71a2d06722f09d36172abbdf5ea6d3846015183ffc62a6e795135c`.
The phase migration verifies successful terminal accounting for all six preparation jobs
and the exact I/O evidence hashes under the submission lock, archives the old workflow
and review, and preserves submission history. No review blocker remains for the recorded
phase-two stage list and first-fit-success dependency.
