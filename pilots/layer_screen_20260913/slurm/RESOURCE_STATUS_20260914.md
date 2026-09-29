# Resource decision after recovery — September14,2026

The numerical issue is resolved: all18 predeclared cases (six representations × workers0/2/4) passed the original deterministic parity, resume, state and gradient checks. Timings use the ordinary production backend. All models, features, splits and numerical tolerances remain unchanged. The user approved the four-fit final-layer versus four-layer-union comparison with an80GPU-hour reservation budget. Full production capture is running and the downstream admission, dispatcher and failure guardian are queued on Slurm. The original14-fit scope remains deferred.

The earlier resume discrepancy also occurred in an in-memory continuation with no disk load. It was isolated to ordinary CUDA reduction nondeterminism, not checkpoint serialization corruption. Deterministic audit contexts restore backend and RNG state before measuring ordinary execution. Original failed/TIMEOUT artifacts remain preserved.

## Verified jobs

- 892060: named resume localization, completed0:0.
- 892080: worker benchmark timed out; retained five complete passing cases; its overall report remains false.
- 892122 array: five disjoint missing-case tasks all completed0:0,11:00:57–11:06:06 Israel,4m53–5m09 each. Thirteen missing cases passed; combined coverage18/18.
- 892167: CPU merge/budget, completed0:0 at11:17:44.
- 892178: CPU options report, completed0:0 at11:19:54.
- 11:20:38 queue snapshot: no active jobs for omrifahn.

## Reservations and estimates

Current resource policy: maximum8GPUs concurrently and64GPU-hours of nominal allocation caps. Diagnostic reserve is265minutes; conservative full-capture cap is710minutes. The latter comes from the36-record gross extraction extrapolation:7.54hours before headroom. The first-use ViT kernel component was not isolated, so this is not a warmed throughput measurement.

The corrected estimator charges the already-full-cohort dataset constructor once and performs one final full validation pass. It retains60epochs, adds1.5headroom and rounds each cap upward to five minutes. First-hash costs for fresh validation workers recur each epoch. A conservative scenario charges new-shard I/O serially; an optimistic scenario removes all additional new-shard/hash costs. Neither scenario is a measured full-fit runtime or a mathematical bound on actual runtime.

| Scope | Conservative total reserved GPU-hours |
|---|---:|
| Original14 fits |252h05m|
| Four single layers, two seeds each |120h45m|
| Final layer versus all12-layer union, two seeds each |111h35m|
| Final layer versus four-layer union, two seeds each |75h15m|
| Four single layers, one seed each |68h30m|
| Fixed first pair: final layer versus all12-layer union, seed7 |63h55m|

All rows include the same265-minute diagnostic and710-minute capture reservations. The75h15m four-fit option is independently reviewed arithmetic from per-fit caps690minutes for block11 and1080minutes for union4, each repeated twice. It was added after the immutable CPU options report; that original report remains unchanged. The parent proposed an80-hour budget for this four-fit option. The64-hour initial-pair option has only five minutes of cap slack and one paired seed, so it is a pilot with limited seed-level evidence. The user subsequently selected the four-fit block11/union4 option with an80-hour budget; capture and its durable dependent chain are now submitted.

Per-fit conservative caps, each using the lowest conservative setting (workers0): block2=930m,block5=760m,block8=755m,block11=690m,union4=1080m,union12=2170m. The first pair's uninflated conservative fit scenario is31.75GPU-hours; its47.67-hour fit reservation includes headroom and rounding. This is still an extrapolation.

Four workers improve warmed one-shard materialization substantially, but full-cache hash/validation-worker startup costs are uncertain. The original14-fit's four-worker optimistic no-new-I/O reservation is126.42GPU-hours. It includes headroom/setup reserves and is not a claim that actual execution must take at least126hours. No full-cohort or eight-job shared-filesystem throughput has been established.

## Evidence and reproducibility

Remote experiment root: `/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_screen_20260913`.

Primary receipts under `manifests/`: `profile_workers_recovery_20260914.json`, five `workers_continue_*_20260914.json`, `worker_cases_merged_20260914.json`, `admission_workers_20260914.json`, `resource_options_20260914.json`, and `execution/`. Submission IDs remain in `submissions.tsv`. Local compact copies are under the task workspace `work/september13/ops/profiling_20260914/worker_budget/`; heavy tensors stay on Slurm.

Budget release06e1896df4222f82; workflowSHA96686837d67e25f01c3010f2efb130e090cedb226712eceb9918d2af8f391edc. Options release66840268a7a52037; workflowSHAd3c15991e785bcfd8f0d19771dfd1e2b56a8fc6fde87225026570c1fc2291b5e. Options input admission SHA507b3247384ecb1cfd40654d32de163511d7d3b1af7051e9d9c2175f988ffade. Parent worker release7702e1de15063208 and continuation release7c07c68d49057230 share the same numerical core.

Current action: the approved four-fit chain runs on Slurm. Capture job892193 produces the unchanged development cache. Admission892205 waits afterok892193; dispatcher892206 waits afterok892205. Independent guardian892207 waits afterany892193:892205:892206, preserving failure/partial backup even if an earlier stage fails. Invalid dependencies are cancelled rather than left pending indefinitely. The dispatcher submits block11/seed7 and union4/seed7; after both have two full epochs including validation/audit/save and pass resource-only gates, the ramp submits the seed17 pair. Four-run validation summary and private Hugging Face backup follow remotely. Actual fit IDs will be recorded by the allocated dispatcher/ramp; they are not yet submitted. Runs use isolated runs_four_20260914, workers0 and original scientific settings. No held-out test is evaluated.

Capture source66840268a7a52037/workflowfa5db859ef3604d9a099a0bc2cd81bae5ba6c2ccddfc0ec743ed04ab55d20316. Downstream source67561626e19e29a0/workflowd30fa316a4dae73234fa679320b6aa39682ac155fd03df0e085447f3be2758bd; all96 source files and final independently reviewed dispatcher/ramp hashes were verified before submission. At11:37:01,128 of28,800 records had been durably captured, weights200/200 loaded, with no error. The captured model/data are active on the GPU and shared storage; initial import/I/O startup has ended. All heavy work and artifacts stay on Slurm, so laptop closure does not interrupt this chain.
