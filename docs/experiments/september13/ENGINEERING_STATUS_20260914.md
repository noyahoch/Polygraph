# Layer screen: engineering status, September 14, 2026

The new layer screen has **no completed scientific comparison results** at this handoff. All 18 worker audit/timing cases passed after the selected continuation; the original 14-fit plan still exceeded its budget. The user then explicitly authorized the recommended **four-fit comparison within 80 GPU-hours**: block11 versus union4, seeds 7/17. The former pending-approval state is superseded; the 14-fit plan is deferred. This is a dated snapshot, not a live scheduler view.

Ops reports full development capture **892193 running since 11:28:33 Israel**, immutable release `66840268a7a52037`. The four-fit reservation is 4515 of 4800 GPU-minutes, including 265 diagnostic and 710 capture minutes; it uses zero loader workers and six CPUs per fit. These are conservative caps, not elapsed training time. The first two fits are block11/seed7 and union4/seed7. Only after both have at least two published complete epochs, inclusive of validation/audits/saves, and fit their existing resource caps may the two seed17 fits launch. The gate never uses scientific scores.

The versioned dispatcher and ramp preserve the data, feature construction, model, hyperparameters and original 60-epoch maximum. The active run directory is `runs_four_20260914`; finalization requires exactly four validation outputs, reports paired comparisons by seed, and creates no new ensemble or test evaluation. Durable submission receipts and failure guardians allow the chain and private backup to continue after laptop closure. Actual job IDs and terminal status belong to the Ops handoff. The earlier sections below retain the chronology and limitations at each stage; their pending checks and 14-fit execution scope are historical.

## What happened

- September 13: the diagnostic cache was extracted for 36 development records. All six representations passed CUDA forward/backward, finite-gradient and checkpoint checks. The initial scheduler workflow rejected an estimated 2,430.75 GPU-hours against the 64-hour limit, so it did not submit the 14 planned fits. That extrapolation mixed startup and recurring costs; it is not a measured experiment duration.
- September 14, 08:51:57 Israel: Ops verified an empty queue, the original source release intact, and no overnight profiling or training. The diagnostic cache remained one 68,212,640-byte shard.
- September 14: the separately reviewed timing profiler ran as Slurm job **891747**, from immutable source release **fa3df78e67cc5781**. It completed with exit **0:0** at **08:59:44 Israel**, after approximately four minutes of allocation. All six arms completed; the measured profiling phase took **51.4 seconds**. The source/data pipeline and model definitions were unchanged.
- Ops' corrected estimate for all 14 fits at maximum 60 epochs was still above the 64 GPU-hour cap: **73.23 hours for training alone**, **96.52 hours including validation/audits before I/O, startup and capture**, and **196.5 hours with conservative allocation margins**. These are projections from diagnostic measurements, not consumed GPU-hours.
- Root authorized one final 15-minute diagnostic for optional 2 and 4 loader workers, with one GPU and six CPUs. Job **891833**, immutable release **deb4869e24cc32e7**, ran **09:13:05–09:16:47 Israel**, using **222 seconds of allocation**. It failed before completing its first case; the diagnostic phase lasted **21.02 seconds**. No scientific fits were created and no test data was evaluated.
- At 10:21 Israel, Ops verified an empty queue and no jobs since that failure. At that point no further retry was scheduled.
- The user then explicitly asked to fix and retry. Localization job **892060**, release **450b06809de915e8**, completed with exit **0:0** at **10:35:56 Israel**, using approximately **131 seconds of allocation**. Its measured diagnostic phase took **24.62 seconds**. It read only the same 36 development records and created disposable states, with no scientific fits or test evaluation.

The completed September 10 experiment is separate: its [scientific results](../september10/RESULTS.md), [review](../september10/FINAL_SCIENTIFIC_REVIEW.md), and [35-model catalog](../../models/SEPTEMBER10_CATALOG.md) remain the current completed research evidence.

Historical implementation handoffs and pre-execution reviews remain archived with their original scope. In particular, the early `CORE_HANDOFF.md` describes the superseded per-graph rewiring gate and checks that were still pending when it was written. For final behavior and completed checks, follow the later [rewiring decision](../september10/REWIRING_DECISION.md) and [results/reproduction report](../september10/RESULTS.md), not that initial handoff. A historical review's "pending execution" wording is not the current run status.

## Timing measurement and limitations

The new profiler separates dataset construction, first-file verification, deserialization, CPU graph construction/collation, host-to-GPU transfer, first and warmed training steps, evaluation-mode forward passes and checkpoint serialization. It calls the production loader and forward functions, including finite-input and gradient checks. Its model updates are disposable; no scientific checkpoint, model selection, test scoring or fitted model result is produced.

Ops reported representative warmed components per 24-record batch:

| Representation | CPU construction/collation | Transfer | Training step |
| --- | ---: | ---: | ---: |
| Final hidden + layer11 attention | 0.180 seconds | 0.006 seconds | 0.028 seconds |
| Final hidden + all12-layer attention union | 0.654 seconds | 0.121 seconds | 0.087 seconds |

These values are transcribed from Ops' measured profile, not recomputed locally. CPU graph construction remains a recurring cost. A single shard cannot establish full-cohort or concurrent NFS throughput. Clearing the dataset's one-shard cache does not clear OS/NFS caches. Production deserializes a shard again after eviction; file SHA verification is cached only within one process while the file fingerprint remains unchanged. Each fresh fit pays its first-read verification cost. Creating a fresh DataLoader iterator for each profiled batch also adds some conservative overhead relative to the persistent production epoch iterator.

Do not multiply first CUDA initialization or whole-dataset startup by the number of batches or epochs. Equally, do not remove real recurring deserialization, graph construction, validation or checkpoint costs to force the estimate under the cap. The full matrix and maximum 60-epoch budget remain unchanged pending review.

## Failed worker diagnostic: what is known

The first attempted case was `block11`, **workers=0**. After two disposable training epochs, the fixture reloaded the first-epoch model, optimizer, sampler and random-generator states and replayed epoch 2. At `profile_workers.py:189`, comparison of the resulting model parameters against uninterrupted epoch 2 failed for an **unnamed one-element tensor**: absolute difference `0.00012578070163726807`, relative difference `0.0033463670406490564`, with absolute and relative tolerances both `1e-5`.

The preceding input-hash, loss and gradient comparisons returned successfully, as established by the traceback reaching the subsequent model-state assertion. Thus the resumed batch tensors/order matched exactly and losses/gradients were within those checks' tolerances. This is a control-flow inference, not a separately saved completed case. Final optimizer-state, sampler-state and RNG comparisons were not reached. The JSON contains `passed=false` and `cases={}`; **neither the 2-worker nor 4-worker configuration was tested**, and no throughput case completed.

At the first read-only review, engineering and independent review found no provable snapshot-aliasing or restore defect. The original log lacked a parameter name, so that report correctly left the cause unknown. The subsequent named diagnostic below supplies new evidence; the original logs and failed source remain intact.

This failure does not establish that multiprocessing or checkpoint serialization is broken. The diagnostic demands closely matching model parameters after resumed optimization; the completed September 10 study's restoration audit instead checks restored validation predictions. Its archived fits, results and immutable releases were not modified. The new opt-in worker path remains **unvalidated**. Its default is still 0 workers, using the ordinary graph loader and unchanged graph materialization. New runtime settings and implementation files are hash-bound, so existing checkpoints must retain their matching frozen source/configuration rather than silently using this changed release.

Local commit `6cb00af` archives the first executed source as an unsuccessful validation attempt. No parameter exemption, tolerance relaxation, model change or scientific fit was introduced in the subsequent recovery. Current evidence still does not justify admitting the unchanged matrix below the resource cap.

## Named localization and focused correction

Job 892060 compared uninterrupted epoch2 with three continuations from the same epoch 1 model, optimizer, input order and random states: a GPU in-memory copy with no serialization, disk reload into the already-used pair (matching the failed fixture), and disk reload into a fresh pair. Every immediate starting-state comparison was exact. All inputs matched exactly and all original loss/gradient checks passed under ordinary CUDA. The **only final state leaf outside the original tolerance was `model.gate.2.bias`**:

| Ordinary CUDA continuation | Absolute difference in that bias |
| --- | ---: |
| Same-checkpoint in-memory replay, no save/load | 0.0004491135 |
| Disk resume into reused model/optimizer | 0.0003561489 |
| Disk resume into fresh model/optimizer | 0.0002478436 |

The maximum difference in fixed evaluation logits was **2.3841858e-7** in every continuation. The no-load control's bias gradients were on the order of 1e-9, differing across repeated CUDA execution. This evidence points to numerical repeatability, not checkpoint corruption, as the cause of the failed assertion. The source adds this scalar bias equally to all node scores before a graph-wise softmax; it cancels in exact arithmetic. Small floating-point gradient differences can nevertheless affect AdamW's parameter updates. The diagnostic does not identify an individual CUDA kernel or establish a universal numerical-error bound.

With deterministic CUDA enabled, **all three continuation routes had exact final states, gradients and fixed evaluation logits**, and every original 1e-5 check passed. No CPU fallback was needed. These observations are limited to the recorded 36-row, block11, seed 7 diagnostic.

The minimal correction scopes deterministic CUDA to the **parity/resume audit**, preserving all parameters and the same tolerances. The audit context restores the prior backend flags and parent RNG in `finally`. A separate freshly seeded disposable model measures **ordinary production-mode throughput**, with deterministic algorithms disabled. Both the benchmark and any future production wrapper must export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python. This numerical runtime is recorded in new run configurations so resume cannot silently change it. The existing production saved-prediction restoration audit is unchanged; ordinary CUDA training is not claimed to have bitwise identical resumed parameter trajectories.

Each worker case records its audit outcome even if it fails; independent timing can then remain informative without being treated as a passing case. All 18 audit/timing cases must pass before complete-matrix validation can be claimed, and resource admission remains a separate review. The optional worker implementation, architecture, data, graph materialization and scientific matrix are unchanged by this correction. No numerical work ran locally.

## Reproduction and evidence

The native result is `manifests/profile_timing_20260914.json` under the new experiment's remote root. Scheduler stdout/stderr are `logs/profile_v2_891747.out` and `.err`. The operation manifest and source release establish the machine-specific paths without embedding credentials or personal filesystem locations here.

Profiler SHA256:

```text
9934884b2f66364b0ce2456f8d75d66d24c2f0fd3298c612cbbfd86e127112ce
```

The executed command, from the complete frozen source with the pinned Slurm environment, was:

```bash
python -m pilots.layer_screen_20260913.profile_timing \
  --cache "$EXPERIMENT_ROOT/preflight_cache" \
  --out "$EXPERIMENT_ROOT/manifests/profile_timing_20260914.json" \
  --batch-size 24 --warmup-rounds 2 --repeats 5 --max-seconds 720
```

`EXPERIMENT_ROOT` denotes the existing September 13 Slurm experiment, not a local Mac directory. The allocation had a 15-minute ceiling; no numerical tests or ML workloads ran locally. The one-shard diagnostic result does not itself authorize full training.

The failed worker diagnostic used the same 36-record, one-shard diagnostic cache, an RTX2080Ti and Torch `2.14.0+cu126`. Its intended timing phase would replay the same records for 288 presentations; that phase was never reached. This is the executed command for archival reproduction, **not an instruction to retry**:

```bash
python -m pilots.layer_screen_20260913.profile_workers \
  --cache "$EXPERIMENT_ROOT/preflight_cache" \
  --out "$EXPERIMENT_ROOT/manifests/profile_workers_20260914.json" \
  --max-seconds 720
```

Evidence under the remote experiment root:

| Artifact | SHA256 |
| --- | --- |
| `releases/deb4869e24cc32e7/pilots/layer_screen_20260913/profile_workers.py` | `bc3dba4163ec83e8aa457cb8c66120e344090f55e2850f3713caad2ef940b116` |
| `manifests/profile_workers_20260914.json` | `436af630a45285f242cd3dc397a50bd76f43479a422f0a0c6981c3bfb8951f2c` |
| `logs/workers_891833.err` | `a5ec28e4739307681cb32e067f66f4ccbf04dfedf78f7d7177c33bf423d7a8e4` |
| `logs/workers_891833.out` | `1afd334017d97abd81c742abb0566cb3d2dd3796dc44f7d805b85bac5534faf3` |

Raw files remain intact in the Ops archive. These checksums were read from the preserved files; no numerical analysis was run on the Mac.

Localization evidence is in `manifests/resume_recovery_20260914/summary.json` (SHA256 `207ae55cb35654904cafcde5da44b98d3ea508151cec260803068d2fbf3db1a9`) and the adjacent phase directories with per-step tensor states and named comparison JSON. Frozen source `diagnose_resume.py` has SHA256 `4c057abb56e3eaccfc488d13844fc26fd266ed0425fbef989091782e1fc2b4f5`; local commit `6c10971` archives its source and recovery authorization. The full as-executed dependencies are preserved in release `450b06809de915e8`.

The reviewed localization command was:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pilots.layer_screen_20260913.diagnose_resume \
  --cache "$EXPERIMENT_ROOT/preflight_cache" \
  --out-dir "$EXPERIMENT_ROOT/manifests/resume_recovery_20260914" \
  --max-seconds 600
```

The corrected worker diagnostic used the same pinned environment and a separate output, `manifests/profile_workers_recovery_20260914.json`, preserving all earlier failure evidence. Ops submitted job 892080 from reviewed immutable release `7702e1de15063208`, with one GPU, six CPUs and a 15-minute cap. It ended **TIMEOUT at 10:55:48 Israel**, consuming 902 seconds of allocation. The last saved JSON recorded 701.84 seconds of profiling and five complete passing cases; there was no recorded mathematical/audit failure. The profiler SHA256 is `84c0b744c3d948876d66210a63063c9fe7971c23177f014dd0344705dec20030`; trainer SHA256 is `8e58cbade79bb545f6e1b2dd7796995a5d9da926f6a84ae1cb3e066e58089c17`. The terminal result SHA256 is `fee534900794bdef95f4f4c3c4ae0d4a93b21d42a8d1aab51d433f321395f3ca`.

## Partial worker results and missing-case continuation

These are native measurements from the five completed cases. Warm batch time is the recorded continuous wall time per 24-record batch, including waiting for the loader, transfer and optimizer work. Validation lifecycle includes its fresh workers and shutdown. No timings were recomputed locally.

| Representation | Workers | Audit | Warm batch, seconds | Validation lifecycle, seconds |
| --- | ---: | --- | ---: | ---: |
| Final hidden + layer11 attention | 0 | Pass | 0.287754 | 0.400099 |
| Final hidden + layer11 attention | 2 | Pass | 0.235556 | 60.156466 |
| Final hidden + layer11 attention | 4 | Pass | 0.117599 | 14.329699 |
| Final hidden + all12-layer union | 0 | Pass | 1.098223 | 1.533136 |
| Final hidden + all12-layer union | 2 | Pass | 0.924913 | 38.460564 |

The layer11 four-worker case has substantially lower warmed batch time, but process startup and validation-worker creation remain material costs. Their variability is visible even within this short run. These 36-record, single-shard measurements do not establish full-cohort or concurrent NFS throughput, and missing cases prevent a complete-matrix speedup claim. The overall JSON correctly remains `passed=false` because coverage is incomplete.

After the job became terminal, Ops froze five completed and 13 missing cases in `ops/frozen_missing_cases_892080.json` (SHA256 `0df1e6e934cab5a6944523ecbc6dae125d485ef76aa4b29f014e841a509a45b7`). The missing list is workers 0/2/4 for each of block2, block5, block8 and union4, plus workers4 for union12. The prior five cases must not be replaced or retimed.

The new `continue_workers.py` wraps the frozen profiler helpers with arm/worker selection and a per-job deadline. It verifies the parent result, all 11 core source hashes, cache/protocol identity, numerical runtime, Torch, GPU model and six-CPU allocation. Case files carry those bindings and can be reused on restart. If the missing case needs a workers0 gradient/input reference, that **audit alone** is regenerated and its cost reported, since the parent JSON does not store those tensors; earlier workers0 timing remains unchanged.

Root authorized up to five per-arm continuation jobs, each capped at 20 minutes with an 840-second internal default, below the global eight-GPU concurrency limit. The existing scientific matrix, audit criteria and measurement helpers remain byte-identical. Independent review passed for wrapper SHA256 `ea332a9cbc628ceae841c343443f9d89e1f3f2eeea6c0d270fc7f525013b86a1`; Ops owns submissions. Merge acceptance requires the original five plus 13 disjoint compatible passing cases, with every original case covered exactly once and all startup/support allocations counted. Full production admission still requires the subsequent budget review; no validation-lifecycle optimization is part of this continuation.

The [protocol source](../../../pilots/layer_screen_20260913/protocol.py) defines the six representations, 14 fits, fixed frozen classifier, train/validation-only cohort and checkpoint-selection policy. [STATE.md](../../../pilots/layer_screen_20260913/STATE.md) records operating constraints and the current restart boundary. Source and reports are being preserved through small local Git commits following the user's September 14 instruction; no remote Git push is authorized or claimed.
