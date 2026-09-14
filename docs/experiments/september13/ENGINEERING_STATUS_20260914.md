# Layer screen: engineering status, September 14, 2026

The new layer screen has **no completed scientific fits or scientific comparison results**. Its six GPU correctness checks passed, and the follow-up timing diagnosis completed successfully. Production training still awaits reviewed resource admission. This is a dated status snapshot, not a live scheduler view.

## What happened

- September 13: the diagnostic cache was extracted for 36 development records. All six representations passed CUDA forward/backward, finite-gradient and checkpoint checks. The initial scheduler workflow rejected an estimated 2,430.75 GPU-hours against the 64-hour limit, so it did not submit the 14 planned fits. That extrapolation mixed startup and recurring costs; it is not a measured experiment duration.
- September 14, 08:51:57 Israel: Ops verified an empty queue, the original source release intact, and no overnight profiling or training. The diagnostic cache remained one 68,212,640-byte shard.
- September 14: the separately reviewed timing profiler ran as Slurm job **891747**, from immutable source release **fa3df78e67cc5781**. It completed with exit **0:0** at **08:59:44 Israel**, after approximately four minutes of allocation. All six arms completed; the measured profiling phase took **51.4 seconds**. The source/data pipeline and model definitions were unchanged.
- After profiling, Ops again confirmed an empty queue and no submitted scientific fits. Revised admission remains pending.

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

Do not multiply first CUDA initialization or whole-dataset startup by the number of batches or epochs. Equally, do not remove real recurring deserialization, graph construction, validation or checkpoint costs to force the estimate under the cap. The full matrix and maximum60-epoch budget remain unchanged pending review.

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

The [protocol source](../../../pilots/layer_screen_20260913/protocol.py) defines the six representations,14 fits, fixed frozen classifier, train/validation-only cohort and checkpoint-selection policy. [STATE.md](../../../pilots/layer_screen_20260913/STATE.md) records operating constraints and the current restart boundary. Source and reports are being preserved through small local Git commits following the user's September 14 instruction; no remote Git push is authorized or claimed.
