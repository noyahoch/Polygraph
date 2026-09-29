# Reusing the fixed LogitDynamics artifacts

Use the three seed-specific auxiliary heads, error probes and scalers together.
Each complete pipeline contains 922,800 auxiliary-head parameters and 86 probe
parameters, in addition to the frozen backbone. Higher output means greater
predicted classifier error. The output is a raw error logit; the weighted
training objective does not establish calibrated probabilities.

## Pinned published files

Private repository: `omrifahn/polygraph-experiments`, model repository type.
Pin the original weight revision
`86605781c0528e286483e305072f7787393da860`, with this prefix:

`logit_dynamics_20260919_114500/snapshot_910575/`

[Browse the pinned snapshot](https://huggingface.co/omrifahn/polygraph-experiments/tree/86605781c0528e286483e305072f7787393da860/logit_dynamics_20260919_114500/snapshot_910575).

Within that prefix, for each `seed` in `7`, `17`, `27`:

| File | Purpose |
| --- | --- |
| `runs/seed{seed}/heads/model.safetensors` | Twelve linear 768-to-100 auxiliary heads |
| `runs/seed{seed}/probe/model.safetensors` | Selected linear 85-to-1 error probe |
| `runs/seed{seed}/probe/normalizer.json` | That seed's probe-training mean and population scale |
| `runs/seed{seed}/heads/config.json` and `probe/config.json` | Recipe, feature order, training/selection bindings |
| `runs/seed{seed}/heads/history.json` and `probe/history.json` | Complete fitting and validation histories |
| `runs/seed{seed}/probe/validation.npz` | Reference selected-checkpoint validation scores and IDs |
| `runs/seed{seed}/predictions/dev_eval.npz` | Reference development scores and IDs |
| `source/pilots/logit_dynamics_20260919/` | Exact scientific implementation used for the run |
| `source/source_manifest.json` and `backup_manifest.json` | Source and published-artifact checksums |

The selected probe epochs are 100, 95 and 100 for seeds 7, 17 and 27. Auxiliary
heads always use epoch 16. Avoid substituting one seed's scaler or heads into
another seed's probe. Use the pinned revision and hashes rather than repository
`main` as an artifact identity. Later documentation may be added at a newer
commit without changing these original weights.

The snapshot contains the small models, references, source and provenance.
**Raw images and extracted CLS feature tensors are excluded.** Existing native
`.pt` checkpoints/resume state are preserved for provenance and execution
recovery; the portable weight files use safetensors.

## Saved-CLS inference and replay

The existing implementation expects a tensor `[N, 12, 768]` containing only
post-block CLS states, ordered from `hidden_states[1]` through
`hidden_states[12]`. Match the training representation: store in FP16, then
promote to FP32 for the heads. Supply the original frozen classifier's ordered
100 FP32 logits separately. Do not add an extra LayerNorm or pool patch tokens.

The audited loading path uses `safetensors.torch.load_file`,
`pilots.logit_dynamics_20260919.train.LayerHeads`, and `torch.nn.Linear(85, 1)`.
Head outputs have shape `[N, 12, 100]`. The existing
`features.build_features(head_logits, final_logits, layers=12, k=5)` creates the
85 named features, and `features.normalize(features, normalizer)` applies the
saved seed-specific scaler. The loaded probe then emits one raw error logit per
record. Keep the numeric competitor exclusion, raw top-five dynamics, terminal
original-classifier logits and feature order exactly as in the pinned source.
These imports were exercised by CPU audit 910647: all six portable weight files
loaded and matched their native counterparts exactly, and all three train-only
scalers were reproduced within the fixed tolerance. That audit stopped at a
seed-17 validation score failure. The September 21 completion audits **915652**
(CPU) and **915653** (CUDA) subsequently replayed all six seed/role combinations.
On CUDA all original validation and development score values matched exactly.
On CPU, all development scores and validation for seeds 7/27 passed the original
`atol=rtol=1e-4`. One seed-17 validation score still failed, with maximum absolute
difference `0.0007970333099365234`. Thus replay execution is complete; universal
CPU score equivalence is not certified. Original scores and tolerance are unchanged.

The September 19 diagnostic 910651 verified the saved 14-vector metrics, exact metadata and
historical scores, and stored-bootstrap aggregation. It isolated one seed 17
validation record (ID 15321, photograph 659) whose depth-4 top-five membership
changes across CPU precisions at a near tie. FP64 CPU heads reproduced its
saved GPU score, but the historical GPU intermediate ranks are unavailable.
This sensitivity diagnosis does not clear the CPU tolerance failure. The later
completion audit checked all previously unreached scores and independently
recomputed every weighted bootstrap draw. Minor CPU metric differences are
listed in the audit document; all original scientific results remain unchanged.
No numerical check remains pending within the current bounded replay scope.

All numerical execution for this project remains inside Slurm. The existing
production command `python -m pilots.logit_dynamics_20260919.train predict`
expects the original campaign layout, native checkpoints, complete CLS cache
and all-three-seed freeze. It is not an arbitrary-file safetensors inference
CLI. The audit script provides the loading/replay path for freshly fetched
portable files with the original saved-CLS references. No separate inference
implementation was introduced.

## Raw-image extraction and complete reproduction

The backbone and processor are pinned independently:

- Model: `edumunozsala/vit_base-224-in21k-ft-cifar100`, revision
  `b0c51e4a5e5bda35cc922419a28df93bb87e6efa`.
- Processor: `google/vit-base-patch16-224-in21k`, revision
  `b4569560a39a0f1af58e3ddaf17facf20ab919b0`, fast processor enabled.
- Frozen FP32 eager-attention backbone in evaluation mode, gradients disabled;
  raw post-block CLS states 1–12, original classifier logits retained.

The original `extract` command reads the canonical CIFAR-100 clean test images
and four corruption families at severities 3/5 through `OfficialImages`,
verifies their recorded provenance and the exact processor configuration, and
compares final logits/labels/predictions/CLS12 against the existing immutable
cache. It then writes the new compact CLS shards. Those raw inputs and caches
must remain available or be reconstructed from their bound original sources;
the small-model snapshot alone cannot reproduce raw-image extraction.

The campaign commands are `protocol prepare`, `extract --mode full`,
`train fit --seed SEED` for all three seeds, `protocol freeze`,
`train predict --seed SEED`, and `evaluate`. This describes the original fixed
workflow, not an instruction to rerun completed fits. Exact command vectors,
absolute paths, Slurm resources and failed/successful attempts are in the
published operator records. Moving files to a new machine does not make the
absolute-path-bound campaign immediately runnable. Preserve the old manifests;
a relocated reproduction needs a separately documented namespace and restored
baseline/cache/environment dependencies.

## Original CPU audit command — September 19 (historical)

The operator runs this inside the pinned existing server environment, after
staging the checksum-verified source release and the published files. The fresh
published directory contains a hash-bound `FETCH_RECEIPT.json`; downloading the
models and scalers did not perform numerical work. The original models and CLS
cache remain at the experiment root.

```sh
CUDA_VISIBLE_DEVICES='' \
/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_ensemble_20260914/env/bin/python -B \
  /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500/ops/audit_reproducibility_v1.py \
  --root /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500 \
  --release /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500/releases/release-400f1fbbc105a370 \
  --published-dir /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500/audit/published_86605781 \
  --out /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/logit_dynamics_20260919_114500/audit/reproducibility_cpu_v1.json
```

This command also needs the original staged `PYTHONPATH`/overlay package
runtime supplied by the operator runner. It is not a clean-environment setup
command. The original import bundle pins Transformers 5.16.1 and PyG 2.6.1;
native libraries remain in the existing pinned environment. A newly installed
machine, dependency migration, or fresh raw-image end-to-end inference was not
tested by this audit. Repeating the audit requires a new output filename; it
refuses to replace an existing audit receipt.

See [AUDIT_REPRODUCIBILITY.md](AUDIT_REPRODUCIBILITY.md) for checked properties,
upfront numerical tolerances and explicit limits. Interpret results using
[PROTOCOL.md](PROTOCOL.md): this is a fixed-budget retrospective development
comparison, not an untouched-test result or a full paper replication.

The immutable audit receipts are `audit/reproducibility_cpu_v1.json` (SHA-256
`68caa27e659bbe46f7d71c2e5e8d6bcf2a2f15065a32662c0194fe369e5c0d53`)
and `audit/cpu_precision_diagnostic_v1.json` (SHA-256
`9c6ddea512a82f0624eed238d8100d23e0c0b1e4eb016c49563bd380a82c2343`).
The latter is a diagnostic completion receipt, not a successful full replay.

## Completed CPU and CUDA replay — September 21

The new audit source is
`pilots/logit_dynamics_20260919/ops/audit_replay_completion.py`, SHA-256
`582a308a4b8b0ba4f1ef0f0114129761989434d2768f5d663e8bc589ab3dde55`.
It was staged as `ops/audit_replay_completion_20260921_v1.py` on the server.
Both commands use the same `--root`, `--release` and `--published-dir` shown
above, adding `--backend cpu` or `--backend cuda`, and new outputs
`audit/replay_completion_20260921_cpu_v1.json` and
`audit/replay_completion_20260921_cuda_v1.json`. Exact command vectors and
the unchanged environment overlay are in
`run_records/replay_completion_20260921/config_replay_completion_v1.json`.
These are completed historical commands, not instructions to resubmit them.

The audit uses production feature/scoring routines and the original batch size
512; the CPU companion independently computes all 2,000 bootstrap draws with
scikit-learn. New per-record replay scores and bootstrap evidence remain under
separate audit directories on Slurm. Read [the completion record](REPLAY_COMPLETION_20260921.md)
and its private preservation receipt for the later audit-code/evidence snapshot;
the original model revision remains the loading target.
