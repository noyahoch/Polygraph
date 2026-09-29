# Core review before full extraction and training

Date: 2026-09-11. Scope: `protocol.py`, `data.py`, `extract.py`, `models.py`,
`rewire.py`, `validate.py`, `train.py`, and the imported graph/model/feature helpers.
The reviewer did not author the original core/training implementation. This was a
source review against [PROTOCOL.md](PROTOCOL.md), followed by the orchestrator's
authorization for the two focused repairs below. It does not replace independent
execution checks or a second review of the repaired paths.

**Release disposition: conditional, not yet a numerical pass.** Do not launch the
35-fit matrix until the new focused CPU tests, GPU model/capture checks and measured
readiness checks pass on the exact released source. No numerical code ran on the Mac.

## Blocking findings and authorized repairs

1. **The rewiring gate implemented a different experiment.** The original
   `rewire_one` required every graph to complete 2E swaps and achieve changed fraction
   at least 0.8, including held-out graphs. One weak or nonrewirable graph invalidated
   the complete control. The registered rule is the mean changed fraction among
   train/validation graphs with E >= 20, with all graphs retained.

   Repaired `rewire.py`: preserve all targets; treat 2E as the attempted construction
   goal under the unchanged 20E attempt cap; gate only the unweighted development
   mean; retain small and weak graphs. Per-graph development diagnostics and summaries
   include quantiles, weak fractions, accepted/attempted swaps and CLS-neighbor changes.
   Held-out records contain opaque identities and integrity status, with no test
   changed-fraction summaries influencing the gate. Missing eligible development
   support is non-diagnostic. Exact endpoint validity, degree, self-loop and duplicate
   checks remain mandatory. Completed rewiring manifests are reused without changing
   their bytes. Old gate-schema sidecars are rejected rather than silently reused.

2. **Input provenance could change without the trained/frozen identity noticing.**
   `CachedDataset._load` trusted original and rewired tensor paths without checking
   manifest hashes. The original training identity did not bind the rewiring manifest.
   A one-time validation of original shards did not protect subsequent changed files.
   Extraction also retained an old `data_provenance.json` on partial resume without
   comparing current image-array hashes; actual processor configuration was only in
   an overwritten environment file.

   Repaired `data.py`/`validate.py`: verify original and rewired tensor hashes lazily
   once per process/file identity; inode, size, mtime or ctime changes force a new
   check, including already memory-cached files. Bind index offsets to shard record
   IDs, validate all rewired shard hashes, and require the registered configuration
   and diagnostic gate schema. Frozen matrices containing `full_rewired` require
   `rewire_manifest_sha256`. The training/evaluation owner separately implements that
   exact key in config and freeze identity; coordinated integration must be tested.

   Repaired `extract.py` within the same provenance correction: partial resume
   compares current source checksums before appending presentations; processor ID,
   immutable revision, effective configuration and use-fast choice are frozen in
   `processor.json` and the cache manifest. Capture numerical conventions are explicit
   metadata. Per-attempt environment records no longer overwrite canonical processor
   provenance. Reusing a completed cache verifies bound cached outputs and explicitly
   reports that original source files were not rechecked; this permits archive removal
   without allowing mixed-source partial capture.

## Static findings that match the protocol

| Area | Source assessment | Still required on Slurm |
|---|---|---|
| Cohort and labels | SHA-256 source ordering, 2,400/800/800 photo groups, all nine views, all seven arms and five fixed seeds; error is predicted class != true class | Exact archive/source checks, record alignment, counts and group disjointness |
| Feature extraction | Pinned frozen ViT, eval mode, eager attention, final value hook, attention block 11 and hidden state 12; all non-self edges strictly above 0.02, no top-K slicing | Actual installed-transformers hook/API smoke and independent legacy-path agreement |
| Feature fairness | Same node/edge arrays across full arms; rewiring replaces only target incidence; attention-only arms omit hidden/class-conditioned additions; logit arm gets all 100 logits | Numerical feature equality and batching tests; observed extraction finite/shape integrity |
| Architecture controls | S1 pools node/edge sets with CLS access; S2 uses endpoint-local records and no iterative neighborhood updates; graph uses edge-gated updates | Actual parameter counts, within-5% matching, permutation invariance, empty-edge and finite-gradient tests |
| Training | Fresh seeded models; AdamW and BCE positive weight from training counts; logit scaler fitted only on training; strict validation-AUROC improvement, tied scores keep earlier checkpoint | Real training smoke and finite updates under the release environment |
| Resume | Model, optimizer, best state, histories, sampler epoch, loader generator and Python/NumPy/torch/CUDA RNG are saved; RNG restored after audit passes | Interrupted-versus-uninterrupted equivalence on the same device/dtype/layout |
| Threshold and restore | Native error logits; strict greater-than threshold at ceil(0.95*n) correct-validation rank; saved scaler/threshold hash bindings; tolerance and separated-pair/decision checks | Actual save/load and alternative-partition agreement at atol=rtol=1e-5 |
| Test boundary | Training reads train/validation; prediction requires freeze for test; integrity reporting does not print test performance aggregates | Exact training/evaluation/freeze integration and source-hash checks before final scoring |

No additional architecture or scientific feature changes are proposed. Exact parameter
counts and numerical agreement are untested by this review, not assumed from constants.
Numba remains the compiled rewiring backend; Ops reports a compatible NumPy/Numba pair.
There is no reason to introduce a C++ backend unless the actual bounded preflight fails.

## Required release checks

Run inside Slurm on the fresh immutable release:

```bash
python -m pilots.topology_20260910.test_core_integrity
python -m pilots.topology_20260910.validate --device cuda --out /absolute/slurm/preflight-models.json
```

Then run the existing small separate-cache extraction/rewiring/validation/training
preflight, including the training owner's resume tests. The focused regression suite
covers the mean gate, retention of weak/small graphs, exclusion of test diagnostics,
endpoint integrity, checksum-cache invalidation, and changed source/processor rejection
before capture. Full extraction and the production matrix remain gated on actual results.
No claims about graph benefit, classifier accuracy, test error counts, or final performance
are made in this review.
