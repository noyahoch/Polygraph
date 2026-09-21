# LogitDynamics replay completion - September 21, 2026

This private addendum preserves completed verification of the already fitted
September 19 models. No training, scientific settings, original predictions,
published results or original numerical tolerance changed.

CUDA replay from the pinned published safetensors reproduced the saved score
values exactly for all three seeds and both validation/development roles.
Every CPU role/seed replay also executed: all development scores met the
unchanged tolerance, while the previously recorded one-row seed-17 validation
exception remains. The earlier failed CPU audit is preserved, not cleared.
Independent CPU recalculation verified all 2,000 paired-photo bootstrap draws.

These are cached-CLS inference checks in the existing pinned environment.
They are not a fresh-install, raw-image end-to-end, or training reproduction.

Start with RESULTS_EN.md or RESULTS_HE.md, then
REPLAY_COMPLETION_20260921.md, AUDIT_REPRODUCIBILITY.md and
run_records/replay_completion_20260921/INDEPENDENT_REVIEW.md.
The scoped documentation tree is preserved with its relative links.

The audit/ directory contains exact new result receipts and all 13 per-row
or bootstrap NPZ evidence files. Fields referring to original absolute server
paths correspond to the same audit/ suffix in this bundle. source/ contains
the exact new audit script; operator/, ops/ and logs/ preserve configurations,
submission/execution evidence and immutable-original checks.
REPLAY_MANIFEST.json records every file checksum and source provenance.

The original models, source and experiment outputs remain at private revision
86605781c0528e286483e305072f7787393da860, prefix
logit_dynamics_20260919_114500/snapshot_910575.
The original review and failed audit remain at private revision
075380b8975fb29f95fa8727234f98cf00e43eaa, prefix
logit_dynamics_20260919_114500/review_20260919.
Both are in omrifahn/polygraph-experiments. Weights are referenced rather than
duplicated here. Original images and CLS feature caches remain on Slurm.
