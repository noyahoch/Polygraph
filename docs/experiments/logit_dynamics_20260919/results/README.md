# Original Slurm-produced evaluation

These three files are byte-for-byte copies from the completed September 19
analysis, not locally recomputed or edited reports. The full JSON includes all
training histories; for a concise interpretation start with `../RESULTS_HE.md`.

The private immutable artifact revision is
`86605781c0528e286483e305072f7787393da860` in
`omrifahn/polygraph-experiments`, prefix
`logit_dynamics_20260919_114500/snapshot_910575/evaluation/`.
`complete.json` binds the exact report and underlying scores/bootstrap hashes.
Heavy prediction arrays and models remain on Slurm and in the private snapshot.

Original report SHA-256:
`d8daf3374f3fb2bb5279b6383a48a933ce26a1126d7efe66bddb2b9a2ff79112`.

Later audits are separate: their findings must not overwrite these original
results. In particular the first CPU portable replay exceeded its fixed
numerical tolerance; consult `../AUDIT_REPRODUCIBILITY.md` for its disposition.
