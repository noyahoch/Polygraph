# Recorded coordination context — preflight retry only

This is an authored preservation note, not a copied scheduler or signed approval
receipt. The coordinating root agent's preservation instruction identified
`config_science_v4.json` as approved and job `924803` as submitted. The sole Slurm
operator separately confirmed the exact approved config SHA-256:

`38bd1734a7a45e7e0a4c5e5ffadc1b54f0a9c493902678d41a7b10b646c3da4c`

The authorization covers a **45-minute `preflight_r2` operational retry** with
unchanged `science-source-v2`, existing campaign, protocol, data, parity criteria,
model and runtime. It does not authorize changing scientific settings. The
full phase remains conditional on successful preflight and measured feasibility
review through the current coordinating task.

Config v3 was prepared and statically reviewed with a 30-minute cap, but the
operator confirmed it was never submitted. It is retained as superseded intent.
Version v4 accounts for the previous allocation's recorded 608 GPU-seconds.
The actual scheduler submission/intention files are copied separately and take
precedence for job identity, requested resources and submission timestamp.

No independent approval file existed in the retrieved operator snapshots at
preservation time; this note records the explicit distinction rather than
inventing one. It is historical evidence, not a new permission to submit.
