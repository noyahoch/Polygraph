# Step 3 — Graph vs probe on corruption unseen slices (supporting lookup)

Honest prior (pre-registered): the probe's absolute number may still win on unseen; reported
either way. Result: **clean negative — the hoped-for "structure generalizes where
representation memorizes" is not supported.**

Main plan (test n=17,000), graph vs probe (cls_mlp), bootstrap 95% CI on the difference:

| slice | n | graph | cls_mlp | cls_seq | graph − cls_mlp (CI) |
|---|---:|---:|---:|---:|---|
| all | 17000 | 0.842 | 0.874 | 0.876 | −0.033 [−0.037, −0.028] |
| seen corruptions | 13286 | 0.842 | 0.876 | 0.878 | −0.034 [−0.039, −0.029] |
| **UNSEEN extra family** | 3538 | 0.843 | 0.871 | 0.871 | **−0.028 [−0.038, −0.018]** |
| severity 1→5 (graph−cls_mlp) | — | — | — | — | −0.027, −0.027, −0.035, −0.029, −0.042 |

The graph loses to the probe on the unseen slice (CI excludes zero), and the deficit does not
shrink with severity — it is stable-to-growing. In the corruption (state-failure) regime the
representation probe wins on every slice including unseen. This reinforces the pre-registered
**negative** column (corruptions = state = graph near-redundant); it does NOT provide a P2
supporting finding. Dies cheaply, as designed.
