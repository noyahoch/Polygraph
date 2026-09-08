# Step 4 — POPE category-holdout shift guard (the replacement transfer axis)

The popular↔adversarial transfer was invalid (76% shared questions). This restores the
exposure-confound guard by holding out object CATEGORIES: split the 79 COCO object categories
in half, train the probe on questions about one half, test on the disjoint half (adversarial
split, out-of-category test n=1298, 203 errors).

| slice | n | errors | probe | output | probe − output (95% CI) |
|---|---:|---:|---:|---:|---|
| all | 1298 | 203 | 0.660 | 0.775 | **−0.115 [−0.159, −0.070]** |
| confident (p>=.9) | 714 | 36 | 0.712 | 0.522 | **+0.190 [+0.044, +0.331]** |

**Two-part finding:**
1. The probe's OVERALL edge is largely category FAMILIARITY — it collapses under category
   shift (−0.115), the same exposure confound that sank representation detectors on the
   corruption weather holdout. The average confident-error headroom is partly disguised
   familiarity, as feared.
2. The CONFIDENT-ERROR edge SURVIVES category shift (+0.19, CI excludes zero) — on object
   categories the probe never trained on. So the confident-error law itself is robust; it is
   NOT merely familiarity. It is a real property of the confident-error regime.

This is the honest resolution of the exposure guard: the *headline* confident-error law holds
under shift; the *broad* probe advantage does not.
