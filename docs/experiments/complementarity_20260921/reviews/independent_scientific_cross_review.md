# Independent fusion/statistics cross-review

**Verdict: ACCEPT; no blocking scientific implementation/report discrepancy identified.** This reviewer implemented the cache/readout components, not the fusion or statistics modules. Independence here is from those modules' implementation. The review is static source and saved-receipt inspection, not an independent numerical reproduction. No numerical calculation, SSH connection or new job was performed.

Reviewed sources reside under:

`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/complementarity-20260921/ops/`

Specifically: the frozen `releases/release-v1/pilots/complementarity_20260921/{fusion,statistics}.py`, frozen `releases/release-v1/docs/experiments/complementarity_20260921/PROTOCOL.md`, `final_compact/report/REPORT.md`, `final_compact/statistics/results.json`, draw manifest and completion receipts. The accepted gate and preservation reviews in this directory establish the reviewed release/receipt bindings.

## Requirements checked

1. **Point estimates use the original fitted models.** `fusion.fit` (line 143) saves the nine original models, and `fusion.predict` (175) uses those saved states after the global gate. `statistics.point_results` (193) takes the frozen fusion/ablation prediction matrices, computes each seed's AUROC/AP, and averages metrics and within-seed contrasts. It does not pool predictions across seeds or substitute bootstrap means. `intervals` preserves the original point-estimate fields. The report states this correctly.

2. **Photographs, not individual views or training seeds, are resampled.** `prepare_draws` (23) uses `SeedSequence(20260921).spawn(3)` in the declared order: fusion fit400, fusion assessment400, ablation assessment800. Photo lists are ascending; all nine rows receive their photo's integer multiplicity. In `one_draw` (291), each role's weight vector is constructed once before the seed/arm loops and shared by every compared method and seed. Seeds remain 7/17/27; the two assessment cohorts use separate streams. The saved draw manifest agrees.

3. **Every fusion bootstrap draw refits the scaler and combiner.** `one_draw` calls `fit_combiner` anew for each seed and DG/DS/DDprime arm. `fit_combiner` (fusion line 83) requires nonnegative integer multiplicities summing to 3,600, applies the same weights to `StandardScaler.fit` and logistic regression, and retains the fixed L2/C/intercept/lbfgs settings. Weights are not normalized, so the intended regularization scale is preserved. Missing weighted classes, nonfinite inputs and convergence problems are treated as failures. The A/B/C/D ablation scores stay fixed inside bootstrap, as specified.

4. **Rankings follow each newly fitted score vector.** `predict_combiner` (fusion line 116) recomputes raw decision values from the draw-specific scaler and coefficients. `one_draw` passes those new predictions to sklearn's weighted AUROC. It does not reuse the original ordering. The fixed-draw weighted/literal checks at statistics line 219 independently address scaler/decision equivalence and same-score AUROC/AP equivalence; cross-fit near-tie differences remain explicit diagnostics. No score rounding or merging is introduced.

5. **The exact family and failure policy are retained.** The three primary contrasts are DG−D, DG−DS and C−B. `intervals` (339) uses exactly `[0.0083333333, 0.9916666667]` with NumPy linear interpolation, and descriptive `[0.025, 0.975]` for the other three. It requires all fixed IDs 0–1999, never resamples seeds or averages fewer seeds, and withholds an affected interval if any required draw is invalid. `run` (356) preserves failures by their original IDs, does not draw replacements, and resumes the retained first50 rather than discarding them. Saved results report 2,000 valid draws for all six contrasts and no failed IDs. This is recorded completion evidence; raw draw calculations were not rerun by this reviewer.

6. **Cohorts and interpretations remain separate.** The report clearly separates 400-photo fusion and 800-photo decomposition tables. It names development-data reuse, conditional uncertainty, overlapping cyclic Dprime pairs, the absence of compute matching, the six-feature nature of A, the class-identity content of D−C, the sparse primary tails and the distinction between nonsignificance and equivalence. It does not claim topology necessity, causal effects of depth or validation on an untouched test set.

## Result wording supported by the saved report

- DG−D is reported as +0.007203 AUROC, with its prespecified interval `[0.003415, 0.011330]`. This supports additional ranking utility from the fitted graph score under this fixed exploratory fusion protocol.
- DG−DS is +0.000043, interval `[-0.001311, 0.001508]`. A graph-specific advantage over matched set-score fusion is not established; this does not prove equivalence. DS has a higher observed point AUROC than D, but no separate DS−D interval was planned/reported, so one must not invent such an inferential result.
- C−B is +0.004135, interval `[0.000428, 0.007911]`. The claim should remain usefulness of intermediate auxiliary projections under this particular readout recipe. Its small positive lower endpoint warrants the report's explicit finite-bootstrap-tail caution.
- D−C is a small negative descriptive contrast, −0.000900, interval `[-0.001750, −0.000048]`. It is not a general finding that dynamics features are harmful. B−A and DG−DDprime also remain secondary/descriptive, not additional family-controlled claims.

These values are transcribed from the completed report/JSON, not independently recomputed. The final report's methodological caveats are adequate for these narrow statements. No method change, extra experiment or result-driven retuning is required by this cross-review.
