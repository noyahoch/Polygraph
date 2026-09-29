# September 24 manuscript: LogitDynamics additions only

This is the historical September 24 snapshot, finalized in `99c4951`. The subsequently authorized claim/length revision is documented in [September 26 submission revision](submission_revision_20260926.md). At the user's later request, the September 24 TeX and PDF were [restored byte-for-byte](manuscript_restoration_20260926.md). A subsequent [claim-only revision](ld_claim_revision_20260926.md) changes only the abstract and conclusion without shortening. The scope statements and validation hashes below describe the historical September 24 snapshot, not the active files.

## Original and current scope

The colleague export at `/Users/omrifahn/Downloads/poly24/overleaf24/Polygraph_report/` was imported unchanged in commit `da2cb097697409eaa5d3295faba70628ea7a88c1`. The [import manifest](overleaf_import_20260924.json) records all ten supplied files. The stable manuscript directory remains `tex/overleaf_20260921/`.

Following the user's scope correction, all stylistic, shortening and other edits from commit `87f17e8` have been reversed. That version remains in Git history. The current TeX differs from the colleague baseline by exactly three insertions:

1. A paragraph describing the canonical LogitDynamics adaptation, training allocation and alignment with the original test records.
2. One main-table row: LD AUROC **0.8903**, AP **0.8733**, AURC **0.2100**, averaged across seeds 1/2/7.
3. A results paragraph giving the mean within-seed Polygraph-minus-LD AUROC difference **+0.00361**, paired-photo 95% interval **[−0.00213, 0.00933]** from 2,000 draws, and the comparison's limitations.

Removing these three insertions reproduces the original TeX byte-for-byte. No original prose, caption, table row, equation, comment, formatting or supporting file is changed. Earlier ensemble, fusion and decomposition experiments are not added. No general proofreading or reassessment of the colleagues' existing wording was performed in this correction.

## Evidence and verification

- [Frozen canonical protocol](../experiments/canonical_logit_dynamics_20260924/PROTOCOL.md).
- [Completed results and limitations](../experiments/canonical_logit_dynamics_20260924/RESULTS.md).
- [Machine-readable results](../experiments/canonical_logit_dynamics_20260924/run_records/final_comparison/evaluation/report.json).
- [Independent scientific review](../experiments/canonical_logit_dynamics_20260924/SCIENTIFIC_REVIEW.md).
- [Artifact hashes and preservation checks](canonical_ld_update_20260924_validation.json).

An independent reviewer confirmed that the current diff contains exactly the three insertion hunks and no edits to original lines. The numerical results are unchanged. At the user's request, the LD method paragraph now describes the same frozen classifier, layer-wise readouts, separate fitting subsets, validation selection and matched test views without the detailed counts, seed IDs or unused meta-validation allocation. The final replication/topology-disclaimer sentence was removed; the result, interval, conditional uncertainty and development-data qualification remain. Full implementation details remain in the linked frozen protocol and results report. No original colleague text was edited.

The PDF compiled successfully using the existing build helper, with no undefined references/citations or overfull boxes. Pages 4-6 were visually checked after the shortening; pages 1-3 render identically to the previously inspected version. It has **six physical pages**: main text now continues onto page 6, where references also begin. No unrelated text was shortened to restore a five-page main-text limit. All nine supporting source files match the original import checksums.

Only local document work and a local commit accompany this correction; no experiments, remote Overleaf changes, push or merge.
