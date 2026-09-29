# Original-to-current manuscript diff review, September 26, 2026

## Scope and outcome

Reviewed the untouched colleague import `da2cb097697409eaa5d3295faba70628ea7a88c1` against manuscript revision `099d20d`. The original TeX blob matches the imported SHA-256 manifest and the supplied export still on disk. This review does not edit the manuscript or rerun experiments.

The seven changed manuscript locations match the requested scope: three LD additions, two claim qualifications, and two prose reductions. No integral scientific content was lost through the reductions. Two surrounding statements need small scope clarifications following LD's inclusion; they remain unresolved in the reviewed revision.

## Intended changes verified

| Location in reviewed TeX | Change | Review |
| --- | --- | --- |
| Abstract, line 40 | Qualify the general strongest-performance claim | Output-baseline advantage and descriptive comparable performance to the tested LD baseline are supported. No formal equivalence claim is made. |
| Introduction overview, line 60 | Shorten explanatory prose | Retains final-block graph, hidden-state nodes, attention/class edge features, gated MPNN, held-out fusion, complementarity and modest connectivity/alignment contribution. Detailed definitions remain in Methodology. |
| Figure caption, lines 126–128 | Shorten prose repeated in the diagram and method | Retains both experts, their scores and fusion. Explicit readout explanation remains in the figure and equation/text at lines 293–321. |
| LD method, lines 586–593 | Add fixed ViT-B baseline and matched-data description | Consistent with the frozen canonical protocol and completed run. Separate fitting-photo pools, linear probe, validation AP and fixed-length training are correctly described. |
| Main table, line 874 | Add LD row | AUROC 0.8903, AP 0.8733 and AURC 0.2100 match the unmodified result JSON at printed precision. |
| Results, lines 892–896 | Add paired comparison and qualifications | Difference +0.00361, interval [−0.00213, 0.00933] and 2,000 draws match the result JSON. Conditional uncertainty and prior development exposure remain explicit. |
| Conclusion, line 1049 | Add concise LD comparison | Descriptive comparison is consistent with the detailed statement that neither superiority nor equivalence is established. |

## Remaining integration findings

1. **Training rule needs a narrower subject (lines 581–582).** The existing opening says models are selected by validation AUROC with early stopping, then refers to all learned detectors. The new LD paragraph correctly describes validation AP and fixed-length training. Restrict the opening selection statement to the graph/output experts so the reader does not apply it to LD. This is an integration contradiction from adding a method with a different selection rule, not a consequence of shortening.
2. **Severity ranking needs an explicit comparison set (lines 899–901).** The original sentence says Polygraph has the highest mean AUROC at every severity. The severity table contains the original methods and no LD results; the completed canonical LD report provides the overall comparison. Now that LD is in the main table, qualify this as highest among the methods shown in the severity table. This is scope ambiguity, not evidence that the original severity results are wrong or that an LD severity advantage was measured.

Suggested minimal edits, not applied by this review: specify the graph and output experts as the subject of the validation-AUROC rule; add “among the methods in Table 2” to the severity claim. No additional caveats, changed numbers or broad proofreading are recommended.

## Preservation and independent checks

- All nine original supporting source files, including bibliography, style and diagram, match the import hashes.
- The preamble, commented history and every displayed equation are unchanged. All five active tables are unchanged apart from the one LD row; original numerical cells are preserved.
- A separate reviewer checked the semantic retention of the shortened introduction/caption and the abstract/conclusion claims. A second reviewer checked LD additions and their surrounding integration against the canonical protocol, results and implementation records.
- The baseline-to-current manuscript directory diff also contains the regenerated PDF and the user-requested `AGENTS.md` export convention; neither adds manuscript content.
- The latest compiled PDF remains the previously verified six-page file, with five main-text pages and references on page 6. No recompile was required for this read-only manuscript review.

## Evidence and artifact identity

- [Import manifest](overleaf_import_20260924.json).
- [Canonical LD protocol](../experiments/canonical_logit_dynamics_20260924/PROTOCOL.md).
- [Completed canonical results](../experiments/canonical_logit_dynamics_20260924/RESULTS.md).
- [Original numerical report](../experiments/canonical_logit_dynamics_20260924/run_records/final_comparison/evaluation/report.json).
- [Current length/restoration record](targeted_length_revision_20260926.md).

Reviewed TeX SHA-256: `1b51f31183744a59c3d41d4b67e56409701db646d4671d520636357d2c67bb73`.

Reviewed PDF SHA-256: `1f74707d595f9066a40b530aaffd40f4c141d2585d5147032413387d42c2e0e6`.
