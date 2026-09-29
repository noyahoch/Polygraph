# Submission revision — September 26, 2026

Status: historical revision, superseded by the user's requested [restoration of the September 24 manuscript](manuscript_restoration_20260926.md). The checks below describe `e82752f`, not the currently restored files.

## Scope and source of the decision

Starting point: local commit `99c4951`, containing the colleague manuscript plus the concise canonical LogitDynamics comparison. In the WhatsApp messages supplied by Omri in this task, Yishai accepted the comparison on September 24, requested a softer overall claim and compliance with the five-page limit, and Omri agreed on September 25 to finish the edit before Noya's review. The September 26 request authorizes this focused revision.

The authoritative course rule is in `Lectures/Lec8- Project.pdf`, physical PDF page 27, under “Final submission”: “The report should be limited to 5 pages (not including references).” Five main-text pages plus a references-only sixth page comply with that length rule.

## Scientific scope

Retain Polygraph's demonstrated improvement over the tested output-only baselines. Replace the general strongest-baseline claim with competitive performance against the fixed LogitDynamics adaptation. The observed AUROC gap is positive, but its paired-photo interval contains zero; this is not evidence of superiority or equivalence.

LD was evaluated on the canonical main benchmark only. Weather and severity statements must retain their actual comparator scope. Preserve the concise LD method and result, including the numerical gap and uncertainty. No new experiments or numerical analyses are part of this edit.

## Evidence

- [Canonical results](../experiments/canonical_logit_dynamics_20260924/RESULTS.md).
- [Frozen protocol](../experiments/canonical_logit_dynamics_20260924/PROTOCOL.md).
- [Scientific review of the completed comparison](../experiments/canonical_logit_dynamics_20260924/SCIENTIFIC_REVIEW.md).
- [September 24 manuscript history](canonical_ld_update_20260924.md).

The compiled PDF and single main TeX remain at `tex/overleaf_20260921/acl_latex.pdf` and `tex/overleaf_20260921/acl_latex.tex`. The stable directory name predates this revision. Changes remain local for Noya's review; shared Overleaf and remote Git are outside this edit.


## Changes and final checks

Five prose locations remain changed relative to `99c4951`: the abstract, the figure caption, the scope of validation-AUROC/early-stopping language, the severity comparison's scope, and the conclusion ending. The abstract and conclusion state close main-test performance against adapted LD without an established advantage. After reviewing the first shortened version (`a42bd78`), Omri requested restoring more of the authors' wording. The introduction opening and first two conclusion sentences are now restored verbatim from `99c4951`; the concise LD conclusion sentence remains. The figure caption stays shortened to meet the page limit.

All five table environments, nine displayed equations, numerical values, existing LD method/result paragraphs, commented history, preamble and nine supporting source files are unchanged. No font, margin, figure-size or spacing adjustment was used to meet the length limit.

Compilation succeeded with no undefined citations/references or overfull boxes. The main text and conclusion end on page 5. References now start on page 6, which contains references only. This satisfies the verified course page-limit rule.

The editor inspected all six rendered pages; the coordinator independently inspected pages 1, 5 and 6. A separate scientific reviewer approved the final restoration and the five remaining prose changes, including the restricted weather/severity claims and the absence of superiority/equivalence claims against LD. [Validation and artifact hashes](submission_revision_20260926_validation.json) record the final checks.
