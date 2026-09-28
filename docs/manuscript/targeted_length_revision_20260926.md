# Targeted five-page revision, September 26, 2026

Starting from `723a513`, the user requested a small amount of targeted shortening to fit the course page limit. This revision removes 137 whitespace-delimited words from four prose locations:

1. Introduction opening: condense the motivation about patch-token interactions and output confidence.
2. Introduction overview: retain the LD distinction, graph features, fusion procedure and topology finding with less repetition.
3. Figure caption: retain the frozen ViT, graph/MLP experts, scores and conditional gate in a shorter description.
4. Repeated fusion explanation: retain the disjoint, photo-grouped meta-validation requirement; the preceding five-fold procedure remains unchanged.

The accepted abstract and conclusion are unchanged, including “It performs comparably to the tested LogitDynamics baseline.” All experimental setup and results text, including the LD method, table row and detailed statistical paragraph, is unchanged. No font, margin, figure-size or spacing adjustment was made.

## Validation

- The course rule in `Lectures/Lec8- Project.pdf`, physical page 27, excludes references from the five-page limit.
- The rebuilt PDF has **five main-text pages and six physical pages**. The conclusion ends on page 5; references begin below it and continue onto a references-only page 6.
- All five active table environments and nine displayed equations are byte-identical to the starting version, as are the preamble, comments, citations, abstract, conclusion, experimental setup and results.
- Compilation with `sh tex/overleaf_20260921/build.sh` passed, with no undefined citations/references or overfull boxes. The existing warning about disabled shell escape remains.
- All six rendered pages were inspected. Independent review passed for the scientific scope of the four edits and the rendering of pages 4–6. No clipping, overlap or broken tables were observed.
- No experiments, push, merge or shared Overleaf changes were performed.

## Initial artifact identity (`7711d1e`)

- `tex/overleaf_20260921/acl_latex.tex`: SHA-256 `4eff040faab26d3e10d30f8d720dcf74d1de3fb7a631d08e3344ab54f1cd588c`.
- `tex/overleaf_20260921/acl_latex.pdf`: SHA-256 `bd8509cc2aeb8e214a0153b9ce3015240a24c4b796c7015c4270091a6cd5b9e7`.

The longer manuscript remains available in `723a513`. Earlier, more extensive shortening attempts remain historical and were not restored.

## Follow-up: restore more original wording

The user requested restoring original content to fill page 5 more fully. The introduction opening and the paragraph explaining the gate and disjoint meta-validation were restored verbatim from `723a513`, adding 43 whitespace-delimited words. Only the introduction overview and figure caption remain shortened relative to that version. The accepted abstract/conclusion wording, LD comparison, tables, equations and results are unchanged.

Exact byte comparison confirms that the two restorations are the only TeX changes from `45347c4`. Compilation passed without undefined references/citations or overfull boxes. All six rendered pages were inspected, with independent review of pages 4–6 and the scope of the changes. Main text and the complete conclusion end on page 5; references begin on page 6. Fonts, margins, figure dimensions and spacing settings are unchanged.

Current artifact hashes:

- TeX SHA-256: `1b51f31183744a59c3d41d4b67e56409701db646d4671d520636357d2c67bb73`.
- PDF SHA-256: `1f74707d595f9066a40b530aaffd40f4c141d2585d5147032413387d42c2e0e6`.
