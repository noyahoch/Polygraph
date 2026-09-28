# LogitDynamics claim-only revision, September 26, 2026

At the user's request, retain the longer manuscript and make only small claim corrections reflecting the canonical LogitDynamics comparison. No five-page compression or unrelated proofreading was performed.

## Scope

Base: `ac9d21f6f46493c0d1205ec1eab714bf92ec5b57`, which restored the September 24 TeX and PDF from `99c4951`.

Two manuscript passages changed:

1. The abstract's broad strongest-performance claim now identifies the improvement over the output-only baseline, including unseen corruption families, and states that main-test AUROC is close to the tested LogitDynamics adaptation with no statistically significant difference.
2. The otherwise-original conclusion receives the same main-test comparison sentence.

The existing LD methods paragraph, numerical results, table row and uncertainty qualifications are unchanged. Close observed performance does not establish equivalence; the retained results paragraph explicitly states that neither superiority nor equivalence is established. The LD comparison is limited to the main test, not the unseen-corruption evaluation.

## Verification

- Exact byte comparison: applying only the two specified substitutions to the base TeX reproduces the current TeX. Every other byte, including comments, formatting and EOF, is unchanged.
- Independent scientific and scope review passed.
- `sh tex/overleaf_20260921/build.sh` compiled successfully. The PDF has **6 total pages**, with main text continuing onto page 6. This revision intentionally does not enforce the submission length limit.
- All six rendered pages were visually checked; no clipping, overlap, broken tables or missing glyphs were observed. Pages 4–6 were also independently reviewed.
- No overfull boxes or undefined references/citations were reported. The existing epstopdf warning states that shell escape is disabled.
- Supporting manuscript source files and experimental evidence were not changed. No experiments, push, merge or shared Overleaf edits were performed.

## Artifact identity

| File | SHA-256 |
| --- | --- |
| `tex/overleaf_20260921/acl_latex.tex` | `49f6288a5d2c5c9e8275ab065aafadc61773698d9fbef63fda8eb11735bf5bd0` |
| `tex/overleaf_20260921/acl_latex.pdf` | `41a93192a172a8c7e395885a39aaeb468414ac893fc799ed7be26c19bd70eeca` |

Scientific source: [canonical results](../experiments/canonical_logit_dynamics_20260924/RESULTS.md) and [independent scientific review](../experiments/canonical_logit_dynamics_20260924/SCIENTIFIC_REVIEW.md).

## Follow-up: concise comparison wording

At the user's next request, both detailed comparison sentences from commit `aeb2738` were replaced with: “It performs comparably to the tested LogitDynamics baseline.” The output-baseline claims remain intact. The detailed statistical paragraph in Results is unchanged, including the statement that neither superiority nor equivalence is established. This is a descriptive performance summary, not a formal equivalence claim.

Exact byte comparison confirms that these two sentence substitutions are the only TeX edits. The PDF compiled successfully to six pages with no undefined references/citations or overfull boxes. Scientific wording review passed; all six rendered pages were visually checked, with pages 4–6 independently reviewed. No unrelated manuscript content was shortened.

Artifact hashes at `723a513` (the earlier table above records the prior revision):

- TeX SHA-256: `377d74c028c0921741f0a61b79d53ce95f1187990a85bfae9cdc04313aad7aca`.
- PDF SHA-256: `b489cb62185e97295ccecdb8cde56f4653827335692934b169f322b945294242`.

The subsequent [targeted length revision](targeted_length_revision_20260926.md) retains these abstract/conclusion sentences and shortens four other prose locations to meet the five-page main-text limit.
