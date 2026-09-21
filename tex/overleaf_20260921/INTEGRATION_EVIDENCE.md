# Focused integration of completed experiments

This update starts from the unchanged September 21, 20:50 colleagues' export in local commit `418652a`. The active entrypoint is [acl_latex.tex](acl_latex.tex), not the earlier proposal or commented drafts. It integrates completed experiments only; no new scientific computation or external publication was performed.

## Change-to-evidence map

| Edited region | Addition and supporting record |
| --- | --- |
| Abstract closing sentences | Scope the original benchmark claim and summarize LD score complementarity and readout decomposition. [September 21 report](../../docs/experiments/complementarity_20260921/results/report/REPORT.md), fusion and paired-contrast tables. |
| Introduction bridge | Extend existing internal-feature, fusion and non-graph-control work with a literature-based comparator and controlled follow-ups. The original related-work organization and claims remain unchanged. |
| Experimental Setup qualifier | Confine error balancing, validation-AUROC early stopping and held-out-test statements to the original experiments. Contrast the later [G/S protocol](../../docs/experiments/final_comparison_20260916/PROTOCOL.md), [LD protocol](../../docs/experiments/logit_dynamics_20260919/PROTOCOL.md) and [fusion/decomposition protocol](../../docs/experiments/complementarity_20260921/PROTOCOL.md). |
| Follow-ups: matched-study context | [Rich36 study results](../../docs/experiments/september10/RESULTS.md): graph-minus-set `+0.006233 [0.002618, 0.009836]`; attention-only `-0.003686 [-0.008349, 0.001045]`, five seeds. The later raw12 graph/set result is `+0.000336 [-0.002800, 0.003409]`, with G/S means `0.888895/0.888559` and the fitted final-layer control `0.883127`; native sources below. Distinct cohorts, budgets and representations prevent a causal cross-study reading. |
| Follow-ups: architecture distinction | [Layer cache extraction](../../pilots/layer_screen_20260913/extract.py), [graph construction](../../pilots/layer_screen_20260913/data.py), [arm selection](../../pilots/final_comparison_20260916/models.py), [endpoint removal](../../pilots/final_comparison_20260916/data.py) and [detector definitions](../../polygraph/training/models.py). Four attention layers share H12; G/S retain identical values but differ in endpoint association and pooling. Per-detector counts are 130,434/131,126. |
| Table 6 and LD comparison/decomposition | [September 19 report](../../docs/experiments/logit_dynamics_20260919/results/REPORT.md) supplies G/S/O/D and G-minus-D. [September 21 report](../../docs/experiments/complementarity_20260921/results/report/REPORT.md) supplies A/B/C/D and their paired increments. All use the same 800 development photographs, 7,200 views and 2,271 errors. The [LD protocol](../../docs/experiments/logit_dynamics_20260919/PROTOCOL.md) supplies 1,200/800/400/800 roles, 16/100 epochs, AP selection and the fixed ViT-Base adaptation limitation. |
| Table 7 and fusion contrasts | [September 21 report](../../docs/experiments/complementarity_20260921/results/report/REPORT.md): a separate 400-photo assessment; DG-minus-D `+0.007203 [0.003415, 0.011330]`, DG-minus-DS `+0.000043 [-0.001311, 0.001508]`, and secondary DG-minus-DDprime `+0.005213 [0.001942, 0.008673]`. [Fusion implementation](../../pilots/complementarity_20260921/fusion.py) preserves D's raw error logit and G/S mean sigmoid scores. |
| Uncertainty paragraph and conclusion | [September 21 protocol](../../docs/experiments/complementarity_20260921/PROTOCOL.md) and [report](../../docs/experiments/complementarity_20260921/results/report/REPORT.md): paired source-photo resampling, combiner refitting, fixed-model decomposition, multiplicity correction and prior development exposure. The conclusion is restricted to tested internal-feature scores and does not infer graph necessity or equivalence. |

## Distinctions retained outside the compact narrative

- The raw12 cache keeps each off-diagonal head value only when it is strictly above 0.02; missing head channels are zero-filled. Original rich36 graph construction selects edges by maximum attention and retains all head values. Diagonal attention remains a node attribute. Rich36's other groups are attention-weighted projected-value magnitudes and signed predicted-versus-runner-up support; see [sidecars.py](../../polygraph/data/sidecars.py) and the [rich36 protocol](../../docs/experiments/september10/PROTOCOL.md).
- The raw12 G-minus-S interval uses **10,000** paired draws and a **six-primary** family: individual 99 1/6% intervals, quantiles `0.004166666666666667` / `0.9958333333333333`. This is not the September 21 correction. G_last_only is the separately fitted logistic control on a single block-11 raw score; it is not a four-detector final-layer ensemble.
- September 19 G-minus-D uses its own ordinary 95% interval. September 21's three primary contrasts DG-minus-D, DG-minus-DS and C-minus-B use **2,000** draws and exact linear-percentile endpoints `0.0083333333` / `0.9916666667`, yielding 98 1/3% individual intervals. Approximately 17 draws lie in each tail. Secondary intervals use 0.025/0.975 and are descriptive.
- The reported B-minus-A increment has a recorded secondary 95% interval `[0.025273, 0.037483]`; the compact manuscript reports only the observed increment. No DS-minus-D or DDprime-minus-D interval is asserted.
- LD AP means scikit-learn average precision. Best readout validation AP selects the checkpoint, retaining the earliest epoch on an exact tie. G/S use validation AUROC. A is six selected original-classifier values, not the full-logit MLP. D and the auxiliary heads remain preserved; D is not replaced post hoc by C in fusion.
- All point estimates are original fitted-model metrics averaged over seeds, not bootstrap averages or pooled-seed prediction metrics. Fusion intervals condition on base detectors and the fixed partition; decomposition intervals condition on all fitted models. Neither corrects prior development exposure or measures complete training variability.

Native later graph/set records are locally preserved under:
`/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/`
(`REPORT.md`, `primary_contrasts.csv`, `report.json`). The repository's [fixed comparison protocol](../../docs/experiments/final_comparison_20260916/PROTOCOL.md) defines the controls and uncertainty. These preserved results were read, not recalculated.

## Preservation and build

- The unified [integration.diff](integration.diff) is against `418652a` and covers only `acl_latex.tex`.
- Original title/authors, preamble, style files, bibliography, comments, method equations, figure and original results/table source blocks are unchanged. Table numbering and pagination change naturally when the new section is added.
- Only two tables are added. All changed regions are an abstract qualification/addition, introduction bridge, original-setup scope, the follow-up section and conclusion synthesis.
- Build: `./build.sh`, using the existing pdfLaTeX/latexmk setup with shell escape disabled. The final [PDF](acl_latex.pdf) has **8 pages including references**; the original had 6 (five main-text pages and a references page). All pages were rendered and visually checked. There are no undefined references/citations or overfull boxes; nonfatal underfull-box and disabled-shell-escape warnings remain.
- TeXcount reports 2,715 body words versus 1,973 in the base (net +742), excluding headings, captions and mathematical expressions. Existing typography was not changed to force a five-page fit. Meeting that target would require a separately authorized reduction of existing material or scope.

## Source discrepancies and unresolved scope

No supplied numerical anchor conflicts were found. Rounding is to four decimals for table means and six for contrasts. The rich36, raw12, original benchmark and fusion/decomposition protocols remain explicitly distinct.

This is a focused integration, not a fresh experiment reproduction or comprehensive audit of colleagues' unchanged claims/bibliography. Historical commented drafts were preserved and not treated as current evidence. The page-budget tradeoff remains unresolved; no unrelated scientific or formatting correction was made.
