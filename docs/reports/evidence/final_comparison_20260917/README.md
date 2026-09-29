# Preserved September 16–17 fixed20 comparison

This compact evidence copy supports the [team report](../../PROJECT_SYNTHESIS_2026-09-22.md). It was added on September 22, 2026, without rerunning experiments or statistics.

- [Native report](REPORT.md): all 25 reported methods, six primary contrasts, training diagnostics and claim limits.
- [Method summary](method_summary.csv): exact mean metrics and observed seed variation.
- [Primary contrasts](primary_contrasts.csv): all six registered paired AUROC comparisons, ordinary and multiplicity-adjusted intervals.
- [Completion receipt](complete.json): identities and hashes of the original output inventory, fitted models, predictions, source files and statistical specification.
- [Frozen protocol](../../../experiments/final_comparison_20260916/PROTOCOL.md) and [scientific synthesis, section 7](../../../experiments/scientific_synthesis_20260917/HANDOFF.md#7-september-1617-final-benchmark-the-most-informative-extension).

## Provenance and verification

The four files were copied byte-for-byte from the previously downloaded Slurm artifacts at:

```text
/Users/omrifahn/Documents/Codex/2026-09-05/read-this-exported-whatsapp-chat-and/work/results-final-20260917/raw/evaluation/
```

Original remote namespace:

```text
/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/final_comparison_20260916_215019/evaluation/
```

The preserved receipt identifies analysis job `904074`, scope `final_comparison_20260916_fixed20`, and input identity `b51ad29f90453bc6c3554df75b98ee7579eea33c14c0903589c2451d20a65fe6`. Historical job identifiers are provenance, not a claim about the current scheduler.

On September 22, local source and destination bytes were compared, and SHA-256 values for the report and both CSVs matched the existing completion receipt. The four copied files total 40,479 bytes. They were inspected as experimental documentation/metadata; they contain no credentials or private chat material. No Slurm access, fresh statistical calculations, model loading, prediction replay or independent end-to-end reproduction was performed for this copy.

The directory's `.gitattributes` disables text normalization for these four native artifacts so Git preserves their original bytes, including CSV line endings. Staged artifact hashes were checked against the local preserved copies before committing.

| File | SHA-256 | Verification |
|---|---|---|
| `REPORT.md` | `4adfe72991eef1ecff00e4077572243299f26c3b7c6c80c3dafdb9231e742121` | Matches preserved receipt |
| `method_summary.csv` | `b3d9d08d4b46849f0a589fe8459b3c4af304602df2954f9c5f006d30d3e1fad3` | Matches preserved receipt |
| `primary_contrasts.csv` | `55d04a4fed2bb98dbe5c4fab61b728fffbe85b8d14748a3833bcbf53e1e4f038` | Matches preserved receipt |
| `complete.json` | `24dae88f64290852264890e457b046b373646cd7d8dd7883d3fcea1291543aa3` | Hash recorded on copy; no external receipt hash revalidated here |

This is a compact numerical evidence package, not the full model/cache/prediction archive. Native relative references inside `REPORT.md` describe the original remote experiment tree and may point to files not included in this subset. The completion receipt lists a larger output inventory than is copied here; it does not imply that every listed artifact is available in this directory.

## Interpretation to preserve

The evaluation used 800 development photographs with nine views each, previously exposed during project development. Thirty-nine neural fits comprise 12 historical graph imports and 27 fresh hidden/set/output fits; they are not 39 independent new replications. Seeds are 7, 17 and 27. Imported graph seed 7 retains its earlier late-diagnostic status.

Graph and same-value set ensembles achieved similar scores; the registered comparison did not establish an additional graph advantage or equivalence. The graph-versus-output result concerns the entire tested feature/model/ensemble recipe. Graph-versus-hidden comparisons do not hold hidden-layer exposure identical. All intervals are conditional on fitted models and the reused development cohort; training-seed and previous-selection uncertainty remain outside them.

The earlier September 14 deadline miss and later prediction/bookkeeping repairs remain in the historical record. Completing and preserving this comparison does not retroactively change those earlier statuses.
