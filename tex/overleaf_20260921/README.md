# Overleaf proposal and controlled follow-ups - September 21, 2026

This folder contains **Project Proposal: Machine Learning with Graphs**, with
its original bibliography and a locally compiled PDF. The original download was
preserved in local commit `0304212`. The subsequent contribution-specific revision
adds a separate completed-follow-up section and a few scope clarifications;
it retains the colleagues' original proposal rather than rewriting it as the
implementation used in the follow-ups.

- `project_template.tex`: revised document source.
- `references.bib`: unchanged original bibliography.
- [project_template.pdf](project_template.pdf): compiled current document, 7 pages.
- `build.sh`: reproducible build command; temporary files stay in ignored `.build/`.
- [CONTRIBUTION_SOURCES.md](CONTRIBUTION_SOURCES.md): source-to-claim mapping,
  integration edits, verification boundaries and remaining document issues.

## Source identity

The user supplied `/Users/omrifahn/Downloads/overleaf/sep21th - Project_Proposal_GNN/`.
During import, the download was moved to
`/Users/omrifahn/Downloads/overleaf/21th 1821/`. The two files were copied from
that location and checked against the corresponding members of
`Project_Proposal_GNN.zip` there.

Original import SHA-256 values (the current TeX is intentionally revised):

```text
a95061134da9a9b58b411e99f4f6b19cd87a3000eec370f571d8aa82bff84501  project_template.tex
aba7be15d880a5f68c8d2e55fad815f41289e81b8df1ccaf224b956204baa3c3  references.bib
```

## Build

From this folder, run:

```sh
./build.sh
```

Requires `latexmk`, pdfLaTeX, BibTeX, the `acmart` class and
`ACM-Reference-Format` bibliography style. Built with TeX Live 2025,
latexmk 4.86a and acmart 2.12. Shell escape is disabled.

The original ACM template emits nonfatal metadata/one-sided-header warnings.
These do not prevent PDF generation. The contribution update checks new claims
against the preserved experiment reports and implementation; it does not
fact-check or revise the original literature discussion. No new experiment is
part of this manuscript revision.

The revised build resolves all citations and cross-references and has no
overfull boxes. All seven rendered pages were visually inspected. The inherited
template warnings and underfull-page warnings do not indicate clipped content.
