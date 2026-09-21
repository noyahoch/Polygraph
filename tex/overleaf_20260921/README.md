# Overleaf proposal snapshot - September 21, 2026

This folder preserves the supplied **Project Proposal: Machine Learning with
Graphs**, with its bibliography, and a locally compiled PDF. The scientific
text and formatting in the downloaded sources are unchanged. This import does
not incorporate experiment results or revise the proposal into a final paper.

- `project_template.tex`: original document source.
- `references.bib`: original bibliography.
- [project_template.pdf](project_template.pdf): compiled document, 4 pages.
- `build.sh`: reproducible build command; temporary files stay in ignored `.build/`.

## Source identity

The user supplied `/Users/omrifahn/Downloads/overleaf/sep21th - Project_Proposal_GNN/`.
During import, the download was moved to
`/Users/omrifahn/Downloads/overleaf/21th 1821/`. The two files were copied from
that location and checked against the corresponding members of
`Project_Proposal_GNN.zip` there.

SHA-256:

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

The final build resolves all citations and produces no overfull boxes. The
unchanged ACM template emits nonfatal metadata/one-sided-header warnings and
one underfull-page warning. These do not prevent PDF generation. Source content
and references have not been scientifically fact-checked as part of this import.
