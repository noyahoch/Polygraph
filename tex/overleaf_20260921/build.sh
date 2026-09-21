#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
mkdir -p .build
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=.build '-pdflatex=pdflatex -no-shell-escape %O %S' project_template.tex
cp .build/project_template.pdf project_template.pdf
