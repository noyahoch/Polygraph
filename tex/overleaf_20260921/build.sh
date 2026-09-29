#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
mkdir -p .build
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=.build '-pdflatex=pdflatex -no-shell-escape %O %S' acl_latex.tex
cp .build/acl_latex.pdf acl_latex.pdf
