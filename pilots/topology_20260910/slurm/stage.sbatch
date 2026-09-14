#!/bin/bash
# Allocation is supplied explicitly by submit.py. No GPU is held by the scheduler.
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
STAGE=${2:?stage required}
RELEASE=${3:?immutable release required}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-topology-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache"
export NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export HF_XET_CACHE="$ROOT/cache/huggingface/xet"
export HF_ASSETS_CACHE="$ROOT/cache/huggingface/assets"
export HF_TOKEN_PATH="$ROOT/private/huggingface/token"
export TORCH_HOME="$ROOT/cache/torch"
export TMPDIR="$OMRI_JOB_CACHE/tmp"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
printf 'JobID=%s Host=%s Stage=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$STAGE" "$(date -Is)"
/usr/bin/python3 -u "$RELEASE/pilots/topology_20260910/slurm/stage.py" --root "$ROOT" --stage "$STAGE"
