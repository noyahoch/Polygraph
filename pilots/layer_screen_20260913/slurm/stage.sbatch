#!/bin/bash
set -euo pipefail
umask 077
ROOT=${1:?new experiment root required}
OLDROOT=${2:?reusable experiment environment required}
STAGE=${3:?stage required}
RELEASE=${4:?immutable release required}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
# Same numerical environment as the reviewed ordinary-CUDA throughput measurements.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "$CUBLAS_WORKSPACE_CONFIG"
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-layers-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache"
export NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions"
export HF_HUB_CACHE="$OLDROOT/cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export HF_XET_CACHE="$OMRI_JOB_CACHE/xet"
export HF_ASSETS_CACHE="$OMRI_JOB_CACHE/assets"
export HF_TOKEN_PATH="$OLDROOT/private/huggingface/token"
export TORCH_HOME="$OMRI_JOB_CACHE/torch"
export TMPDIR="$OMRI_JOB_CACHE/tmp"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
printf 'JobID=%s Host=%s Stage=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$STAGE" "$(date -Is)"
printf 'Scratch snapshot (read-only):\n'
timeout 10 df -B1 "$OMRI_JOB_CACHE" || true
timeout 10 findmnt -T "$OMRI_JOB_CACHE" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
/usr/bin/python3 -u "$RELEASE/pilots/layer_screen_20260913/slurm/runner.py" --root "$ROOT" --stage "$STAGE"
