#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=studentbatch
#SBATCH --job-name=omri-topology-timing
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?}
RELEASE=${2:?}
CACHE=${3:?}
RUNNER=${4:?}
IMPORT_SHA=${5:?}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-topology-measure-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache" NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions" TMPDIR="$OMRI_JOB_CACHE/tmp"
mkdir -p "$TMPDIR"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub" HF_XET_CACHE="$ROOT/cache/huggingface/xet"
export HF_ASSETS_CACHE="$ROOT/cache/huggingface/assets" HF_TOKEN_PATH="$ROOT/private/huggingface/token"
export TORCH_HOME="$ROOT/cache/torch"
printf 'JobID=%s Host=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/usr/bin/python3 -u "$RUNNER" --root "$ROOT" --release "$RELEASE" --cache "$CACHE" --import-sha "$IMPORT_SHA"
