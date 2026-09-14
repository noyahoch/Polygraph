#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=studentbatch
#SBATCH --job-name=omri-topology-preflight
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
RELEASE=${2:?immutable code release required}
PYTHON=${3:?existing graph environment required}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-topology-preflight-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache"
export NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions"
export TMPDIR="$OMRI_JOB_CACHE/tmp"
mkdir -p "$TMPDIR"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export HF_XET_CACHE="$ROOT/cache/huggingface/xet"
export HF_ASSETS_CACHE="$ROOT/cache/huggingface/assets"
export HF_TOKEN_PATH="$ROOT/private/huggingface/token"
export TORCH_HOME="$ROOT/cache/torch"
printf 'JobID=%s Host=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
EXTRA=()
if [[ ${4:-} == publish ]]; then EXTRA+=(--publisher-tests); fi
if [[ -n ${5:-} ]]; then EXTRA+=(--prior-job "$5" --supplemental-check-job "$6" --import-bundle-sha "$7"); fi
/usr/bin/python3 -u "$RELEASE/pilots/topology_20260910/slurm/preflight.py" --root "$ROOT" --release "$RELEASE" --python "$PYTHON" "${EXTRA[@]}"
