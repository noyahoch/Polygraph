#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=cpu-killable
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
RELEASE=${2:?immutable release required}
MODE=${3:-evaluation}
IMPORT_SHA=${4:-}
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-topology-check-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache"
export NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
printf 'JobID=%s Host=%s Stage=evaluator_recovery_checks Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
/usr/bin/python3 -u "$RELEASE/pilots/topology_20260910/slurm/source_checks.py" --root "$ROOT" --release "$RELEASE" --mode "$MODE" --import-bundle-sha "$IMPORT_SHA"
