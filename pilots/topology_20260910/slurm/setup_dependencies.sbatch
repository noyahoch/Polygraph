#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=cpu-killable
#SBATCH --job-name=omri-topology-deps
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
OLD_ROOT=${2:?existing root required}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PIP_CACHE_DIR="$ROOT/cache/pip"
printf 'JobID=%s Host=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
exec /usr/bin/python3 -u "$ROOT/ops/setup_dependencies.py" --root "$ROOT" --old-root "$OLD_ROOT"
