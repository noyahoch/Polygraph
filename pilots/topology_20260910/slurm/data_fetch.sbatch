#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=cpu-killable
#SBATCH --job-name=omri-topology-data
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
OLD_ROOT=${2:?existing September root required}
printf 'JobID=%s Host=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
exec /usr/bin/python3 -u "$ROOT/ops/data_fetch.py" --root "$ROOT/data" --existing-inputs "$OLD_ROOT/inputs"
