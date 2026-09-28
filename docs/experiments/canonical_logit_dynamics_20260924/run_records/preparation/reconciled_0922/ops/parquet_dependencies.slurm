#!/usr/bin/env bash
#SBATCH --account=gpu-students
#SBATCH --partition=cpu-killable
#SBATCH --job-name=canonical0924-parquet-deps
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=logs/parquet_deps_%j.out
#SBATCH --error=logs/parquet_deps_%j.err
set -euo pipefail
export HF_HOME=/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200/data/hf_runtime
export HF_XET_CACHE="$HF_HOME/xet"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec /home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_ensemble_20260914/env/bin/python -B ops/install_parquet_overlay.py
