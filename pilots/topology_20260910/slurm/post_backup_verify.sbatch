#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=studentbatch
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?root}
ADDENDUM=${2:?immutable addendum directory}
OMRI_JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/omri-restore-${SLURM_JOB_ID}.XXXXXX")
export OMRI_JOB_CACHE
trap 'rm -rf -- "$OMRI_JOB_CACHE"' EXIT
export PYTHONPYCACHEPREFIX="$OMRI_JOB_CACHE/pycache" NUMBA_CACHE_DIR="$OMRI_JOB_CACHE/numba"
export TORCH_EXTENSIONS_DIR="$OMRI_JOB_CACHE/torch_extensions" TMPDIR="$OMRI_JOB_CACHE"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub" HF_XET_CACHE="$ROOT/cache/huggingface/xet"
export HF_ASSETS_CACHE="$ROOT/cache/huggingface/assets" HF_TOKEN_PATH="$ROOT/private/huggingface/token"
export TORCH_HOME="$ROOT/cache/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
/usr/bin/python3 -u - "$ROOT" "$ADDENDUM" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
root, addendum=map(pathlib.Path,sys.argv[1:])
manifest=json.loads((addendum/'manifest.json').read_text())
assert manifest['decision']=='GO' and manifest['afterok_job']=='879968'
for name, expected in manifest['files'].items():
    assert hashlib.sha256((addendum/name).read_bytes()).hexdigest()==expected,name
release=root/'releases/81b4e83e8bc2a823'
sys.path.insert(0,str(release/'pilots/topology_20260910/slurm'))
from imports import prepare
environment, staging=prepare(root,os.environ['OMRI_JOB_CACHE'],'9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f')
# Avoid any inherited scientific checkout path; only the verified staged third-party trees.
environment['PYTHONPATH']=staging['local_package_root']
print(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'import_staging':staging,'verification_only':True}),flush=True)
raise SystemExit(subprocess.call([str(root/'env/bin/python'),'-u',str(addendum/'post_backup_verify.py'),'--root',str(root)],cwd=addendum,env=environment))
PY
