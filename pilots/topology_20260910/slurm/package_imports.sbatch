#!/bin/bash
#SBATCH --account=gpu-students
#SBATCH --partition=cpu-killable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
ROOT=${1:?experiment root required}
BASE=/home/yandex/MLWG2026/omrifahn/polygraph_september_2026
MAIN="$BASE/env/lib/python3.12/site-packages"
GRAPH="$BASE/env-overnight-graph/lib/python3.12/site-packages"
OUT="$ROOT/import_bundle"
mkdir -p "$OUT"
printf 'JobID=%s Host=%s Start=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
du -sh "$ROOT/env" "$BASE/env-overnight-graph" "$BASE/env" > "$OUT/environment_sizes.txt"
cat "$OUT/environment_sizes.txt"
# The measured bottleneck is Python module discovery in Transformers/PyG.
# Archive these exact pure-Python trees; native CUDA libraries remain unchanged
# at their existing locations instead of copying gigabytes per GPU job.
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$OUT/imports.tar.partial" -C "$MAIN" transformers transformers-5.16.1.dist-info
tar --exclude='__pycache__' --exclude='*.pyc' -rf "$OUT/imports.tar.partial" -C "$GRAPH" torch_geometric torch_geometric-2.6.1.dist-info
mv "$OUT/imports.tar.partial" "$OUT/imports.tar"
sha256sum "$OUT/imports.tar" > "$OUT/imports.sha256"
export OMRI_IMPORT_BUNDLE_OUT="$OUT"
/usr/bin/python3 - <<'PY'
import datetime,hashlib,json,os
from pathlib import Path
p=Path(os.environ['OMRI_IMPORT_BUNDLE_OUT'])
archive=p/'imports.tar'
record={'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'packages':{'transformers':'5.16.1','torch-geometric':'2.6.1'},
        'scope':'Exact installed Python package trees and dist-info; bytecode excluded; native libraries and original environments unchanged'}
(p/'complete.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
