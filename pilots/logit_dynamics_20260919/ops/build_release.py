"""Freeze a source-only release locally; no scientific imports or execution."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
repo=a.repository.resolve();ops=Path(__file__).resolve().parent
prefixes=['pilots/logit_dynamics_20260919','pilots/final_comparison_20260916','pilots/layer_ensemble_20260914','pilots/layer_screen_20260913','pilots/topology_20260910','polygraph']
paths={str(f.relative_to(repo)):f for prefix in prefixes for f in (repo/prefix).rglob('*.py') if '__pycache__' not in f.parts}
for name in ['pilots/__init__.py','docs/experiments/logit_dynamics_20260919/PROTOCOL.md','docs/experiments/logit_dynamics_20260919/SESSION.md']:
    f=repo/name
    if f.is_file():paths[name]=f
for name in ['stage_runner.py','submit.py','status.py','backup.py','validate_baseline.py']:
    f=ops/name
    if f.is_file():paths['ops/'+name]=f
if not any(n.startswith('pilots/logit_dynamics_20260919/') for n in paths):raise RuntimeError('New scientific package missing')
data={name:path.read_bytes() for name,path in sorted(paths.items())}
inventory={name:hashlib.sha256(content).hexdigest() for name,content in data.items()}
identity=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
release=a.output/('release-'+identity[:16]);release.mkdir(parents=True,exist_ok=False)
for name,content in data.items():
    target=release/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content);target.chmod(0o400)
manifest={'files':inventory,'source_identity_sha256':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'source files only; no data, caches or credentials'}
path=release/'source_manifest.json';path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');path.chmod(0o400)
archive=release.with_suffix('.tar.gz')
with tarfile.open(archive,'x:gz') as stream:
    stream.add(release,arcname=release.name)
summary={'release_id':release.name,'release_path':str(release),'archive_path':str(archive),'source_identity_sha256':identity,'source_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'file_count':len(inventory),'archive_bytes':archive.stat().st_size}
(a.output/(release.name+'.json')).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2))
