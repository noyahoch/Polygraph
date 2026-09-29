"""Read-only login-node metadata inventory; never imports numerical libraries."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path('/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments')
PARENT = BASE / 'logit_dynamics_20260919_114500'
RELEASE = PARENT / 'releases/release-400f1fbbc105a370'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=20)
    return dict(command=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


manifest_path = RELEASE / 'source_manifest.json'
manifest = json.loads(manifest_path.read_text())
failures = [name for name, expected in manifest['files'].items() if sha(RELEASE / name) != expected]
required = ['campaign.json', 'role_map.json', 'evaluation_gate.json', 'cls/index.json', 'cls/manifest.json', 'evaluation/scores.npz']
for seed in (7, 17, 27):
    required.extend(f'runs/seed{seed}/{name}' for name in
                    ['heads/checkpoint.pt', 'probe/checkpoint.pt', 'probe/normalizer.json',
                     'probe/validation.npz', 'predictions/dev_eval.npz'])
files = {name: dict(exists=(PARENT / name).is_file(), bytes=(PARENT / name).stat().st_size if (PARENT / name).is_file() else None)
         for name in required}
indices = json.loads((PARENT / 'cls/index.json').read_text())
shards = sorted((PARENT / 'cls/shards').glob('shard_*.pt'))
new_namespaces = []
for path in sorted(BASE.glob('complementarity_20260921_*')):
    new_namespaces.append(dict(root=str(path), intents=[str(p.relative_to(path)) for p in sorted(path.glob('ops/submissions/*'))],
                               configs=[str(p.relative_to(path)) for p in path.glob('ops/*config*.json')]))
vfs = os.statvfs(BASE)
output = dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(), user=os.environ.get('USER'), hostname=os.uname().nodename,
              queue=command(['squeue', '-u', 'omrifahn', '-h', '-o', '%i|%j|%T|%V|%a|%R']),
              recent_accounting=command(['sacct', '-X', '-S', '2026-09-21T00:00:00', '-u', 'omrifahn', '-n', '-P',
                                         '-o', 'JobIDRaw,JobName%70,User,Account,Submit,Start,End,State,ElapsedRaw,AllocTRES%100']),
              new_namespaces=new_namespaces, parent_source_manifest_sha256=sha(manifest_path),
              source_file_count=len(manifest['files']), source_hash_failures=failures,
              parent_scores_sha256=sha(PARENT / 'evaluation/scores.npz'), required_files=files,
              cls_shards=len(shards), cls_shard_bytes=sum(path.stat().st_size for path in shards),
              parent_role_map_sha256=sha(PARENT / 'role_map.json'), cls_index_sha256=sha(PARENT / 'cls/index.json'),
              cls_manifest_sha256=sha(PARENT / 'cls/manifest.json'),
              python_exists=(BASE / 'layer_ensemble_20260914/env/bin/python').exists(),
              overlay_manifest_sha256=sha(BASE / 'layer_ensemble_20260914/dependencies/overlay/complete.json'),
              storage_available_bytes=vfs.f_bavail * vfs.f_frsize,
              quota_note='Filesystem availability only; personal quota not queried because quota hangs on this service.',
              numerical_work=False, mutations=False)
print(json.dumps(output, indent=2, sort_keys=True))
