"""Run one reviewed stage in an allocation; preserve immutable execution receipts."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or sha(a.config) != a.config_sha256:
        raise RuntimeError('An allocation and exact reviewed configuration are required')
    config = json.loads(a.config.read_text())
    root, release = Path(config['root']), Path(config['release'])
    stage = config['stages'][a.stage]
    if not root.name.startswith('canonical_ld_20260924_'):
        raise RuntimeError('Wrong campaign namespace')
    job = os.environ['SLURM_JOB_ID']
    receipts = root / 'ops' / 'stage_receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / (a.stage + '_' + job + '.json')
    if receipt.exists():
        raise RuntimeError('Same-job receipt exists; reconcile before rerun')
    state = dict(stage=a.stage, job_id=job, user=os.environ.get('USER'),
                 hostname=os.uname().nodename, started_utc=now(), status='running',
                 configuration_sha256=a.config_sha256,
                 source_manifest_sha256=config['source_manifest_sha256'], commands=[])
    save(receipt, state)
    start = time.monotonic()
    try:
        manifest_path = release / 'source_manifest.json'
        if sha(manifest_path) != config['source_manifest_sha256']:
            raise RuntimeError('Source manifest changed')
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            path = release / name
            if path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise RuntimeError('Executable source changed: ' + name)
        sys.path.insert(0, str(release))
        from pilots.topology_20260910.slurm.imports import prepare
        pilot = Path(config['environment']['pilot_root'])
        overlay = pilot / 'dependencies' / 'overlay' / 'complete.json'
        if sha(overlay) != config['environment']['overlay_manifest_sha256']:
            raise RuntimeError('Pinned overlay identity changed')
        dependency = json.loads(overlay.read_text())
        if dependency.get('complete') is not True:
            raise RuntimeError('Dependency overlay incomplete')
        scratch = Path(tempfile.mkdtemp(prefix='canonical-' + job + '-', dir='/tmp'))
        environment, imports = prepare(pilot, scratch / 'imports',
                                       config['environment']['import_bundle_sha256'])
        extra_sites = []
        parquet_spec = config['environment'].get('parquet_overlay')
        if parquet_spec is not None:
            parquet_path = Path(parquet_spec['manifest'])
            if sha(parquet_path) != parquet_spec['sha256']:
                raise RuntimeError('Pinned parquet dependency manifest changed')
            parquet = json.loads(parquet_path.read_text())
            if parquet.get('complete') is not True:
                raise RuntimeError('Parquet runtime/reader verification incomplete')
            for name, expected in parquet['files'].items():
                if sha(Path(parquet['site']) / name) != expected:
                    raise RuntimeError('Parquet dependency source changed: ' + name)
            extra_sites.append(parquet['site'])
            state['parquet_overlay_manifest_sha256'] = parquet_spec['sha256']
        environment['PYTHONPATH'] = os.pathsep.join(
            [str(release), *extra_sites, dependency['site'], environment.get('PYTHONPATH', '')])
        threads = int(stage.get('numerical_threads', stage['cpus']))
        if not 1 <= threads <= stage['cpus']:
            raise RuntimeError('Invalid numerical thread count')
        for name in ('tmp', 'cache'):
            (scratch / name).mkdir(exist_ok=True)
        hub_runtime = root / 'data' / 'hf_runtime'
        hub_runtime.mkdir(parents=True, exist_ok=True)
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                           CANONICAL_CONFIG_SHA256=a.config_sha256,
                           OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                           MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
                           CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           HF_HOME=str(hub_runtime), HF_XET_CACHE=str(hub_runtime / 'xet'),
                           XDG_CACHE_HOME=str(scratch / 'cache'), TMPDIR=str(scratch / 'tmp'),
                           HF_HUB_CACHE=config['environment']['hf_hub_cache'])
        if stage.get('offline', True):
            environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        else:
            environment.pop('HF_HUB_OFFLINE', None)
            environment.pop('TRANSFORMERS_OFFLINE', None)
        state['imports'] = imports
        save(receipt, state)
        for command in stage['commands']:
            begin = time.monotonic()
            result = subprocess.run(command, cwd=release, env=environment, check=False)
            state['commands'].append(dict(argv=command, returncode=result.returncode,
                                           elapsed_seconds=time.monotonic() - begin))
            save(receipt, state)
            if result.returncode:
                raise RuntimeError('Scientific command failed with code ' + str(result.returncode))
        state.update(status='complete', completed_utc=now())
    except BaseException as exc:
        state.update(status='failed', failed_utc=now(), error=repr(exc))
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - start
        save(receipt, state)


if __name__ == '__main__':
    main()
