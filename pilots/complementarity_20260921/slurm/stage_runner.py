"""Immutable-source, allocation-only runner for the new complementarity scope."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific execution requires a Slurm allocation')
    if sha(args.config) != args.config_sha256:
        raise RuntimeError('Execution configuration changed')
    config = json.loads(args.config.read_text())
    stage = config['stages'][args.stage]
    root, release = Path(config['root']), Path(config['release'])
    job = os.environ['SLURM_JOB_ID']
    receipt = root / 'ops' / 'stage_receipts' / (args.stage + '_' + job + '.json')
    if receipt.exists():
        raise RuntimeError('Existing same-job receipt; automatic requeue is prohibited')
    state = dict(stage=args.stage, job_id=job, started_utc=now(), status='running',
                 configuration_sha256=args.config_sha256,
                 source_manifest_sha256=config['source_manifest_sha256'],
                 hostname=os.uname().nodename, gpu=stage['gpu'])
    atomic(receipt, state)
    started = time.monotonic()
    try:
        manifest_path = release / 'source_manifest.json'
        if sha(manifest_path) != config['source_manifest_sha256']:
            raise RuntimeError('Source inventory changed')
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            path = release / name
            if not path.is_file() or path.is_symlink() or sha(path) != expected:
                raise RuntimeError('Source changed: ' + name)
        original = Path(config['parent_release']) / 'source_manifest.json'
        if sha(original) != config['parent_source_manifest_sha256']:
            raise RuntimeError('Parent source inventory changed')
        for name, expected in json.loads(original.read_text())['files'].items():
            if sha(Path(config['parent_release']) / name) != expected:
                raise RuntimeError('Parent source changed: ' + name)
            if manifest['files'].get(name) != expected:
                raise RuntimeError('New release did not preserve parent source: ' + name)
        pilot = Path(config['environment']['pilot_root'])
        overlay = pilot / 'dependencies/overlay/complete.json'
        if sha(overlay) != config['environment']['overlay_manifest_sha256']:
            raise RuntimeError('Pinned overlay inventory changed')
        dependency = json.loads(overlay.read_text())
        if dependency.get('complete') is not True:
            raise RuntimeError('Pinned overlay is incomplete')
        scratch = Path(tempfile.mkdtemp(prefix='complementarity-' + job + '-', dir='/tmp'))
        sys.path.insert(0, str(release))
        from pilots.topology_20260910.slurm.imports import prepare
        environment, imported = prepare(pilot, scratch / 'imports', config['environment']['import_bundle_sha256'])
        environment['PYTHONPATH'] = os.pathsep.join((str(release), dependency['site'], environment.get('PYTHONPATH', '')))
        for name in ['tmp', 'hf', 'torch', 'torch_extensions', 'numba']:
            (scratch / name).mkdir()
        threads = int(stage.get('numerical_threads', stage['cpus']))
        if threads < 1 or threads > int(stage['cpus']):
            raise RuntimeError('Numerical thread count exceeds CPU allocation')
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                           COMPLEMENTARITY_CONFIG_SHA256=args.config_sha256,
                           OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                           OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR=str(scratch / 'tmp'),
                           HF_HOME=str(scratch / 'hf'), HF_HUB_CACHE=config['environment']['hf_hub_cache'],
                           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           TORCH_HOME=str(scratch / 'torch'), TORCH_EXTENSIONS_DIR=str(scratch / 'torch_extensions'),
                           NUMBA_CACHE_DIR=str(scratch / 'numba'))
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN_PATH'):
            environment.pop(key, None)
        if not stage['gpu']:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        state['environment'] = dict(python=config['python'], overlay_manifest_sha256=sha(overlay),
                                    imports=imported, scratch=str(scratch), numerical_threads=threads)
        inventory = root / 'ops/environments' / (args.stage + '_' + job + '.json')
        subprocess.run([config['python'], '-B', str(release / 'ops/complementarity_environment_inventory.py'),
                        '--out', str(inventory)], cwd=release, env=environment, check=True)
        state['environment']['actual_inventory_path'] = str(inventory)
        state['environment']['actual_inventory_sha256'] = sha(inventory)
        atomic(receipt, state)
        for index, command in enumerate(stage['commands']):
            print(json.dumps(dict(stage=args.stage, command_index=index, started_utc=now(), command=command)), flush=True)
            subprocess.run(command, cwd=release, env=environment, check=True)
        state.update(status='complete', complete=True)
    except BaseException as error:
        state.update(status='failed', complete=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        atomic(receipt, state)


if __name__ == '__main__':
    main()
