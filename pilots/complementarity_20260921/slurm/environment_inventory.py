"""Capture actual imported dependency versions inside an allocated environment."""
import argparse
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Environment validation requires a Slurm allocation')
    versions = {}
    for name in ['numpy', 'scipy', 'sklearn', 'torch', 'safetensors', 'huggingface_hub']:
        module = importlib.import_module(name)
        versions[name] = dict(version=module.__version__, imported_from=module.__file__)
        if name == 'torch':
            versions[name]['compiled_cuda'] = module.version.cuda
            versions[name]['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    value = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), job_id=os.environ['SLURM_JOB_ID'],
                 hostname=os.uname().nodename, python=sys.version, executable=sys.executable,
                 platform=platform.platform(), packages=versions,
                 numerical_threads={key: os.environ.get(key) for key in
                                    ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
                 cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
