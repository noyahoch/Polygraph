"""Allocation-only syntax/import/runtime gate before freezing a campaign."""
import argparse
import ast
import importlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocation required')
    from pilots.canonical_logit_dynamics_20260924.protocol import atomic_json, runtime
    source = Path(__file__).resolve().parents[1]
    checked = []
    for path in sorted(source.rglob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        checked.append(str(path.relative_to(source)))
    modules = ('protocol', 'data_sources', 'extract', 'data', 'features', 'train', 'evaluate', 'preflight')
    for module in modules:
        importlib.import_module('pilots.canonical_logit_dynamics_20260924.' + module)
    result = dict(complete=True, slurm_job_id=os.environ['SLURM_JOB_ID'],
                  syntax_files=checked, imported_modules=list(modules), runtime=runtime())
    atomic_json(args.root / 'ops' / 'import_check.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
