"""Run prescribed modules in one allocated process, retaining import/hash caches."""
import argparse
import datetime as dt
import json
from pathlib import Path
import importlib
import sys
from .runtime import atomic_json, load_workflow, require_slurm


def module(name, args):
    original = sys.argv
    sys.argv = [name, *[str(arg) for arg in args]]
    try:
        try:
            # Keep smoke's spawn targets under their importable module name.
            # Calling main also reuses process imports across prediction stages.
            importlib.import_module('pilots.layer_ensemble_20260914.'+name).main()
        except SystemExit as error:
            if error.code not in (0, None):
                raise
    finally:
        sys.argv = original


def main():
    require_slurm()
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('preflight', 'postprocess'))
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    r = args.root
    common = ['--cache', r/'feature_cache', '--execution', r/'execution.json', '--roles', r/'role_map.json']
    if args.mode == 'preflight':
        module('protocol', ['--cache', r/'feature_cache', '--out-root', r])
        module('smoke', [*common, '--out', r/'manifests/smoke.json'])
        return
    workflow = load_workflow()
    if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(workflow['deadlines']['predictions']):
        raise RuntimeError('Prediction deadline already passed')
    # The scientific predictor owns verification and the four-base freeze.
    module('predict', [*common, '--run-root', r/'runs', '--base-freeze', r/'base_freeze.json', '--freeze-base',
                       '--role', 'meta', '--out', r/'predictions/meta.npz'])
    module('combine', ['--root', r, '--predictions', r/'predictions/meta.npz', '--out', r/'heads'])
    module('predict', [*common, '--run-root', r/'runs', '--base-freeze', r/'base_freeze.json',
                       '--role', 'dev_eval', '--out', r/'predictions/dev_eval.npz',
                       '--heads-freeze', r/'heads_freeze.json'])
    module('evaluate', ['--root', r, '--predictions', r/'predictions/dev_eval.npz',
                        '--heads', r/'heads', '--out', r/'evaluation'])
    atomic_json(r/'manifests/postprocess.json', {'status': 'complete', 'complete': True,
                 'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
