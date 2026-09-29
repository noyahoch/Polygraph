"""Slurm-only entrypoints for the approved fixed complementarity follow-up."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .common import SPLIT_SALT, freeze_all, prepare, receipt, require_slurm


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'tests-cpu', 'tests', 'cache', 'fit-ablation', 'freeze-ablation',
                                         'fusion-fit', 'freeze', 'predict-ablation', 'fusion-predict', 'statistics', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-ld-root', type=Path)
    parser.add_argument('--parent-ld-release', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--seed', type=int, choices=(7, 17, 27))
    parser.add_argument('--stop-after', type=int, choices=(50,))
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not all((args.parent_ld_root, args.parent_ld_release, args.protocol)):
            parser.error('prepare requires both parent paths and the frozen protocol file')
        prepare(args.root, args.parent_ld_root, args.parent_ld_release, SPLIT_SALT, args.protocol)
        from .statistics import prepare_draws
        prepare_draws(args.root)
    elif args.stage in ('tests', 'tests-cpu'):
        from .test_cache import run_tests
        run_tests(args.root, cuda=args.stage == 'tests')
        if args.stage == 'tests-cpu':
            from .test_stats import run_tests as run_stats_tests
            checks = run_stats_tests()
            receipt(args.root, args.root / 'preflight/statistics_tests.json', [], checks=checks)
    elif args.stage == 'cache':
        from .cache import build
        build(args.root)
    elif args.stage in ('fit-ablation', 'freeze-ablation', 'predict-ablation'):
        from . import ablation
        if args.stage == 'fit-ablation':
            if args.seed is None:
                parser.error('fit-ablation requires --seed')
            ablation.fit(args.root, args.seed)
        elif args.stage == 'freeze-ablation':
            ablation.freeze(args.root)
        else:
            ablation.predict(args.root)
    elif args.stage in ('fusion-fit', 'fusion-predict'):
        from . import fusion
        (fusion.fit if args.stage == 'fusion-fit' else fusion.predict)(args.root)
    elif args.stage == 'freeze':
        freeze_all(args.root)
    elif args.stage == 'statistics':
        from .statistics import run
        run(args.root, stop_after=args.stop_after)
    elif args.stage == 'report':
        from .report import run
        run(args.root)
    print(json.dumps({'stage': args.stage, 'command_completed': True, 'root': str(args.root)}), flush=True)


if __name__ == '__main__':
    main()
