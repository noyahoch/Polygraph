"""Write an execution proposal for exact-hash coordinator review; never submit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--local-release', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = '/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments'
    parent = base + '/logit_dynamics_20260919_114500'
    parent_release = parent + '/releases/release-400f1fbbc105a370'
    python = base + '/layer_ensemble_20260914/env/bin/python'
    root, release = args.root, args.release
    remote_config = root + '/ops/config_v1.json'
    entry = [python, '-B', '-m', 'pilots.complementarity_20260921.run']
    def run(action, *extra):
        return entry + [action, '--root', root, *extra]
    def stage(minutes, commands, afterok=(), gpu=False, cpus=6):
        return dict(minutes=minutes, commands=commands, afterok=list(afterok), gpu=gpu,
                    cpus=cpus, memory_mb=16000, numerical_threads=1 if not gpu else 6,
                    exclude_nodes=['s-004'])
    stages = {}
    stages['prepare'] = stage(10, [run('prepare', '--parent-ld-root', parent, '--parent-ld-release', parent_release,
                                               '--protocol', release + '/docs/experiments/complementarity_20260921/PROTOCOL.md'),
                                   run('tests-cpu')])
    stages['cache'] = stage(20, [run('tests'), run('cache')], ['prepare'], gpu=True)
    for seed in (7, 17, 27):
        stages['fit' + str(seed)] = stage(25, [run('fit-ablation', '--seed', str(seed))], ['cache'], gpu=True)
    stages['fusion_fit'] = stage(10, [run('fusion-fit')], ['cache'])
    stages['freeze'] = stage(5, [run('freeze-ablation'), run('freeze')], ['fit7', 'fit17', 'fit27', 'fusion_fit'])
    stages['predict'] = stage(10, [run('predict-ablation')], ['freeze'], gpu=True)
    stages['fusion_predict'] = stage(5, [run('fusion-predict')], ['freeze'])
    stages['statistics50'] = stage(15, [run('statistics', '--stop-after', '50'),
                                        [python, '-B', release + '/ops/complementarity_advance_after_timing.py',
                                         '--config', remote_config]], ['predict', 'fusion_predict'], cpus=1)
    stages['statistics_rest'] = stage(150, [run('statistics')], ['statistics50'], cpus=1)
    stages['report'] = stage(5, [run('report')], ['statistics_rest'])
    stages['backup'] = stage(30, [[python, '-B', release + '/ops/complementarity_backup.py',
                                  '--root', root, '--config', remote_config]], ['report'])
    configuration = dict(schema_version=1, scope='complementarity_20260921', root=root, release=release,
                         parent_root=parent, parent_release=parent_release, python=python,
                         parent_source_manifest_sha256='d78438ba6a0c512ab66e5b59a16a772b0c4d972e39e9bf57ceb043b8936144b3',
                         parent_scores_sha256='8b17c6a55c26beb0dc9ac54c574849b61d93c8feedd576677f5e96c20f6dff75',
                         source_manifest_sha256=sha(args.local_release / 'source_manifest.json'),
                         operator_submit_sha256=sha(args.local_release / 'ops/complementarity_submit.py'),
                         job_prefix='comp0921-' + Path(root).name.rsplit('_', 1)[-1],
                         max_concurrent_gpus=3, aggregate_gpu_seconds_ceiling=7200, aggregate_cpu_wall_seconds_ceiling=14400,
                         prior_attempts=dict(gpu_seconds=0, cpu_wall_seconds=0),
                         gpu_serial_groups=[['cache'], ['fit7', 'fit17', 'fit27'], ['predict']],
                         environment=dict(pilot_root=base + '/layer_ensemble_20260914',
                                          overlay_manifest_sha256='66e9b9f841830177525f5baf7c806713fe23a25de704bdce8ac2c2fcd48fab6c',
                                          import_bundle_sha256='9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f',
                                          hf_hub_cache=base + '/topology_20260910/cache/huggingface/hub'),
                         phases=dict(initial=[name for name in stages if name not in ['statistics_rest', 'report', 'backup']],
                                     remaining=['statistics_rest', 'report', 'backup']),
                         timing_gated_phases=['remaining'], stages=stages,
                         authority='Exact configuration requires root review before initial submission; remaining phase is preauthorized only through the retained-draw throughput gate.',
                         protocol_commit='47179a9', science_policy='All scientific commands in Slurm; immutable old outputs; no new ViT, head or GNN training; no paid resources')
    with args.out.open('x') as stream:
        json.dump(configuration, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(config=str(args.out), sha256=sha(args.out),
                          gpu_reserved_seconds=sum(stage['minutes'] * 60 for stage in stages.values() if stage['gpu']),
                          cpu_reserved_wall_seconds=sum(stage['minutes'] * 60 for stage in stages.values() if not stage['gpu'])), indent=2))


if __name__ == '__main__':
    main()
