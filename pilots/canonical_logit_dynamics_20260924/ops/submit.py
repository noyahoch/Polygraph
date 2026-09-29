"""Single operator, write-once intents and phase-specific reviewed hashes."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def budget(config):
    if config['max_concurrent_gpus'] != 3:
        raise RuntimeError('GPU concurrency ceiling changed')
    prior = config['prior_attempts']
    gpu = prior['gpu_seconds'] + sum(60 * s['minutes'] for s in config['stages'].values() if s['gpu'])
    cpu = prior['cpu_wall_seconds'] + sum(60 * s['minutes'] for s in config['stages'].values() if not s['gpu'])
    if gpu > 43200 or cpu > 7200:
        raise RuntimeError('Entire planned reservations plus consumed attempts exceed 12 GPU-hours / 2 CPU-hours')
    groups = config['gpu_serial_groups']
    all_gpu = {k for k, s in config['stages'].items() if s['gpu']}
    flattened = [k for group in groups for k in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != all_gpu:
        raise RuntimeError('GPU serial groups must cover every GPU stage exactly once')
    def ancestors(name, trail=()):
        if name in trail:
            raise RuntimeError('Cyclic stage dependencies')
        parents = config['stages'][name].get('afterok', [])
        result = set(parents)
        for parent in parents:
            result |= ancestors(parent, trail + (name,))
        return result
    earlier = set()
    for group in groups:
        if len(group) > 3:
            raise RuntimeError('Too many simultaneous GPU stages')
        for name in group:
            if not earlier.issubset(ancestors(name)):
                raise RuntimeError('GPU groups are not synchronized: ' + name)
        earlier.update(group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--phase', required=True)
    p.add_argument('--approved-config-sha256', required=True)
    a = p.parse_args()
    identity = sha(a.config)
    if identity != a.approved_config_sha256:
        raise RuntimeError('Configuration differs from the explicitly reviewed hash')
    config = json.loads(a.config.read_text())
    if sha(__file__) != config['operator_submit_sha256']:
        raise RuntimeError('Submitter implementation changed')
    budget(config)
    root = Path(config['root'])
    if not root.name.startswith('canonical_ld_20260924_'):
        raise RuntimeError('Wrong submission namespace')
    ops = root / 'ops'
    (ops / 'submissions').mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    with (ops / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in config['phases'][a.phase]:
            stage = config['stages'][name]
            receipt = ops / 'submissions' / (name + '.json')
            intent = ops / 'submissions' / (name + '.intent.json')
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous['configuration_sha256'] != identity:
                    raise RuntimeError('Existing stage uses another configuration; reconcile explicitly')
                print(json.dumps(dict(stage=name, already_submitted=previous['job_id'])), flush=True)
                continue
            if intent.exists():
                raise RuntimeError('Ambiguous previous intent must be reconciled: ' + name)
            dependencies, sealed = [], []
            for dep in stage.get('afterok', []):
                previous = json.loads((ops / 'submissions' / (dep + '.json')).read_text())
                marker = ops / 'stage_receipts' / (dep + '_' + previous['job_id'] + '.json')
                if marker.exists():
                    state = json.loads(marker.read_text())
                    if state.get('status') == 'complete' and state['configuration_sha256'] == previous['configuration_sha256']:
                        sealed.append(dict(stage=dep, job_id=previous['job_id'], receipt_sha256=sha(marker)))
                        continue
                    if state.get('status') == 'failed':
                        raise RuntimeError('Predecessor failed; do not submit successor: ' + dep)
                dependencies.append(previous['job_id'])
            job_name = config['job_prefix'] + '-' + name
            wrapper = 'exec ' + shlex.join([config['python'], '-B',
                str(Path(config['release']) / 'pilots/canonical_logit_dynamics_20260924/ops/stage_runner.py'),
                '--config', str(a.config), '--config-sha256', identity, '--stage', name])
            command = ['sbatch', '--parsable', '--account=gpu-students', '--no-requeue',
                       '--partition=' + ('studentbatch' if stage['gpu'] else 'cpu-killable'),
                       '--job-name=' + job_name, '--time=' + str(stage['minutes']), '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=' + str(stage['cpus']), '--mem=' + str(stage['memory_mb']),
                       '--output=' + str(root / 'logs' / (name + '_%j.out')),
                       '--error=' + str(root / 'logs' / (name + '_%j.err')),
                       '--chdir=' + config['release'], '--kill-on-invalid-dep=yes']
            if stage['gpu']:
                command += ['--gpus=1', '--constraint=geforce_rtx_2080']
            if stage.get('exclude_nodes'):
                command += ['--exclude=' + ','.join(stage['exclude_nodes'])]
            if dependencies:
                command += ['--dependency=afterok:' + ':'.join(dependencies)]
            command += ['--wrap', wrapper]
            payload = dict(stage=name, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           configuration_sha256=identity, user='omrifahn', account='gpu-students',
                           job_name=job_name, command=command, dependencies=dependencies,
                           sealed_completed_dependencies=sealed,
                           reserved_gpu_seconds=60 * stage['minutes'] if stage['gpu'] else 0,
                           reserved_cpu_wall_seconds=60 * stage['minutes'] if not stage['gpu'] else 0)
            save_new(intent, payload)
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                save_new(ops / 'submissions' / (name + '.rejected.json'),
                         dict(**payload, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                raise RuntimeError('Submission rejected; intent preserved')
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Ambiguous scheduler response; intent preserved')
            payload.update(job_id=job, response=result.stdout.strip())
            save_new(receipt, payload)
            print(json.dumps(dict(stage=name, job_id=job, dependencies=dependencies)), flush=True)


if __name__ == '__main__':
    main()
