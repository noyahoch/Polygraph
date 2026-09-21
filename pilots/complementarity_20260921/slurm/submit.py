"""One operator; write-once intents, immutable approvals, bounded reservations."""
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


def save(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_resources(config):
    gpu = sum(s['minutes'] * 60 for s in config['stages'].values() if s['gpu'])
    cpu = sum(s['minutes'] * 60 for s in config['stages'].values() if not s['gpu'])
    ledger = config.get('prior_attempts', {'gpu_seconds': 0, 'cpu_wall_seconds': 0})
    if gpu + ledger['gpu_seconds'] > 7200 or cpu + ledger['cpu_wall_seconds'] > 14400:
        raise RuntimeError('Full reservations plus prior attempts exceed the authorized budget')
    if config['max_concurrent_gpus'] != 3:
        raise RuntimeError('Approved GPU concurrency ceiling changed')
    # The approved DAG groups synchronize all predecessors, never four concurrent GPU stages.
    groups = config['gpu_serial_groups']
    flattened = [name for group in groups for name in group]
    expected = {name for name, stage in config['stages'].items() if stage['gpu']}
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise RuntimeError('GPU synchronization groups do not cover exactly the GPU stages')
    ancestors = {}
    def ancestors_of(name):
        if name in ancestors:
            return ancestors[name]
        parents = config['stages'][name].get('afterok', [])
        result = set(parents)
        for parent in parents:
            result.update(ancestors_of(parent))
        ancestors[name] = result
        return result
    earlier = set()
    for group in groups:
        if len(group) > 3:
            raise RuntimeError('A GPU group exceeds three parallel allocations')
        for name in group:
            if not earlier.issubset(ancestors_of(name)):
                raise RuntimeError('GPU groups are not synchronized: ' + name)
        earlier.update(group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--approved-config-sha256', required=True)
    args = parser.parse_args()
    configuration_sha = sha(args.config)
    if configuration_sha != args.approved_config_sha256:
        raise RuntimeError('Configuration differs from the exact approved hash')
    config = json.loads(args.config.read_text())
    if sha(__file__) != config['operator_submit_sha256']:
        raise RuntimeError('Submitter differs from the reviewed implementation')
    validate_resources(config)
    root = Path(config['root'])
    if not root.name.startswith('complementarity_20260921_'):
        raise RuntimeError('Submission outside the new namespace prohibited')
    phase = config['phases'][args.phase]
    if args.phase in config.get('timing_gated_phases', []):
        gate = json.loads((root / 'ops' / 'timing_gate.json').read_text())
        if gate.get('complete') is not True or gate.get('within_budget') is not True or gate.get('continuation_allowed') is not True:
            raise RuntimeError('Retained-draw timing gate has not approved continuation')
        if gate.get('configuration_sha256') != configuration_sha:
            raise RuntimeError('Timing gate is bound to a different configuration')
    ops = root / 'ops'
    (ops / 'submissions').mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    with (ops / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in phase:
            stage = config['stages'][name]
            receipt = ops / 'submissions' / (name + '.json')
            intent = ops / 'submissions' / (name + '.intent.json')
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['configuration_sha256'] != configuration_sha:
                    raise RuntimeError('Existing stage receipt uses a different configuration')
                print(json.dumps(dict(stage=name, already_submitted=old['job_id'])), flush=True)
                continue
            if intent.exists():
                raise RuntimeError('Ambiguous prior intent must be reconciled, not retried: ' + name)
            dependencies = [json.loads((ops / 'submissions' / (dep + '.json')).read_text())['job_id']
                            for dep in stage.get('afterok', [])]
            job_name = config['job_prefix'] + '-' + name
            wrapper = 'exec ' + shlex.join([config['python'], '-B', str(Path(config['release']) / 'ops/complementarity_stage_runner.py'),
                                           '--config', str(args.config), '--config-sha256', configuration_sha, '--stage', name])
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
                           configuration_sha256=configuration_sha, user='omrifahn', account='gpu-students',
                           job_name=job_name, command=command, dependencies=dependencies,
                           reserved_gpu_seconds=stage['minutes'] * 60 if stage['gpu'] else 0,
                           reserved_cpu_wall_seconds=stage['minutes'] * 60 if not stage['gpu'] else 0)
            save(intent, payload)
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode != 0:
                save(ops / 'submissions' / (name + '.rejected.json'),
                     dict(**payload, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                raise RuntimeError('Scheduler rejected submission; intent retained: ' + name)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Ambiguous scheduler response; intent retained')
            payload.update(job_id=job, response=result.stdout.strip())
            save(receipt, payload)
            print(json.dumps(dict(stage=name, job_id=job, afterok=dependencies)), flush=True)


if __name__ == '__main__':
    main()
