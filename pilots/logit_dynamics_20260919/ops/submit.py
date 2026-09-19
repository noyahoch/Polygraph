"""Submit only explicitly selected stages; preserve intents and refuse ambiguous retries."""
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
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--phase',choices=['preflight','validation','full'],required=True)
    p.add_argument('--approved-config-sha256',required=True)
    a=p.parse_args()
    config_sha=sha(a.config)
    if config_sha!=a.approved_config_sha256:
        raise RuntimeError('Execution configuration differs from approved hash')
    config=json.loads(a.config.read_text()); root=Path(config['root']); ops=root/'ops'
    if sum(s['minutes'] for s in config['stages'].values() if s['gpu']) > 720:
        raise RuntimeError('Declared aggregate GPU cap exceeds the 12-hour envelope')
    stages=config['phases'][a.phase]
    (ops/'submissions').mkdir(parents=True,exist_ok=True); (root/'logs').mkdir(exist_ok=True)
    with (ops/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in stages:
            stage=config['stages'][name]
            receipt=ops/'submissions'/f'{name}.json'; intent=ops/'submissions'/f'{name}.intent.json'
            if receipt.exists():
                print(json.dumps({'stage':name,'already_submitted':json.loads(receipt.read_text())['job_id']}),flush=True)
                continue
            if intent.exists():
                raise RuntimeError('Ambiguous prior submission; reconcile scheduler before retry: '+name)
            dependencies=[]
            for dep in stage.get('afterok',[]):
                previous=json.loads((ops/'submissions'/f'{dep}.json').read_text())
                dependencies.append(previous['job_id'])
            job_name=config['job_prefix']+'-'+name
            wrapper='exec '+shlex.join([config['python'],'-B',str(Path(config['release'])/'ops/stage_runner.py'),
                                       '--config',str(a.config),'--config-sha256',config_sha,'--stage',name])
            command=['sbatch','--parsable','--account=gpu-students','--partition='+('studentbatch' if stage['gpu'] else 'cpu-killable'),
                     '--job-name='+job_name,'--time='+str(stage['minutes']),'--nodes=1','--ntasks=1',
                     '--cpus-per-task='+str(stage['cpus']),'--mem='+str(stage['memory_mb']),
                     '--output='+str(root/'logs'/(name+'_%j.out')),'--error='+str(root/'logs'/(name+'_%j.err')),
                     '--chdir='+config['release'],'--kill-on-invalid-dep=yes']
            if stage['gpu']:
                command += ['--gpus=1','--constraint=geforce_rtx_2080']
            if dependencies:
                command += ['--dependency=afterok:'+':'.join(dependencies)]
            command += ['--wrap',wrapper]
            payload=dict(stage=name,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),configuration_sha256=config_sha,
                         job_name=job_name,command=command,dependencies=dependencies)
            save(intent,payload)
            result=subprocess.run(command,text=True,capture_output=True,check=True)
            job=result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Ambiguous sbatch response')
            payload.update(job_id=job,response=result.stdout.strip())
            save(receipt,payload)
            print(json.dumps({'stage':name,'job_id':job,'afterok':dependencies}),flush=True)

if __name__=='__main__': main()
