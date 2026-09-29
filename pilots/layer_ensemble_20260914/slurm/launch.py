"""Initial operator-authorized submission only; no ML imports or computation."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .runtime import atomic_json, file_sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workflow',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    a=p.parse_args()
    if file_sha256(a.workflow)!=a.sha256: raise RuntimeError('Unreviewed workflow')
    w=json.loads(a.workflow.read_text()); release=Path(w['code_root']); root=release.parent.parent
    if w['scope_id']!='layer_ensemble_20260914_core_seed7': raise RuntimeError('Wrong scope')
    if w['matrix']!=[[arm,7] for arm in ('block2','block5','block8','block11')]: raise RuntimeError('Wrong matrix')
    if sum(s['resources']['minutes'] for s in w['stages'].values() if s['resources']['gpu'])+975>3000:
        raise RuntimeError('Budget violation')
    for name,sha in w['source_sha256'].items():
        if file_sha256(release/name)!=sha: raise RuntimeError('Changed source: '+name)
    old=root.parent/'topology_20260910'
    os.environ['OMRI_WORKFLOW_PATH']=str(a.workflow)
    os.environ['OMRI_WORKFLOW_SHA256']=a.sha256
    (root/'manifests/submission_intents').mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(parents=True,exist_ok=True)
    def submit(stage,dependency=None):
        row=w['stages'][stage]['resources']
        command=['sbatch','--parsable','--account=gpu-students',
            '--partition='+('studentbatch' if row['gpu'] else 'cpu-killable'),
            '--cpus-per-task='+str(row['cpus']),'--mem='+('32000M' if row['gpu'] else '8000M'),
            '--time='+str(row['minutes']),'--job-name=omri-core-'+stage,
            '--no-requeue','--kill-on-invalid-dep=yes',
            '--output='+str(root/'logs'/f'{stage}_%j.out'), '--error='+str(root/'logs'/f'{stage}_%j.err')]
        if row['gpu']: command += ['--gpus=1','--constraint=geforce_rtx_2080']
        if dependency: command += ['--dependency='+dependency]
        command += [str(release/'pilots/layer_ensemble_20260914/slurm/stage.sbatch'),str(root),str(old),stage,str(release)]
        intent=root/'manifests/submission_intents'/(hashlib.sha256(stage.encode()).hexdigest()+'.json')
        with intent.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if intent.exists():
                previous=json.loads(intent.read_text())
                if previous.get('status')=='recorded' and previous.get('command')==command:
                    return previous['job_id']
                raise RuntimeError('Unresolved or different initial submission intent; reconcile first: '+str(intent))
            value={'status':'submitting','stage':stage,'command':command,'workflow_sha256':a.sha256,
                   'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            atomic_json(intent,value)
            job=subprocess.check_output(command,text=True).strip().split(';')[0]
            if not job.isdigit(): raise RuntimeError('Unrecognized job receipt')
            with (root/'manifests/submissions.tsv').open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                stream.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()}\t{job}\t{stage}\t{row['minutes']}min\t{int(row['gpu'])}gpu\t{row['cpus']}cpu\n")
                stream.flush(); os.fsync(stream.fileno())
            value.update(status='recorded',job_id=job);atomic_json(intent,value)
            return job
    # Independent guardian is accepted before the success-dependent chain.
    guardian=submit('guardian')
    preflight=submit('preflight','afterok:'+w['capture_job_id'])
    dispatch=submit('dispatch','afterok:'+preflight)
    result={'guardian':guardian,'preflight':preflight,'dispatch':dispatch,
            'capture_preserved':w['capture_job_id'],'workflow_sha256':a.sha256,'release':str(release),
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(root/'manifests/launch.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
