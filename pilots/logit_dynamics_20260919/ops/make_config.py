"""Write a reviewed-phase configuration from a frozen release receipt; no science."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','validation','full'],required=True);p.add_argument('--release-receipt',type=Path,required=True);p.add_argument('--root',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments';root=a.root;meta=json.loads(a.release_receipt.read_text())
release=root+'/releases/'+meta['release_id'];python=b+'/layer_ensemble_20260914/env/bin/python';module='pilots.logit_dynamics_20260919.'
def command(mod,*args):return [python,'-B','-m',module+mod,*args]
def stage(minutes,gpu,commands,afterok=None):return {'minutes':minutes,'gpu':gpu,'cpus':6 if gpu else 4,'memory_mb':32000 if gpu else 16000,'commands':commands,'afterok':afterok or []}
if a.phase=='preflight':
    smoke=root+'/smoke/preflight_v1'
    stages={'preflight':stage(15,True,[[python,'-B','-m','pytest','-q','pilots/logit_dynamics_20260919'],
               command('protocol','prepare','--root',smoke,'--baseline-root',b+'/final_comparison_20260916_215019'),
               command('extract','--root',smoke,'--data-root',b+'/topology_20260910/data','--mode','preflight'),
               command('preflight','--root',smoke)])}
elif a.phase=='validation':
    stages={'validation':stage(20,False,[[python,'-B','-m','pytest','-q','-p','no:cacheprovider','pilots/logit_dynamics_20260919/test_science.py'],
            command('protocol','prepare','--root',root,'--baseline-root',b+'/final_comparison_20260916_215019'),
            [python,'-B',release+'/ops/validate_baseline.py','--root',root]])}
else:
    stages={'extract':stage(270,True,[command('protocol','prepare','--root',root,'--baseline-root',b+'/final_comparison_20260916_215019'),command('extract','--root',root,'--data-root',b+'/topology_20260910/data','--mode','full')])}
    for seed in [7,17,27]:stages['fit'+str(seed)]=stage(120,True,[command('train','fit','--root',root,'--seed',str(seed))],['extract'])
    stages['freeze']=stage(15,False,[command('protocol','freeze','--root',root)],['fit7','fit17','fit27'])
    stages['predictions']=stage(30,True,[command('train','predict','--root',root,'--seed',str(seed)) for seed in [7,17,27]],['freeze'])
    stages['analysis']=stage(120,False,[command('evaluate','--root',root)],['predictions'])
    stages['backup']=stage(30,False,[[python,'-B',release+'/ops/backup.py','--root',root,'--release',release]],['analysis'])
config={'schema_version':1,'scope':'logit_dynamics_20260919','root':root,'release':release,'python':python,
        'job_prefix':'ld0919-'+root.rsplit('_',1)[-1],'source_manifest_sha256':meta['source_manifest_sha256'],
        'source_identity_sha256':meta['source_identity_sha256'],'environment':{'pilot_root':b+'/layer_ensemble_20260914',
        'overlay_manifest_sha256':'66e9b9f841830177525f5baf7c806713fe23a25de704bdce8ac2c2fcd48fab6c',
        'import_bundle_sha256':'9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f',
        'hf_hub_cache':b+'/topology_20260910/cache/huggingface/hub'},'stages':stages,'phases':{a.phase:list(stages)},
        'authorization':'Root-coordinated implementation of user-approved LogitDynamics follow-up on 2026-09-19; root approves each submitted configuration hash',
        'max_concurrent_gpus':3,'aggregate_gpu_hour_ceiling_including_preflight_and_retries':12,
        'scientific_compute_policy':'Slurm jobs only; preserve all old artifacts',
        'full_submission_gate':'Preflight successful and reviewed by root; no evaluation-driven tuning'}
if a.output.exists():raise RuntimeError('Refusing configuration overwrite')
a.output.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
print(json.dumps({'configuration_path':str(a.output),'configuration_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'phase':a.phase,'gpu_cap_minutes':sum(s['minutes'] for s in stages.values() if s['gpu'])},indent=2))
