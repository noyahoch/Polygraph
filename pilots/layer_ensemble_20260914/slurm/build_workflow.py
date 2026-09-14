"""Write source-only immutable job declarations; does not execute numerical work."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    parent='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments'
    root=parent+'/layer_ensemble_20260914'
    old=parent+'/topology_20260910'
    release=root+'/releases/'+manifest['release_id']
    python=old+'/env/bin/python'
    module='pilots.layer_ensemble_20260914.'
    control=['--root',root,'--old-root',old,'--release',release]
    stages={
      'preflight':{'command':[python,'-u','-m',module+'slurm.steps','preflight','--root',root],
                   'resources':{'minutes':30,'gpu':True,'cpus':6},'requires':['feature_cache/manifest.json']},
      'dispatch':{'command':['/usr/bin/python3','-u','-m',module+'slurm.control','dispatch',*control],
                  'resources':{'minutes':5,'gpu':False,'cpus':2},'stdlib_only':True,
                  'requires':['manifests/smoke.json']},
      'postprocess':{'command':[python,'-u','-m',module+'slurm.steps','postprocess','--root',root],
                     'resources':{'minutes':300,'gpu':True,'cpus':6},'requires':['manifests/admission.json']},
      'guardian':{'command':[python,'-u','-m',module+'slurm.control','guardian',*control],
                   'resources':{'minutes':1440,'gpu':False,'cpus':1},'stdlib_only':True,'requires':[]},
    }
    for arm,minutes in (('block2',385),('block5',290),('block8',285),('block11',265)):
        stages['fit_'+arm]={'command':[python,'-u','-m',module+'train','--cache',root+'/feature_cache',
             '--run-root',root+'/runs','--execution',root+'/execution.json','--roles',root+'/role_map.json',
             '--arm',arm,'--seed','7','--device','cuda','--num-workers','0'],
             'resources':{'minutes':minutes,'gpu':True,'cpus':6},'requires':['manifests/admission.json']}
    workflow={'scope_id':'layer_ensemble_20260914_core_seed7',
       'code_root':release,'source_sha256':manifest['source_sha256'],
       'import_bundle':{'sha256':'9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f'},
       'matrix':[[arm,7] for arm in ('block2','block5','block8','block11')],
       'epochs':20,'early_stopping':False,'roles':{'base_train':1600,'checkpoint':400,'meta':400,'dev_eval':800},
       'capture_job_id':'892193','capture_cap_minutes':710,'historical_diagnostic_cap_minutes':265,
       'historical_diagnostic_actual_seconds':4014,'budget_gpu_minutes':3000,'max_concurrent_fit_gpus':4,
       'postprocess_cap_breakdown_minutes':{'meta':140,'cpu_heads_gpu_reserved':15,'dev_eval_and_metrics':145},
       'deadlines':{'base':'2026-09-14T23:00:00+03:00','predictions':'2026-09-15T04:00:00+03:00',
                    'report_target':'2026-09-15T05:00:00+03:00','report_latest':'2026-09-15T07:00:00+03:00'},
       'deadline_policy':'Independent running CPU guardian writes complete or explicitly incomplete report; no truncated-model ranking or subset ensemble.',
       'stages':stages}
    workflow['planned_gpu_cap_minutes_including_historical_reserves']=975+sum(s['resources']['minutes'] for s in stages.values() if s['resources']['gpu'])
    if workflow['planned_gpu_cap_minutes_including_historical_reserves']>3000:
        raise RuntimeError('Reserved allocations exceed approved50GPU-hour ceiling')
    args.out.write_text(json.dumps(workflow,indent=2)+'\n')
    print(hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__=='__main__': main()
