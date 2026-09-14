"""Source/metadata-only workflow writer; it executes no numerical work."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text())
root='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_screen_20260913'
old='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910'
release=root+'/releases/'+m['release_id'];admission=root+'/manifests/admission_four_20260914.json';py=old+'/env/bin/python'
base=['--root',root,'--old-root',old,'--release',release,'--admission',admission]
stages={
 'admit_four_20260914':{'command':['/usr/bin/python3','-u','-m','pilots.layer_screen_20260913.slurm.admit_four','--root',root,'--release',release],'stdlib_only':True,'requires':['feature_cache/manifest.json']},
 'dispatch_four_20260914':{'command':['/usr/bin/python3','-u','-m','pilots.layer_screen_20260913.slurm.dispatch_four']+base,'stdlib_only':True,'requires':['feature_cache/manifest.json','manifests/admission_four_20260914.json']},
 'ramp_four_20260914':{'command':['/usr/bin/python3','-u','-m','pilots.layer_screen_20260913.slurm.ramp']+base+['--dispatch',root+'/manifests/dispatch_four_20260914.json'],'stdlib_only':True,'requires':['manifests/admission_four_20260914.json']},
 'finalize_four_20260914':{'command':[py,'-u','-m','pilots.layer_screen_20260913.slurm.finalize_four']+base,'requires':[]},
 'guardian_four_20260914':{'command':['/usr/bin/python3','-u','-m','pilots.layer_screen_20260913.slurm.finalize_four']+base+['--guardian'],'stdlib_only':True,'requires':[]},
 'guardian_after_ramp_four_20260914':{'command':['/usr/bin/python3','-u','-m','pilots.layer_screen_20260913.slurm.finalize_four']+base+['--guardian','--after-ramp'],'stdlib_only':True,'requires':[]},
}
for arm in ('block11','union4'):
 for seed in (7,17):
  stages[f'fit_four_{arm}_s{seed}']={'command':[py,'-u','-m','pilots.layer_screen_20260913.train','--cache',root+'/feature_cache','--run-root',root+'/runs_four_20260914','--arm',arm,'--seed',str(seed),'--device','cuda','--num-workers','0'],'requires':['feature_cache/manifest.json','manifests/admission_four_20260914.json']}
w={'scope_id':'layer_screen_four_fits_20260914','code_root':release,'source_sha256':m['source_sha256'],'import_bundle':{'sha256':'9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f'},'matrix':[['block11',7],['union4',7],['block11',17],['union4',17]],'budget_gpu_minutes':4800,'reserved_gpu_minutes':4515,'diagnostic_cap_minutes':265,'capture_limit_minutes':710,'capture_job_id':'892193','capture_source_release':'66840268a7a52037','capture_workflow_sha256':'fa5db859ef3604d9a099a0bc2cd81bae5ba6c2ccddfc0ec743ed04ab55d20316','max_concurrent_gpus':8,'stages':stages}
a.out.write_text(json.dumps(w,indent=2)+'\n');print(hashlib.sha256(a.out.read_bytes()).hexdigest())
