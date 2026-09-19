"""Slurm-only schema/identity validation of the frozen historical score bundle."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from pilots.logit_dynamics_20260919.protocol import METADATA, SEEDS, atomic_json, campaign, read, require_slurm, sha256, verify

p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();require_slurm()
c=campaign(a.root);old=Path(c['baseline_root']);score_path=old/'evaluation/scores.npz'
verify(score_path,c['baseline_scores_sha256'])
with np.load(score_path,allow_pickle=False) as archive:
    keys=archive['score_keys'].tolist();scores=archive['scores']
    metadata={key:archive[key] for key in METADATA}
required=[f'{m}/seed{s}' for m in ['G_mean','S_mean','O'] for s in SEEDS]+['MSP','entropy']
if len(keys)!=69 or len(set(keys))!=69 or scores.shape!=(7200,69) or not np.isfinite(scores).all():raise RuntimeError('Historical 69-column score schema invalid')
if not set(required)<=set(keys):raise RuntimeError('Required baseline method keys absent')
roles=read(a.root/'role_map.json')['roles']['dev_eval'];cache=Path(c['cache'])
verify(cache/'index.json',c['cache_index_sha256'])
rows=read(cache/'index.json');by_id={r['record_id']:r for r in rows};ordered=[by_id[i] for i in roles['record_ids']]
for key in METADATA:
    expected=np.asarray([r[key] for r in ordered],dtype=np.int64)
    if metadata[key].dtype!=np.int64 or not np.array_equal(metadata[key],expected):raise RuntimeError('Historical baseline metadata differs: '+key)
images,frequency=np.unique(metadata['image_id'],return_counts=True)
if len(images)!=800 or not np.all(frequency==9) or sorted(images.tolist())!=roles['photo_ids']:raise RuntimeError('Photograph grouping differs')
result={'complete':True,'job_id':os.environ['SLURM_JOB_ID'],'score_sha256':sha256(score_path),'campaign_sha256':sha256(a.root/'campaign.json'),'roles_sha256':sha256(a.root/'role_map.json'),'score_shape':list(scores.shape),'required_keys':required,'all_keys':keys,'metadata_checked':list(METADATA),'source_photographs':800,'views_per_photograph':9,'group':'image_id','new_logit_dynamics_predictions_read':False}
atomic_json(a.root/'ops/baseline_schema_validation.json',result)
print(json.dumps(result),flush=True)
