"""Back up the completed compact follow-up to the existing private HF repository."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated Slurm job required')
    complete=a.root/'evaluation/complete.json'
    if not complete.is_file() or json.loads(complete.read_text()).get('complete') is not True:
        raise RuntimeError('Completed scientific evaluation required for this backup')
    publication=a.root/'publication';publication.mkdir(exist_ok=True)
    bundle=publication/('snapshot_'+os.environ['SLURM_JOB_ID']);bundle.mkdir(exist_ok=False)
    selected=[]
    for name in ['runs','predictions','evaluation','ops','logs']:
        directory=a.root/name
        if directory.is_dir():
            selected += [f for f in directory.rglob('*') if f.is_file() and not f.is_symlink()
                         and f.suffix in ('.json','.jsonl','.csv','.tsv','.txt','.md','.npz','.safetensors','.pt','.out','.err','.sh','.py')
                         and '.tmp' not in f.name and 'hf_backup_cache' not in f.parts]
    selected += [f for f in a.root.iterdir() if f.is_file() and f.suffix in ('.json','.md')]
    total=sum(f.stat().st_size for f in selected)
    if total>1024**3:raise RuntimeError('Compact backup exceeds 1 GiB; inspect selection before upload')
    for source in sorted(set(selected)):
        destination=bundle/source.relative_to(a.root);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    shutil.copytree(a.release,bundle/'source')
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'Source, protocol, fitted small models, predictions, statistics, logs and operational provenance; excludes raw images and extracted feature tensors','files':{str(f.relative_to(bundle)):sha(f) for f in bundle.rglob('*') if f.is_file()}}
    save(bundle/'backup_manifest.json',manifest)
    cache=publication/'hf_backup_cache';cache.mkdir(exist_ok=True)
    os.environ['HF_TOKEN_PATH']='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910/private/huggingface/token'
    os.environ['HF_HOME']=str(cache);os.environ['HF_HUB_CACHE']=str(cache/'hub');os.environ['HF_XET_CACHE']=str(cache/'xet');os.environ['HF_ASSETS_CACHE']=str(cache/'assets')
    os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('HF_DATASETS_OFFLINE',None)
    receipt=publication/'receipt.json'
    try:
        from huggingface_hub import HfApi
        api=HfApi();repo='omrifahn/polygraph-experiments'
        if not api.repo_info(repo,repo_type='model').private:raise RuntimeError('Backup destination is not private')
        uploaded=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(bundle),path_in_repo=a.root.name+'/'+bundle.name,commit_message='Preserve completed LogitDynamics development comparison')
        result={'status':'uploaded','experiment_status':'complete','repo_id':repo,'private':True,'commit_oid':uploaded.oid,'commit_url':uploaded.commit_url,'snapshot':str(bundle),'manifest_sha256':sha(bundle/'backup_manifest.json')}
        save(receipt,result);print(json.dumps(result),flush=True)
    except Exception as error:
        result={'status':'upload_failed','experiment_status':'complete','error_type':type(error).__name__,'snapshot':str(bundle),'manifest_sha256':sha(bundle/'backup_manifest.json')}
        save(receipt,result);print(json.dumps(result),flush=True)
        # The science is already complete. Never rerun fitting because publication failed.
        sys.exit(2)

if __name__=='__main__':main()
