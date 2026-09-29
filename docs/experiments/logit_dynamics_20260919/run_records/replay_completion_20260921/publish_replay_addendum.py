"""Publish scoped September 21 replay evidence, preserving both earlier snapshots."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,urllib.parse,urllib.request
from pathlib import Path

REPO='omrifahn/polygraph-experiments'
ORIGINAL='86605781c0528e286483e305072f7787393da860'
ORIGINAL_PREFIX='logit_dynamics_20260919_114500/snapshot_910575/'
PREVIOUS_REVIEW='075380b8975fb29f95fa8727234f98cf00e43eaa'
PREVIOUS_REVIEW_PREFIX='logit_dynamics_20260919_114500/review_20260919/'
ADDENDUM_PREFIX='logit_dynamics_20260919_114500/replay_completion_20260921'
ORIGINAL_MANIFEST_SHA='f2e17250d314568bf7b5815b5eea7d79720e8c9bb24d35289f58530f974ed572'

def sha(data):return hashlib.sha256(data).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

class ScopedRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  nxt=super().redirect_request(req,fp,code,msg,headers,newurl)
  if nxt and urllib.parse.urlparse(newurl).hostname!='huggingface.co':nxt.remove_header('Authorization')
  return nxt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
 root=a.root.resolve();bundle=a.bundle.resolve();manifest_path=bundle/'REPLAY_MANIFEST.json';data=manifest_path.read_bytes()
 assert sha(data)==a.manifest_sha256,'Reviewed addendum manifest changed'
 manifest=json.loads(data);assert manifest['original_snapshot_revision']==ORIGINAL and manifest['previous_review_revision']==PREVIOUS_REVIEW and manifest['path_in_repo']==ADDENDUM_PREFIX
 actual_names={str(f.relative_to(bundle)) for f in bundle.rglob('*') if f.is_file()}
 assert actual_names==set(manifest['files'])|{'REPLAY_MANIFEST.json'},'Unexpected file in addendum'
 for relative,spec in manifest['files'].items():
  assert not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Unsafe addendum path'
  path=bundle/relative
  assert not path.is_symlink() and sha(path.read_bytes())==spec['sha256'],'Addendum file hash mismatch'
 assert sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())<32*1024*1024,'Review addendum exceeds bounded text size'
 publication=root/'publication/replay_completion_20260921';publication.mkdir(parents=True,exist_ok=True)
 intent=publication/'upload_intent.json';receipt=publication/'receipt.json'
 assert not intent.exists() and not receipt.exists(),'Prior addendum attempt must be reconciled before retry'
 os.environ['HF_TOKEN_PATH']='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910/private/huggingface/token'
 for key in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE']:os.environ.pop(key,None)
 os.environ['HF_HOME']=str(publication/'hf_cache');os.environ['HF_HUB_CACHE']=str(publication/'hf_cache/hub')
 from huggingface_hub import HfApi
 api=HfApi();before=api.repo_info(REPO,repo_type='model',files_metadata=True)
 assert before.private is True,'Backup destination is not private'
 root_readme_before={f.rfilename:f.blob_id for f in before.siblings if f.rfilename=='README.md'}
 all_existing_blobs={f.rfilename:f.blob_id for f in before.siblings}
 assert not any(f.rfilename.startswith(ADDENDUM_PREFIX+'/') for f in before.siblings),'Review prefix already exists; reconcile first'
 original=api.repo_info(REPO,repo_type='model',revision=ORIGINAL,files_metadata=True)
 original_blobs={f.rfilename:f.blob_id for f in original.siblings if f.rfilename.startswith(ORIGINAL_PREFIX)}
 assert original.private is True and original.sha==ORIGINAL and len(original_blobs)==296
 previous=api.repo_info(REPO,repo_type='model',revision=PREVIOUS_REVIEW,files_metadata=True)
 previous_blobs={f.rfilename:f.blob_id for f in previous.siblings if f.rfilename.startswith(PREVIOUS_REVIEW_PREFIX)}
 assert previous.private is True and previous.sha==PREVIOUS_REVIEW and len(previous_blobs)==38
 assert all(all_existing_blobs.get(k)==v for k,v in original_blobs.items()),'Original snapshot changed before upload'
 assert all(all_existing_blobs.get(k)==v for k,v in previous_blobs.items()),'Previous review changed before upload'
 save(intent,{'started_utc':now(),'repo_id':REPO,'prefix':ADDENDUM_PREFIX,'original_snapshot_revision':ORIGINAL,'previous_review_revision':PREVIOUS_REVIEW,'review_manifest_sha256':a.manifest_sha256,'parent_revision':before.sha,'scope':'Only new replay evidence prefix; no replacement or deletion of any existing file'})
 result={'repo_id':REPO,'original_snapshot_revision':ORIGINAL,'previous_review_revision':PREVIOUS_REVIEW,'path_in_repo':ADDENDUM_PREFIX,'review_manifest_sha256':a.manifest_sha256}
 try:
  uploaded=api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(bundle),path_in_repo=ADDENDUM_PREFIX,commit_message='Preserve completed CPU/CUDA replay and independent bootstrap verification',parent_commit=before.sha)
  result.update(status='uploaded_verification_pending',commit_oid=uploaded.oid,commit_url=uploaded.commit_url)
  # Pin every verification read to the new immutable revision.
  info=api.repo_info(REPO,repo_type='model',revision=uploaded.oid,files_metadata=True)
  assert info.private is True and info.sha==uploaded.oid
  assert {f.rfilename:f.blob_id for f in info.siblings if f.rfilename=='README.md'}==root_readme_before,'Repository root README changed'
  current_blobs={f.rfilename:f.blob_id for f in info.siblings if f.rfilename.startswith(ORIGINAL_PREFIX)}
  assert current_blobs==original_blobs,'Original snapshot tree changed'
  after_blobs={f.rfilename:f.blob_id for f in info.siblings}
  assert all(after_blobs.get(k)==v for k,v in all_existing_blobs.items()),'Existing repository content changed'
  assert {k:v for k,v in after_blobs.items() if k.startswith(PREVIOUS_REVIEW_PREFIX)}==previous_blobs,'Previous review tree changed'
  assert set(after_blobs)-set(all_existing_blobs)=={ADDENDUM_PREFIX+'/'+r for r in actual_names},'Unexpected newly published file'
  credential=Path(os.environ['HF_TOKEN_PATH']).read_text().strip();opener=urllib.request.build_opener(ScopedRedirect)
  def fetch(relative):
   url='https://huggingface.co/'+REPO+'/resolve/'+uploaded.oid+'/'+urllib.parse.quote(relative,safe='/')
   request=urllib.request.Request(url,headers={'Authorization':'Bearer '+credential,'User-Agent':'polygraph-review-verification'})
   with opener.open(request,timeout=25) as response:
    value=response.read(32*1024*1024+1)
    assert len(value)<=32*1024*1024
    return value
  assert sha(fetch(ORIGINAL_PREFIX+'backup_manifest.json'))==ORIGINAL_MANIFEST_SHA
  checked={}
  for relative,spec in manifest['files'].items():
   observed=sha(fetch(ADDENDUM_PREFIX+'/'+relative));assert observed==spec['sha256'],'Uploaded review file differs'
   checked[relative]=observed
  assert sha(fetch(ADDENDUM_PREFIX+'/REPLAY_MANIFEST.json'))==a.manifest_sha256
  result.update(status='verified',private=True,verified_file_sha256=checked,verified_file_count=len(checked)+1,repository_root_readme_unchanged=True,original_snapshot_unchanged=True,original_snapshot_blob_count=len(original_blobs),previous_review_unchanged=True,previous_review_blob_count=len(previous_blobs),all_existing_repository_files_unchanged=True,existing_file_count=len(all_existing_blobs),original_manifest_sha256=ORIGINAL_MANIFEST_SHA,finished_utc=now())
 except Exception as error:
  result.update(status='verification_or_upload_failed',error_type=type(error).__name__,finished_utc=now())
 save(receipt,result);print(json.dumps(result,indent=2))
 if result['status']!='verified':raise SystemExit(2)

if __name__=='__main__':main()
