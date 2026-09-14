"""Install the pinned CPU head dependencies in a separate server-only overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Allocated CPU Slurm job required')
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--requirements',type=Path,required=True)
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=False)
    site=a.root/'site';wheels=a.root/'wheels';wheels.mkdir()
    os.environ['PIP_CACHE_DIR']=str(a.root/'pip_cache')
    os.environ['PYTHONPYCACHEPREFIX']=str(a.root/'pycache')
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    state={'status':'running','job_id':os.environ['SLURM_JOB_ID'],'requirements_sha256':sha(a.requirements),
           'requirements':a.requirements.read_text(),'python':str(a.python),'started_unix':time.time()}
    atomic(a.root/'install.json',state)
    try:
        print('Downloading only the four pinned wheels; existing NumPy/Torch are preserved.',flush=True)
        subprocess.run([str(a.python),'-m','pip','download','--only-binary=:all:','--no-deps',
                        '--dest',str(wheels),'-r',str(a.requirements)],check=True,timeout=240)
        subprocess.run([str(a.python),'-m','pip','install','--no-index','--no-deps',
                        '--find-links',str(wheels),'--target',str(site),'-r',str(a.requirements)],check=True,timeout=180)
        if any(site.glob('numpy*')) or any(site.glob('torch*')):
            raise RuntimeError('Overlay must not replace original NumPy or Torch')
        env=os.environ.copy();env['PYTHONPATH']=str(site)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        check='''import json,importlib.metadata as m,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import scipy,joblib,threadpoolctl
from packaging.requirements import Requirement
from packaging.version import Version
names=('scikit-learn','scipy','joblib','threadpoolctl','numpy','torch')
versions={n:m.version(n) for n in names}
for name in names[:4]:
 for raw in m.requires(name) or ():
  req=Requirement(raw)
  if req.marker and not req.marker.evaluate():continue
  if Version(m.version(req.name)) not in req.specifier:raise RuntimeError('Incompatible requirement:'+raw)
x=np.asarray([[0,0],[0,1],[1,0],[1,1],[2,1],[2,2]],dtype=float);y=np.asarray([0,0,1,1,1,0])
z=StandardScaler().fit_transform(x)
model=LogisticRegression(C=1,solver='lbfgs',penalty='l2',max_iter=1000,tol=1e-6).fit(z,y)
assert np.isfinite(model.decision_function(z)).all()
assert roc_auc_score([0,1,0,1],[0.,1.,0.,1.])==1
print(json.dumps({'versions':versions,'cpu_import_and_fit_check':True,'numpy_path':np.__file__}))
'''
        result=subprocess.run([str(a.python),'-c',check],env=env,text=True,capture_output=True,timeout=150)
        if result.stderr:print(result.stderr,flush=True)
        if result.returncode:raise RuntimeError('CPU compatibility check failed: '+result.stdout+' '+result.stderr)
        measured=json.loads(result.stdout.strip().splitlines()[-1])
        manifest={'status':'complete','complete':True,'requirements_sha256':sha(a.requirements),
           'site':str(site),'wheels':{q.name:sha(q) for q in wheels.iterdir() if q.is_file()},
           'files':{str(q.relative_to(site)):sha(q) for q in site.rglob('*') if q.is_file()},
           'verification':measured,'job_id':os.environ['SLURM_JOB_ID'],'completed_unix':time.time()}
        atomic(a.root/'complete.json',manifest)
        state.update(status='complete',completed_unix=time.time(),complete_manifest_sha256=sha(a.root/'complete.json'))
        print(json.dumps({'status':'complete','manifest_sha256':state['complete_manifest_sha256'],**measured}),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),ended_unix=time.time());raise
    finally:atomic(a.root/'install.json',state)


if __name__=='__main__':main()
