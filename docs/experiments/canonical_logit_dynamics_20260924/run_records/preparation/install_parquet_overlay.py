"""One allocated mechanical dependency repair; preserve the old runtime intact."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path('/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/canonical_ld_20260924_085200')
PILOT = Path('/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/layer_ensemble_20260914')
PINS = ['pandas==2.3.3', 'pyarrow==21.0.0', 'python-dateutil==2.9.0.post0',
        'pytz==2025.2', 'tzdata==2025.2', 'six==1.17.0']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocation required')
    destination = ROOT / 'dependencies/parquet_overlay_v1'
    destination.mkdir(parents=True, exist_ok=False)
    wheels, site = destination / 'wheels', destination / 'site'
    wheels.mkdir(); site.mkdir()
    started = time.monotonic()
    pip_python = '/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/env/bin/python'
    environment = os.environ.copy()
    environment.update(PIP_CACHE_DIR=str(destination / 'pip_cache'), PIP_DISABLE_PIP_VERSION_CHECK='1',
                       HF_HOME=str(ROOT / 'data/hf_runtime'), HF_XET_CACHE=str(ROOT / 'data/hf_runtime/xet'))
    subprocess.run([pip_python, '-m', 'pip', 'download', '--index-url', 'https://pypi.org/simple',
                    '--only-binary=:all:', '--no-deps', '--dest', str(wheels), *PINS],
                   check=True, env=environment)
    distributions = {}
    for requirement in PINS:
        name, version = requirement.split('==')
        with urllib.request.urlopen('https://pypi.org/pypi/' + name + '/' + version + '/json', timeout=30) as response:
            metadata = json.load(response)
        distributions[name] = dict(version=version, requires_python=metadata['info']['requires_python'],
                                   requires_dist=metadata['info']['requires_dist'],
                                   wheels={u['filename']: u['digests']['sha256'] for u in metadata['urls']})
    allowed = {filename: digest for spec in distributions.values() for filename, digest in spec['wheels'].items()}
    wheel_hashes = {}
    for path in wheels.glob('*.whl'):
        value = sha(path)
        if allowed.get(path.name) != value:
            raise RuntimeError('Wheel differs from pinned public package metadata')
        wheel_hashes[path.name] = value
    subprocess.run([pip_python, '-m', 'pip', 'install', '--no-deps', '--no-index', '--no-compile',
                    '--find-links', str(wheels), '--target', str(site), *PINS], check=True, env=environment)
    release = ROOT / 'releases/science-source-v1'
    sys.path.insert(0, str(release))
    from pilots.topology_20260910.slurm.imports import prepare
    scratch = Path(tempfile.mkdtemp(prefix='canonical-parquet-' + os.environ['SLURM_JOB_ID'] + '-', dir='/tmp'))
    runtime_environment, imported = prepare(PILOT, scratch / 'imports',
        '9842ee48453ba44e4f043c7cbad41aecef2230da759de238ec45c35319bc262f')
    runtime_environment.update(environment)
    runtime_environment['PYTHONPATH'] = os.pathsep.join((str(release), str(site), str(PILOT / 'dependencies/overlay/site'), imported['local_package_root']))
    runtime_environment.update(HF_HUB_CACHE='/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments/topology_20260910/cache/huggingface/hub',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    verifier = r'''
import importlib, importlib.metadata, json, sys
from pathlib import Path
import torch, numpy as np
from polygraph.data.sources import get_pool
from transformers import AutoImageProcessor
root = Path(sys.argv[1])
modules = {'pandas':'pandas', 'pyarrow':'pyarrow', 'torch':'torch', 'numpy':'numpy',
           'scikit-learn':'sklearn', 'Pillow':'PIL', 'huggingface_hub':'huggingface_hub',
           'torchvision':'torchvision', 'transformers':'transformers', 'safetensors':'safetensors',
           'python-dateutil':'dateutil', 'pytz':'pytz', 'tzdata':'tzdata', 'six':'six'}
versions = {name:dict(version=importlib.metadata.version(name), file=importlib.import_module(module).__file__)
            for name,module in modules.items()}
pool=get_pool('clean_test',0,root/'data')
image=pool.image(0)
first=json.loads((root/'logit_dynamics_reproduction_20260923/data/graph_dataset/scan_records.jsonl').open().readline())
assert first['source']=='clean_test' and first['base_index']==0 and pool.label(0)==first['label']
processor=AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',revision='b4569560a39a0f1af58e3ddaf17facf20ab919b0',use_fast=True,local_files_only=True)
pixels=processor(images=[image],return_tensors='pt')['pixel_values']
assert tuple(pixels.shape)==(1,3,224,224) and torch.isfinite(pixels).all()
assert np.__version__=='2.5.2', 'Do not replace the existing NumPy runtime'
result={'complete':True,'python':sys.version,'distributions':versions,'canonical_reader':{'source':'clean_test','index':0,'label':pool.label(0),'mode':image.mode,'size':list(image.size)},'processor':{'shape':list(pixels.shape),'dtype':str(pixels.dtype),'finite':True},'sys_path':sys.path}
(root/'dependencies/parquet_overlay_v1/runtime_verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'dependency_reader_processor_check':'passed'}))
'''
    subprocess.run([str(PILOT / 'env/bin/python'), '-B', '-c', verifier, str(ROOT)], cwd=release,
                   env=runtime_environment, check=True)
    files = {str(p.relative_to(site)): sha(p) for p in sorted(site.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    result = dict(complete=True, job_id=os.environ['SLURM_JOB_ID'], pins=PINS, site=str(site),
                  wheels=wheel_hashes, distributions=distributions, files=files,
                  verification_sha256=sha(destination / 'runtime_verification.json'),
                  elapsed_seconds=time.monotonic() - started, import_bundle=imported)
    (destination / 'complete.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'complete':True,'elapsed_seconds':result['elapsed_seconds'],'manifest_sha256':sha(destination/'complete.json')}))


if __name__ == '__main__':
    main()
