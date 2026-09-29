"""One allocated GPU preflight; only synthetic checks and a separate diagnostic cache."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--publisher-tests", action="store_true")
    p.add_argument("--prior-job")
    p.add_argument("--supplemental-check-job")
    p.add_argument("--import-bundle-sha")
    a = p.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise SystemExit("Allocated Slurm GPU required")
    os.umask(0o077)
    out = a.root / "preflight" / job
    out.mkdir(parents=True, exist_ok=False)
    data_receipt = json.loads((a.root / "data/download_complete.json").read_text())
    if data_receipt.get("status") != "complete":
        raise RuntimeError("Official data download did not complete")
    environment = json.loads((a.root / "manifests/environment/complete.json").read_text())
    if environment.get("status") != "complete" or environment["python"] != a.python:
        raise RuntimeError("Exact isolated dependency overlay has not passed setup")
    source = json.loads((a.release / "source_manifest.json").read_text())
    for name, expected in source["files"].items():
        if hashlib.sha256((a.release / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Snapshot hash mismatch: {name}")
    python = a.python
    diagnostic = out / "diagnostic_cache"
    prefix = [python, "-u", "-m", "pilots.topology_20260910"]
    commands = [
        ("syntax", ["/usr/bin/python3", "-c", "import ast,pathlib; paths=list(pathlib.Path('polygraph').rglob('*.py'))+list(pathlib.Path('pilots/topology_20260910').glob('*.py'))+list(pathlib.Path('tests').glob('test_topology_*.py')); [ast.parse(p.read_text(),filename=str(p)) for p in paths]; print('Syntax files:',len(paths))"]),
        ("environment", [python, "-u", "-c", "import json,os,torch,torch_geometric,transformers,numba,pytest,safetensors; assert torch.cuda.is_available(), 'No allocated CUDA'; torch.set_num_threads(2); torch.set_num_interop_threads(2); print(json.dumps({'torch':torch.__version__,'pyg':torch_geometric.__version__,'transformers':transformers.__version__,'numba':numba.__version__,'gpu':torch.cuda.get_device_name(),'visible_gpus':torch.cuda.device_count(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'allocated_job':os.getenv('SLURM_JOB_ID')}),flush=True)"]),
        ("data_shapes", prefix[:-1] + ["pilots.topology_20260910.extract", "prepare", "--data-root", str(a.root / "data")]),
        ("core_integrity_tests", prefix[:-1] + ["pilots.topology_20260910.test_core_integrity"]),
        ("model_validation", prefix[:-1] + ["pilots.topology_20260910.validate", "--device", "cuda", "--out", str(out / "model_preflight.json")]),
        ("training_evaluation_tests", [python, "-u", "-m", "pytest", "-q", "tests/test_topology_training.py", "tests/test_topology_evaluation.py", "--basetemp", str(out / "pytest"), "-o", "cache_dir=" + str(out / "pytest_cache")]),
        ("diagnostic_capture", prefix[:-1] + ["pilots.topology_20260910.slurm.benchmark", "capture", "--data-root", str(a.root / "data"), "--cache", str(diagnostic)]),
        ("diagnostic_rewire", prefix[:-1] + ["pilots.topology_20260910.rewire", "--cache", str(diagnostic), "--workers", "2"]),
        ("diagnostic_validation", prefix[:-1] + ["pilots.topology_20260910.validate", "--cache", str(diagnostic), "--device", "cuda", "--require-rewire"]),
        ("diagnostic_training_timing", prefix[:-1] + ["pilots.topology_20260910.slurm.benchmark", "train", "--cache", str(diagnostic)]),
    ]
    if a.publisher_tests:
        commands.append(("publication_mock_tests", prefix[:-1] + ["pilots.topology_20260910.test_publish"]))
    inherited = []
    environment = os.environ.copy()
    staging = None
    if a.prior_job:
        prior_path = a.root / "preflight" / a.prior_job / "status.json"
        prior = json.loads(prior_path.read_text())
        expected = ["syntax", "environment", "data_shapes", "core_integrity_tests", "model_validation", "training_evaluation_tests"]
        if (prior.get("failed_stage") != "diagnostic_capture"
                or [entry["stage"] for entry in prior["stages"] if entry["exit_code"] == 0] != expected):
            raise RuntimeError("This continuation requires the exact six passed pre-capture gates")
        old_source = json.loads((Path(prior["release"]) / "source_manifest.json").read_text())["files"]
        protected = [name for name in old_source if name.startswith("polygraph/")]
        protected += ["pilots/topology_20260910/" + name for name in
                      ("protocol.py", "models.py", "data.py", "extract.py", "rewire.py", "validate.py", "train.py", "test_core_integrity.py")]
        protected += ["tests/test_topology_training.py"]
        if any(source["files"].get(name) != old_source[name] for name in protected):
            raise RuntimeError("A previously passed scientific implementation changed")
        check = json.loads((a.root / "checks" / a.supplemental_check_job / "status.json").read_text())
        if check.get("status") != "complete" or check.get("import_staging", {}).get("archive_sha256") != a.import_bundle_sha:
            raise RuntimeError("Node-local import/source supplement has not passed")
        checked_source = json.loads((Path(check["release"]) / "source_manifest.json").read_text())["files"]
        checked_files = ["pilots/topology_20260910/" + name for name in
                         ("publish.py", "test_publish.py", "slurm/imports.py", "slurm/test_imports.py")]
        if any(source["files"].get(name) != checked_source.get(name) for name in checked_files):
            raise RuntimeError("Import/publisher supplement code changed after its tests")
        statistical_job = json.loads((a.root / "manifests/evaluator_checks_job.json").read_text())["job_id"]
        statistical = json.loads((a.root / "checks" / statistical_job / "status.json").read_text())
        stat_source = json.loads((Path(statistical["release"]) / "source_manifest.json").read_text())["files"]
        if (statistical.get("status") != "complete" or any(source["files"].get(name) != stat_source.get(name)
                for name in ("pilots/topology_20260910/evaluate.py", "tests/test_topology_evaluation.py"))):
            raise RuntimeError("Current evaluator is not bound to the successful statistical checks")
        from imports import prepare
        environment, staging = prepare(a.root, os.environ["OMRI_JOB_CACHE"], a.import_bundle_sha)
        inherited = [dict(entry, inherited_from_job=a.prior_job,
                          prior_receipt_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest())
                     for entry in prior["stages"] if entry["exit_code"] == 0]
        commands = [(name, command) for name, command in commands if name not in expected]
        destination_check = """import json,os,pathlib,tempfile,shutil; from huggingface_hub import constants as c
paths={'hub':c.HF_HUB_CACHE,'xet':c.HF_XET_CACHE,'assets':c.HF_ASSETS_CACHE,'tmp':tempfile.gettempdir()}
for key,path in paths.items():
 p=pathlib.Path(path); p.mkdir(parents=True,exist_ok=True)
 expected={'hub':'HF_HUB_CACHE','xet':'HF_XET_CACHE','assets':'HF_ASSETS_CACHE','tmp':'TMPDIR'}[key]
 assert p==pathlib.Path(os.environ[expected])
 with tempfile.TemporaryFile(dir=p) as out: out.write(b'0'*4096);out.flush();os.fsync(out.fileno())
 assert shutil.disk_usage(p).free > 2*1024**3
print(json.dumps({'passed':True,'writable_cache_destinations':paths,'effective_auth_token_path':c.HF_TOKEN_PATH}),flush=True)
"""
        commands.insert(0, ("cache_destinations", [python, "-u", "-c", destination_check]))
    record = {"status": "running", "job_id": job, "release": str(a.release),
              "source_manifest_sha256": hashlib.sha256((a.release / "source_manifest.json").read_bytes()).hexdigest(),
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "stages": inherited,
              "prior_job": a.prior_job, "import_staging": staging,
              "production_capture": False, "diagnostic_cache": str(diagnostic)}
    receipt = out / "status.json"
    atomic(receipt, record)
    for name, command in commands:
        start = time.time()
        print(json.dumps({"stage": name, "event": "start", "utc": dt.datetime.now(dt.timezone.utc).isoformat()}), flush=True)
        code = subprocess.call(command, cwd=a.release, env=environment)
        result = {"stage": name, "exit_code": code, "elapsed_seconds": time.time() - start}
        record["stages"].append(result)
        if code:
            record.update(status="failed", failed_stage=name)
            atomic(receipt, record)
            raise SystemExit(code)
        atomic(receipt, record)
        print(json.dumps(dict(result, event="complete")), flush=True)
    record.update(status="completed", ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic(receipt, record)
    print(json.dumps({"status": "completed", "receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
