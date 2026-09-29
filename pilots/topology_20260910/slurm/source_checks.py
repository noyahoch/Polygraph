"""Run the separately reviewed evaluator recovery tests on allocated Slurm CPUs."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--release", type=Path, required=True)
p.add_argument("--mode", choices=["evaluation", "publisher_imports"], default="evaluation")
p.add_argument("--import-bundle-sha")
a = p.parse_args()
job = os.environ.get("SLURM_JOB_ID")
if not job:
    raise SystemExit("Allocated Slurm job required")
source = json.loads((a.release / "source_manifest.json").read_text())
for name, expected in source["files"].items():
    if hashlib.sha256((a.release / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Immutable source mismatch: {name}")
out = a.root / "checks" / job
out.mkdir(parents=True, exist_ok=False)
record = {"status": "running", "job_id": job, "release": str(a.release),
          "source_manifest_sha256": hashlib.sha256((a.release / "source_manifest.json").read_bytes()).hexdigest(),
          "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "scope": a.mode}
receipt = out / "status.json"
receipt.write_text(json.dumps(record, indent=2) + "\n")
command = [str(a.root / "env/bin/python"), "-u", "-m", "pytest", "-q",
           "tests/test_topology_evaluation.py",
           "tests/test_topology_training.py::test_threshold_conservative_ties_and_auroc",
           "--basetemp", str(out / "pytest"), "-o", "cache_dir=" + str(out / "pytest_cache")]
started = time.monotonic()
environment = os.environ.copy()
if a.mode == "publisher_imports":
    from imports import prepare
    environment, staged = prepare(a.root, os.environ["OMRI_JOB_CACHE"], a.import_bundle_sha)
    record["import_staging"] = staged
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    prefix = [str(a.root / "env/bin/python"), "-u"]
    proof = """import hashlib,json,pathlib,sys,time; t=time.monotonic(); import torch,torch_geometric,transformers,numba
root=pathlib.Path(sys.argv[1]); staged=pathlib.Path(sys.argv[2]); base=root.parents[1]
assert transformers.__version__=='5.16.1' and torch_geometric.__version__=='2.6.1'
assert torch.__version__=='2.14.0+cu126' and numba.__version__=='0.67.0'
assert pathlib.Path(torch.__file__).is_relative_to(base/'env')
assert pathlib.Path(numba.__file__).is_relative_to(root/'env')
for module,original in [(transformers,base/'env/lib/python3.12/site-packages/transformers/__init__.py'),(torch_geometric,base/'env-overnight-graph/lib/python3.12/site-packages/torch_geometric/__init__.py')]:
 assert pathlib.Path(module.__file__).is_relative_to(staged)
 assert hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
print(json.dumps({'passed':True,'imports_seconds':time.monotonic()-t,'transformers':transformers.__file__,'pyg':torch_geometric.__file__,'torch':torch.__file__,'numba':numba.__file__}),flush=True)
"""
    commands = [prefix + ["-c", proof, str(a.root), staged["local_package_root"]],
                prefix + ["-m", "unittest", "pilots.topology_20260910.slurm.test_imports"],
                prefix + ["-m", "unittest",
                          "pilots.topology_20260910.test_publish.PublisherTests.test_terminal_wait_preserves_interval_without_real_sleep",
                          "pilots.topology_20260910.test_publish.PublisherTests.test_terminal_wakes_final_upload_and_retains_single_publisher_lock"]]
else:
    commands = [command]
code = 0
for command in commands:
    print(json.dumps({"command": command[:5], "event": "start"}), flush=True)
    code = subprocess.call(command, cwd=a.release, env=environment)
    if code:
        break
record.update(status="complete" if code == 0 else "failed", exit_code=code,
              elapsed_seconds=time.monotonic() - started,
              ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(code)
