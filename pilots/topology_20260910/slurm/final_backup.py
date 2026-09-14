"""One serialized final publication, or a clearly partial backup after failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--release", type=Path, required=True)
a = p.parse_args()
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Allocated Slurm job required")
terminal = a.root / "status/terminal.json"
if not terminal.is_file():
    raise SystemExit("All scientific jobs must be terminal before final or partial backup")
command = [str(a.root / "env/bin/python"), "-u", "-m", "pilots.topology_20260910.publish",
           "--repo-id", "omrifahn/polygraph-experiments", "--run-root", str(a.root / "runs"),
           "--receipt-dir", str(a.root / "publication"), "--code-root", str(a.release),
           "--cache-metadata-root", str(a.root / "feature_cache"), "--once"]
completion = a.root / "results/evaluation_complete.json"
mode = "partial"
if completion.exists():
    # An existing inconsistent completion is an error, never grounds to demote
    # final results to a later partial snapshot. publish performs full hash checks.
    if json.loads(completion.read_text()).get("status") != "complete":
        raise SystemExit("Existing evaluation completion is invalid; refusing partial overwrite")
    command += ["--freeze", str(a.root / "freeze.json"), "--results-root", str(a.root / "results")]
    mode = "final"
print(json.dumps({"backup_mode": mode, "job_id": os.environ["SLURM_JOB_ID"]}), flush=True)
raise SystemExit(subprocess.call(command, cwd=a.release))
