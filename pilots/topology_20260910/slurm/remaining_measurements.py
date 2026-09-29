"""Independent disposable graph timing and publisher mocks; no rewire/test access."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--import-sha", required=True)
    a = p.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    out = a.root / "checks" / job
    out.mkdir(parents=True, exist_ok=False)
    source_path = a.release / "source_manifest.json"
    source = json.loads(source_path.read_text())
    for name, sha in source["files"].items():
        if hashlib.sha256((a.release / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError("Immutable source mismatch: " + name)
    sys.path.insert(0, str(a.release / "pilots/topology_20260910/slurm"))
    from imports import prepare
    env, staging = prepare(a.root, os.environ["OMRI_JOB_CACHE"], a.import_sha)
    py = str(a.root / "env/bin/python")
    commands = [
        ("diagnostic_training_timing", [py, "-u", "-m", "pilots.topology_20260910.slurm.benchmark", "train", "--cache", str(a.cache), "--out", str(out / "training_timing.json")]),
        ("publication_mock_tests", [py, "-u", "-m", "pilots.topology_20260910.test_publish"]),
    ]
    record = {"status": "running", "job_id": job, "release": str(a.release),
              "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cache": str(a.cache), "test_opened": False, "model_saved": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "import_staging": staging, "stages": []}
    def save():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(out / "status.json")
    save()
    failures = []
    for name, command in commands:
        started = time.monotonic()
        print(json.dumps({"stage": name, "event": "start"}), flush=True)
        code = subprocess.call(command, cwd=a.release, env=env)
        item = {"stage": name, "exit_code": code, "seconds": time.monotonic() - started}
        record["stages"].append(item)
        if code:
            failures.append(name)
        save()
        print(json.dumps(item), flush=True)
    record.update(status="failed" if failures else "complete", failed_stages=failures,
                  ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
