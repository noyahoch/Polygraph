"""Run one immutable, reviewed command under Slurm and record its exit status."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", required=True)
    a = p.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise SystemExit("Allocated Slurm job required")
    os.umask(0o077)
    plan_path = a.root / "manifests" / "workflow.json"
    payload = plan_path.read_bytes()
    workflow = json.loads(payload)
    stage = workflow["stages"][a.stage]
    code_root = Path(workflow["code_root"]).resolve()
    release_root = (a.root / "releases").resolve()
    if code_root.parent != release_root or not (code_root / "source_manifest.json").is_file():
        raise SystemExit("Workflow code must name one immutable experiment release")
    expected = os.environ.get("OMRI_WORKFLOW_SHA256")
    if not expected or hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit("Submitted workflow checksum no longer matches")
    for name, sha in workflow["source_sha256"].items():
        path = code_root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"Scientific source changed after review: {name}")
    for required in stage.get("requires", []):
        path = a.root / required
        if not path.is_file():
            raise SystemExit(f"Missing required successful-stage receipt: {required}")
        value = json.loads(path.read_text())
        if required == "feature_cache/rewire/admission.json":
            # Explicit scientific admission is distinct from mixing-quality PASS.
            # The scientific reader subsequently verifies its complete bindings.
            if (value.get("admitted") is not True or value.get("structural_integrity_passed") is not True
                    or value.get("decision", {}).get("sha256") != workflow.get("rewiring_decision", {}).get("sha256")):
                raise SystemExit("Required explicit rewiring admission is invalid")
            continue
        if (value.get("status") not in {"complete", "completed", "passed", "PASS", "ready"}
                and value.get("passed") is not True and value.get("complete") is not True
                and value.get("frozen") is not True):
            raise SystemExit(f"Required receipt is not successful: {required}")
    array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    command = stage["commands"][int(array_index)] if array_index is not None else stage["command"]
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
        raise SystemExit("Command must be an argv list")
    suffix = f"_{array_index}" if array_index is not None else ""
    path = a.root / "manifests" / "execution" / f"{a.stage}_{job}{suffix}.json"
    started = time.time()
    receipt = {"stage": a.stage, "job_id": job, "array_index": array_index,
               "command": command, "workflow_sha256": expected,
               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running"}
    atomic(path, receipt)
    print(json.dumps(receipt), flush=True)
    environment = os.environ.copy()
    if workflow.get("import_bundle"):
        from imports import prepare
        environment, staged = prepare(a.root, os.environ["OMRI_JOB_CACHE"], workflow["import_bundle"]["sha256"])
        receipt["import_staging"] = staged
        atomic(path, receipt)
        print(json.dumps({"import_staging": staged}), flush=True)
    child = subprocess.Popen(command, cwd=code_root, env=environment)

    def forward(signum, _frame):
        child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = child.wait()
    receipt.update(status="completed" if code == 0 else "failed", exit_code=code,
                   elapsed_seconds=time.time() - started,
                   ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
