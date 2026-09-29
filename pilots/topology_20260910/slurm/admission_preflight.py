"""Resume the existing diagnostic cache after the explicit reviewed decision."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--import-sha", required=True)
    a = p.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    out = a.root / "preflight" / job
    out.mkdir(parents=True, exist_ok=False)
    cache = a.root / "preflight/877598/diagnostic_cache"
    source = read(a.release / "source_manifest.json")
    for name, expected in source["files"].items():
        if sha(a.release / name) != expected:
            raise RuntimeError("Immutable source mismatch: " + name)
    prior = read(a.root / "preflight/877598/status.json")
    if prior.get("failed_stage") != "diagnostic_rewire":
        raise RuntimeError("The specified diagnostic did not stop at the known mixing gate")
    old = read(Path(prior["release"]) / "source_manifest.json")["files"]
    protected = [name for name in old if name.startswith("polygraph/")]
    protected += ["pilots/topology_20260910/" + name + ".py" for name in
                  ("protocol", "models", "extract", "rewire")]
    if any(old[name] != source["files"].get(name) for name in protected):
        raise RuntimeError("The validated numerical model/extractor/rewiring implementation changed")
    measurement_path = a.root / "checks/877703/status.json"
    measurement = read(measurement_path)
    if (measurement.get("status") != "complete" or
            {x["stage"] for x in measurement["stages"] if x["exit_code"] == 0} !=
            {"diagnostic_training_timing", "publication_mock_tests"}):
        raise RuntimeError("Independent timing and publication mock evidence is incomplete")
    original_hashes = {name: sha(cache / name) for name in ("manifest.json", "rewire/manifest.json")}
    from imports import prepare
    env, staging = prepare(a.root, os.environ["OMRI_JOB_CACHE"], a.import_sha)
    py = str(a.root / "env/bin/python")
    prefix = [py, "-u", "-m"]
    cache_check = """import json,pathlib,sys
from pilots.topology_20260910.protocol import atomic_json,file_sha256,require_slurm
from pilots.topology_20260910.validate import cache_checks
require_slurm(); cache=pathlib.Path(sys.argv[1]); prior=pathlib.Path(sys.argv[2])
evidence=json.loads(prior.read_text()); assert evidence['passed'] is True
result={'passed':True,'model_checks':evidence['model_checks'],'model_checks_inherited_from_job':'877551','model_evidence_sha256':file_sha256(prior),'cache_checks':cache_checks(cache,True)}
atomic_json(cache/'validation.json',result)
print(json.dumps(result),flush=True)
"""
    commands = [
        ("admission_boundary_tests", prefix + ["pytest", "-q", "tests/test_topology_rewiring_admission.py",
                                              "tests/test_topology_training.py", "tests/test_topology_evaluation.py",
                                              "--basetemp", str(out / "pytest"), "-o", "cache_dir=" + str(out / "pytest_cache")]),
        ("core_integrity_updated_readers", prefix + ["pilots.topology_20260910.test_core_integrity"]),
        ("explicit_rewiring_admission", prefix + ["pilots.topology_20260910.rewiring_decision", "--cache", str(cache),
                                               "--decision", str(a.release / "docs/experiments/september10/REWIRING_DECISION.md")]),
        ("diagnostic_cache_validation", [py, "-u", "-c", cache_check, str(cache),
                                         str(a.root / "preflight/877551/model_preflight.json")]),
        ("publication_updated_bindings", prefix + ["pilots.topology_20260910.test_publish"]),
    ]
    record = {"status": "running", "job_id": job, "release": str(a.release),
              "source_manifest_sha256": sha(a.release / "source_manifest.json"), "diagnostic_cache": str(cache),
              "prior_job": "877598", "prior_receipt_sha256": sha(a.root / "preflight/877598/status.json"),
              "measurement_job": "877703", "measurement_receipt_sha256": sha(measurement_path),
              "original_cache_hashes": original_hashes, "import_staging": staging,
              "model_checks_repeated": False, "recapture": False, "rewire_repeated": False,
              "production_capture": False, "test_scoring": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "stages": []}
    def save():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(out / "status.json")
    save()
    for name, command in commands:
        started = time.monotonic()
        print(json.dumps({"stage": name, "event": "start"}), flush=True)
        code = subprocess.call(command, cwd=a.release, env=env)
        record["stages"].append({"stage": name, "exit_code": code, "seconds": time.monotonic() - started})
        if code:
            record.update(status="failed", failed_stage=name)
            save()
            raise SystemExit(code)
        save()
    if any(sha(cache / name) != expected for name, expected in original_hashes.items()):
        raise RuntimeError("Admission changed the original cache or failed-quality manifest")
    record.update(status="complete", ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
