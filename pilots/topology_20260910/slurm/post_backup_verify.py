"""Separate, validation-only restoration audit after the immutable phase2 DAG.

This file is a reviewed addendum, not a replacement of the submitted source release.
No model training, detector test scoring, upload, or dependency installation occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORKFLOW_SHA = "1bb59c6e0bffd7b170e8737a8954678fd9c725cff7816321ca8779d8b7d98914"
SOURCE_RELEASE = "81b4e83e8bc2a823"
BACKUP_JOB = "879968"
REPO = "omrifahn/polygraph-experiments"
PREFIX = "topology_20260910"
ARMS = ("full_graph", "full_rewired", "full_set", "full_endpoint", "raw_graph", "raw_set", "logit")
EXPECTED_RUNS = {f"{arm}/seed{seed}" for arm in ARMS for seed in (1, 2, 7, 17, 27)}


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def native(download, cache, audit, environment_receipt):
    """Fresh process: import scientific code exclusively from the downloaded bundle."""
    source = (download / PREFIX / "code/source").resolve()
    sys.path.insert(0, str(source))
    from pilots.topology_20260910 import publish, train, data, models
    from polygraph.training import train as original_train, models as original_models
    from polygraph.data import sidecars
    modules = (publish, train, data, models, original_train, original_models, sidecars)
    require(all(Path(module.__file__).resolve().is_relative_to(source) for module in modules),
            "A scientific module was imported outside downloaded source")
    environment = read(download / PREFIX / "code/environment.json")
    require(sys.version.split()[0] == environment["python"], "Python version differs from backup")
    observed = {name: importlib.metadata.version(name) for name in environment["packages"]}
    require(observed == environment["packages"], "Effective dependency versions differ from backup")
    lock = "".join(f"{name}=={version}\n" for name, version in sorted(environment["packages"].items()))
    require((download / PREFIX / "code/requirements.lock.txt").read_text() == lock,
            "Dependency lock and environment manifest disagree")
    atomic(environment_receipt, {
        "status": "passed", "python": environment["python"], "packages_checked": len(observed),
        "environment_sha256": checksum(download / PREFIX / "code/environment.json"),
        "source_root": str(source), "scientific_modules_from_download": [module.__name__ for module in modules],
        "scope": "Existing pinned Slurm dependencies verified against backup; no clean dependency installation claimed"})
    publish.smoke_restoration(download, "full_graph/seed1", cache, "cuda", audit)
    # The reused audit compares 48 validation scores and threshold decisions, and
    # repeats with a different batch partition. It never constructs test data.
    require(read(audit).get("status") == "passed" and read(audit).get("split") == "val",
            "Native restoration did not pass validation-only audit")
    publish.verify_local(download)


def orchestrate(root):
    started = time.monotonic()
    job = os.environ["SLURM_JOB_ID"]
    directory = root / "restoration" / job
    directory.mkdir(parents=True, exist_ok=False)
    receipt = directory / "verification.json"
    record = {"schema": 1, "status": "running", "stage": "final_completeness_guards",
              "job_id": job, "workflow_sha256": WORKFLOW_SHA,
              "addendum_sha256": checksum(__file__), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "test_rescored": False, "new_training": False, "upload_performed": False}
    atomic(receipt, record)
    try:
        workflow_path = root / "manifests/workflow.json"
        require(checksum(workflow_path) == WORKFLOW_SHA, "Submitted workflow changed")
        workflow = read(workflow_path)
        release = root / "releases" / SOURCE_RELEASE
        require(Path(workflow["code_root"]) == release, "Unexpected scientific release")
        execution = list((root / "manifests/execution").glob(f"final_backup_{BACKUP_JOB}*.json"))
        require(len(execution) == 1, "Exactly one authoritative final backup execution required")
        backup = read(execution[0])
        require(backup.get("status") == "completed" and backup.get("exit_code") == 0
                and backup.get("workflow_sha256") == WORKFLOW_SHA, "Final backup did not succeed")
        require(read(root / "status/terminal.json").get("scientific_status") == "complete",
                "Scientific pipeline is incomplete; a partial backup is not completion")
        freeze_hash = checksum(root / "freeze.json")
        freeze = read(root / "freeze.json")
        require(freeze.get("frozen") is True and freeze.get("matrix") == "full"
                and set(freeze.get("runs", {})) == EXPECTED_RUNS, "Full35 frozen matrix required")
        completion = read(root / "results/evaluation_complete.json")
        require(completion.get("status") == "complete" and completion.get("freeze_sha256") == freeze_hash,
                "Final scientific completion is missing or mismatched")
        publication_path = root / "publication/latest.json"
        publication_hash = checksum(publication_path)
        publication = read(publication_path)
        require(publication.get("backup_complete") is True and publication.get("freeze_sha256") == freeze_hash
                and set(publication.get("completed_runs", {})) == EXPECTED_RUNS,
                "Authoritative HF receipt is not the complete35 final backup")
        for name, expected in workflow["source_sha256"].items():
            require(checksum(release / name) == expected, f"Approved source changed: {name}")
        sys.path.insert(0, str(release))
        from pilots.topology_20260910 import publish
        record.update(stage="download_exact_final_revision", revision=publication["revision"],
                      artifact_revision=publication["artifact_revision"], snapshot_id=publication["snapshot_id"],
                      freeze_sha256=freeze_hash, publication_receipt_sha256=publication_hash)
        atomic(receipt, record)
        download = directory / "download"
        payload = publish.download_verified(REPO, publication["revision"], download)
        require(payload.get("snapshot_id") == publication["snapshot_id"]
                and payload.get("freeze_sha256") == freeze_hash
                and payload.get("test_results_status") == "complete_evaluation_available"
                and set(payload.get("runs", {})) == EXPECTED_RUNS
                and all(v.get("status") == "training_complete" for v in payload["runs"].values()),
                "Downloaded snapshot is not the complete final experiment")
        published_root = download / PREFIX
        downloaded_freeze = read(published_root / "freeze.json")
        publish.check_final_freeze(downloaded_freeze)
        require(checksum(published_root / "freeze.json") == freeze_hash, "Downloaded freeze changed")
        require(checksum(published_root / "results/evaluation_complete.json") ==
                checksum(root / "results/evaluation_complete.json"), "Downloaded completion differs")
        for name, expected in completion["artifacts"].items():
            require(checksum(publish.safe_file(published_root / "results", name)) == expected,
                    f"Downloaded final artifact mismatch: {name}")
        bundle = read(published_root / "code/bundle.json")
        expected_source = {name: sha for name, sha in workflow["source_sha256"].items()
                           if publish.source_allowed(name)}
        require(bundle["provenance"]["allowlisted_release_files"] == expected_source,
                "Downloaded source inventory differs from approved scientific release")
        for name, expected in expected_source.items():
            require(checksum(publish.safe_file(published_root / "code/source", name)) == expected,
                    f"Downloaded source mismatch: {name}")
        record.update(stage="fresh_process_native_validation_restore", downloaded_files=len(payload["files"]),
                      downloaded_source_files=len(expected_source), model="full_graph/seed1")
        atomic(receipt, record)
        audit, environment_receipt = directory / "native_validation.json", directory / "environment.json"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--native", "--download", str(download),
                   "--cache", str(root / "feature_cache"), "--audit", str(audit),
                   "--environment-receipt", str(environment_receipt)]
        subprocess.run(command, cwd=published_root / "code/source", check=True)
        require(checksum(publication_path) == publication_hash, "Authoritative publication changed during audit")
        record.update(status="passed", stage="complete", native_validation_sha256=checksum(audit),
                      environment_receipt_sha256=checksum(environment_receipt),
                      scope="Isolated server download, all artifact/source/dependency checks,48 validation rows from representative full_graph/seed1. No fresh dependency install or real network-cut recovery claimed.",
                      elapsed_seconds=time.monotonic()-started, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic(receipt, record)
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started,
                      ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic(receipt, record)
        print(json.dumps(record), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    require(bool(os.environ.get("SLURM_JOB_ID")), "Allocated Slurm job required")
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path)
    p.add_argument("--native", action="store_true")
    p.add_argument("--download", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--environment-receipt", type=Path)
    args = p.parse_args()
    if args.native:
        native(args.download, args.cache, args.audit, args.environment_receipt)
    else:
        require(args.root is not None, "Experiment root required")
        orchestrate(args.root)
