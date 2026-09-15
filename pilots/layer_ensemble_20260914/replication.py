"""Immutable, separately authorized seed-17/27 replication; no job submission."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from .combine import RECIPE, atomic_json
from .protocol import (ARMS, SCOPE as PILOT_SCOPE, TRAINING, file_sha256,
                       implementation_identity, make_roles, read, require_slurm)

SCOPE = "layer_ensemble_replication_seed17_27"
SEEDS = (17, 27)
DEADLINE_NAMES = ("base_complete_before", "predictions_complete_before",
                  "evaluation_complete_before")
ANALYSIS = {
    "primary_seeds": [17, 27], "descriptive_seeds": [7, 17, 27],
    "contrast": "mean of seed-wise AUROC(stack) minus AUROC(last_only)",
    "seed_standard_deviation": "sample; ddof=1",
    "bootstrap_draws": 2000, "bootstrap_seed": 20260914,
    "bootstrap_group": "image_id", "bootstrap_shared_across_seeds": True,
    "percentile_method": "linear", "cross_seed_prediction_ensemble": False,
    "original_test_access": False, "seed7_status": "complete_late_diagnostic",
}
SOURCE_FILES = (
    "role_map.json", "execution.json", "base_freeze.json", "heads_freeze.json",
    "predictions/meta.json", "predictions/dev_eval.json",
    "evaluation/report.json", "evaluation/REPORT.md", "evaluation/complete.json",
    "evaluation/late_diagnostic.json", "evaluation/scores.npz",
    "evaluation/bootstrap.json", "evaluation/bootstrap_source_counts.npz",
)


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Every replication cutoff requires an explicit timezone")
    return parsed.timestamp()


def normalized_cutoffs(base, prediction, evaluation):
    values = (base, prediction, evaluation)
    return {name: dt.datetime.fromtimestamp(timestamp(value), ZoneInfo("Asia/Jerusalem")).isoformat()
            for name, value in zip(DEADLINE_NAMES, values)}


def validate_contract(value):
    if value.get("schema_version") != 1 or value.get("scope_id") != SCOPE:
        raise RuntimeError("Unknown replication protocol")
    if not isinstance(value.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value["run_id"]):
        raise RuntimeError("An explicit, safe replication run ID is required")
    if value.get("seeds") != list(SEEDS) or value.get("matrix") != [[arm, seed] for seed in SEEDS for arm in ARMS]:
        raise RuntimeError("Only the fixed eight seed-17/27 fits are authorized")
    if value.get("training") != TRAINING or value.get("head_recipe") != RECIPE or value.get("head_random_state") != 7:
        raise RuntimeError("Replication must retain the frozen detector and head recipes")
    if value.get("analysis") != ANALYSIS or value.get("original_test_access") is not False:
        raise RuntimeError("Replication analysis or original-test prohibition changed")
    created = value.get("created_unix")
    if not isinstance(created, (int, float)) or not math.isfinite(created) or created <= 0:
        raise RuntimeError("Invalid replication creation timestamp")
    deadlines = value.get("deadlines", {})
    if set(deadlines) != set(DEADLINE_NAMES):
        raise RuntimeError("All three explicit replication cutoffs are required")
    cutoffs = [timestamp(deadlines[name]) for name in DEADLINE_NAMES]
    if not created < cutoffs[0] < cutoffs[1] < cutoffs[2]:
        raise RuntimeError("Cutoffs must be strictly ordered and future at protocol creation")
    if set(value.get("source_files", {})) != set(SOURCE_FILES):
        raise RuntimeError("Incomplete seed-7 provenance inventory")


def check_deadline(manifest, name):
    if time.time() >= timestamp(manifest["deadlines"][name]):
        raise TimeoutError("Replication cutoff reached: " + name + "; preserve incomplete status")


def _source_bindings(source):
    marker = read(source / "evaluation/late_diagnostic.json")
    report = read(source / "evaluation/report.json")
    complete = read(source / "evaluation/complete.json")
    if (marker.get("status") != "complete_late_diagnostic" or marker.get("complete") is not True
            or marker.get("scope_id") != PILOT_SCOPE
            or marker.get("reused_on_time_meta_export") is not True
            or not marker.get("original_protocol_status", "").startswith("incomplete")
            or not marker.get("warning")):
        raise RuntimeError("The pilot must remain an explicitly late, formally incomplete diagnostic")
    if (report.get("seed") != 7 or report.get("scope_id") != PILOT_SCOPE
            or report.get("complete") is not True or report.get("test_evaluated") is not False
            or report.get("records") != 7200 or report.get("base_epochs") != 20
            or complete.get("complete") is not True or complete.get("inputs") != report.get("inputs")):
        raise RuntimeError("Seed-7 report/completion identity is invalid")
    for key, name in (("roles_sha256", "role_map.json"), ("execution_sha256", "execution.json"),
                      ("base_freeze_sha256", "base_freeze.json"), ("heads_freeze_sha256", "heads_freeze.json"),
                      ("dev_prediction_sidecar_sha256", "predictions/dev_eval.json")):
        if report["inputs"].get(key) != file_sha256(source / name):
            raise RuntimeError("Seed-7 evaluation uses different frozen inputs: " + key)
    if report["inputs"].get("cache_manifest_sha256") != file_sha256(source / "feature_cache/manifest.json"):
        raise RuntimeError("Seed-7 evaluation uses a different feature cache")
    required = {"report.json", "REPORT.md", "scores.npz", "bootstrap.json", "bootstrap_source_counts.npz"}
    if set(complete.get("files", {})) != required:
        raise RuntimeError("Seed-7 evaluation inventory is incomplete")
    for name, wanted in complete["files"].items():
        if file_sha256(source / "evaluation" / name) != wanted:
            raise RuntimeError("Seed-7 evaluation changed: " + name)
    return {name: file_sha256(source / name) for name in SOURCE_FILES}


def validate_manifest(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "replication.json")
    validate_contract(value)
    source = Path(value["source_root"]).resolve()
    if value.get("root") != str(root) or root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise RuntimeError("Replication and historical source roots must be separate")
    if value.get("implementation") != implementation_identity():
        raise RuntimeError("Replication source changed after protocol freeze")
    for name, wanted in value["source_files"].items():
        if file_sha256(source / name) != wanted:
            raise RuntimeError("Frozen pilot provenance changed: " + name)
    if value["roles_sha256"] != value["source_files"]["role_map.json"] or file_sha256(root / "role_map.json") != value["roles_sha256"]:
        raise RuntimeError("Replication must reuse the exact original role-map bytes")
    cache = Path(value["cache_path"]).resolve()
    if cache != (source / "feature_cache").resolve():
        raise RuntimeError("Replication cache path changed")
    if file_sha256(cache / "manifest.json") != value["cache_manifest_sha256"]:
        raise RuntimeError("Replication cache manifest changed")
    manifest = read(cache / "manifest.json")
    for name in ("protocol_sha256", "cohort_sha256"):
        if value["cache_" + name] != manifest[name]:
            raise RuntimeError("Replication cache identity changed: " + name)
    return value


def seed_execution(root, seed, manifest):
    if type(seed) is not int or seed not in SEEDS:
        raise RuntimeError("Only training seeds 17 and 27 belong to this replication")
    root = Path(root).resolve()
    return {
        "schema_version": 2, "scope_id": SCOPE, "seed": seed,
        "matrix": [[arm, seed] for arm in ARMS], "training": TRAINING,
        "deadlines": manifest["deadlines"], "roles_sha256": manifest["roles_sha256"],
        "cache_manifest_sha256": manifest["cache_manifest_sha256"],
        "cache_protocol_sha256": manifest["cache_protocol_sha256"],
        "cache_cohort_sha256": manifest["cache_cohort_sha256"],
        "training_role": "base_train", "selection_role": "checkpoint",
        "head_training_role": "meta", "evaluation_role": "dev_eval",
        "heads": {"stack": {"arms": list(ARMS)}, "last_only": {"arms": ["block11"]}},
        "head_recipe": RECIPE, "head_random_state": 7, "num_workers": 0,
        "precision": "float32", "cublas_workspace_config": ":4096:8",
        "test_evaluated": False, "replication_root": str(root),
        "replication_manifest_sha256": file_sha256(root / "replication.json"),
        "future_seeds": "no automatic expansion",
    }


def validate_seed_execution(execution_path, roles_path):
    execution_path, roles_path = Path(execution_path).resolve(), Path(roles_path).resolve()
    value = read(execution_path)
    root = Path(value["replication_root"]).resolve()
    manifest = validate_manifest(root)
    expected_root = root / f"seed{value['seed']}"
    if (execution_path != expected_root / "execution.json" or roles_path != expected_root / "role_map.json"
            or expected_root.resolve() != expected_root):
        raise RuntimeError("Each replication seed requires its own canonical artifact namespace")
    if file_sha256(roles_path) != manifest["roles_sha256"] or value != seed_execution(root, value["seed"], manifest):
        raise RuntimeError("Seed execution identity, roles or fixed recipe changed")
    if (expected_root / "feature_cache").resolve() != Path(manifest["cache_path"]):
        raise RuntimeError("Per-seed cache is not the immutable shared cache")
    return value


def _frozen_json(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Refusing to replace a frozen replication artifact: " + str(path))
    else:
        atomic_json(path, value)


def _copy_roles(source, target):
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError("Refusing to replace an existing role map")
        return
    temporary = target.with_name(target.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def prepare(source_root, root, run_id, deadlines):
    require_slurm()
    source, root = Path(source_root).resolve(), Path(root).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise RuntimeError("Never prepare a replication inside the historical experiment")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "replication.json").exists():
            manifest = validate_manifest(root)
            if manifest["source_root"] != str(source) or manifest["run_id"] != run_id or manifest["deadlines"] != deadlines:
                raise RuntimeError("Existing replication cannot change identity or cutoffs")
        else:
            if any(path.name != ".prepare.lock" for path in root.iterdir()):
                raise RuntimeError("New replication requires an empty, separate root")
            cache = (source / "feature_cache").resolve()
            cache_manifest = read(cache / "manifest.json")
            if cache_manifest.get("complete") is not True or cache_manifest.get("diagnostic_only") is not False or cache_manifest.get("records") != 28800:
                raise RuntimeError("A complete immutable development-only cache is required")
            roles = read(source / "role_map.json")
            if roles != make_roles(read(cache / "cohort.json")):
                raise RuntimeError("The original source-image role map does not match the cache")
            manifest = {
                "schema_version": 1, "scope_id": SCOPE, "run_id": run_id,
                "root": str(root), "source_root": str(source), "cache_path": str(cache),
                "created_unix": time.time(), "deadlines": deadlines,
                "seeds": list(SEEDS), "matrix": [[arm, seed] for seed in SEEDS for arm in ARMS],
                "training": TRAINING, "head_recipe": RECIPE, "head_random_state": 7,
                "analysis": ANALYSIS, "original_test_access": False,
                "source_files": _source_bindings(source),
                "roles_sha256": file_sha256(source / "role_map.json"),
                "cache_manifest_sha256": file_sha256(cache / "manifest.json"),
                "cache_protocol_sha256": cache_manifest["protocol_sha256"],
                "cache_cohort_sha256": cache_manifest["cohort_sha256"],
                "implementation": implementation_identity(),
            }
            validate_contract(manifest)
            _copy_roles(source / "role_map.json", root / "role_map.json")
            _frozen_json(root / "replication.json", manifest)
        check_deadline(manifest, "base_complete_before")
        for seed in SEEDS:
            directory = root / f"seed{seed}"
            directory.mkdir(exist_ok=True)
            if directory.resolve() != directory:
                raise RuntimeError("Seed namespace may not be redirected by a symlink")
            _copy_roles(root / "role_map.json", directory / "role_map.json")
            link = directory / "feature_cache"
            if not link.exists() and not link.is_symlink():
                link.symlink_to(manifest["cache_path"], target_is_directory=True)
            if link.resolve() != Path(manifest["cache_path"]):
                raise RuntimeError("Existing per-seed cache path differs")
            _frozen_json(directory / "execution.json", seed_execution(root, seed, manifest))
            validate_seed_execution(directory / "execution.json", directory / "role_map.json")
        return manifest


def _joint_bindings(root):
    from .combine import validate_heads
    manifest = validate_manifest(root)
    result = {}
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        execution = validate_seed_execution(directory / "execution.json", directory / "role_map.json")
        if execution["seed"] != seed:
            raise RuntimeError("Joint freeze seed identity mismatch")
        validate_heads(directory)
        result[str(seed)] = {name: file_sha256(directory / name) for name in
                             ("execution.json", "base_freeze.json", "heads_freeze.json",
                              "heads/stack.json", "heads/last_only.json")}
    return manifest, result


def freeze_heads(root):
    require_slurm()
    root = Path(root).resolve()
    with (root / ".joint-heads.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "joint_heads_freeze.json"
        if path.exists():
            return validate_joint_heads_freeze(root)
        manifest, bindings = _joint_bindings(root)
        check_deadline(manifest, "predictions_complete_before")
        if any((root / f"seed{s}" / name).exists() for s in SEEDS for name in
               ("predictions/dev_eval.npz", "predictions/dev_eval.json", "evaluation/report.json")):
            raise RuntimeError("Both seed pipelines must freeze before any new dev-evaluation output")
        frozen = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                  "seeds": list(SEEDS), "replication_manifest_sha256": file_sha256(root / "replication.json"),
                  "files": bindings, "frozen_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"],
                  "original_test_access": False}
        _frozen_json(path, frozen)
        return validate_joint_heads_freeze(root)


def validate_joint_heads_freeze(root):
    require_slurm()
    root = Path(root).resolve()
    manifest, bindings = _joint_bindings(root)
    frozen = read(root / "joint_heads_freeze.json")
    expected = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                "seeds": list(SEEDS), "replication_manifest_sha256": file_sha256(root / "replication.json"),
                "files": bindings, "original_test_access": False}
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Both replications' complete head freezes are required before dev_eval")
    when = frozen.get("frozen_unix", 0)
    if not manifest["created_unix"] < when < timestamp(manifest["deadlines"]["predictions_complete_before"]):
        raise RuntimeError("Joint head freeze missed its prospective cutoff")
    for seed in SEEDS:
        head_freeze = read(root / f"seed{seed}" / "heads_freeze.json")
        if not manifest["created_unix"] < timestamp(head_freeze["frozen_utc"]) <= when:
            raise RuntimeError("Each seed's heads must precede the joint freeze")
    return frozen


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("prepare")
    init.add_argument("--source-root", type=Path, required=True)
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    for name in ("base-cutoff", "prediction-cutoff", "evaluation-cutoff"):
        init.add_argument("--" + name, required=True)
    frozen = commands.add_parser("freeze-heads")
    frozen.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.root, args.run_id,
                         normalized_cutoffs(args.base_cutoff, args.prediction_cutoff, args.evaluation_cutoff))
    else:
        result = freeze_heads(args.root)
    print(json.dumps({"scope_id": SCOPE, "command": args.command, "root": str(args.root),
                      "seeds": result["seeds"]}), flush=True)


if __name__ == "__main__":
    main()
