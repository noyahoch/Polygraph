"""Bounded seed-17/27 orchestration. Planning is inert; submission needs approval."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .runtime import atomic_json, file_sha256, require_slurm

SCOPE = "layer_ensemble_replication_seed17_27"
MODULE = "pilots.layer_ensemble_20260914."
SELF = "pilots/layer_ensemble_20260914/slurm/replicate.py"
SEEDS = (17, 27)
ARMS = ("block2", "block5", "block8", "block11")
MATRIX = [[arm, seed] for seed in SEEDS for arm in ARMS]
ORDER = ("guardian", "prepare", "fits", "meta_heads", "joint_heads",
         "dev_eval", "evaluate", "aggregate")
DEADLINE_KEYS = {"base": "base_complete_before",
                 "predictions": "predictions_complete_before",
                 "evaluation": "evaluation_complete_before"}
EVALUATION_FILES = {"report.json", "REPORT.md", "scores.npz", "bootstrap.json",
                    "bootstrap_source_counts.npz"}
AGGREGATE_FILES = EVALUATION_FILES | {"bootstrap_draws.npz"}
HISTORY_FILES = {"role_map.json", "execution.json", "base_freeze.json", "heads_freeze.json",
                 "predictions/meta.json", "predictions/dev_eval.json", "evaluation/complete.json",
                 "evaluation/late_diagnostic.json", *("evaluation/" + name for name in EVALUATION_FILES)}
REQUIRED_SOURCE = {
    SELF, "pilots/layer_ensemble_20260914/slurm/runtime.py",
    "pilots/topology_20260910/slurm/imports.py",
    *("pilots/layer_ensemble_20260914/" + name + ".py" for name in
      ("replication", "protocol", "smoke", "train", "predict", "combine", "evaluate",
       "aggregate_replications")),
}
FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
          "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
          "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"}
HEX = re.compile(r"[0-9a-f]{64}")
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")


class PlanError(ValueError):
    pass


class AmbiguousSubmission(RuntimeError):
    """An unresolved intent is never permission to call sbatch again."""


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def timestamp(value):
    if not isinstance(value, str):
        raise PlanError("A timezone-qualified ISO timestamp is required")
    try:
        result = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise PlanError("Invalid ISO timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise PlanError("Naive timestamps are forbidden")
    return result


def utc_now(now=None):
    result = dt.datetime.now(dt.timezone.utc) if now is None else now
    if isinstance(result, str):
        result = timestamp(result)
    if not isinstance(result, dt.datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise PlanError("A timezone-aware clock is required")
    return result.astimezone(dt.timezone.utc)


def integer(value, label, minimum=1, maximum=4320):
    if type(value) is not int or not minimum <= value <= maximum:
        raise PlanError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def absolute(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in "\n\r\0"):
        raise PlanError("Invalid " + label)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value or value == "/":
        raise PlanError(label + " must be a canonical absolute path")
    return path


def _separate(left, right):
    return left != right and left not in right.parents and right not in left.parents


def _config(config):
    if not isinstance(config, dict):
        raise PlanError("Configuration must be a JSON object")
    required = {"run_id", "root", "source_root", "control_root", "code_root",
                "python", "source_sha256", "environment", "deadlines", "resources",
                "caps", "max_concurrent_gpus", "external_reserved_gpus",
                "gpu_budget_minutes", "external_gpu_minutes", "submission_grace_minutes",
                "guardian_grace_minutes", "readiness_max_seconds"}
    if set(config) != required:
        raise PlanError("Configuration fields differ: " + str(sorted(set(config) ^ required)))
    value = copy.deepcopy(config)
    if not isinstance(value["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value["run_id"]):
        raise PlanError("An explicit safe, new run_id is required")
    paths = {name: absolute(value[name], name) for name in
             ("root", "source_root", "control_root", "code_root", "python")}
    for a, b in (("root", "source_root"), ("control_root", "source_root"), ("root", "control_root")):
        if not _separate(paths[a], paths[b]):
            raise PlanError(a + " and " + b + " must be disjoint namespaces")
    if (paths["code_root"].parent != paths["control_root"] / "releases"
            or not SAFE.fullmatch(paths["code_root"].name)):
        raise PlanError("code_root must name one immutable control_root/releases/RELEASE")
    if paths["python"] != paths["source_root"] / "env/bin/python":
        raise PlanError("Use the original experiment's pinned env/bin/python")
    inventory = value["source_sha256"]
    if not isinstance(inventory, dict) or not REQUIRED_SOURCE <= set(inventory):
        raise PlanError("Incomplete immutable source inventory")
    for name, sha in inventory.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != name
                or relative.suffix not in {".py", ".md", ".json", ".toml", ".txt", ".sbatch", ".sh"}
                or not isinstance(sha, str) or not HEX.fullmatch(sha)):
            raise PlanError("Unsafe source inventory entry")
    env = value["environment"]
    if (not isinstance(env, dict) or set(env) != {"import_bundle_sha256", "dependency_manifest_sha256"}
            or any(not isinstance(sha, str) or not HEX.fullmatch(sha) for sha in env.values())):
        raise PlanError("Both existing runtime manifests must be explicitly hash-pinned")
    deadlines = value["deadlines"]
    if not isinstance(deadlines, dict) or set(deadlines) != set(DEADLINE_KEYS):
        raise PlanError("Explicit base, predictions and evaluation cutoffs are required")
    for name in DEADLINE_KEYS:
        deadlines[name] = timestamp(deadlines[name]).astimezone(ZoneInfo("Asia/Jerusalem")).isoformat()
    if not timestamp(deadlines["base"]) < timestamp(deadlines["predictions"]) < timestamp(deadlines["evaluation"]):
        raise PlanError("Cutoffs must be strictly ordered")
    resources = value["resources"]
    expected = {"account", "gpu_partition", "cpu_partition", "gpu_constraint",
                "gpu_cpus", "cpu_cpus", "gpu_memory_mb", "cpu_memory_mb"}
    if not isinstance(resources, dict) or set(resources) != expected:
        raise PlanError("All resource classes must be explicitly declared")
    for name in ("account", "gpu_partition", "cpu_partition", "gpu_constraint"):
        if not isinstance(resources[name], str) or not SAFE.fullmatch(resources[name]):
            raise PlanError("Invalid explicit resource class: " + name)
    for name in ("gpu_cpus", "cpu_cpus"):
        integer(resources[name], name, maximum=6)
    for name in ("gpu_memory_mb", "cpu_memory_mb"):
        integer(resources[name], name, maximum=32000)
    if not isinstance(value["caps"], dict) or set(value["caps"]) != set(ORDER):
        raise PlanError("Exactly eight stage allocation caps are required")
    for name, minutes in value["caps"].items():
        integer(minutes, name + " minutes")
    integer(value["readiness_max_seconds"], "readiness_max_seconds", minimum=60, maximum=1500)
    if 2 * value["readiness_max_seconds"] >= value["caps"]["prepare"] * 60:
        raise PlanError("Prepare must reserve both sequential readiness checks plus source validation")
    integer(value["max_concurrent_gpus"], "max_concurrent_gpus", maximum=8)
    integer(value["external_reserved_gpus"], "external_reserved_gpus", minimum=0, maximum=7)
    if value["max_concurrent_gpus"] + value["external_reserved_gpus"] > 8:
        raise PlanError("Workflow plus external GPU reservations exceed the eight-GPU ceiling")
    integer(value["gpu_budget_minutes"], "gpu_budget_minutes", maximum=8 * 4320 * 3)
    integer(value["external_gpu_minutes"], "external_gpu_minutes", minimum=0, maximum=8 * 4320 * 3)
    integer(value["submission_grace_minutes"], "submission_grace_minutes", maximum=10)
    integer(value["guardian_grace_minutes"], "guardian_grace_minutes", maximum=30)
    return value


def _stages(config):
    root = PurePosixPath(config["root"])
    python = config["python"]
    cap = config["max_concurrent_gpus"]
    resources = config["resources"]

    def command(module, *args):
        return [python, "-B", "-u", "-m", MODULE + module, *map(str, args)]

    def stage(tasks, gpu, cutoff, dependency=None, dependency_type="afterok", throttle=1):
        return {"tasks": tasks, "gpu": gpu,
                "cpus": resources["gpu_cpus"] if gpu else resources["cpu_cpus"],
                "memory_mb": resources["gpu_memory_mb"] if gpu else resources["cpu_memory_mb"],
                "cutoff": cutoff, "dependency": dependency, "dependency_type": dependency_type,
                "throttle": throttle}

    prepare = command("replication", "prepare", "--source-root", config["source_root"],
                      "--root", root, "--run-id", config["run_id"],
                      "--base-cutoff", config["deadlines"]["base"],
                      "--prediction-cutoff", config["deadlines"]["predictions"],
                      "--evaluation-cutoff", config["deadlines"]["evaluation"])
    stages = {
        "guardian": stage([{"key": "guardian", "commands": []}], False, "evaluation"),
        "prepare": stage([{"key": "prepare", "commands": [prepare]}], True, "base",
                         "guardian", "after"),
    }
    fits, meta, predictions, evaluations = [], [], [], []
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        common = ["--cache", directory / "feature_cache", "--execution", directory / "execution.json",
                  "--roles", directory / "role_map.json"]
        predictor = [*common, "--run-root", directory / "runs",
                     "--base-freeze", directory / "base_freeze.json"]
        stages["prepare"]["tasks"][0]["commands"].append(
            command("smoke", *common, "--out", directory / "manifests/readiness.json",
                    "--max-seconds", config["readiness_max_seconds"]))
        for arm in ARMS:
            fits.append({"key": f"seed{seed}_{arm}", "seed": seed, "arm": arm,
                         "commands": [command("train", *common, "--run-root", directory / "runs",
                                              "--arm", arm, "--seed", seed, "--device", "cuda",
                                              "--num-workers", 0)]})
        meta.append({"key": f"seed{seed}", "seed": seed, "commands": [
            command("predict", *predictor, "--freeze-base", "--role", "meta",
                    "--out", directory / "predictions/meta.npz"),
            command("combine", "--root", directory, "--predictions", directory / "predictions/meta.npz",
                    "--out", directory / "heads")]})
        predictions.append({"key": f"seed{seed}", "seed": seed, "commands": [
            command("predict", *predictor, "--role", "dev_eval", "--heads-freeze",
                    directory / "heads_freeze.json", "--out", directory / "predictions/dev_eval.npz")]})
        evaluations.append({"key": f"seed{seed}", "seed": seed, "commands": [
            command("evaluate", "--root", directory, "--predictions", directory / "predictions/dev_eval.npz",
                    "--heads", directory / "heads", "--out", directory / "evaluation")]})
    stages["fits"] = stage(fits, True, "base", "prepare", throttle=cap)
    stages["fits"]["requires"] = [f"seed{seed}/manifests/readiness.json" for seed in SEEDS]
    stages["meta_heads"] = stage(meta, True, "predictions", "fits", throttle=min(2, cap))
    stages["joint_heads"] = stage([{"key": "joint_heads", "commands": [
        command("replication", "freeze-heads", "--root", root)]}],
        False, "predictions", "meta_heads")
    stages["dev_eval"] = stage(predictions, True, "predictions", "joint_heads", throttle=min(2, cap))
    stages["evaluate"] = stage(evaluations, False, "evaluation", "dev_eval", throttle=2)
    stages["aggregate"] = stage([{"key": "aggregate", "commands": [
        command("aggregate_replications", "--root", root, "--out", root / "evaluation",
                "--late-seed7-root", config["source_root"])]}], False, "evaluation", "evaluate")
    stages["guardian"]["cpus"] = 1
    for name in ORDER:
        stages[name]["minutes"] = config["caps"][name]
    return stages


def build_plan(config, *, now=None, check_time=True):
    """Pure deterministic planning: no imports of ML, filesystem writes or Slurm calls."""
    config = _config(config)
    stages = _stages(config)
    gpu_minutes = sum(row["minutes"] * len(row["tasks"]) for row in stages.values() if row["gpu"])
    total_gpu_minutes = gpu_minutes + config["external_gpu_minutes"]
    if total_gpu_minutes > config["gpu_budget_minutes"]:
        raise PlanError("All array-task GPU caps plus external reservations exceed the approved budget")
    cap = config["max_concurrent_gpus"]
    caps = config["caps"]
    base = caps["prepare"] + math.ceil(8 / cap) * caps["fits"]
    predictions = base + math.ceil(2 / cap) * (caps["meta_heads"] + caps["dev_eval"]) + caps["joint_heads"]
    critical = {"base": base, "predictions": predictions,
                "evaluation": predictions + caps["evaluate"] + caps["aggregate"]}
    plan = {
        "schema_version": 1, "scope_id": SCOPE, "config": config,
        "run_id": config["run_id"], "matrix": copy.deepcopy(MATRIX),
        "epochs": 20, "fresh_fits": 8, "original_test_access": False, "automatic_retries": 0,
        "submission_count": len(ORDER), "allocated_task_count": sum(len(s["tasks"]) for s in stages.values()),
        "order": list(ORDER), "stages": stages, "critical_path_cap_minutes": critical,
        "release_sha256": digest(config["source_sha256"]),
        "resource_claims": {
            "resources": copy.deepcopy(config["resources"]), "caps": copy.deepcopy(caps),
            "readiness_max_seconds": config["readiness_max_seconds"],
            "gpu_allocation_counts": {name: len(row["tasks"]) for name, row in stages.items() if row["gpu"]},
            "max_concurrent_gpus": cap, "external_reserved_gpus": config["external_reserved_gpus"],
            "gpu_ceiling_including_external": cap + config["external_reserved_gpus"],
            "workflow_gpu_minutes": gpu_minutes, "external_gpu_minutes": config["external_gpu_minutes"],
            "reserved_gpu_minutes": total_gpu_minutes, "gpu_budget_minutes": config["gpu_budget_minutes"],
            "allocated_cpu_minutes": sum(row["minutes"] * row["cpus"] * len(row["tasks"]) for row in stages.values()),
            "submission_count": len(ORDER),
            "allocated_task_count": sum(len(s["tasks"]) for s in stages.values()),
        },
    }
    if check_time:
        _admit_time(plan, utc_now(now))
    return plan


def _admit_time(plan, now):
    config = plan["config"]
    for name, minutes in plan["critical_path_cap_minutes"].items():
        if now + dt.timedelta(minutes=minutes) >= timestamp(config["deadlines"][name]):
            raise PlanError("Insufficient prospective time for full allocation caps before " + name)
    guardian_end = timestamp(config["deadlines"]["evaluation"]) + dt.timedelta(minutes=config["guardian_grace_minutes"])
    if now + dt.timedelta(minutes=config["caps"]["guardian"]) < guardian_end:
        raise PlanError("Guardian cap must cover evaluation cutoff and its bounded reporting grace")


def authorization_template(plan):
    validate_plan(plan, check_time=False)
    return {
        "schema_version": 1, "scope_id": SCOPE, "approved": False,
        "purpose": "scientific_replication", "run_id": plan["run_id"],
        "root": plan["config"]["root"], "source_root": plan["config"]["source_root"],
        "workflow_sha256": digest(plan), "release_sha256": plan["release_sha256"],
        "matrix": copy.deepcopy(MATRIX), "deadlines": copy.deepcopy(plan["config"]["deadlines"]),
        "resource_claims": copy.deepcopy(plan["resource_claims"]),
        "automatic_retries": 0, "original_test_access": False,
        "authorized_at": None, "expires_at": None,
    }


def validate_authorization(plan, authorization, *, now=None, current=True):
    if not isinstance(authorization, dict):
        raise PlanError("Explicit approved JSON authorization is required")
    expected = authorization_template(plan)
    expected["approved"] = True
    for key, value in expected.items():
        if key not in {"authorized_at", "expires_at"} and authorization.get(key) != value:
            raise PlanError("Authorization does not match frozen " + key)
    if set(authorization) != set(expected) or authorization.get("approved") is not True:
        raise PlanError("Authorization fields or explicit approval are invalid")
    issued = timestamp(authorization["authorized_at"])
    expires = timestamp(authorization["expires_at"])
    evaluation = timestamp(plan["config"]["deadlines"]["evaluation"])
    if not issued < expires <= evaluation:
        raise PlanError("Authorization must have a finite expiry no later than evaluation cutoff")
    if current and not issued <= utc_now(now) < expires:
        raise PlanError("Authorization is not yet active or has expired")
    return authorization


def validate_plan(plan, authorization=None, *, now=None, check_time=True):
    if not isinstance(plan, dict) or "config" not in plan:
        raise PlanError("Invalid workflow object")
    expected = build_plan(plan["config"], now=now, check_time=check_time)
    if plan != expected:
        raise PlanError("Workflow graph, commands, matrix or resource claims changed")
    if authorization is not None:
        validate_authorization(plan, authorization, now=now)
        if (check_time and utc_now(now) + dt.timedelta(minutes=plan["critical_path_cap_minutes"]["evaluation"])
                >= timestamp(authorization["expires_at"])):
            raise PlanError("Authorization expires before the complete allocation-cap chain can finish")
    return plan


def verify_release(plan, *, executing=False):
    config = plan["config"]
    for name in ("root", "control_root", "code_root", "source_root"):
        if Path(config[name]).resolve() != Path(config[name]):
            raise PlanError("Redirected namespace: " + name)
    release = Path(config["code_root"])
    if executing and Path(__file__).resolve() != release / SELF:
        raise PlanError("Stage runner is not executing from the authorized immutable release")
    manifest_path = release / "source_manifest.json"
    if manifest_path.is_symlink():
        raise PlanError("Release manifest must not be redirected")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_sha256") != config["source_sha256"] or manifest.get("release_id") != release.name:
        raise PlanError("Source release manifest differs from the reviewed inventory")
    present = {str(path.relative_to(release)) for path in release.rglob("*")
               if path.is_file() and path != manifest_path and "__pycache__" not in path.parts}
    if present != set(config["source_sha256"]):
        raise PlanError("Release must contain exactly the inventoried source files")
    for relative, wanted in config["source_sha256"].items():
        path = release / relative
        if path.stat().st_size > 1024 * 1024 and not os.environ.get("SLURM_JOB_ID"):
            raise PlanError("Large release verification requires an allocated Slurm job")
        if path.resolve() != path or not path.is_file() or file_sha256(path) != wanted:
            raise PlanError("Immutable source changed: " + relative)


def _bound_json(path, expected):
    if not isinstance(expected, str) or not HEX.fullmatch(expected) or file_sha256(path) != expected:
        raise PlanError("Changed hash-bound file: " + str(path))
    return json.loads(Path(path).read_text())


def _frozen_json(path, value):
    if path.exists():
        if path.read_bytes() != json_bytes(value):
            raise PlanError("Refusing to replace immutable workflow/authorization: " + str(path))
    else:
        atomic_json(path, value)


def _control(plan):
    return Path(plan["config"]["control_root"])


def _name(plan, stage):
    return "omri-rep-" + plan["run_id"][:24] + "-" + stage + "-" + digest(plan)[:12]


def _comment(plan, stage):
    return "polygraph-replication:" + digest(plan) + ":" + stage


def sbatch_command(plan, stage, jobs, authorization_sha256):
    if stage not in ORDER:
        raise PlanError("Undeclared stage")
    row = plan["stages"][stage]
    config = plan["config"]
    resources = config["resources"]
    control = _control(plan)
    runner = ["/usr/bin/python3", "-B", "-u", "-m", MODULE + "slurm.replicate", "run-stage",
              "--workflow", str(control / "workflow.json"), "--sha256", digest(plan),
              "--authorization", str(control / "authorization.json"),
              "--authorization-sha256", authorization_sha256, "--stage", stage]
    command = ["sbatch", "--parsable", "--account=" + resources["account"],
               "--partition=" + resources["gpu_partition" if row["gpu"] else "cpu_partition"],
               "--cpus-per-task=" + str(row["cpus"]), "--mem=" + str(row["memory_mb"]) + "M",
               "--time=" + str(row["minutes"]), "--job-name=" + _name(plan, stage),
               "--comment=" + _comment(plan, stage), "--no-requeue", "--kill-on-invalid-dep=yes",
               "--export=NONE", "--chdir=" + config["code_root"],
               "--output=" + str(control / "logs" / (stage + "_%A_%a.out")),
               "--error=" + str(control / "logs" / (stage + "_%A_%a.err"))]
    if row["gpu"]:
        command += ["--gpus=1", "--constraint=" + resources["gpu_constraint"]]
    if len(row["tasks"]) > 1:
        command += [f"--array=0-{len(row['tasks']) - 1}%{row['throttle']}"]
    if row["dependency"] is not None:
        dependency = jobs.get(row["dependency"], "")
        if not isinstance(dependency, str) or not dependency.isdigit():
            raise PlanError("Missing durable predecessor ID for " + stage)
        command += ["--dependency=" + row["dependency_type"] + ":" + dependency]
    return [*command, "--wrap", "exec " + shlex.join(runner)]


class Scheduler:
    """Finite scheduler calls. Every uncertain submission remains durable."""

    @staticmethod
    def _run(command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout

    def submit(self, command):
        return self._run(command).strip()

    def reconcile(self, intent):
        start = (timestamp(intent["created_utc"]) - dt.timedelta(days=1)).date().isoformat()
        commands = [
            ["squeue", "--noheader", "--array", "--user=" + getpass.getuser(),
             "--name=" + intent["job_name"], "--format=%F|%j|%k"],
            ["sacct", "--noheader", "--parsable2", "--allocations", "--user=" + getpass.getuser(),
             "--starttime=" + start, "--name=" + intent["job_name"],
             "--format=JobID%64,JobName%160,Comment%256"],
        ]
        found = set()
        for command in commands:
            for line in self._run(command).splitlines():
                row = line.split("|")
                if len(row) >= 3 and row[1] == intent["job_name"] and row[2] == intent["comment"]:
                    job = row[0].split("_")[0].split(".")[0]
                    if job.isdigit():
                        found.add(job)
        return sorted(found)

    def states(self, ids):
        if not ids:
            return {}
        output = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--array",
                            "--jobs=" + ",".join(ids), "--format=JobID%64,State%32,ExitCode,ElapsedRaw"])
        result = {}
        for line in output.splitlines():
            row = line.split("|")
            if len(row) >= 4 and "." not in row[0] and row[0].split("_")[0] in ids:
                result[row[0]] = {"state": row[1].split()[0].rstrip("+"),
                                  "exit_code": row[2], "elapsed_seconds": row[3]}
        return result

    def cancel(self, ids):
        if ids:
            self._run(["scancel", *ids])


def _intent_path(plan, stage):
    return _control(plan) / "manifests/submission_intents" / (stage + ".json")


def _record_job(plan, path, intent, job, *, reconciled=False):
    ledger = _control(plan) / "manifests/submissions.tsv"
    with ledger.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write("\t".join((utc_now().isoformat(), job, intent["stage"],
                               digest(plan), intent["authorization_sha256"])) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    intent.update(status="recorded", job_id=job, reconciled=reconciled)
    atomic_json(path, intent)
    return job


def submit_one(plan, stage, jobs, authorization_sha256, scheduler, *, now=None):
    command = sbatch_command(plan, stage, jobs, authorization_sha256)
    path = _intent_path(plan, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            intent = json.loads(path.read_text())
            if (intent.get("command") != command or intent.get("workflow_sha256") != digest(plan)
                    or intent.get("authorization_sha256") != authorization_sha256):
                raise AmbiguousSubmission("Different immutable intent; never resubmit " + stage)
            if intent.get("status") == "recorded" and str(intent.get("job_id", "")).isdigit():
                return str(intent["job_id"])
            matches = scheduler.reconcile(intent)
            if len(matches) != 1:
                raise AmbiguousSubmission("Intent has zero or multiple scheduler matches; no retry: " + stage)
            return _record_job(plan, path, intent, matches[0], reconciled=True)
        intent = {"status": "submitting", "stage": stage, "command": command,
                  "job_name": _name(plan, stage), "comment": _comment(plan, stage),
                  "created_utc": utc_now(now).isoformat(), "workflow_sha256": digest(plan),
                  "authorization_sha256": authorization_sha256}
        atomic_json(path, intent)
        try:
            receipt = scheduler.submit(command)
            if not isinstance(receipt, str) or not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", receipt.strip()):
                raise AmbiguousSubmission("Unrecognized sbatch receipt")
            job = receipt.strip().split(";")[0]
        except Exception as error:
            intent.update(status="ambiguous", error_type=type(error).__name__)
            atomic_json(path, intent)
            raise AmbiguousSubmission("Submission may have been accepted; reconcile without retry: " + stage) from error
        return _record_job(plan, path, intent, job)


def recorded_jobs(plan, authorization_sha256=None):
    result = {}
    for stage in ORDER:
        path = _intent_path(plan, stage)
        if not path.exists():
            continue
        intent = json.loads(path.read_text())
        authorization_sha = intent.get("authorization_sha256", "")
        if (intent.get("workflow_sha256") != digest(plan) or intent.get("stage") != stage
                or intent.get("job_name") != _name(plan, stage)
                or intent.get("comment") != _comment(plan, stage)
                or not isinstance(authorization_sha, str) or not HEX.fullmatch(authorization_sha)
                or (authorization_sha256 is not None and authorization_sha != authorization_sha256)
                or intent.get("command") != sbatch_command(plan, stage, result, authorization_sha)):
            raise PlanError("Foreign submission intent")
        if intent.get("status") == "recorded" and str(intent.get("job_id", "")).isdigit():
            result[stage] = str(intent["job_id"])
    if len(result.values()) != len(set(result.values())):
        raise PlanError("One Slurm ID is bound to multiple stages")
    return result


def reconcile_intents(plan, scheduler, authorization_sha256):
    """Record unique matches only. Absence, accounting lag and errors never retry."""
    jobs = recorded_jobs(plan, authorization_sha256)
    unresolved = []
    for stage in ORDER:
        path = _intent_path(plan, stage)
        if not path.exists() or stage in jobs:
            continue
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            intent = json.loads(path.read_text())
            if (intent.get("workflow_sha256") != digest(plan)
                    or intent.get("authorization_sha256") != authorization_sha256
                    or intent.get("command") != sbatch_command(plan, stage, jobs, authorization_sha256)):
                raise PlanError("Cannot reconcile an altered intent")
            matches = scheduler.reconcile(intent)
            if len(matches) == 1:
                jobs[stage] = _record_job(plan, path, intent, matches[0], reconciled=True)
            else:
                unresolved.append({"stage": stage, "matches": matches})
    return {"job_ids": jobs, "unresolved": unresolved, "submitted_new_jobs": False}


def submit_plan(plan, authorization, scheduler=None, *, now=None):
    validate_plan(plan, authorization, now=now)
    verify_release(plan)
    scheduler = scheduler or Scheduler()
    control = _control(plan)
    control.mkdir(parents=True, exist_ok=True)
    if control.resolve() != control:
        raise PlanError("Control namespace is redirected")
    with (control / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / "manifests/terminal.json").exists():
            raise PlanError("Terminal replication cannot be relaunched")
        for name in ("logs", "manifests/submission_intents"):
            (control / name).mkdir(parents=True, exist_ok=True)
        _frozen_json(control / "workflow.json", plan)
        _frozen_json(control / "authorization.json", authorization)
        authorization_sha = digest(authorization)
        launch_path = control / "manifests/launch.json"
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if (launch.get("workflow_sha256") != digest(plan)
                    or launch.get("authorization_sha256") != authorization_sha):
                raise PlanError("Existing launch identity differs")
            if launch.get("status") == "incomplete_submission":
                raise PlanError("Failed static launch is terminal; reconcile IDs, do not expand it")
        else:
            launch = {"status": "submitting", "workflow_sha256": digest(plan),
                      "authorization_sha256": authorization_sha, "created_utc": utc_now(now).isoformat(),
                      "run_id": plan["run_id"], "expected_stages": list(ORDER)}
            atomic_json(launch_path, launch)
        jobs = {}
        try:
            for stage in ORDER:
                validate_authorization(plan, authorization, now=now)
                jobs[stage] = submit_one(plan, stage, jobs, authorization_sha, scheduler, now=now)
        except Exception as error:
            launch.update(status="incomplete_submission", job_ids=recorded_jobs(plan, authorization_sha),
                          error_type=type(error).__name__)
            atomic_json(launch_path, launch)
            raise
        launch.update(status="submitted", job_ids=jobs)
        atomic_json(launch_path, launch)
        return launch


def required_evidence_paths():
    paths = {"replication.json", "role_map.json", "joint_heads_freeze.json", "evaluation/complete.json"}
    paths.update("evaluation/" + name for name in AGGREGATE_FILES)
    for seed in SEEDS:
        prefix = f"seed{seed}/"
        paths.update(prefix + name for name in
                     ("execution.json", "role_map.json", "base_freeze.json", "heads_freeze.json",
                      "heads/stack.json", "heads/last_only.json", "predictions/dev_eval.json",
                      "predictions/dev_eval.npz", "evaluation/complete.json", "manifests/readiness.json"))
        paths.update(prefix + "evaluation/" + name for name in EVALUATION_FILES)
    return paths


def _artifact(root, relative):
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise PlanError("Missing or redirected required artifact: " + relative)
    return path


def verify_readiness(plan, expected_job_id=None):
    require_slurm()
    root = Path(plan["config"]["root"])
    manifest = json.loads(_artifact(root, "replication.json").read_text())
    observed = time.time()
    base_cutoff = timestamp(plan["config"]["deadlines"]["base"]).timestamp()
    result = {}
    for seed in SEEDS:
        prefix = f"seed{seed}/"
        relative = prefix + "manifests/readiness.json"
        report = json.loads(_artifact(root, relative).read_text())
        if (report.get("scope_id") != SCOPE or report.get("seed") != seed
                or report.get("passed") is not True or report.get("diagnostic_only") is not True
                or report.get("scientific_fits_created") is not False
                or report.get("meta_or_dev_predictions") is not False
                or report.get("source_sha256") != plan["config"]["source_sha256"]["pilots/layer_ensemble_20260914/smoke.py"]
                or report.get("execution_sha256") != file_sha256(_artifact(root, prefix + "execution.json"))
                or report.get("roles_sha256") != file_sha256(_artifact(root, prefix + "role_map.json"))
                or set(report.get("arms", {})) != set(ARMS)
                or (expected_job_id is not None and report.get("job_id") != expected_job_id)):
            raise PlanError("Both seed-specific, source-bound readiness checks are required")
        if (type(report.get("completed_unix")) not in (int, float)
                or not manifest["created_unix"] < report["completed_unix"] < base_cutoff
                or report["completed_unix"] > observed
                or type(report.get("elapsed_seconds")) not in (int, float)
                or not 0 < report["elapsed_seconds"] <= plan["config"]["readiness_max_seconds"]):
            raise PlanError("Readiness check missed its bounded prospective window")
        probes = report.get("two_loader_probes", [])
        if (not isinstance(probes, list) or any(not isinstance(probe, dict) for probe in probes)
                or [probe.get("role") for probe in probes] != ["base_train", "checkpoint"]):
            raise PlanError("Readiness may inspect only base/checkpoint roles")
        checks = ("finite_gradient", "immediate_state_exact", "deterministic_resume_original_tolerance_passed",
                  "record_and_hidden_parity", "sampler_epoch_resume")
        if (any(not isinstance(report["arms"][arm], dict) for arm in ARMS)
                or any(report["arms"][arm].get(key) is not True for arm in ARMS for key in checks)):
            raise PlanError("All four prescribed readiness audits must pass for each seed")
        result[relative] = file_sha256(root / relative)
    return result


def completion_evidence(plan, authorization=None, *, prepare_job_id=None):
    """Hash existing artifacts on an allocated CPU; never load arrays or compute metrics."""
    require_slurm()
    root = Path(plan["config"]["root"])
    try:
        files = {name: file_sha256(_artifact(root, name)) for name in sorted(required_evidence_paths())}

        def read(name):
            return json.loads(_artifact(root, name).read_text())

        manifest = read("replication.json")
        expected_deadlines = {DEADLINE_KEYS[key]: value for key, value in plan["config"]["deadlines"].items()}
        if (manifest.get("scope_id") != SCOPE or manifest.get("run_id") != plan["run_id"]
                or manifest.get("root") != str(root) or manifest.get("source_root") != plan["config"]["source_root"]
                or manifest.get("seeds") != list(SEEDS) or manifest.get("matrix") != MATRIX
                or manifest.get("deadlines") != expected_deadlines or manifest.get("original_test_access") is not False):
            raise PlanError("Replication completion has a different identity")
        if (manifest.get("roles_sha256") != files["role_map.json"]
                or manifest.get("source_files", {}).get("role_map.json") != files["role_map.json"]
                or any(files[f"seed{seed}/role_map.json"] != files["role_map.json"] for seed in SEEDS)):
            raise PlanError("Replication must retain the exact original role-map bytes")
        readiness = verify_readiness(plan, prepare_job_id)
        if any(files[name] != sha for name, sha in readiness.items()):
            raise PlanError("Readiness receipts changed during validation")
        observed = time.time()
        created = manifest["created_unix"]
        latest = timestamp(plan["config"]["deadlines"]["evaluation"]).timestamp()
        if authorization is not None:
            validate_authorization(plan, authorization, current=False)
            latest = min(latest, timestamp(authorization["expires_at"]).timestamp())
        if type(created) not in (int, float) or not math.isfinite(created) or not 0 < created < latest:
            raise PlanError("Invalid replication creation time")
        joint = read("joint_heads_freeze.json")
        if (joint.get("complete") is not True or joint.get("scope_id") != SCOPE
                or joint.get("seeds") != list(SEEDS) or joint.get("original_test_access") is not False
                or joint.get("replication_manifest_sha256") != files["replication.json"]
                or set(joint.get("files", {})) != {str(seed) for seed in SEEDS}):
            raise PlanError("Both-seed head freeze is missing or unbound")
        if not created < joint.get("frozen_unix", 0) <= observed:
            raise PlanError("Invalid joint head-freeze timestamp")
        seed_completions = {}
        for seed in SEEDS:
            prefix = f"seed{seed}/"
            bindings = joint["files"][str(seed)]
            names = {"execution.json", "base_freeze.json", "heads_freeze.json",
                     "heads/stack.json", "heads/last_only.json"}
            if set(bindings) != names or any(bindings[name] != files[prefix + name] for name in names):
                raise PlanError("Joint freeze no longer binds both heads for both seeds")
            complete = read(prefix + "evaluation/complete.json")
            seed_completions[str(seed)] = complete
            report = read(prefix + "evaluation/report.json")
            if (complete.get("complete") is not True or complete.get("scope_id") != SCOPE
                    or complete.get("seed") != seed or not created < complete.get("completed_unix", 0) < latest
                    or complete.get("completed_unix", float("inf")) > observed
                    or report.get("scope_id") != SCOPE or report.get("seed") != seed
                    or report.get("complete") is not True or report.get("test_evaluated") is not False
                    or report.get("base_epochs") != 20 or report.get("records") != 7200
                    or complete.get("inputs") != report.get("inputs")
                    or set(complete.get("files", {})) != EVALUATION_FILES):
                raise PlanError("Incomplete, late or foreign per-seed evaluation")
            for name, wanted in complete["files"].items():
                if wanted != files[prefix + "evaluation/" + name]:
                    raise PlanError("Per-seed evaluation file changed: " + name)
            inputs = complete["inputs"]
            expected_inputs = {
                "replication_manifest_sha256": files["replication.json"],
                "joint_heads_freeze_sha256": files["joint_heads_freeze.json"],
                "execution_sha256": files[prefix + "execution.json"],
                "roles_sha256": files[prefix + "role_map.json"],
                "base_freeze_sha256": files[prefix + "base_freeze.json"],
                "heads_freeze_sha256": files[prefix + "heads_freeze.json"],
                "dev_prediction_sidecar_sha256": files[prefix + "predictions/dev_eval.json"],
                "dev_prediction_npz_sha256": files[prefix + "predictions/dev_eval.npz"],
                "cache_manifest_sha256": manifest.get("cache_manifest_sha256"),
                "implementation": {
                    name: plan["config"]["source_sha256"]["pilots/layer_ensemble_20260914/" + name]
                    for name in ("combine.py", "evaluate.py", "protocol.py", "replication.py")},
            }
            if set(inputs) != set(expected_inputs) or any(inputs.get(key) != value for key, value in expected_inputs.items()):
                raise PlanError("Per-seed evaluation is not bound to current frozen inputs")
            prediction = read(prefix + "predictions/dev_eval.json")
            if (prediction.get("seed") != seed or prediction.get("role") != "dev_eval"
                    or prediction.get("joint_heads_freeze_sha256") != files["joint_heads_freeze.json"]
                    or prediction.get("npz_sha256") != files[prefix + "predictions/dev_eval.npz"]
                    or not joint.get("frozen_unix", float("inf")) < prediction.get("completed_unix", 0)
                    < timestamp(plan["config"]["deadlines"]["predictions"]).timestamp()
                    or prediction.get("completed_unix", float("inf")) > complete["completed_unix"]):
                raise PlanError("Development export missed its bound cross-seed gate or cutoff")
        complete = read("evaluation/complete.json")
        report = read("evaluation/report.json")
        if (complete.get("complete") is not True or complete.get("scope_id") != SCOPE
                or report.get("complete") is not True or report.get("scope_id") != SCOPE
                or report.get("status") != "complete" or report.get("included_seeds") != [17, 27, 7]
                or complete.get("primary_seeds") != list(SEEDS) or report.get("primary_seeds") != list(SEEDS)
                or report.get("primary", {}).get("seeds") != list(SEEDS)
                or report.get("test_evaluated") is not False or report.get("records_per_seed") != 7200
                or report.get("source_photographs") != 800 or report.get("base_epochs") != 20
                or report.get("all3_descriptive_complete") is not True
                or report.get("all3_descriptive", {}).get("status") != "complete"
                or report.get("all3_descriptive", {}).get("seeds") != [7, 17, 27]
                or set(report.get("per_seed", {})) != {"7", "17", "27"}
                or report.get("per_seed", {}).get("7", {}).get("status") != "complete_late_diagnostic"
                or complete.get("inputs") != report.get("inputs")
                or set(complete.get("files", {})) != AGGREGATE_FILES
                or not created < complete.get("completed_unix", 0) < latest
                or complete.get("completed_unix", float("inf")) > observed):
            raise PlanError("Aggregate completion inventory or identity is invalid")
        for name, wanted in complete["files"].items():
            if wanted != files["evaluation/" + name]:
                raise PlanError("Aggregate evaluation file changed: " + name)
        bootstrap = read("evaluation/bootstrap.json")
        if any(report.get(key) != bootstrap.get(key) for key in ("primary", "all3_descriptive", "per_seed")):
            raise PlanError("Aggregate report and bound bootstrap metadata disagree")
        inputs = complete["inputs"]
        expected_keys = {"replication_manifest_sha256", "roles_sha256", "joint_heads_freeze_sha256",
                         "primary_seeds", "late_seed7_root", "late_seed7_source_files",
                         "seed_evaluations", "implementation"}
        implementation = {
            name: plan["config"]["source_sha256"]["pilots/layer_ensemble_20260914/" + name]
            for name in ("aggregate_replications.py", "evaluate.py", "combine.py", "replication.py", "protocol.py")}
        if (set(inputs) != expected_keys or inputs.get("implementation") != implementation
                or inputs.get("replication_manifest_sha256") != files["replication.json"]
                or inputs.get("roles_sha256") != files["role_map.json"]
                or inputs.get("joint_heads_freeze_sha256") != files["joint_heads_freeze.json"]
                or inputs.get("primary_seeds") != list(SEEDS)
                or inputs.get("late_seed7_root") != plan["config"]["source_root"]
                or inputs.get("late_seed7_source_files") != manifest.get("source_files")
                or set(inputs.get("seed_evaluations", {})) != {"7", "17", "27"}):
            raise PlanError("Aggregate has incomplete replication/late-source bindings")
        for seed in SEEDS:
            binding = inputs["seed_evaluations"][str(seed)]
            seed_complete = seed_completions[str(seed)]
            if (binding.get("seed") != seed or binding.get("scope_id") != SCOPE
                    or binding.get("root") != str(root / f"seed{seed}")
                    or binding.get("complete_sha256") != files[f"seed{seed}/evaluation/complete.json"]
                    or binding.get("files") != seed_complete["files"] or binding.get("inputs") != seed_complete["inputs"]
                    or seed_complete["completed_unix"] > complete["completed_unix"]):
                raise PlanError("Aggregate is not bound to both per-seed evaluation completions")
        late = inputs["seed_evaluations"]["7"]
        marker = report.get("late_seed7_marker", {})
        source = Path(plan["config"]["source_root"])
        if set(manifest.get("source_files", {})) != HISTORY_FILES:
            raise PlanError("Historical provenance inventory is incomplete")
        for name, wanted in manifest["source_files"].items():
            if file_sha256(_artifact(source, name)) != wanted:
                raise PlanError("Frozen historical provenance changed: " + name)
        original_marker = json.loads(_artifact(source, "evaluation/late_diagnostic.json").read_text())
        original_complete = json.loads(_artifact(source, "evaluation/complete.json").read_text())
        if (late.get("seed") != 7 or late.get("scope_id") != "layer_ensemble_20260914_core_seed7"
                or late.get("root") != plan["config"]["source_root"]
                or late.get("complete_sha256") != manifest["source_files"].get("evaluation/complete.json")
                or late.get("files") != original_complete.get("files")
                or late.get("inputs") != original_complete.get("inputs")
                or marker != original_marker
                or marker.get("complete") is not True or marker.get("status") != "complete_late_diagnostic"
                or marker.get("scope_id") != "layer_ensemble_20260914_core_seed7"
                or marker.get("reused_on_time_meta_export") is not True
                or not marker.get("original_protocol_status", "").startswith("incomplete")
                or not isinstance(marker.get("warning"), str) or not marker["warning"].strip()):
            raise PlanError("The all-three descriptive summary must preserve the bound late seed-7 status")
        return {"complete": True, "workflow_sha256": digest(plan), "files": files,
                "source_files": manifest["source_files"],
                "validated_utc": utc_now().isoformat()}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"complete": False, "workflow_sha256": digest(plan),
                "reason": str(error), "error_type": type(error).__name__}


def stage_states(plan, jobs, states):
    result = {}
    for stage in ORDER:
        if stage == "guardian":
            continue
        job = jobs.get(stage)
        count = len(plan["stages"][stage]["tasks"])
        keys = [job] if count == 1 else [f"{job}_{index}" for index in range(count)]
        rows = [states.get(key, {}) for key in keys]
        failed = next((row.get("state") for row in rows if row.get("state") in FAILED), None)
        if failed:
            result[stage] = failed
        elif all(row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0" for row in rows):
            result[stage] = "COMPLETED"
        elif any(row.get("state") == "COMPLETED" and row.get("exit_code") != "0:0" for row in rows):
            result[stage] = "FAILED"
        else:
            result[stage] = next((row["state"] for row in rows if row.get("state") in ACTIVE), "UNKNOWN")
    return result


def guardian_decision(plan, states, evidence=None, *, now=None, launch=None):
    """Pure fail-closed decision; a success marker alone can never complete the run."""
    now = utc_now(now)
    config = plan["config"]
    launch = launch or {}
    if launch.get("status") == "incomplete_submission":
        return {"status": "incomplete", "reason": "incomplete_submission"}
    if launch.get("status") != "submitted":
        started = timestamp(launch["created_utc"]) if launch.get("created_utc") else now
        if now >= started + dt.timedelta(minutes=config["submission_grace_minutes"]):
            return {"status": "incomplete", "reason": "submission_not_committed"}
    failed = {name: state for name, state in states.items() if state in FAILED}
    if failed:
        return {"status": "incomplete", "reason": "required_stage_failed", "failed_stages": failed}
    if all(states.get(name) == "COMPLETED" for name in ORDER if name != "guardian"):
        bound = (isinstance(evidence, dict) and evidence.get("complete") is True
                 and evidence.get("workflow_sha256") == digest(plan)
                 and required_evidence_paths() <= set(evidence.get("files", {}))
                 and all(isinstance(sha, str) and HEX.fullmatch(sha) for sha in evidence.get("files", {}).values())
                 and set(evidence.get("source_files", {})) == HISTORY_FILES
                 and all(isinstance(sha, str) and HEX.fullmatch(sha)
                         for sha in evidence.get("source_files", {}).values()))
        if bound and launch.get("status") == "submitted":
            return {"status": "complete", "reason": "all_bound_evaluations_complete", "evidence": evidence}
        return {"status": "incomplete", "reason": "evaluation_artifacts_unverified", "evidence": evidence}
    for deadline, stage in (("base", "fits"), ("predictions", "dev_eval"), ("evaluation", "aggregate")):
        if now >= timestamp(config["deadlines"][deadline]) and states.get(stage) != "COMPLETED":
            return {"status": "incomplete", "reason": deadline + "_deadline"}
    if now >= timestamp(config["deadlines"]["evaluation"]):
        return {"status": "incomplete", "reason": "evaluation_unverified_at_cutoff"}
    return {"status": "watching"}


def _guardian_job(plan, authorization_sha256, scheduler=None, *, wait_for_ack=False,
                  clock=None, sleep=None):
    require_slurm()
    actual = os.environ["SLURM_JOB_ID"]
    if (not actual.isdigit() or os.environ.get("SLURM_ARRAY_TASK_ID") is not None
            or os.environ.get("SLURM_ARRAY_JOB_ID") is not None):
        raise PlanError("Guardian requires its declared singleton Slurm allocation")
    clock, sleep = clock or time.time, sleep or time.sleep
    launch = json.loads((_control(plan) / "manifests/launch.json").read_text())
    if (launch.get("workflow_sha256") != digest(plan)
            or launch.get("authorization_sha256") != authorization_sha256):
        raise PlanError("Guardian launch identity differs")
    grace = plan["config"]["submission_grace_minutes"] * 60
    until = min(clock() + grace, timestamp(launch["created_utc"]).timestamp() + grace)
    next_reconciliation = clock() + 5
    path = _intent_path(plan, "guardian")
    expected = {"stage": "guardian", "workflow_sha256": digest(plan),
                "authorization_sha256": authorization_sha256,
                "job_name": _name(plan, "guardian"), "comment": _comment(plan, "guardian"),
                "command": sbatch_command(plan, "guardian", {}, authorization_sha256)}

    def read_intent():
        if not path.exists():
            return None
        if path.resolve() != path:
            raise PlanError("Guardian intent is redirected")
        intent = json.loads(path.read_text())
        if any(intent.get(key) != value for key, value in expected.items()):
            raise PlanError("Guardian submission intent differs")
        if intent.get("status") == "recorded":
            if intent.get("job_id") != actual:
                raise PlanError("Guardian allocation is not its recorded Slurm job")
        elif intent.get("status") not in {"submitting", "ambiguous"}:
            raise PlanError("Invalid guardian submission state")
        return intent

    while True:
        intent = read_intent()
        if intent is not None and intent["status"] == "recorded":
            return actual
        if not wait_for_ack or clock() >= until:
            raise PlanError("Bounded guardian acknowledgement wait expired; allocation is unbound")
        if scheduler is not None and intent is not None and (
                intent["status"] == "ambiguous" or clock() >= next_reconciliation):
            matches = scheduler.reconcile(intent)
            if matches and matches != [actual]:
                raise PlanError("Guardian intent resolves to a different or ambiguous allocation")
            if matches == [actual]:
                with path.with_suffix(".lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        current = read_intent()
                        if current is not None and current["status"] == "recorded":
                            return actual
                        if current is not None and clock() < until:
                            _record_job(plan, path, current, actual, reconciled=True)
                            return actual
            next_reconciliation = clock() + 5
        remaining = until - clock()
        if remaining <= 0:
            raise PlanError("Bounded guardian acknowledgement wait expired; allocation is unbound")
        sleep(min(5, remaining))


def _terminal_receipt(plan, authorization_sha256, guardian_job_id):
    path = _control(plan) / "manifests/terminal.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.resolve() != path:
        raise PlanError("Terminal receipt is redirected")
    receipt = json.loads(path.read_text())
    expected = {"schema_version": 1, "scope_id": SCOPE, "run_id": plan["run_id"],
                "workflow_sha256": digest(plan), "authorization_sha256": authorization_sha256,
                "guardian_job_id": guardian_job_id}
    if (any(receipt.get(key) != value for key, value in expected.items())
            or receipt.get("status") not in {"complete", "incomplete"}
            or receipt.get("complete") is not (receipt.get("status") == "complete")):
        raise PlanError("Existing terminal receipt has a different or invalid identity")
    return receipt


@contextmanager
def _terminal_lock(path, *, clock=None, sleep=None):
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    lock_path = path.with_suffix(".lock")
    if lock_path.resolve() != lock_path:
        raise PlanError("Terminal publisher lock is redirected")
    until = clock() + 35
    with lock_path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = until - clock()
                if remaining <= 0:
                    raise PlanError("Bounded terminal publisher lock wait expired")
                sleep(min(0.05, remaining))
        yield


def _finish_guardian(plan, decision, jobs, scheduler, authorization_sha256, states=None):
    actual = _guardian_job(plan, authorization_sha256)
    path = _control(plan) / "manifests/terminal.json"
    with _terminal_lock(path):
        previous = _terminal_receipt(plan, authorization_sha256, actual)
        if previous is not None:
            return 0 if previous["complete"] else 2
        if decision.get("status") not in {"complete", "incomplete"}:
            raise PlanError("Only a terminal decision can be committed")
        cancellation_error = None
        if decision["status"] != "complete":
            ids = sorted({job for stage, job in jobs.items() if stage != "guardian"})
            try:
                scheduler.cancel(ids)
            except Exception as error:
                cancellation_error = type(error).__name__
        receipt = {**decision, "schema_version": 1, "complete": decision["status"] == "complete",
                   "scope_id": SCOPE, "run_id": plan["run_id"], "workflow_sha256": digest(plan),
                   "authorization_sha256": authorization_sha256, "job_ids": jobs,
                   "guardian_job_id": actual, "states": states or {},
                   "created_utc": utc_now().isoformat(), "automatic_retries": 0}
        if cancellation_error:
            receipt["cancellation_error_type"] = cancellation_error
        atomic_json(path, receipt)
        return 0 if receipt["complete"] else 2


def guardian(plan, authorization, scheduler=None):
    require_slurm()
    validate_plan(plan, check_time=False)
    validate_authorization(plan, authorization, current=False)
    scheduler = scheduler or Scheduler()
    control = _control(plan)
    authorization_sha = digest(authorization)
    actual = _guardian_job(plan, authorization_sha, scheduler, wait_for_ack=True)
    previous_terminal = _terminal_receipt(plan, authorization_sha, actual)
    if previous_terminal is not None:
        return 0 if previous_terminal["complete"] else 2
    end = timestamp(plan["config"]["deadlines"]["evaluation"]) + dt.timedelta(
        minutes=plan["config"]["guardian_grace_minutes"])
    stopped = []
    previous = {}

    def stop(signum, _frame):
        stopped.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, stop)
    jobs = {}
    try:
        while utc_now() < end and not stopped:
            try:
                _guardian_job(plan, authorization_sha)
                previous_terminal = _terminal_receipt(plan, authorization_sha, actual)
                if previous_terminal is not None:
                    return 0 if previous_terminal["complete"] else 2
                jobs = recorded_jobs(plan, authorization_sha)
                _bound_json(control / "workflow.json", digest(plan))
                _bound_json(control / "authorization.json", authorization_sha)
                launch = json.loads((control / "manifests/launch.json").read_text())
                if (launch.get("workflow_sha256") != digest(plan)
                        or launch.get("authorization_sha256") != authorization_sha):
                    raise PlanError("Guardian launch identity changed")
                states = scheduler.states(list(jobs.values()))
                statuses = stage_states(plan, jobs, states)
                evidence = (completion_evidence(plan, authorization, prepare_job_id=jobs.get("prepare"))
                            if statuses.get("aggregate") == "COMPLETED" else None)
                decision = guardian_decision(plan, statuses, evidence, launch=launch)
                if states.get(actual, {}).get("state") in FAILED:
                    decision = {"status": "incomplete", "reason": "guardian_allocation_failed"}
                elif decision["status"] == "complete" and states.get(actual, {}).get("state") != "RUNNING":
                    decision = {"status": "incomplete", "reason": "guardian_allocation_not_running"}
                if decision["status"] == "watching" and utc_now() >= timestamp(authorization["expires_at"]):
                    decision = {"status": "incomplete", "reason": "authorization_expired"}
                atomic_json(control / "manifests/guardian.json", {
                    "status": decision["status"], "workflow_sha256": digest(plan),
                    "authorization_sha256": authorization_sha, "job_id": os.environ["SLURM_JOB_ID"],
                    "checked_utc": utc_now().isoformat(), "job_ids": jobs, "stage_states": statuses})
                if decision["status"] != "watching":
                    reconciled = reconcile_intents(plan, scheduler, authorization_sha)
                    jobs = reconciled["job_ids"]
                    if reconciled["unresolved"]:
                        decision = {"status": "incomplete", "reason": "unresolved_submission_intent",
                                    "unresolved": reconciled["unresolved"]}
                    return _finish_guardian(plan, decision, jobs, scheduler, authorization_sha, states)
            except Exception as error:
                decision = {"status": "incomplete", "reason": "guardian_observation_failed",
                            "error_type": type(error).__name__, "detail": str(error)}
                return _finish_guardian(plan, decision, jobs, scheduler, authorization_sha)
            time.sleep(30)
        jobs = recorded_jobs(plan, authorization_sha)
        return _finish_guardian(plan, {"status": "incomplete",
                                "reason": "guardian_interrupted" if stopped else "guardian_bound_reached"},
                                jobs, scheduler, authorization_sha)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _stage_environment(plan, job_directory, gpu):
    source = Path(plan["config"]["source_root"])
    pinned = plan["config"]["environment"]
    dependency = _bound_json(source / "dependencies/overlay/complete.json",
                             pinned["dependency_manifest_sha256"])
    site = Path(dependency["site"]).resolve()
    if (dependency.get("complete") is not True or not site.is_relative_to((source / "dependencies").resolve())
            or not isinstance(dependency.get("files"), dict) or not dependency["files"]):
        raise PlanError("Original dependency overlay is not complete")
    for name, wanted in dependency["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PlanError("Invalid installed dependency path")
        path = site / name
        if not path.resolve().is_relative_to(site) or file_sha256(path) != wanted:
            raise PlanError("Pinned installed dependency changed")
    from pilots.topology_20260910.slurm.imports import prepare
    environment, imported = prepare(source, job_directory / "import_staging",
                                    pinned["import_bundle_sha256"])
    environment["PYTHONPATH"] = os.pathsep.join((plan["config"]["code_root"], str(site),
                                                environment.get("PYTHONPATH", "")))
    for name in ("bytecode", "numba", "torch_extensions", "torch", "work", "huggingface"):
        (job_directory / name).mkdir()
    environment.update({
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(job_directory / "bytecode"),
        "NUMBA_CACHE_DIR": str(job_directory / "numba"),
        "TORCH_EXTENSIONS_DIR": str(job_directory / "torch_extensions"),
        "TORCH_HOME": str(job_directory / "torch"), "TMPDIR": str(job_directory / "work"),
        "HF_HOME": str(job_directory / "huggingface"), "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    if not gpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment, imported


def _run_command(command, cwd, environment, stop_at):
    remaining = stop_at - time.time()
    if remaining <= 0:
        raise TimeoutError("Stage allocation, authorization or scientific cutoff reached")
    child = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True)
    handlers = {}

    def forward(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, forward)
    try:
        try:
            return child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            forward(signal.SIGTERM, None)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forward(signal.SIGKILL, None)
                child.wait(timeout=15)
            raise TimeoutError("Stage stopped at its allocation/authorization/deadline bound")
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _await_committed_launch(plan, authorization, stop_at):
    control = _control(plan)
    while time.time() < stop_at:
        if (control / "manifests/terminal.json").exists():
            raise PlanError("Terminal workflow cannot perform scientific work")
        launch = json.loads((control / "manifests/launch.json").read_text())
        if (launch.get("workflow_sha256") != digest(plan)
                or launch.get("authorization_sha256") != digest(authorization)):
            raise PlanError("Stage launch identity changed")
        if launch.get("status") == "submitted":
            jobs = recorded_jobs(plan, digest(authorization))
            if set(jobs) != set(ORDER) or launch.get("job_ids") != jobs:
                raise PlanError("Committed launch does not bind exactly eight durable submissions")
            return jobs
        if launch.get("status") != "submitting":
            raise PlanError("Static submission did not complete")
        limit = timestamp(launch["created_utc"]).timestamp() + plan["config"]["submission_grace_minutes"] * 60
        if time.time() + 5 >= min(limit, stop_at):
            break
        time.sleep(5)
    raise PlanError("Bounded submission-commit wait expired; no scientific work started")


def run_stage(plan, authorization, stage):
    require_slurm()
    validate_plan(plan, check_time=False)
    validate_authorization(plan, authorization, current=stage != "guardian")
    verify_release(plan, executing=True)
    if stage not in ORDER:
        raise PlanError("Undeclared stage")
    if stage == "guardian":
        return guardian(plan, authorization)
    started = time.time()
    control = _control(plan)
    if (control / "manifests/terminal.json").exists():
        raise PlanError("Terminal workflow cannot perform more scientific work")
    row = plan["stages"][stage]
    raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if len(row["tasks"]) > 1:
        if raw_index is None or not raw_index.isdigit() or not 0 <= int(raw_index) < len(row["tasks"]):
            raise PlanError("The exact declared array index is required")
        index = int(raw_index)
    else:
        if raw_index is not None:
            raise PlanError("A singleton stage must not be expanded into an array")
        index = 0
    cutoff = timestamp(plan["config"]["deadlines"][row["cutoff"]]).timestamp()
    if started + row["minutes"] * 60 >= cutoff:
        raise PlanError("Queue delay left insufficient time for the full stage cap")
    stop_at = min(started + row["minutes"] * 60, cutoff, timestamp(authorization["expires_at"]).timestamp())
    jobs = _await_committed_launch(plan, authorization, stop_at)
    actual_job = os.environ.get("SLURM_ARRAY_JOB_ID") if len(row["tasks"]) > 1 else os.environ["SLURM_JOB_ID"]
    if actual_job != jobs[stage]:
        raise PlanError("Stage allocation is not bound to its durable submission intent")
    if stage == "fits":
        verify_readiness(plan, jobs["prepare"])
    for attempt in range(7):
        heartbeat = control / "manifests/guardian.json"
        if heartbeat.is_file():
            value = json.loads(heartbeat.read_text())
            if (value.get("status") == "watching" and value.get("workflow_sha256") == digest(plan)
                    and value.get("authorization_sha256") == digest(authorization)
                    and value.get("job_id") == recorded_jobs(plan, digest(authorization)).get("guardian")
                    and 0 <= time.time() - timestamp(value["checked_utc"]).timestamp() <= 90):
                break
        if attempt == 6 or time.time() + 10 >= stop_at:
            raise PlanError("Independent guardian is not live; refusing scientific work")
        time.sleep(10)
    task = row["tasks"][index]
    key = f"{stage}_{actual_job}_{index}"
    directory = control / "job_work" / key
    directory.mkdir(parents=True, exist_ok=False)
    receipt_path = control / "manifests/execution" / (key + ".json")
    receipt = {"status": "running", "workflow_sha256": digest(plan), "authorization_sha256": digest(authorization),
               "stage": stage, "task": task["key"], "job_id": actual_job, "array_index": index,
               "commands": task["commands"], "started_utc": utc_now().isoformat()}
    atomic_json(receipt_path, receipt)
    try:
        environment, imported = _stage_environment(plan, directory, row["gpu"])
        receipt["import_staging"] = imported
        atomic_json(receipt_path, receipt)
        for command in task["commands"]:
            validate_authorization(plan, authorization)
            code = _run_command(command, plan["config"]["code_root"], environment, stop_at)
            if code != 0:
                raise RuntimeError("Declared stage command failed with exit " + str(code))
        if stage == "prepare":
            verify_readiness(plan, jobs["prepare"])
        receipt.update(status="completed", exit_code=0, completed_utc=utc_now().isoformat())
        atomic_json(receipt_path, receipt)
        return 0
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, detail=str(error),
                       completed_utc=utc_now().isoformat())
        atomic_json(receipt_path, receipt)
        raise
    finally:
        shutil.rmtree(directory)


def main(argv=None, *, scheduler=None, now=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="plan",
                        choices=("plan", "validate", "reconcile", "run-stage"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--stage", choices=ORDER)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.submit and args.out:
        parser.error("--submit cannot be combined with --out; freeze and review the plan first")
    if bool(args.config) == bool(args.workflow):
        parser.error("Supply exactly one of --config and --workflow")
    if args.mode != "plan" and not args.workflow:
        parser.error("This mode requires a frozen --workflow and --sha256")
    if args.workflow:
        if not args.sha256:
            parser.error("--workflow requires its reviewed --sha256")
        plan = _bound_json(args.workflow, args.sha256)
    else:
        plan = build_plan(json.loads(args.config.read_text()), now=now)
    if args.mode in {"run-stage", "reconcile"} and (args.submit or args.out):
        parser.error("Internal execution/reconciliation never submits or rewrites workflows")
    validate_plan(plan, now=now, check_time=args.mode in {"plan", "validate"})
    authorization = None
    if args.authorization:
        authorization = (_bound_json(args.authorization, args.authorization_sha256)
                         if args.authorization_sha256 else json.loads(args.authorization.read_text()))
    if args.mode == "run-stage":
        if not args.authorization_sha256 or not args.stage:
            parser.error("Allocated execution requires bound authorization and a declared stage")
        return run_stage(plan, authorization, args.stage)
    if args.mode == "reconcile":
        if not args.authorization_sha256:
            parser.error("Reconciliation requires the original authorization checksum")
        validate_authorization(plan, authorization, current=False)
        result = reconcile_intents(plan, scheduler or Scheduler(), digest(authorization))
        print(json.dumps(result, sort_keys=True))
        return 2 if result["unresolved"] else 0
    if args.submit:
        if authorization is None:
            parser.error("--submit requires a separate approved --authorization JSON")
        result = submit_plan(plan, authorization, scheduler, now=now)
    else:
        if authorization is not None:
            validate_authorization(plan, authorization, now=now)
        result = {"status": "validated_not_submitted", "workflow_sha256": digest(plan),
                  "plan": plan, "authorization_template": authorization_template(plan)}
    if args.out:
        target = args.out.resolve()
        for name in ("root", "source_root"):
            protected = Path(plan["config"][name])
            if target == protected or target.is_relative_to(protected):
                raise PlanError("Do not write planning artifacts inside scientific or historical roots")
        _frozen_json(target, plan)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
