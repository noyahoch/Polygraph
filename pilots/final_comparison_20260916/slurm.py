"""Approval-bound final-comparison DAG; the default CLI only prints a plan.

Two independent, bounded CPU guards surround the static DAG. The first guard
locks the six phase receipts; its afterany guard alone can certify success after
checking the first guard's accounting history. Neither guard submits retries.
Submission intents, allocation ACK handling and terminal publication preserve
the tested replication operator's fail-closed semantics.

Prepare a preflight configuration only after SOURCE READY. Benchmark configuration
additionally requires the resulting measured receipt, explicit new GPU/CPU
budgets and prospective cutoffs. No default budget inherits an older campaign.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import getpass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import sys
import time

from pilots.layer_ensemble_20260914.slurm import replicate as legacy
from pilots.layer_ensemble_20260914.slurm.runtime import atomic_json, file_sha256, require_slurm

PlanError = legacy.PlanError
AmbiguousSubmission = legacy.AmbiguousSubmission
json_bytes = legacy.json_bytes
digest = legacy.digest
timestamp = legacy.timestamp
utc_now = legacy.utc_now
integer = legacy.integer
absolute = legacy.absolute
_terminal_lock = legacy._terminal_lock
_run_command = legacy._run_command
HEX = legacy.HEX
SAFE = legacy.SAFE
FAILED = legacy.FAILED
ACTIVE = legacy.ACTIVE

SCOPE = "final_comparison_20260916_fixed20"
MODULE = "pilots.final_comparison_20260916."
SELF = "pilots/final_comparison_20260916/slurm.py"
# Mirrors protocol.SEEDS (amended 2026-09-16 to three seeds); this submit-host module stays import-light.
SEEDS = (7, 17, 27)
IMPORTED_SEEDS = (7, 17, 27)
ARMS = ("block2", "block5", "block8", "block11")
METHODS = tuple(f"{family}_{name}" for family in ("G", "H", "S")
                for name in (*ARMS, "mean", "stack", "last_only")) + ("O", "L", "MSP", "entropy")
CONTRASTS = (("G_stack", "S_stack"), ("G_mean", "S_mean"),
             ("G_stack", "H_stack"), ("G_mean", "H_mean"),
             ("G_stack", "O"), ("G_mean", "O"))
GUARDS = ("guardian", "failure_guard")
PHASES = {
    "preparation": ("prepare", "hidden_full"),
    "bases": ("gpu_fits", "cpu_fits", "linear_fit", "base_freeze"),
    "meta": ("meta_gpu", "meta_import", "heads"),
    "gate": ("evaluation_gate",),
    "predictions": ("dev_gpu", "dev_import", "dev_o", "static_export"),
    "evaluation": ("analysis",),
}
ORDER = (*GUARDS, *(name for group in PHASES.values() for name in group))
PREFLIGHT_ORDER = (*GUARDS, "cpu_validation", "gpu_preflight")
PRODUCTION_MODULES = ("protocol", "freeze", "data", "models", "extract", "train", "predict",
                      "preflight", "heads", "linear", "diagnostics", "evaluate", "slurm")
REQUIRED_SOURCE = {
    *("pilots/final_comparison_20260916/" + name + ".py" for name in PRODUCTION_MODULES),
    "pilots/final_comparison_20260916/__init__.py",
    "pilots/final_comparison_20260916/test_slurm.py",
    "pilots/layer_ensemble_20260914/slurm/replicate.py",
    "pilots/layer_ensemble_20260914/slurm/runtime.py",
    "pilots/topology_20260910/slurm/imports.py",
}
MINIMUM_EVALUATION_FILES = {"complete.json", "report.json", "scores.npz"}
RESOURCE_CLASS = {
    "account": "gpu-students", "gpu_partition": "studentbatch",
    "cpu_partition": "cpu-killable", "gpu_constraint": "geforce_rtx_2080",
    "gpu_cpus": 6, "gpu_memory_mb": 32000,
}


def neural_matrix():
    return [{"family": family, "arm": arm, "seed": seed}
            for family in ("G", "H", "S", "O") for seed in SEEDS
            for arm in (("logits",) if family == "O" else ARMS)]


def is_import(row):
    return row["family"] == "G" and row["seed"] in IMPORTED_SEEDS


NEURAL_FITS = len(neural_matrix())
IMPORTED_NEURAL_FITS = sum(map(is_import, neural_matrix()))
NEW_NEURAL_FITS = NEURAL_FITS - IMPORTED_NEURAL_FITS
HEAD_NAMES = ("stack", "last_only")
META_HEADS = 3 * len(SEEDS) * len(HEAD_NAMES)
NEW_META_HEADS = META_HEADS - len(IMPORTED_SEEDS) * len(HEAD_NAMES)


def _sha(value, label):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise PlanError("An explicit SHA256 is required: " + label)
    return value


def _relative(name):
    if not isinstance(name, str) or not name:
        raise PlanError("Unsafe relative artifact/source path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise PlanError("Unsafe relative artifact/source path")
    return path


def _config(config):
    fields = {
        "mode", "approval_id", "run_id", "root", "control_root", "code_root", "pilot_root",
        "replication_root", "cache", "data_root", "python", "source_sha256", "environment",
        "resources", "caps", "deadlines", "max_concurrent_gpus", "external_reserved_gpus",
        "gpu_budget_minutes", "cpu_budget_minutes", "external_gpu_minutes", "external_cpu_minutes",
        "submission_grace_minutes", "guardian_grace_minutes", "campaign_binding", "preflight",
        "test_selectors", "evaluation_files",
    }
    if not isinstance(config, dict) or set(config) != fields:
        raise PlanError("Configuration fields must exactly match the documented planner contract")
    value = copy.deepcopy(config)
    if value["mode"] not in ("preflight", "benchmark"):
        raise PlanError("Choose bounded preflight or separately approved benchmark")
    for name in ("approval_id", "run_id"):
        if not isinstance(value[name], str) or not SAFE.fullmatch(value[name]):
            raise PlanError("An explicit safe and unique " + name + " is required")
    paths = {name: absolute(value[name], name) for name in
             ("root", "control_root", "code_root", "pilot_root", "replication_root",
              "cache", "data_root", "python")}
    for new in ("root", "control_root"):
        for old in ("pilot_root", "replication_root", "cache", "data_root"):
            if not legacy._separate(paths[new], paths[old]):
                raise PlanError("New control/scientific roots must not overlap historical inputs")
    if not legacy._separate(paths["root"], paths["control_root"]):
        raise PlanError("Scientific and control roots must be disjoint")
    if (paths["code_root"].parent != paths["control_root"] / "releases"
            or not SAFE.fullmatch(paths["code_root"].name)):
        raise PlanError("Use a sealed control_root/releases/RELEASE namespace")
    if paths["python"] != paths["pilot_root"] / "env/bin/python":
        raise PlanError("Use the existing pilot's pinned Python; no environment upgrade")
    inventory = value["source_sha256"]
    if not isinstance(inventory, dict) or not REQUIRED_SOURCE <= set(inventory):
        raise PlanError("Missing required sealed source files")
    for name, sha in inventory.items():
        if _relative(name).suffix not in {".py", ".json", ".toml", ".txt", ".md", ".sh", ".sbatch"}:
            raise PlanError("Only source/configuration files belong in a release")
        _sha(sha, name)
    env = value["environment"]
    if not isinstance(env, dict) or set(env) != {
        "import_bundle_sha256", "dependency_manifest_sha256", "hf_hub_cache",
    }:
        raise PlanError("Pin the existing overlay, import bundle and offline model-cache path")
    for name in ("import_bundle_sha256", "dependency_manifest_sha256"):
        _sha(env[name], name)
    absolute(env["hf_hub_cache"], "offline HF model cache")
    resources = value["resources"]
    if (not isinstance(resources, dict)
            or set(resources) != set(RESOURCE_CLASS) | {"cpu_cpus", "cpu_memory_mb"}
            or any(resources.get(k) != v for k, v in RESOURCE_CLASS.items())):
        raise PlanError("Only the observed six-CPU/32000M existing RTX2080 resource class is allowed")
    integer(resources["cpu_cpus"], "cpu_cpus", maximum=6)
    integer(resources["cpu_memory_mb"], "cpu_memory_mb", maximum=32000)
    integer(value["max_concurrent_gpus"], "max_concurrent_gpus", maximum=8)
    integer(value["external_reserved_gpus"], "external_reserved_gpus", minimum=0, maximum=7)
    if value["max_concurrent_gpus"] + value["external_reserved_gpus"] > 8:
        raise PlanError("Workflow and external reservations exceed eight concurrent GPUs")
    for name in ("gpu_budget_minutes", "cpu_budget_minutes", "external_gpu_minutes", "external_cpu_minutes"):
        integer(value[name], name, minimum=0, maximum=10**9)
    integer(value["submission_grace_minutes"], "submission_grace_minutes", maximum=10)
    integer(value["guardian_grace_minutes"], "guardian_grace_minutes", maximum=30)
    order = PREFLIGHT_ORDER if value["mode"] == "preflight" else ORDER
    if not isinstance(value["caps"], dict) or set(value["caps"]) != set(order):
        raise PlanError("Every declared stage needs its own explicit allocation cap")
    for name, minutes in value["caps"].items():
        integer(minutes, name + " minutes", maximum=7200 if name in GUARDS else 4320)
    deadline_keys = ("validation",) if value["mode"] == "preflight" else ("base", "predictions", "evaluation")
    if not isinstance(value["deadlines"], dict) or set(value["deadlines"]) != set(deadline_keys):
        raise PlanError("Explicit prospective stage cutoffs are required")
    deadlines = [timestamp(value["deadlines"][name]) for name in deadline_keys]
    if deadlines != sorted(set(deadlines)):
        raise PlanError("Cutoffs must be strictly ordered")
    selectors = value["test_selectors"]
    if not isinstance(selectors, list) or len(selectors) != len(set(selectors)):
        raise PlanError("Test selectors must be an explicit unique list")
    for selector in selectors:
        if (not isinstance(selector, str)
                or not re.fullmatch(r"pilots\.[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*\.test_[a-zA-Z0-9_]+", selector)
                or selector.replace(".", "/") + ".py" not in inventory):
            raise PlanError("Synthetic validation selectors must be source-bound test modules")
    files = value["evaluation_files"]
    if not isinstance(files, list) or len(files) != len(set(files)):
        raise PlanError("Evaluation inventory must be an explicit unique list")
    for name in files:
        _relative(name)
    if value["mode"] == "preflight":
        if (value["campaign_binding"] is not None or value["preflight"] is not None
                or files or MODULE + "test_slurm" not in selectors):
            raise PlanError("Preflight prepares a static campaign; it cannot authorize scientific execution")
        if (value["caps"]["gpu_preflight"] > 30 or value["max_concurrent_gpus"] != 1
                or value["caps"]["cpu_validation"] > 60 or value["caps"]["guardian"] > 180
                or value["caps"]["failure_guard"] > 30):
            raise PlanError("Preflight is bounded: one GPU <=30min, CPU validation <=60min, guards <=180/30min")
    else:
        if selectors or not MINIMUM_EVALUATION_FILES <= set(files):
            raise PlanError("Scientific launch requires the full declared evaluation inventory, not more tests")
        binding = value["campaign_binding"]
        if not isinstance(binding, dict) or set(binding) != {"campaign_sha256", "roles_sha256", "reuse_sha256"}:
            raise PlanError("Bind the prepared campaign, original roles and complete reuse registry")
        for name, sha in binding.items():
            _sha(sha, name)
        preflight = value["preflight"]
        if not isinstance(preflight, dict) or set(preflight) != {
            "terminal_path", "terminal_sha256", "workflow_sha256", "reserved_gpu_minutes", "allocated_cpu_minutes",
        }:
            raise PlanError("A measured, source-matched allocated preflight receipt is required")
        absolute(preflight["terminal_path"], "preflight terminal")
        _sha(preflight["terminal_sha256"], "preflight terminal")
        _sha(preflight["workflow_sha256"], "preflight workflow")
        integer(preflight["reserved_gpu_minutes"], "preflight GPU reservation", maximum=30)
        integer(preflight["allocated_cpu_minutes"], "preflight CPU reservation", maximum=10000)
    return value


def _stages(config):
    root, python = config["root"], config["python"]
    cap, resources = config["max_concurrent_gpus"], config["resources"]

    def command(module, *args):
        return [python, "-B", "-u", "-m", MODULE + module, *map(str, args)]

    def task(key, commands, **identity):
        return {"key": key, "commands": commands, **identity}

    def stage(tasks, *, gpu=False, cutoff="base", dependencies=(), dependency_type="afterok", throttle=1):
        if not tasks:
            raise PlanError("A declared stage has no tasks; the fixed seed/import matrix is inconsistent")
        return {"tasks": tasks, "gpu": gpu, "cutoff": cutoff, "dependencies": list(dependencies),
                "dependency_type": dependency_type, "throttle": throttle,
                "cpus": resources["gpu_cpus"] if gpu else resources["cpu_cpus"],
                "memory_mb": resources["gpu_memory_mb"] if gpu else resources["cpu_memory_mb"]}

    final = "validation" if config["mode"] == "preflight" else "evaluation"
    stages = {
        "guardian": stage([task("guardian", [])], cutoff=final),
        "failure_guard": stage([task("failure_guard", [])], cutoff=final,
                               dependencies=("guardian",), dependency_type="afterany"),
    }
    for name in GUARDS:
        stages[name]["cpus"] = 1
    prepare = command("protocol", "--root", root, "--cache", config["cache"],
                      "--pilot-root", config["pilot_root"], "--replication-root", config["replication_root"],
                      "--run-id", config["run_id"])
    if config["mode"] == "preflight":
        stages["cpu_validation"] = stage([task("cpu_validation", [
            [python, "-B", "-u", "-m", "unittest", *config["test_selectors"]], prepare,
        ])], cutoff="validation", dependencies=("guardian",), dependency_type="after")
        stages["gpu_preflight"] = stage([task("gpu_preflight", [
            command("preflight", "--root", root, "--data-root", config["data_root"],
                    "--out", root + "/preflight.json"),
        ])], gpu=True, cutoff="validation", dependencies=("cpu_validation",))
    else:
        stages["prepare"] = stage([task("prepare", [prepare])],
                                  dependencies=("guardian",), dependency_type="after")
        stages["hidden_full"] = stage([task("hidden_full", [
            command("extract", "--root", root, "--data-root", config["data_root"], "--mode", "full"),
        ])], gpu=True, dependencies=("prepare",))
        missing = [row for row in neural_matrix() if not is_import(row)]
        for name, gpu in (("gpu_fits", True), ("cpu_fits", False)):
            rows = [row for row in missing if (row["family"] != "O") == gpu]
            tasks = [task(f"{row['family']}_{row['arm']}_seed{row['seed']}", [
                command("train", "--root", root, "--family", row["family"], "--arm", row["arm"],
                        "--seed", row["seed"], "--device", "cuda" if gpu else "cpu"),
            ], **row) for row in rows]
            stages[name] = stage(tasks, gpu=gpu, dependencies=("hidden_full",),
                                 throttle=cap if gpu else 5)
        stages["linear_fit"] = stage([task("linear_fit", [
            command("linear", "fit", "--root", root),
        ])], dependencies=("hidden_full",))
        stages["base_freeze"] = stage([task("base_freeze", [
            command("freeze", "base", "--root", root), command("diagnostics", "--root", root),
        ])], dependencies=("gpu_fits", "cpu_fits", "linear_fit"))
        for role, predecessor in (("meta", "base_freeze"), ("dev_eval", "evaluation_gate")):
            prefix = "meta" if role == "meta" else "dev"
            for suffix, imported in (("gpu", False), ("import", True)):
                rows = [{"family": family, "seed": seed}
                        for family in ("G", "H", "S") for seed in SEEDS
                        if is_import({"family": family, "seed": seed}) == imported]
                tasks = [task(f"{row['family']}_seed{row['seed']}", [
                    command("predict", "--root", root, "--role", role,
                            "--family", row["family"], "--seed", row["seed"]),
                ], imported=imported, **row) for row in rows]
                stages[prefix + "_" + suffix] = stage(tasks, gpu=not imported, cutoff="predictions",
                                                      dependencies=(predecessor,),
                                                      throttle=3 if imported else cap)
        stages["heads"] = stage([
            task(f"{family}_seed{seed}", [
                command("heads", "--root", root, "--family", family, "--seed", seed),
            ], family=family, seed=seed, imported=is_import({"family": family, "seed": seed}))
            for family in ("G", "H", "S") for seed in SEEDS
        ], cutoff="predictions", dependencies=("meta_gpu", "meta_import"), throttle=5)
        stages["evaluation_gate"] = stage([task("evaluation_gate", [
            command("freeze", "heads", "--root", root), command("freeze", "evaluation", "--root", root),
        ])], cutoff="predictions", dependencies=("heads",))
        stages["dev_o"] = stage([task(f"O_seed{seed}", [
            command("predict", "--root", root, "--role", "dev_eval", "--family", "O", "--seed", seed),
        ], family="O", seed=seed) for seed in SEEDS], cutoff="predictions",
            dependencies=("evaluation_gate",), throttle=5)
        stages["static_export"] = stage([task("static_export", [
            command("linear", "export", "--root", root),
        ])], cutoff="predictions", dependencies=("evaluation_gate",))
        stages["analysis"] = stage([task("analysis", [
            command("evaluate", "--root", root, "--out", root + "/evaluation"),
        ])], cutoff="evaluation", dependencies=("dev_gpu", "dev_import", "dev_o", "static_export"))
    order = PREFLIGHT_ORDER if config["mode"] == "preflight" else ORDER
    for name in order:
        stages[name]["minutes"] = config["caps"][name]
    return {name: stages[name] for name in order}


def build_plan(config, *, now=None, check_time=True):
    """Pure planning; no scientific imports, filesystem writes or scheduler calls."""
    config = _config(config)
    stages = _stages(config)
    gpu = sum(row["minutes"] * len(row["tasks"]) for row in stages.values() if row["gpu"])
    cpu = sum(row["minutes"] * row["cpus"] * len(row["tasks"]) for row in stages.values())
    prior = config["preflight"] or {"reserved_gpu_minutes": 0, "allocated_cpu_minutes": 0}
    total_gpu = gpu + prior["reserved_gpu_minutes"] + config["external_gpu_minutes"]
    total_cpu = cpu + prior["allocated_cpu_minutes"] + config["external_cpu_minutes"]
    if total_gpu > config["gpu_budget_minutes"] or total_cpu > config["cpu_budget_minutes"]:
        raise PlanError("All array tasks, both guards, preflight and external reservations must fit the NEW budgets")
    critical, finish = {}, {name: 0 for name in GUARDS}
    for name, row in stages.items():
        if name in GUARDS:
            continue
        finish[name] = max((finish[dep] for dep in row["dependencies"]), default=0) + (
            math.ceil(len(row["tasks"]) / row["throttle"]) * row["minutes"])
        critical[row["cutoff"]] = max(critical.get(row["cutoff"], 0), finish[name])
    plan = {
        "schema_version": 1, "scope_id": SCOPE, "mode": config["mode"],
        "run_id": config["run_id"], "approval_id": config["approval_id"], "config": config,
        "matrix": neural_matrix(), "imports": [row for row in neural_matrix() if is_import(row)],
        "new_matrix": [row for row in neural_matrix() if not is_import(row)],
        "epochs": 20, "neural_fits": NEURAL_FITS, "imported_neural_fits": IMPORTED_NEURAL_FITS,
        "new_neural_fits": NEW_NEURAL_FITS, "linear_fits": 1, "meta_heads": META_HEADS,
        "new_meta_heads": NEW_META_HEADS,
        "reported_methods": list(METHODS), "primary_contrasts": [list(row) for row in CONTRASTS],
        "original_test_access": False, "automatic_retries": 0,
        "order": list(stages), "stages": stages,
        "phases": {name: list(group) for name, group in PHASES.items()} if config["mode"] == "benchmark" else {},
        "critical_path_cap_minutes": critical, "release_sha256": digest(config["source_sha256"]),
        "resource_claims": {
            "resources": config["resources"], "caps": config["caps"],
            "gpu_allocation_counts": {name: len(row["tasks"]) for name, row in stages.items() if row["gpu"]},
            "workflow_gpu_minutes": gpu, "workflow_cpu_minutes": cpu,
            "preflight_gpu_minutes": prior["reserved_gpu_minutes"],
            "preflight_cpu_minutes": prior["allocated_cpu_minutes"],
            "external_gpu_minutes": config["external_gpu_minutes"],
            "external_cpu_minutes": config["external_cpu_minutes"],
            "reserved_gpu_minutes": total_gpu, "allocated_cpu_minutes": total_cpu,
            "gpu_budget_minutes": config["gpu_budget_minutes"], "cpu_budget_minutes": config["cpu_budget_minutes"],
            "max_concurrent_gpus": config["max_concurrent_gpus"],
            "external_reserved_gpus": config["external_reserved_gpus"],
            "gpu_ceiling_including_external": config["max_concurrent_gpus"] + config["external_reserved_gpus"],
            "submission_count": len(stages), "allocated_task_count": sum(len(row["tasks"]) for row in stages.values()),
        },
    }
    if check_time:
        _admit_time(plan, utc_now(now))
    return plan


def _admit_time(plan, now):
    config = plan["config"]
    for key, minutes in plan["critical_path_cap_minutes"].items():
        if now + dt.timedelta(minutes=minutes + config["submission_grace_minutes"]) >= timestamp(config["deadlines"][key]):
            raise PlanError("Insufficient prospective time for all allocation caps before " + key)
    end = max(map(timestamp, config["deadlines"].values())) + dt.timedelta(minutes=config["guardian_grace_minutes"])
    if now + dt.timedelta(minutes=config["caps"]["guardian"]) < end:
        raise PlanError("CPU guardian must cover the final cutoff plus its explicit grace")


def authorization_template(plan):
    return {
        "schema_version": 1, "scope_id": SCOPE, "approval_id": plan["approval_id"], "run_id": plan["run_id"],
        "mode": plan["mode"], "workflow_sha256": digest(plan), "release_sha256": plan["release_sha256"],
        "matrix_sha256": digest(plan["matrix"]), "campaign_binding": plan["config"]["campaign_binding"],
        "resource_limits": plan["resource_claims"], "deadlines": plan["config"]["deadlines"],
        "preflight": plan["config"]["preflight"], "original_test_access": False, "automatic_retries": 0,
        "approved": False, "source_ready_at": None, "full_ready_at": None,
        "measured_resource_approval": False, "authorized_at": None, "expires_at": None,
    }


def validate_authorization(plan, authorization, *, now=None, current=True):
    template = authorization_template(plan)
    mutable = {"approved", "source_ready_at", "full_ready_at", "measured_resource_approval", "authorized_at", "expires_at"}
    if (not isinstance(authorization, dict) or set(authorization) != set(template)
            or any(authorization.get(k) != v for k, v in template.items() if k not in mutable)
            or authorization.get("approved") is not True):
        raise PlanError("Separate explicit approval must bind source, matrix, reuse, roles, resources and cutoffs")
    issued, expires = timestamp(authorization["authorized_at"]), timestamp(authorization["expires_at"])
    source_ready = timestamp(authorization["source_ready_at"])
    if (source_ready > issued or issued >= min(map(timestamp, plan["config"]["deadlines"].values()))
            or expires != max(map(timestamp, plan["config"]["deadlines"].values()))):
        raise PlanError("SOURCE READY and prospective execution cutoffs must precede submission")
    if plan["mode"] == "benchmark":
        if (timestamp(authorization["full_ready_at"]) > issued
                or authorization["measured_resource_approval"] is not True):
            raise PlanError("Full scientific launch requires FULL READY and explicit measured resource approval")
    elif authorization["full_ready_at"] is not None or authorization["measured_resource_approval"] is not False:
        raise PlanError("Implementation preflight cannot claim scientific-launch approval")
    if current and not issued <= utc_now(now) < expires:
        raise PlanError("Approval is not currently active")
    return authorization


def validate_plan(plan, authorization=None, *, now=None, check_time=True):
    if not isinstance(plan, dict) or plan != build_plan(plan.get("config"), now=now, check_time=check_time):
        raise PlanError("Plan differs from the exact fixed comparison DAG")
    if authorization is not None:
        validate_authorization(plan, authorization, now=now, current=check_time)
    return plan


def _control(plan):
    return Path(plan["config"]["control_root"])


def _bound_json(path, expected):
    path = Path(path)
    if path.resolve() != path or not path.is_file() or file_sha256(path) != _sha(expected, str(path)):
        raise PlanError("Missing, redirected or changed hash-bound JSON: " + str(path))
    return json.loads(path.read_text())


def _frozen_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _terminal_lock(path):
        if path.exists():
            if path.resolve() != path or path.read_bytes() != json_bytes(value):
                raise PlanError("Refusing to overwrite a frozen artifact: " + str(path))
        else:
            atomic_json(path, value)


def verify_release(plan, *, executing=False):
    config = plan["config"]
    for name in ("root", "control_root", "code_root", "pilot_root", "replication_root", "cache", "data_root"):
        if Path(config[name]).resolve() != Path(config[name]):
            raise PlanError("Redirected namespace: " + name)
    release = Path(config["code_root"])
    if executing and Path(__file__).resolve() != release / SELF:
        raise PlanError("Runner is not executing from the sealed approved release")
    manifest_path = release / "source_manifest.json"
    if manifest_path.resolve() != manifest_path:
        raise PlanError("Redirected release manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_sha256") != config["source_sha256"] or manifest.get("release_id") != release.name:
        raise PlanError("Release differs from its exact approved source inventory")
    present = {str(path.relative_to(release)) for path in release.rglob("*")
               if path.is_file() and path != manifest_path and "__pycache__" not in path.parts
               and path != release / "source_manifest.json.lock"}
    if present != set(config["source_sha256"]):
        raise PlanError("Uninventoried or missing source in the sealed release")
    for name, sha in config["source_sha256"].items():
        path = release / name
        if path.stat().st_size > 1024 * 1024 and not os.environ.get("SLURM_JOB_ID"):
            raise PlanError("Heavy source verification belongs in Slurm")
        if path.resolve() != path or file_sha256(path) != sha:
            raise PlanError("Changed sealed source: " + name)


def campaign_binding(root):
    root = Path(root)
    campaign = json.loads((root / "campaign.json").read_text())
    return {"campaign_sha256": file_sha256(root / "campaign.json"),
            "roles_sha256": file_sha256(root / "role_map.json"),
            "reuse_sha256": digest({"reuse": campaign["reuse"], "reuse_groups": campaign["reuse_groups"]})}


def verify_launch_inputs(plan):
    """Metadata/source only on the submit host; old numeric verification is allocated."""
    if plan["mode"] == "preflight":
        if (Path(plan["config"]["root"]) / "execution.json").exists():
            raise PlanError("An already authorized scientific root cannot become a new preflight")
        return
    config = plan["config"]
    root = Path(config["root"])
    binding = config["campaign_binding"]
    campaign = _bound_json(root / "campaign.json", binding["campaign_sha256"])
    if campaign_binding(root) != binding:
        raise PlanError("Prepared campaign/role/reuse bytes changed")
    if (campaign.get("run_id") != plan["run_id"] or campaign.get("root") != str(root)
            or campaign.get("scope_id") != SCOPE or campaign.get("neural_matrix") != neural_matrix()
            or campaign.get("missing_matrix") != plan["new_matrix"]
            or campaign.get("original_test_access") is not False
            or set(campaign.get("reuse", {})) != {
                f"G/{arm}/seed{seed}" for seed in IMPORTED_SEEDS for arm in ARMS}):
        raise PlanError("Prepared scientific identity or imported-G reuse registry differs")
    sources = campaign.get("source_identity", {})
    if (not sources or any(config["source_sha256"].get(name) != sha for name, sha in sources.items())
            or any("pilots/final_comparison_20260916/" + name + ".py" not in sources
                   for name in PRODUCTION_MODULES)):
        raise PlanError("Prepared campaign source differs from this sealed release")
    prior = config["preflight"]
    receipt = _bound_json(prior["terminal_path"], prior["terminal_sha256"])
    if (receipt.get("complete") is not True or receipt.get("mode") != "preflight"
            or receipt.get("workflow_sha256") != prior["workflow_sha256"]
            or receipt.get("release_sha256") != plan["release_sha256"]
            or receipt.get("run_id") != plan["run_id"]
            or receipt.get("evidence", {}).get("campaign_binding") != binding
            or receipt.get("resource_claims", {}).get("reserved_gpu_minutes") != prior["reserved_gpu_minutes"]
            or receipt.get("resource_claims", {}).get("allocated_cpu_minutes") != prior["allocated_cpu_minutes"]):
        raise PlanError("Measured allocated preflight does not certify this source/campaign/resource reservation")


def execution_record(plan, authorization):
    return {
        "schema_version": 1, "scope_id": SCOPE, "run_id": plan["run_id"], "approved": True,
        "approval_id": plan["approval_id"], "campaign_sha256": plan["config"]["campaign_binding"]["campaign_sha256"],
        "authorized_at": authorization["authorized_at"], "deadlines": plan["config"]["deadlines"],
        "resource_approval": plan["resource_claims"], "workflow_sha256": digest(plan),
        "authorization_sha256": digest(authorization), "source_sha256": plan["release_sha256"],
        "original_test_access": False, "automatic_retries": 0,
    }


def _name(plan, stage):
    return "omri-final-" + plan["run_id"][:20] + "-" + stage + "-" + digest(plan)[:12]


def _comment(plan, stage):
    return "polygraph-final:" + digest(plan) + ":" + stage


def sbatch_command(plan, stage, jobs, authorization_sha256):
    if stage not in plan["order"]:
        raise PlanError("Undeclared stage")
    config, row, control = plan["config"], plan["stages"][stage], _control(plan)
    resources = config["resources"]
    runner = ["/usr/bin/python3", "-B", "-u", "-m", MODULE + "slurm", "run-stage",
              "--workflow", str(control / "workflow.json"), "--sha256", digest(plan),
              "--authorization", str(control / "authorization.json"), "--authorization-sha256",
              authorization_sha256, "--stage", stage]
    result = [
        "sbatch", "--parsable", "--account=" + resources["account"],
        "--partition=" + resources["gpu_partition" if row["gpu"] else "cpu_partition"],
        "--cpus-per-task=" + str(row["cpus"]), "--mem=" + str(row["memory_mb"]) + "M",
        "--time=" + str(row["minutes"]), "--job-name=" + _name(plan, stage),
        "--comment=" + _comment(plan, stage), "--no-requeue", "--kill-on-invalid-dep=yes",
        "--export=NONE", "--chdir=" + config["code_root"],
        "--output=" + str(control / "logs" / (stage + "_%A_%a.out")),
        "--error=" + str(control / "logs" / (stage + "_%A_%a.err")),
    ]
    if row["gpu"]:
        result += ["--gpus=1", "--constraint=geforce_rtx_2080"]
    if len(row["tasks"]) > 1:
        result += [f"--array=0-{len(row['tasks']) - 1}%{row['throttle']}"]
    if row["dependencies"]:
        predecessors = [jobs.get(name, "") for name in row["dependencies"]]
        if any(not isinstance(job, str) or not job.isdigit() for job in predecessors):
            raise PlanError("Every dependency needs its durable recorded job ID")
        result += ["--dependency=" + row["dependency_type"] + ":" + ":".join(predecessors)]
    return [*result, "--wrap", "exec " + shlex.join(runner)]


class Scheduler(legacy.Scheduler):
    """Reuse finite submission/accounting calls; add conservative external-GPU admission."""

    # Slurm job IDs on this cluster are reused (e.g. 898776-898778 also name another user's
    # 2025 FAILED jobs). Before slurmdbd records a fresh job, an unfiltered `sacct --jobs`
    # returns that historical row, which falsely failed preflight 898775. Accounting is
    # therefore restricted to this user and a bounded recent window, and foreign rows ignored.
    ACCOUNTING_WINDOW = "now-8days"

    def states(self, ids):
        if not ids:
            return {}
        user = getpass.getuser()
        output = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--array",
                            "--user=" + user, "--starttime=" + self.ACCOUNTING_WINDOW,
                            "--jobs=" + ",".join(ids),
                            "--format=JobID%64,State%32,ExitCode,ElapsedRaw,User%64"])
        result = {}
        for line in output.splitlines():
            row = line.split("|")
            if (len(row) >= 5 and "." not in row[0] and row[0].split("_")[0] in ids
                    and row[4].strip() == user):
                result[row[0]] = {"state": row[1].split()[0].rstrip("+"),
                                  "exit_code": row[2], "elapsed_seconds": row[3]}
        return result

    def external_gpu_count(self, owned):
        output = self._run(["squeue", "--noheader", "--array", "--user=" + getpass.getuser(), "--format=%F|%b"])
        count = 0
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) != 2 or not fields[0].isdigit():
                raise PlanError("Unrecognized scheduler reservation row")
            if fields[0] in owned:
                continue
            for resource in fields[1].split(","):
                if "gpu" in resource:
                    if not resource.startswith(("gpu:", "gres/gpu:", "gres:gpu:")):
                        raise PlanError("Unrecognized external GPU reservation")
                    last = resource.rsplit(":", 1)[-1].split("(")[0]
                    if not last.isdigit():
                        raise PlanError("Unrecognized external GPU reservation")
                    count += int(last)
        return count


def _check_external(plan, scheduler, jobs):
    count = scheduler.external_gpu_count(set(jobs.values()))
    if count > plan["config"]["external_reserved_gpus"]:
        raise PlanError("Live external GPU reservations exceed the approved allowance")


def _intent_path(plan, stage):
    return _control(plan) / "manifests/submission_intents" / (stage + ".json")


def _record_job(plan, path, intent, job, *, reconciled=False):
    if not isinstance(job, str) or not job.isdigit():
        raise AmbiguousSubmission("A numeric singleton/array parent ID is required")
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


def _intent_expected(plan, stage, jobs, authorization_sha256):
    return {"stage": stage, "workflow_sha256": digest(plan), "authorization_sha256": authorization_sha256,
            "job_name": _name(plan, stage), "comment": _comment(plan, stage),
            "command": sbatch_command(plan, stage, jobs, authorization_sha256)}


def submit_one(plan, stage, jobs, authorization_sha256, scheduler, *, now=None):
    path = _intent_path(plan, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _intent_expected(plan, stage, jobs, authorization_sha256)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            intent = json.loads(path.read_text())
            if any(intent.get(key) != value for key, value in expected.items()):
                raise AmbiguousSubmission("Altered immutable intent; never resubmit " + stage)
            if intent.get("status") == "recorded" and str(intent.get("job_id", "")).isdigit():
                return intent["job_id"]
            matches = scheduler.reconcile(intent)
            if len(matches) != 1:
                raise AmbiguousSubmission("Zero/multiple accounting matches never authorize a retry: " + stage)
            return _record_job(plan, path, intent, matches[0], reconciled=True)
        intent = {**expected, "status": "submitting", "created_utc": utc_now(now).isoformat()}
        atomic_json(path, intent)
        try:
            receipt = scheduler.submit(expected["command"])
            if not isinstance(receipt, str) or not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", receipt.strip()):
                raise AmbiguousSubmission("Unrecognized sbatch receipt")
        except Exception as error:
            intent.update(status="ambiguous", error_type=type(error).__name__)
            atomic_json(path, intent)
            raise AmbiguousSubmission("Possibly accepted submission; reconcile before any further action") from error
        return _record_job(plan, path, intent, receipt.strip().split(";")[0])


def recorded_jobs(plan, authorization_sha256):
    result = {}
    for stage in plan["order"]:
        path = _intent_path(plan, stage)
        if not path.exists():
            continue
        if path.resolve() != path:
            raise PlanError("Redirected submission intent")
        intent = json.loads(path.read_text())
        if any(intent.get(k) != v for k, v in _intent_expected(plan, stage, result, authorization_sha256).items()):
            raise PlanError("Foreign or altered submission intent")
        if intent.get("status") == "recorded" and isinstance(intent.get("job_id"), str) and intent["job_id"].isdigit():
            result[stage] = intent["job_id"]
        elif intent.get("status") not in {"submitting", "ambiguous"}:
            raise PlanError("Invalid durable submission status")
    if len(result) != len(set(result.values())):
        raise PlanError("One Slurm allocation cannot represent two stages")
    return result


def reconcile_intents(plan, scheduler, authorization_sha256):
    """Resolve unique accepted allocations, including after disconnect; never sbatch."""
    jobs = recorded_jobs(plan, authorization_sha256)
    unresolved = []
    for stage in plan["order"]:
        path = _intent_path(plan, stage)
        if not path.exists() or stage in jobs:
            continue
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            intent = json.loads(path.read_text())
            if any(intent.get(k) != v for k, v in _intent_expected(plan, stage, jobs, authorization_sha256).items()):
                raise PlanError("Changed intent cannot be reconciled")
            matches = scheduler.reconcile(intent)
            if len(matches) == 1:
                jobs[stage] = _record_job(plan, path, intent, matches[0], reconciled=True)
            else:
                unresolved.append({"stage": stage, "matches": matches})
    return {"job_ids": jobs, "unresolved": unresolved, "submitted_new_jobs": False}


def submit_plan(plan, authorization, scheduler=None, *, now=None):
    validate_plan(plan, authorization, now=now)
    verify_release(plan)
    verify_launch_inputs(plan)
    scheduler = scheduler or Scheduler()
    control = _control(plan)
    control.mkdir(parents=True, exist_ok=True)
    with (control / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / "manifests/terminal.json").exists():
            raise PlanError("A terminal workflow cannot be relaunched")
        for name in ("logs", "manifests/submission_intents"):
            (control / name).mkdir(parents=True, exist_ok=True)
        _frozen_json(control / "workflow.json", plan)
        _frozen_json(control / "authorization.json", authorization)
        authorization_sha = digest(authorization)
        reconciled = reconcile_intents(plan, scheduler, authorization_sha)
        if reconciled["unresolved"]:
            raise AmbiguousSubmission("Unresolved prior intent; no new jobs submitted")
        _check_external(plan, scheduler, reconciled["job_ids"])
        launch_path = control / "manifests/launch.json"
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if (launch.get("workflow_sha256") != digest(plan)
                    or launch.get("authorization_sha256") != authorization_sha):
                raise PlanError("Existing launch identity changed")
            if launch.get("status") == "incomplete_submission":
                raise PlanError("A failed static submission is terminal; reconcile, never expand it")
            if launch.get("status") == "submitted":
                if launch.get("job_ids") != reconciled["job_ids"] or set(reconciled["job_ids"]) != set(plan["order"]):
                    raise PlanError("Committed launch lost a durable dependency ID")
                return launch
        else:
            launch = {"status": "submitting", "workflow_sha256": digest(plan),
                      "authorization_sha256": authorization_sha, "created_utc": utc_now(now).isoformat(),
                      "run_id": plan["run_id"], "expected_stages": plan["order"]}
            atomic_json(launch_path, launch)
        if plan["mode"] == "benchmark":
            _frozen_json(Path(plan["config"]["root"]) / "execution.json", execution_record(plan, authorization))
        jobs = dict(reconciled["job_ids"])
        try:
            for stage in plan["order"]:
                validate_authorization(plan, authorization, now=now)
                if (control / "manifests/terminal.json").exists():
                    raise PlanError("Guardian terminated this launch; no further submissions")
                jobs[stage] = submit_one(plan, stage, jobs, authorization_sha, scheduler, now=now)
            if (control / "manifests/terminal.json").exists():
                raise PlanError("Guardian terminated before static launch commit")
        except Exception as error:
            launch.update(status="incomplete_submission", job_ids=recorded_jobs(plan, authorization_sha),
                          error_type=type(error).__name__)
            atomic_json(launch_path, launch)
            raise
        launch.update(status="submitted", job_ids=jobs, committed_utc=utc_now(now).isoformat(),
                      server_durable=True, laptop_close_ready=False)
        atomic_json(launch_path, launch)
        return launch


def _bound_guard_job(plan, authorization_sha256, scheduler=None, *, stage="guardian",
                     wait_for_ack=False, clock=None, sleep=None):
    require_slurm()
    if stage not in GUARDS:
        raise PlanError("Only declared guards may publish control receipts")
    actual = os.environ["SLURM_JOB_ID"]
    if (not actual.isdigit() or os.environ.get("SLURM_ARRAY_TASK_ID") is not None
            or os.environ.get("SLURM_ARRAY_JOB_ID") is not None):
        raise PlanError("Guard requires its exact singleton Slurm allocation")
    clock, sleep = clock or time.time, sleep or time.sleep
    launch = json.loads((_control(plan) / "manifests/launch.json").read_text())
    if (launch.get("workflow_sha256") != digest(plan)
            or launch.get("authorization_sha256") != authorization_sha256):
        raise PlanError("Guard launch identity differs")
    grace = plan["config"]["submission_grace_minutes"] * 60
    until = min(clock() + grace, timestamp(launch["created_utc"]).timestamp() + grace)
    if stage == "failure_guard":
        until = clock() + grace
    next_reconciliation = clock() + 5
    path = _intent_path(plan, stage)
    jobs = {} if stage == "guardian" else recorded_jobs(plan, authorization_sha256)
    expected = _intent_expected(plan, stage, jobs, authorization_sha256)

    def read_intent():
        if not path.exists():
            return None
        if path.resolve() != path:
            raise PlanError("Guard submission intent is redirected")
        intent = json.loads(path.read_text())
        if any(intent.get(key) != value for key, value in expected.items()):
            raise PlanError("Guard submission intent differs")
        if intent.get("status") == "recorded":
            if intent.get("job_id") != actual:
                raise PlanError("Guard allocation is not its recorded Slurm job")
        elif intent.get("status") not in {"submitting", "ambiguous"}:
            raise PlanError("Invalid guard submission state")
        return intent

    while True:
        intent = read_intent()
        if intent is not None and intent["status"] == "recorded":
            return actual
        if not wait_for_ack or clock() >= until:
            raise PlanError("Bounded guard acknowledgement wait expired; allocation is unbound")
        if scheduler is not None and intent is not None and (
                intent["status"] == "ambiguous" or clock() >= next_reconciliation):
            matches = scheduler.reconcile(intent)
            if matches and matches != [actual]:
                raise PlanError("Guard intent resolves to a different/ambiguous allocation")
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
        sleep(min(5, until - clock()))


def stage_states(plan, jobs, states):
    result = {}
    for name, stage in plan["stages"].items():
        if name in GUARDS:
            continue
        job = jobs.get(name)
        keys = [job] if len(stage["tasks"]) == 1 else [f"{job}_{index}" for index in range(len(stage["tasks"]))]
        rows = [states.get(key, {}) for key in keys]
        failed = next((row.get("state") for row in rows if row.get("state") in FAILED), None)
        if failed:
            result[name] = failed
        elif all(row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0" for row in rows):
            result[name] = "COMPLETED"
        elif any(row.get("state") == "COMPLETED" and row.get("exit_code") != "0:0" for row in rows):
            result[name] = "FAILED"
        else:
            result[name] = next((row["state"] for row in rows if row.get("state") in ACTIVE), "UNKNOWN")
    return result


def guardian_decision(plan, states, evidence=None, *, now=None, launch=None):
    now, launch = utc_now(now), launch or {}
    if launch.get("status") == "incomplete_submission":
        return {"status": "incomplete", "reason": "incomplete_submission"}
    if launch.get("status") != "submitted":
        start = timestamp(launch["created_utc"]) if launch.get("created_utc") else now
        if now >= start + dt.timedelta(minutes=plan["config"]["submission_grace_minutes"]):
            return {"status": "incomplete", "reason": "submission_not_committed"}
    failed = {name: value for name, value in states.items() if value in FAILED}
    if failed:
        return {"status": "incomplete", "reason": "required_stage_failed", "failed_stages": failed}
    if all(states.get(name) == "COMPLETED" for name in plan["order"] if name not in GUARDS):
        if launch.get("status") == "submitted" and evidence_is_complete(plan, evidence):
            return {"status": "complete", "reason": "all_bound_outputs_complete", "evidence": evidence}
        return {"status": "incomplete", "reason": "outputs_unverified"}
    for deadline, cutoff in plan["config"]["deadlines"].items():
        required = [name for name, row in plan["stages"].items() if row["cutoff"] == deadline and name not in GUARDS]
        if now >= timestamp(cutoff) and any(states.get(name) != "COMPLETED" for name in required):
            return {"status": "incomplete", "reason": deadline + "_deadline"}
    if now >= max(map(timestamp, plan["config"]["deadlines"].values())):
        return {"status": "incomplete", "reason": "unverified_at_final_cutoff"}
    return {"status": "watching"}


def _receipt_path(plan, stage, job, index):
    return _control(plan) / "manifests/execution" / f"{stage}_{job}_{index}.json"


def _verify_execution_receipts(plan, authorization, jobs, stages=None):
    files = {}
    for name in stages or [name for name in plan["order"] if name not in GUARDS]:
        row = plan["stages"][name]
        for index, task in enumerate(row["tasks"]):
            path = _receipt_path(plan, name, jobs[name], index)
            value = json.loads(path.read_text())
            expected = {"status": "completed", "exit_code": 0, "stage": name, "task": task["key"],
                        "workflow_sha256": digest(plan), "authorization_sha256": digest(authorization),
                        "job_id": jobs[name], "array_index": index, "commands": task["commands"]}
            if path.resolve() != path or any(value.get(k) != v for k, v in expected.items()):
                raise PlanError("Incomplete, foreign or changed task execution receipt")
            when = timestamp(value["completed_utc"])
            if not timestamp(authorization["authorized_at"]) <= when < timestamp(plan["config"]["deadlines"][row["cutoff"]]):
                raise PlanError("Task completion missed its prospective cutoff")
            files[str(path.relative_to(_control(plan)))] = file_sha256(path)
    return files


def _freeze_phases(plan, authorization, jobs, statuses):
    previous = None
    for phase, stages in plan["phases"].items():
        if not all(statuses.get(name) == "COMPLETED" for name in stages):
            break
        files = _verify_execution_receipts(plan, authorization, jobs, stages)
        path = _control(plan) / "manifests/phases" / (phase + ".json")
        value = {"complete": True, "phase": phase, "workflow_sha256": digest(plan),
                 "authorization_sha256": digest(authorization), "guardian_job_id": jobs["guardian"],
                 "predecessor_sha256": previous, "files": files}
        _frozen_json(path, value)
        previous = file_sha256(path)


def _terminal_receipt(plan, authorization_sha256, jobs):
    path = _control(plan) / "manifests/terminal.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.resolve() != path:
        raise PlanError("Terminal receipt is redirected")
    value = json.loads(path.read_text())
    expected = {"schema_version": 1, "scope_id": SCOPE, "run_id": plan["run_id"], "mode": plan["mode"],
                "workflow_sha256": digest(plan), "authorization_sha256": authorization_sha256,
                "guardian_job_id": jobs.get("guardian")}
    if (any(value.get(k) != v for k, v in expected.items()) or value.get("status") not in {"complete", "incomplete"}
            or value.get("complete") is not (value["status"] == "complete")
            or any(jobs.get(k) != v for k, v in value.get("job_ids", {}).items())
            or (value.get("failure_guard_job_id") is not None
                and value["failure_guard_job_id"] != jobs.get("failure_guard"))
            or (value.get("complete") is True and value.get("failure_guard_job_id") is None)):
        raise PlanError("Existing terminal receipt has a different/invalid identity")
    return value


def _finish_guardian(plan, decision, jobs, scheduler, authorization_sha256, states=None, *, publisher="guardian"):
    _bound_guard_job(plan, authorization_sha256, stage=publisher)
    path = _control(plan) / "manifests/terminal.json"
    with _terminal_lock(path):
        previous = _terminal_receipt(plan, authorization_sha256, jobs)
        if previous is not None:
            return 0 if previous["complete"] else 2
        if decision.get("status") not in {"complete", "incomplete"}:
            raise PlanError("Only terminal decisions may be committed")
        if decision["status"] == "complete" and publisher != "failure_guard":
            raise PlanError("Only the independent afterany guard may certify successful guardian history")
        if decision["status"] == "complete" and not evidence_is_complete(plan, decision.get("evidence")):
            raise PlanError("A bare success marker cannot certify the final comparison")
        if decision["status"] == "complete":
            guard_state = (states or {}).get(jobs.get("guardian"), {})
            statuses = stage_states(plan, jobs, states or {})
            if (guard_state.get("state") != "COMPLETED" or guard_state.get("exit_code") != "0:0"
                    or set(jobs) != set(plan["order"])
                    or any(status != "COMPLETED" for status in statuses.values())):
                raise PlanError("Successful final publication requires successful guardian and task history")
        cancellation_error = None
        if decision["status"] == "incomplete":
            try:
                scheduler.cancel(sorted({job for name, job in jobs.items() if name not in GUARDS}))
            except Exception as error:
                cancellation_error = type(error).__name__
        receipt = {**decision, "schema_version": 1, "scope_id": SCOPE, "mode": plan["mode"],
                   "run_id": plan["run_id"], "complete": decision["status"] == "complete",
                   "workflow_sha256": digest(plan), "authorization_sha256": authorization_sha256,
                   "release_sha256": plan["release_sha256"], "resource_claims": plan["resource_claims"],
                   "job_ids": jobs, "guardian_job_id": jobs.get("guardian"),
                   "failure_guard_job_id": jobs.get("failure_guard"), "publisher": publisher,
                   "states": states or {}, "created_utc": utc_now().isoformat(), "automatic_retries": 0}
        if cancellation_error:
            receipt["cancellation_error_type"] = cancellation_error
        atomic_json(path, receipt)
        return 0 if receipt["complete"] else 2


def _stage_environment(plan, directory, gpu):
    require_slurm()
    translated = {"config": {"source_root": plan["config"]["pilot_root"],
                              "code_root": plan["config"]["code_root"],
                              "environment": plan["config"]["environment"]}}
    environment, imports = legacy._stage_environment(translated, directory, gpu)
    environment.update({
        "HF_HUB_CACHE": plan["config"]["environment"]["hf_hub_cache"],
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "POLYGRAPH_TEST_ARTIFACT_ROOT": str(directory / "test_artifacts"),
    })
    threads = str(min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = threads
    (directory / "test_artifacts").mkdir()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        environment.pop(name, None)
    return environment, imports


def _artifact(root, relative):
    path = Path(root) / _relative(relative)
    if path.resolve() != path or not path.is_file():
        raise PlanError("Missing or redirected required artifact: " + str(relative))
    return path


def required_evidence_paths(plan):
    if plan["mode"] == "preflight":
        return {"campaign.json", "role_map.json", "preflight.json"}
    names = {"campaign.json", "role_map.json", "execution.json", "base_freeze.json",
             "heads_freeze.json", "evaluation_gate.json", "linear/model.json", "linear/complete.json",
             "diagnostics/training_diagnostics.json", "diagnostics/training_diagnostics.csv",
             "diagnostics/complete.json", "predictions/dev_eval/static.npz", "predictions/dev_eval/static.json"}
    for family in ("G", "H", "S"):
        for seed in SEEDS:
            names.update(f"heads/{family}/seed{seed}/{name}.json" for name in ("stack", "last_only", "freeze"))
            names.update(f"predictions/{role}/{family}/seed{seed}.{suffix}"
                         for role in ("meta", "dev_eval") for suffix in ("npz", "json"))
    names.update(f"predictions/dev_eval/O/seed{seed}.{suffix}" for seed in SEEDS for suffix in ("npz", "json"))
    names.update("evaluation/" + name for name in plan["config"]["evaluation_files"])
    return names


def evidence_is_complete(plan, evidence):
    if (not isinstance(evidence, dict) or evidence.get("complete") is not True
            or evidence.get("workflow_sha256") != digest(plan)
            or evidence.get("release_sha256") != plan["release_sha256"]
            or set(evidence.get("files", {})) != required_evidence_paths(plan)
            or any(not isinstance(sha, str) or not HEX.fullmatch(sha) for sha in evidence["files"].values())):
        return False
    if plan["mode"] == "preflight":
        return (evidence.get("scientific_fits") == 0 and evidence.get("dev_eval_scoring") is False
                and evidence.get("campaign_binding") is not None)
    return (evidence.get("neural_fits") == NEURAL_FITS and evidence.get("new_neural_fits") == NEW_NEURAL_FITS
            and evidence.get("imported_neural_fits") == IMPORTED_NEURAL_FITS
            and evidence.get("meta_heads") == META_HEADS
            and evidence.get("linear_fits") == 1 and evidence.get("diagnostic_histories") == NEURAL_FITS
            and evidence.get("method_names") == list(METHODS)
            and evidence.get("primary_contrasts") == [list(row) for row in CONTRASTS]
            and evidence.get("bootstrap_draws") == 10000 and evidence.get("source_files_verified") is True)


def validate_evaluation_outputs(plan):
    """Fail closed on the declared complete inventory and all 25 registered methods."""
    root = Path(plan["config"]["root"]) / "evaluation"
    complete = json.loads(_artifact(root, "complete.json").read_text())
    report = json.loads(_artifact(root, "report.json").read_text())
    expected = set(plan["config"]["evaluation_files"]) - {"complete.json"}
    if complete.get("complete") is not True or set(complete.get("files", {})) != expected:
        raise PlanError("Evaluation completion must bind the entire declared output inventory")
    for name, sha in complete["files"].items():
        if file_sha256(_artifact(root, name)) != _sha(sha, name):
            raise PlanError("Evaluation output changed after completion: " + name)
    if (report.get("method_names") != list(METHODS)
            or report.get("primary_contrasts") != [list(row) for row in CONTRASTS]
            or report.get("seeds") != list(SEEDS)
            or report.get("bootstrap_draws") != 10000
            or report.get("scope_id") != SCOPE
            or report.get("campaign_sha256") != plan["config"]["campaign_binding"]["campaign_sha256"]):
        raise PlanError("Evaluation report lacks the exact 25-method/fixed-seed/six-contrast/10000-draw contract")
    return complete


def completion_evidence(plan, authorization):
    require_slurm()
    verify_release(plan, executing=True)
    root = Path(plan["config"]["root"])
    from .protocol import read_campaign
    campaign = read_campaign(root)
    jobs = recorded_jobs(plan, digest(authorization))
    receipts = _verify_execution_receipts(plan, authorization, jobs)
    evidence = {
        "complete": True, "workflow_sha256": digest(plan), "release_sha256": plan["release_sha256"],
        "campaign_binding": campaign_binding(root), "execution_receipts": receipts,
        "files": {name: file_sha256(_artifact(root, name)) for name in required_evidence_paths(plan)},
    }
    if plan["mode"] == "preflight":
        preflight = json.loads(_artifact(root, "preflight.json").read_text())
        if preflight.get("complete") is not True:
            raise PlanError("Allocated model/loader/extraction preflight is incomplete")
        if preflight.get("job_id") != jobs["gpu_preflight"]:
            raise PlanError("Preflight receipt belongs to a different GPU allocation")
        if (root / "execution.json").exists() or (root / "predictions/dev_eval").exists():
            raise PlanError("Implementation validation must not perform scientific fits/evaluation")
        evidence.update(scientific_fits=0, dev_eval_scoring=False, measured_preflight=preflight)
    else:
        verify_launch_inputs(plan)
        if json.loads((root / "execution.json").read_text()) != execution_record(plan, authorization):
            raise PlanError("Scientific execution approval changed")
        from .freeze import validate_evaluation_gate
        validate_evaluation_gate(root)
        validate_evaluation_outputs(plan)
        historical = {}
        for key, row in campaign["reuse"].items():
            for name, sha in row["artifacts"].items():
                path = _artifact(row["path"], name)
                if file_sha256(path) != sha:
                    raise PlanError("Imported G fit changed: " + key + "/" + name)
                historical[str(path)] = sha
        for group in campaign["reuse_groups"].values():
            for name, sha in group["files"].items():
                path = _artifact(group["root"], name)
                if file_sha256(path) != sha:
                    raise PlanError("Imported predictions/heads/late provenance changed")
                historical[str(path)] = sha
        evidence.update(neural_fits=NEURAL_FITS, new_neural_fits=NEW_NEURAL_FITS,
                        imported_neural_fits=IMPORTED_NEURAL_FITS, meta_heads=META_HEADS,
                        linear_fits=1, diagnostic_histories=NEURAL_FITS, method_names=list(METHODS),
                        primary_contrasts=[list(row) for row in CONTRASTS], bootstrap_draws=10000,
                        source_files_verified=True, historical_files=historical)
    if not evidence_is_complete(plan, evidence):
        raise PlanError("Independent completion evidence is insufficient")
    return evidence


def _allocated_audit(plan, authorization, publisher, stop_at):
    actual = _bound_guard_job(plan, digest(authorization), stage=publisher)
    control = _control(plan)
    directory = control / "job_work" / f"audit_{publisher}_{actual}"
    directory.mkdir(parents=True, exist_ok=False)
    path = control / "manifests" / ("evidence_" + publisher + ".json")
    try:
        environment, imported = _stage_environment(plan, directory, False)
        atomic_json(control / "manifests" / ("environment_" + publisher + ".json"),
                    {"job_id": actual, "import_staging": imported, "gpu": False})
        command = [plan["config"]["python"], "-B", "-u", "-m", MODULE + "slurm", "audit",
                   "--workflow", str(control / "workflow.json"), "--sha256", digest(plan),
                   "--authorization", str(control / "authorization.json"), "--authorization-sha256",
                   digest(authorization), "--stage", publisher, "--out", str(path)]
        if _run_command(command, plan["config"]["code_root"], environment, stop_at) != 0:
            raise PlanError("Allocated independent artifact audit failed")
        value = json.loads(path.read_text())
        if not evidence_is_complete(plan, value):
            raise PlanError("Allocated independent audit returned incomplete evidence")
        return value
    finally:
        shutil.rmtree(directory)


def guardian(plan, authorization, scheduler=None):
    require_slurm()
    validate_plan(plan, check_time=False)
    validate_authorization(plan, authorization, current=False)
    scheduler = scheduler or Scheduler()
    authorization_sha, control = digest(authorization), _control(plan)
    actual = _bound_guard_job(plan, authorization_sha, scheduler, wait_for_ack=True)
    end = max(map(timestamp, plan["config"]["deadlines"].values())).timestamp() + (
        plan["config"]["guardian_grace_minutes"] * 60)
    stopped, handlers, jobs, states = [], {}, {}, {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, lambda signum, _frame: stopped.append(signum))
    try:
        while time.time() < end and not stopped:
            try:
                _bound_guard_job(plan, authorization_sha)
                jobs = recorded_jobs(plan, authorization_sha)
                previous = _terminal_receipt(plan, authorization_sha, jobs)
                if previous is not None:
                    return 0 if previous["complete"] else 2
                _bound_json(control / "workflow.json", digest(plan))
                _bound_json(control / "authorization.json", authorization_sha)
                launch = json.loads((control / "manifests/launch.json").read_text())
                if (launch.get("workflow_sha256") != digest(plan)
                        or launch.get("authorization_sha256") != authorization_sha):
                    raise PlanError("Guardian launch identity changed")
                states = scheduler.states(list(jobs.values()))
                statuses = stage_states(plan, jobs, states)
                _freeze_phases(plan, authorization, jobs, statuses)
                if launch.get("status") == "submitted":
                    _check_external(plan, scheduler, jobs)
                all_done = all(statuses.get(name) == "COMPLETED" for name in plan["order"] if name not in GUARDS)
                evidence = _allocated_audit(plan, authorization, "guardian", end) if all_done else None
                decision = guardian_decision(plan, statuses, evidence, launch=launch)
                if states.get(actual, {}).get("state") in FAILED:
                    decision = {"status": "incomplete", "reason": "guardian_allocation_failed"}
                elif decision["status"] == "complete" and states.get(actual, {}).get("state") != "RUNNING":
                    decision = {"status": "incomplete", "reason": "guardian_allocation_not_running"}
                atomic_json(control / "manifests/guardian.json", {
                    "status": decision["status"], "workflow_sha256": digest(plan),
                    "authorization_sha256": authorization_sha, "job_id": actual,
                    "checked_utc": utc_now().isoformat(), "job_ids": jobs, "stage_states": statuses,
                })
                if decision["status"] != "watching":
                    reconciliation = reconcile_intents(plan, scheduler, authorization_sha)
                    jobs = reconciliation["job_ids"]
                    if reconciliation["unresolved"]:
                        decision = {"status": "incomplete", "reason": "unresolved_submission_intent",
                                    "unresolved": reconciliation["unresolved"]}
                    if decision["status"] == "complete":
                        _frozen_json(control / "manifests/guardian_result.json", {
                            **decision, "workflow_sha256": digest(plan), "authorization_sha256": authorization_sha,
                            "guardian_job_id": actual, "job_ids": jobs, "states": states,
                        })
                        return 0
                    return _finish_guardian(plan, decision, jobs, scheduler, authorization_sha, states)
            except Exception as error:
                return _finish_guardian(plan, {
                    "status": "incomplete", "reason": "guardian_observation_failed",
                    "error_type": type(error).__name__, "detail": str(error),
                }, jobs, scheduler, authorization_sha, states)
            time.sleep(30)
        jobs = recorded_jobs(plan, authorization_sha)
        return _finish_guardian(plan, {"status": "incomplete",
                                      "reason": "guardian_interrupted" if stopped else "guardian_bound_reached"},
                                jobs, scheduler, authorization_sha, states)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def failure_guard_decision(plan, jobs, states, candidate, evidence=None):
    guardian_state = states.get(jobs.get("guardian"), {})
    if guardian_state.get("state") != "COMPLETED" or guardian_state.get("exit_code") != "0:0":
        return {"status": "incomplete", "reason": "guardian_accounting_not_successful",
                "guardian_accounting": guardian_state}
    statuses = stage_states(plan, jobs, states)
    if (set(jobs) != set(plan["order"])
            or any(statuses.get(name) != "COMPLETED" for name in plan["order"] if name not in GUARDS)):
        return {"status": "incomplete", "reason": "required_task_accounting_not_successful"}
    if (not isinstance(candidate, dict) or candidate.get("status") != "complete"
            or candidate.get("workflow_sha256") != digest(plan)
            or candidate.get("guardian_job_id") != jobs["guardian"]
            or not evidence_is_complete(plan, candidate.get("evidence"))
            or not evidence_is_complete(plan, evidence)
            or candidate["evidence"] != evidence):
        return {"status": "incomplete", "reason": "independent_outputs_unverified"}
    return {"status": "complete", "reason": "guardian_history_and_all_outputs_verified", "evidence": evidence}


def failure_guard(plan, authorization, scheduler=None):
    require_slurm()
    validate_plan(plan, check_time=False)
    validate_authorization(plan, authorization, current=False)
    scheduler = scheduler or Scheduler()
    authorization_sha = digest(authorization)
    _bound_guard_job(plan, authorization_sha, scheduler, stage="failure_guard", wait_for_ack=True)
    jobs, states = recorded_jobs(plan, authorization_sha), {}
    previous = _terminal_receipt(plan, authorization_sha, jobs)
    if previous is not None:
        return 0 if previous["complete"] else 2
    try:
        reconciliation = reconcile_intents(plan, scheduler, authorization_sha)
        jobs = reconciliation["job_ids"]
        if reconciliation["unresolved"]:
            raise PlanError("Unresolved accepted job intent after guardian exit")
        _bound_json(_control(plan) / "workflow.json", digest(plan))
        _bound_json(_control(plan) / "authorization.json", authorization_sha)
        until = time.time() + min(120, plan["config"]["caps"]["failure_guard"] * 15)
        while True:
            states = scheduler.states(list(jobs.values()))
            row = states.get(jobs.get("guardian"), {})
            if row.get("state") in FAILED or row.get("state") == "COMPLETED" or time.time() >= until:
                break
            time.sleep(5)
        candidate_path = _control(plan) / "manifests/guardian_result.json"
        candidate = json.loads(candidate_path.read_text()) if candidate_path.exists() else None
        successful = row.get("state") == "COMPLETED" and row.get("exit_code") == "0:0"
        evidence = _allocated_audit(
            plan, authorization, "failure_guard",
            time.time() + plan["config"]["caps"]["failure_guard"] * 60 - 30,
        ) if successful and candidate is not None else None
        decision = failure_guard_decision(plan, jobs, states, candidate, evidence)
    except Exception as error:
        decision = {"status": "incomplete", "reason": "failure_guard_observation_failed",
                    "error_type": type(error).__name__, "detail": str(error)}
    return _finish_guardian(plan, decision, jobs, scheduler, authorization_sha, states, publisher="failure_guard")


def _await_committed_launch(plan, authorization, stop_at):
    control = _control(plan)
    while time.time() < stop_at:
        if (control / "manifests/terminal.json").exists():
            raise PlanError("A terminal workflow cannot perform further work")
        launch = json.loads((control / "manifests/launch.json").read_text())
        if (launch.get("workflow_sha256") != digest(plan)
                or launch.get("authorization_sha256") != digest(authorization)):
            raise PlanError("Stage launch identity changed")
        if launch.get("status") == "submitted":
            jobs = recorded_jobs(plan, digest(authorization))
            if set(jobs) != set(plan["order"]) or launch.get("job_ids") != jobs:
                raise PlanError("Static launch did not bind every durable dependency")
            return jobs
        if launch.get("status") != "submitting":
            raise PlanError("Static submission failed; no work may start")
        until = timestamp(launch["created_utc"]).timestamp() + plan["config"]["submission_grace_minutes"] * 60
        if time.time() + 5 >= min(until, stop_at):
            break
        time.sleep(5)
    raise PlanError("Bounded launch-commit wait expired")


def _live_guardian(plan, authorization, jobs):
    path = _control(plan) / "manifests/guardian.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    return (value.get("status") == "watching" and value.get("workflow_sha256") == digest(plan)
            and value.get("authorization_sha256") == digest(authorization)
            and value.get("job_id") == jobs.get("guardian")
            and 0 <= time.time() - timestamp(value["checked_utc"]).timestamp() <= 90)


def _await_guardian_phase(plan, authorization, jobs, stage, stop_at):
    predecessor = None
    for phase, stages in plan["phases"].items():
        if stage in stages:
            break
        predecessor = phase
    for attempt in range(13):
        phase_ready = predecessor is None
        if predecessor is not None:
            path = _control(plan) / "manifests/phases" / (predecessor + ".json")
            if path.is_file():
                value = json.loads(path.read_text())
                phase_ready = (value.get("complete") is True and value.get("phase") == predecessor
                               and value.get("workflow_sha256") == digest(plan)
                               and value.get("authorization_sha256") == digest(authorization)
                               and value.get("guardian_job_id") == jobs["guardian"])
        if _live_guardian(plan, authorization, jobs) and phase_ready:
            return
        if attempt == 12 or time.time() + 10 >= stop_at:
            raise PlanError("Independent live guardian and previous global phase freeze are required")
        time.sleep(10)


def _stage_identity(plan, stage, jobs):
    row = plan["stages"][stage]
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if len(row["tasks"]) > 1:
        if raw is None or not raw.isdigit() or not 0 <= int(raw) < len(row["tasks"]):
            raise PlanError("Exact declared Slurm array index required")
        index, actual = int(raw), os.environ.get("SLURM_ARRAY_JOB_ID")
    else:
        if raw is not None or os.environ.get("SLURM_ARRAY_JOB_ID") is not None:
            raise PlanError("Singleton stages cannot be silently expanded into arrays")
        index, actual = 0, os.environ["SLURM_JOB_ID"]
    if actual != jobs.get(stage):
        raise PlanError("Stage allocation is not bound to its recorded Slurm job")
    if os.environ.get("SLURM_RESTART_COUNT", "0") not in ("", "0"):
        raise PlanError("Automatic scientific retries/requeues are forbidden")
    if row["gpu"]:
        if os.environ.get("SLURM_GPUS", "") != "1" and os.environ.get("SLURM_GPUS_ON_NODE", "") != "1":
            raise PlanError("Exactly one allocated GPU is required")
    elif os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS", "0") not in ("", "0"):
        raise PlanError("Output-only, linear, heads and analysis stages are CPU-only")
    return index, actual


def run_stage(plan, authorization, stage, scheduler=None):
    require_slurm()
    validate_plan(plan, check_time=False)
    validate_authorization(plan, authorization, current=stage not in GUARDS)
    verify_release(plan, executing=True)
    if stage not in plan["order"]:
        raise PlanError("Undeclared stage")
    if stage in GUARDS:
        return (guardian if stage == "guardian" else failure_guard)(plan, authorization, scheduler)
    row, started = plan["stages"][stage], time.time()
    cutoff = timestamp(plan["config"]["deadlines"][row["cutoff"]]).timestamp()
    if started + row["minutes"] * 60 >= cutoff:
        raise PlanError("Queue delay leaves insufficient prospective time for the full task cap")
    stop_at = min(started + row["minutes"] * 60 - 30, cutoff, timestamp(authorization["expires_at"]).timestamp())
    jobs = _await_committed_launch(plan, authorization, stop_at)
    index, actual = _stage_identity(plan, stage, jobs)
    _await_guardian_phase(plan, authorization, jobs, stage, stop_at)
    if row["gpu"]:
        _check_external(plan, scheduler or Scheduler(), jobs)
    directory = _control(plan) / "job_work" / f"{stage}_{actual}_{index}"
    directory.mkdir(parents=True, exist_ok=False)
    task = row["tasks"][index]
    path = _receipt_path(plan, stage, actual, index)
    if path.exists():
        raise PlanError("An already attempted task cannot run again")
    receipt = {"status": "running", "workflow_sha256": digest(plan), "authorization_sha256": digest(authorization),
               "stage": stage, "task": task["key"], "job_id": actual, "array_index": index,
               "commands": task["commands"], "started_utc": utc_now().isoformat()}
    atomic_json(path, receipt)
    try:
        environment, imports = _stage_environment(plan, directory, row["gpu"])
        receipt["import_staging"] = imports
        atomic_json(path, receipt)
        for command in task["commands"]:
            _bound_json(_control(plan) / "authorization.json", digest(authorization))
            validate_authorization(plan, authorization)
            if (_control(plan) / "manifests/terminal.json").exists():
                raise PlanError("Terminal guardian prohibits further commands")
            code = _run_command(command, plan["config"]["code_root"], environment, stop_at)
            if code != 0:
                raise PlanError("Declared command failed with exit " + str(code))
        verify_release(plan, executing=True)
        receipt.update(status="completed", exit_code=0, completed_utc=utc_now().isoformat())
        atomic_json(path, receipt)
        return 0
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, detail=str(error),
                       completed_utc=utc_now().isoformat())
        atomic_json(path, receipt)
        raise
    finally:
        shutil.rmtree(directory)


def handoff_state(plan, authorization, scheduler=None):
    """Read-only handoff: queue submission alone is not a laptop-close signal."""
    scheduler = scheduler or Scheduler()
    jobs = recorded_jobs(plan, digest(authorization))
    states = scheduler.states(list(jobs.values()))
    terminal = _terminal_receipt(plan, digest(authorization), jobs)
    launch = json.loads((_control(plan) / "manifests/launch.json").read_text())
    bound = (launch.get("workflow_sha256") == digest(plan)
             and launch.get("authorization_sha256") == digest(authorization)
             and launch.get("status") == "submitted" and launch.get("job_ids") == jobs
             and set(jobs) == set(plan["order"]))
    live = (bound and _live_guardian(plan, authorization, jobs)
            and states.get(jobs["guardian"], {}).get("state") == "RUNNING"
            and states.get(jobs["failure_guard"], {}).get("state") in ACTIVE)
    return {"run_id": plan["run_id"], "mode": plan["mode"], "job_ids": jobs,
            "workflow_sha256": digest(plan), "server_durable": bound,
            "laptop_close_ready": bool(plan["mode"] == "benchmark" and live and terminal is None),
            "terminal": terminal, "stage_states": stage_states(plan, jobs, states),
            "note": "Preflight completion never authorizes the %d new scientific fits" % NEW_NEURAL_FITS}


def main(argv=None, *, scheduler=None, now=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="plan",
                        choices=("plan", "validate", "reconcile", "handoff", "run-stage", "audit"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.workflow):
        parser.error("Supply exactly one configuration or sealed workflow")
    if args.mode != "plan" and not args.workflow:
        parser.error("This mode requires a sealed workflow and checksum")
    if args.submit and (args.out or args.mode not in ("plan", "validate")):
        parser.error("Only explicit plan/validate --submit may mutate the queue; no --out")
    if args.workflow and not args.sha256:
        parser.error("--workflow requires its approved --sha256")
    plan = (_bound_json(args.workflow, args.sha256) if args.workflow
            else build_plan(json.loads(args.config.read_text()), now=now))
    validate_plan(plan, now=now, check_time=args.mode in ("plan", "validate"))
    authorization = None
    if args.authorization:
        authorization = (_bound_json(args.authorization, args.authorization_sha256)
                         if args.authorization_sha256 else json.loads(args.authorization.read_text()))
    internal = args.mode in ("run-stage", "audit", "reconcile", "handoff")
    if internal and (authorization is None or not args.authorization_sha256):
        parser.error("Execution/reconciliation/handoff needs the original bound authorization")
    if args.mode == "run-stage":
        if args.out or args.stage not in plan["order"]:
            parser.error("Execution requires a declared stage, without an output override")
        return run_stage(plan, authorization, args.stage, scheduler)
    if args.mode == "audit":
        require_slurm()
        validate_authorization(plan, authorization, current=False)
        if args.stage not in GUARDS or args.out != _control(plan) / "manifests" / ("evidence_" + args.stage + ".json"):
            parser.error("Audits write only the declared guard's evidence path")
        _bound_guard_job(plan, digest(authorization), stage=args.stage)
        _frozen_json(args.out, completion_evidence(plan, authorization))
        return 0
    if args.mode in ("reconcile", "handoff"):
        if args.out:
            parser.error("Read/reconcile modes cannot rewrite control records through --out")
        validate_authorization(plan, authorization, current=False)
        result = (reconcile_intents(plan, scheduler or Scheduler(), digest(authorization))
                  if args.mode == "reconcile" else handoff_state(plan, authorization, scheduler))
        print(json.dumps(result, sort_keys=True))
        return 2 if result.get("unresolved") else 0
    if args.submit:
        if authorization is None:
            parser.error("--submit requires separate explicit approval")
        result = submit_plan(plan, authorization, scheduler, now=now)
    else:
        if authorization is not None:
            validate_authorization(plan, authorization, now=now)
        result = {"status": "validated_not_submitted", "workflow_sha256": digest(plan),
                  "plan": plan, "authorization_template": authorization_template(plan)}
        if args.out:
            _frozen_json(args.out, plan)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
