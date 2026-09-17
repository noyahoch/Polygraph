"""Resume-only DAG for a benchmark that failed after its evaluation gate.

Re-runs exactly the parent's dev_gpu prediction tasks and the analysis, surrounded by
the same guardian/failure-guard machinery as ``slurm.py``. Nothing scientific is
re-fitted, re-frozen or re-authorized: the parent's execution record, evaluation gate,
completed task receipts and global phase freezes are hash-bound and re-verified, and a
separate write-once resume record discloses the resume without rewriting parent files.

This module deliberately lives outside ``protocol.source_identity()``: every file bound
by the frozen campaign must stay byte-identical to the parent release (checked at launch).
The orchestration logic is the tested ``slurm.py`` code, re-bound to this module's
globals, so that only resume-specific planning/validation is new.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types

from . import slurm as base

_OWN = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__",
        "__builtins__", "__cached__", "annotations"}
for _key, _value in list(vars(base).items()):
    if _key in _OWN:
        continue
    if isinstance(_value, types.FunctionType) and _value.__module__ == base.__name__:
        _clone = types.FunctionType(_value.__code__, globals(), _value.__name__,
                                    _value.__defaults__, _value.__closure__)
        _clone.__kwdefaults__ = copy.copy(_value.__kwdefaults__)
        _clone.__doc__ = _value.__doc__
        globals()[_key] = _clone
    else:
        globals()[_key] = _value
del _key, _value

# Runner/audit commands are built as MODULE + "slurm"; this prefix targets this module.
MODULE = "pilots.final_comparison_20260916.resume_"
SCIENCE = base.MODULE
SELF = "pilots/final_comparison_20260916/resume_slurm.py"
RESUME_FILES = frozenset({SELF, "pilots/final_comparison_20260916/test_resume_slurm.py"})
REQUIRED_SOURCE = frozenset(base.REQUIRED_SOURCE | RESUME_FILES)
RERUN = ("dev_gpu", "analysis")
RESUME_ORDER = (*base.GUARDS, "cpu_validation", "dev_gpu", "analysis")
RESUME_PHASES = {"predictions": ["cpu_validation", "dev_gpu"], "evaluation": ["analysis"]}
RESUME_KEYS = {"parent_control_root", "parent_workflow_sha256", "parent_authorization_sha256",
               "parent_terminal_sha256", "parent_receipts_sha256", "parent_phases", "execution_sha256",
               "evaluation_gate_sha256", "reason"}
FROZEN_PARENT_PHASES = ("preparation", "bases", "meta", "gate")
# certify: CPU-only certification of a resume whose science finished but whose guard audit failed.
SCIENCE_STAGES = ("cpu_validation", "dev_gpu", "analysis")
CERTIFY_ORDER = (*base.GUARDS, "output_check")
CERTIFY_PHASES = {"evaluation": ["output_check"]}
CERTIFY_KEYS = {"resume_control_root", "resume_workflow_sha256", "resume_authorization_sha256",
                "resume_terminal_sha256", "resume_receipts_sha256", "science_jobs", "reason"}
SHARED = ("run_id", "root", "pilot_root", "replication_root", "cache", "data_root", "python",
          "environment", "resources", "campaign_binding", "preflight", "evaluation_files",
          "submission_grace_minutes", "guardian_grace_minutes")

_base_build_plan = build_plan  # noqa: F821 - cloned from slurm
_base_completion_evidence = completion_evidence  # noqa: F821
_base_submit_plan = submit_plan  # noqa: F821
_base_required_evidence_paths = required_evidence_paths  # noqa: F821
_base_evidence_is_complete = evidence_is_complete  # noqa: F821


def _name(plan, stage):
    return "omri-resume-" + plan["run_id"][:20] + "-" + stage + "-" + digest(plan)[:12]


def _comment(plan, stage):
    return "polygraph-final-resume:" + digest(plan) + ":" + stage


def _config(config):
    mode = config.get("mode") if isinstance(config, dict) else None
    if mode == "certify":
        return _certify_config(config)
    return _resume_config(config)


def _resume_config(config):
    fields = {
        "mode", "approval_id", "run_id", "root", "control_root", "code_root", "pilot_root",
        "replication_root", "cache", "data_root", "python", "source_sha256", "environment",
        "resources", "caps", "deadlines", "max_concurrent_gpus", "external_reserved_gpus",
        "gpu_budget_minutes", "cpu_budget_minutes", "external_gpu_minutes", "external_cpu_minutes",
        "submission_grace_minutes", "guardian_grace_minutes", "campaign_binding", "preflight",
        "test_selectors", "evaluation_files", "resume",
    }
    if not isinstance(config, dict) or set(config) != fields or config.get("mode") != "resume":
        raise PlanError("Resume configuration fields must exactly match the resume contract")
    value = copy.deepcopy(config)
    _validate_common(value, RESUME_ORDER)
    integer(value["max_concurrent_gpus"], "max_concurrent_gpus", maximum=6)
    if value["caps"]["dev_gpu"] > 180 or value["caps"]["cpu_validation"] > 60:
        raise PlanError("Resume caps are bounded: dev_gpu <=180min, CPU validation <=60min")
    selectors = value["test_selectors"]
    if (not isinstance(selectors, list) or len(selectors) != len(set(selectors))
            or SCIENCE + "test_resume_slurm" not in selectors or SCIENCE + "test_slurm" not in selectors):
        raise PlanError("Resume validation must run the slurm and resume test modules")
    for selector in selectors:
        if (not isinstance(selector, str) or not selector.startswith(SCIENCE + "test_")
                or selector.replace(".", "/") + ".py" not in value["source_sha256"]):
            raise PlanError("Synthetic validation selectors must be source-bound test modules")
    return value


def _certify_config(config):
    fields = {
        "mode", "approval_id", "run_id", "root", "control_root", "code_root", "pilot_root",
        "replication_root", "cache", "data_root", "python", "source_sha256", "environment",
        "resources", "caps", "deadlines", "max_concurrent_gpus", "external_reserved_gpus",
        "gpu_budget_minutes", "cpu_budget_minutes", "external_gpu_minutes", "external_cpu_minutes",
        "submission_grace_minutes", "guardian_grace_minutes", "campaign_binding", "preflight",
        "test_selectors", "evaluation_files", "resume", "certify",
    }
    if not isinstance(config, dict) or set(config) != fields or config.get("mode") != "certify":
        raise PlanError("Certify configuration fields must exactly match the certify contract")
    value = copy.deepcopy(config)
    _validate_common(value, CERTIFY_ORDER)
    if value["test_selectors"] != [] or value["max_concurrent_gpus"] != 1 or value["external_reserved_gpus"] != 0:
        raise PlanError("Certification runs no tests and reserves no GPU")
    if value["caps"]["output_check"] > 240:
        raise PlanError("Certification output check is bounded to 240 minutes")
    certify = value["certify"]
    if not isinstance(certify, dict) or set(certify) != CERTIFY_KEYS:
        raise PlanError("Bind the resumed workflow, authorization, terminal, receipts and science jobs")
    absolute(certify["resume_control_root"], "resume control root")
    for other in (certify["resume_control_root"], value["resume"]["parent_control_root"]):
        if not legacy._separate(Path(other), Path(value["control_root"])):
            raise PlanError("Certification needs its own control namespace")
    for name in ("resume_workflow_sha256", "resume_authorization_sha256", "resume_terminal_sha256",
                 "resume_receipts_sha256"):
        _sha(certify[name], name)
    jobs = certify["science_jobs"]
    if (not isinstance(jobs, dict) or set(jobs) != set(SCIENCE_STAGES)
            or any(not isinstance(job, str) or not job.isdigit() for job in jobs.values())):
        raise PlanError("Name the resumed allocations that produced the science outputs")
    if not isinstance(certify["reason"], str) or not 10 <= len(certify["reason"]) <= 1000:
        raise PlanError("A disclosed certification reason is required")
    return value


def _validate_common(value, order):
    for name in ("approval_id", "run_id"):
        if not isinstance(value[name], str) or not SAFE.fullmatch(value[name]):
            raise PlanError("An explicit safe and unique " + name + " is required")
    paths = {name: absolute(value[name], name) for name in
             ("root", "control_root", "code_root", "pilot_root", "replication_root",
              "cache", "data_root", "python")}
    for old in ("pilot_root", "replication_root", "cache", "data_root"):
        for new in ("root", "control_root"):
            if not legacy._separate(paths[new], paths[old]):
                raise PlanError("Control/scientific roots must not overlap historical inputs")
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
    if not isinstance(env, dict) or set(env) != {"import_bundle_sha256", "dependency_manifest_sha256", "hf_hub_cache"}:
        raise PlanError("Pin the existing overlay, import bundle and offline model-cache path")
    for name in ("import_bundle_sha256", "dependency_manifest_sha256"):
        _sha(env[name], name)
    absolute(env["hf_hub_cache"], "offline HF model cache")
    resources = value["resources"]
    if (not isinstance(resources, dict)
            or set(resources) != set(RESOURCE_CLASS) | {"cpu_cpus", "cpu_memory_mb"}
            or any(resources.get(k) != v for k, v in RESOURCE_CLASS.items())):
        raise PlanError("Only the existing RTX2080 resource class is allowed")
    integer(resources["cpu_cpus"], "cpu_cpus", maximum=6)
    integer(resources["cpu_memory_mb"], "cpu_memory_mb", maximum=32000)
    integer(value["max_concurrent_gpus"], "max_concurrent_gpus", maximum=6)
    integer(value["external_reserved_gpus"], "external_reserved_gpus", minimum=0, maximum=2)
    if value["max_concurrent_gpus"] + value["external_reserved_gpus"] > 8:
        raise PlanError("Workflow and external reservations exceed eight concurrent GPUs")
    for name in ("gpu_budget_minutes", "cpu_budget_minutes", "external_gpu_minutes", "external_cpu_minutes"):
        integer(value[name], name, minimum=0, maximum=10**9)
    integer(value["submission_grace_minutes"], "submission_grace_minutes", maximum=10)
    integer(value["guardian_grace_minutes"], "guardian_grace_minutes", maximum=30)
    if not isinstance(value["caps"], dict) or set(value["caps"]) != set(order):
        raise PlanError("Every declared stage needs its own explicit allocation cap")
    for name, minutes in value["caps"].items():
        integer(minutes, name + " minutes", maximum=7200 if name in GUARDS else 4320)
    deadlines = value["deadlines"]
    if not isinstance(deadlines, dict) or set(deadlines) != {"predictions", "evaluation"}:
        raise PlanError("Resume keeps exactly the parent's prediction/evaluation cutoffs")
    if timestamp(deadlines["predictions"]) >= timestamp(deadlines["evaluation"]):
        raise PlanError("Cutoffs must be strictly ordered")
    if (not isinstance(value["evaluation_files"], list)
            or len(value["evaluation_files"]) != len(set(value["evaluation_files"]))
            or set(value["evaluation_files"]) != EVALUATION_FILES):
        raise PlanError("Resume must produce the parent's exact evaluation inventory")
    binding = value["campaign_binding"]
    if not isinstance(binding, dict) or set(binding) != {"campaign_sha256", "roles_sha256", "reuse_sha256"}:
        raise PlanError("Bind the prepared campaign, original roles and complete reuse registry")
    for name, sha in binding.items():
        _sha(sha, name)
    if not isinstance(value["preflight"], dict):
        raise PlanError("The parent's measured preflight binding is required")
    resume = value["resume"]
    if not isinstance(resume, dict) or set(resume) != RESUME_KEYS:
        raise PlanError("Bind the failed parent workflow, terminal, receipts, phases, execution and gate")
    absolute(resume["parent_control_root"], "parent control root")
    if not legacy._separate(Path(resume["parent_control_root"]), paths["control_root"]):
        raise PlanError("A resume needs its own control namespace")
    for name in RESUME_KEYS - {"parent_control_root", "parent_phases", "reason"}:
        _sha(resume[name], name)
    if (not isinstance(resume["parent_phases"], dict)
            or set(resume["parent_phases"]) != set(FROZEN_PARENT_PHASES)):
        raise PlanError("The parent must have frozen every phase up to the evaluation gate")
    for name, sha in resume["parent_phases"].items():
        _sha(sha, name)
    if not isinstance(resume["reason"], str) or not 10 <= len(resume["reason"]) <= 1000:
        raise PlanError("A disclosed resume reason is required")


def _stages(config):
    root, python = config["root"], config["python"]
    resources = config["resources"]

    def command(module, *args):
        return [python, "-B", "-u", "-m", SCIENCE + module, *map(str, args)]

    def stage(tasks, *, gpu=False, cutoff, dependencies=(), dependency_type="afterok", throttle=1):
        return {"tasks": tasks, "gpu": gpu, "cutoff": cutoff, "dependencies": list(dependencies),
                "dependency_type": dependency_type, "throttle": throttle,
                "cpus": resources["gpu_cpus"] if gpu else resources["cpu_cpus"],
                "memory_mb": resources["gpu_memory_mb"] if gpu else resources["cpu_memory_mb"]}

    guards = {
        "guardian": stage([{"key": "guardian", "commands": []}], cutoff="evaluation"),
        "failure_guard": stage([{"key": "failure_guard", "commands": []}], cutoff="evaluation",
                               dependencies=("guardian",), dependency_type="afterany"),
    }
    for name in GUARDS:
        guards[name]["cpus"] = 1
    if config["mode"] == "certify":
        # CPU-only: evaluate() on a completed root re-collects every input and re-derives the
        # stored scores; it never rewrites outputs. The guards run the full completion audit.
        stages = {**guards, "output_check": stage([{"key": "output_check", "commands": [
            command("evaluate", "--root", root, "--out", root + "/evaluation")]}],
            cutoff="evaluation", dependencies=("guardian",), dependency_type="after")}
        for name in CERTIFY_ORDER:
            stages[name]["minutes"] = config["caps"][name]
        return {name: stages[name] for name in CERTIFY_ORDER}
    stages = {
        **guards,
        "cpu_validation": stage([{"key": "cpu_validation", "commands": [
            [python, "-B", "-u", "-m", "unittest", *config["test_selectors"]]]}],
            cutoff="predictions", dependencies=("guardian",), dependency_type="after"),
        # Identical task identities/commands to the parent's dev_gpu stage (checked at launch).
        "dev_gpu": stage([
            {"key": f"{family}_seed{seed}", "commands": [
                command("predict", "--root", root, "--role", "dev_eval", "--family", family, "--seed", seed)],
             "imported": False, "family": family, "seed": seed}
            for family in ("G", "H", "S") for seed in SEEDS
            if not (family == "G" and seed in IMPORTED_SEEDS)
        ], gpu=True, cutoff="predictions", dependencies=("cpu_validation",),
            throttle=config["max_concurrent_gpus"]),
        "analysis": stage([{"key": "analysis", "commands": [
            command("evaluate", "--root", root, "--out", root + "/evaluation")]}],
            cutoff="evaluation", dependencies=("dev_gpu",)),
    }
    for name in RESUME_ORDER:
        stages[name]["minutes"] = config["caps"][name]
    return {name: stages[name] for name in RESUME_ORDER}


def build_plan(config, *, now=None, check_time=True):
    plan = _base_build_plan(config, now=now, check_time=check_time)
    if plan["mode"] == "certify":
        plan["phases"] = copy.deepcopy(CERTIFY_PHASES)
        plan["certify"] = {"certified_resume": plan["config"]["certify"], "parent": plan["config"]["resume"],
                           "new_gpu_minutes": 0, "refits": 0, "new_predictions": 0}
        return plan
    plan["phases"] = copy.deepcopy(RESUME_PHASES)
    plan["resume"] = {"rerun_stages": list(RERUN), "parent": plan["config"]["resume"],
                      "refits": 0, "new_meta_heads": 0}
    return plan


def validate_authorization(plan, authorization, *, now=None, current=True):
    template = authorization_template(plan)
    mutable = {"approved", "source_ready_at", "full_ready_at", "measured_resource_approval",
               "authorized_at", "expires_at"}
    if (not isinstance(authorization, dict) or set(authorization) != set(template)
            or any(authorization.get(k) != v for k, v in template.items() if k not in mutable)
            or authorization.get("approved") is not True
            or authorization.get("measured_resource_approval") is not True):
        raise PlanError("Separate explicit resume approval must bind parent, resources and cutoffs")
    issued, expires = timestamp(authorization["authorized_at"]), timestamp(authorization["expires_at"])
    deadlines = list(map(timestamp, plan["config"]["deadlines"].values()))
    if (timestamp(authorization["source_ready_at"]) > issued
            or timestamp(authorization["full_ready_at"]) > issued
            or issued >= min(deadlines) or expires != max(deadlines)):
        raise PlanError("SOURCE READY, FULL READY and the parent's cutoffs must precede submission")
    if current and not issued <= utc_now(now) < expires:
        raise PlanError("Approval is not currently active")
    return authorization


def _parent(plan):
    """Load and re-validate the failed parent benchmark exactly as its own tooling would."""
    resume = plan["config"]["resume"]
    control = Path(resume["parent_control_root"])
    parent = _bound_json(control / "workflow.json", resume["parent_workflow_sha256"])
    approval = _bound_json(control / "authorization.json", resume["parent_authorization_sha256"])
    base.validate_plan(parent, check_time=False)
    base.validate_authorization(parent, approval, current=False)
    if parent["mode"] != "benchmark" or base._control(parent) != control:
        raise PlanError("The resume parent must be a benchmark workflow in its own control root")
    terminal = _bound_json(control / "manifests/terminal.json", resume["parent_terminal_sha256"])
    if (terminal.get("complete") is not False or terminal.get("run_id") != parent["run_id"]
            or terminal.get("workflow_sha256") != resume["parent_workflow_sha256"]
            or terminal.get("authorization_sha256") != resume["parent_authorization_sha256"]):
        raise PlanError("Only a terminally incomplete parent of this workflow can be resumed")
    return parent, approval, terminal


def _parent_receipts(parent, approval):
    jobs = base.recorded_jobs(parent, digest(approval))
    if set(jobs) != set(parent["order"]):
        raise PlanError("The parent launch did not bind every stage")
    reused = [name for name in parent["order"] if name not in GUARDS and name not in RERUN]
    files = base._verify_execution_receipts(parent, approval, jobs, reused)
    for name in RERUN:
        for index in range(len(parent["stages"][name]["tasks"])):
            path = base._receipt_path(parent, name, jobs[name], index)
            if path.exists() and json.loads(path.read_text()).get("status") == "completed":
                raise PlanError("A completed parent task must not be re-run: " + name)
    return {"reused_stages": reused, "files": files, "sha256": digest(files)}


def _parent_phases(parent):
    phases = {}
    for name in FROZEN_PARENT_PHASES:
        path = base._control(parent) / "manifests/phases" / (name + ".json")
        value = json.loads(path.read_text())
        if (value.get("complete") is not True or value.get("phase") != name
                or value.get("workflow_sha256") != digest(parent)):
            raise PlanError("Parent phase freeze is missing or foreign: " + name)
        phases[name] = file_sha256(path)
    return phases


def resume_binding(parent_control_root, reason):
    """Compute the parent binding for a new resume config (metadata/hash reads only)."""
    control = Path(parent_control_root)
    parent_sha = file_sha256(control / "workflow.json")
    approval_sha = file_sha256(control / "authorization.json")
    parent = _bound_json(control / "workflow.json", parent_sha)
    approval = _bound_json(control / "authorization.json", approval_sha)
    root = Path(parent["config"]["root"])
    probe = {"config": {"resume": {"parent_control_root": str(control), "parent_workflow_sha256": parent_sha,
                                   "parent_authorization_sha256": approval_sha,
                                   "parent_terminal_sha256": file_sha256(control / "manifests/terminal.json")}}}
    _parent(probe)
    return {"parent_control_root": str(control), "parent_workflow_sha256": parent_sha,
            "parent_authorization_sha256": approval_sha,
            "parent_terminal_sha256": probe["config"]["resume"]["parent_terminal_sha256"],
            "parent_receipts_sha256": _parent_receipts(parent, approval)["sha256"],
            "parent_phases": _parent_phases(parent),
            "execution_sha256": file_sha256(root / "execution.json"),
            "evaluation_gate_sha256": file_sha256(root / "evaluation_gate.json"),
            "reason": reason}


def execution_record(plan, authorization):
    """The unchanged parent execution record: a resume never re-authorizes the science."""
    parent, approval, _ = _parent(plan)
    return base.execution_record(parent, approval)


def _record_path(plan):
    return Path(plan["config"]["root"]) / "resumes" / (plan["config"]["approval_id"] + ".json")


def resume_record(plan, authorization):
    resume = plan["config"]["resume"]
    return {
        "schema_version": 1, "scope_id": SCOPE, "run_id": plan["run_id"], "mode": "resume",
        "approval_id": plan["approval_id"], "workflow_sha256": digest(plan),
        "authorization_sha256": digest(authorization), "release_sha256": plan["release_sha256"],
        "control_root": plan["config"]["control_root"], "authorized_at": authorization["authorized_at"],
        "parent": resume, "rerun_stages": list(RERUN), "refits": 0, "new_meta_heads": 0,
        "execution_record_unchanged": True, "deadlines_unchanged": plan["config"]["deadlines"],
        "resource_claims": plan["resource_claims"], "original_test_access": False, "automatic_retries": 0,
        "disclosure": ("The parent benchmark failed only in its dev_gpu prediction stage; this resume "
                       "re-ran those six prediction groups and the analysis with the same frozen models, "
                       "heads, evaluation gate, source identity and cutoffs."),
    }


def _certified(plan):
    """Load and re-validate the resume whose outputs are being certified."""
    certify = plan["config"]["certify"]
    control = Path(certify["resume_control_root"])
    resumed = _bound_json(control / "workflow.json", certify["resume_workflow_sha256"])
    approval = _bound_json(control / "authorization.json", certify["resume_authorization_sha256"])
    if resumed.get("mode") != "resume" or _control(resumed) != control:
        raise PlanError("Only a resume workflow in its own control root can be certified")
    validate_plan(resumed, check_time=False)
    validate_authorization(resumed, approval, current=False)
    if resumed["config"]["resume"] != plan["config"]["resume"]:
        raise PlanError("Certification must bind the same parent as the certified resume")
    terminal = _bound_json(control / "manifests/terminal.json", certify["resume_terminal_sha256"])
    if (terminal.get("complete") is not False or terminal.get("run_id") != resumed["run_id"]
            or terminal.get("workflow_sha256") != certify["resume_workflow_sha256"]
            or terminal.get("authorization_sha256") != certify["resume_authorization_sha256"]):
        raise PlanError("Only a terminally incomplete resume of this workflow can be certified")
    jobs = recorded_jobs(resumed, digest(approval))
    if (set(jobs) != set(resumed["order"])
            or {name: jobs[name] for name in SCIENCE_STAGES} != certify["science_jobs"]):
        raise PlanError("The certified resume did not bind the declared science allocations")
    files = _verify_execution_receipts(resumed, approval, jobs, list(SCIENCE_STAGES))
    if digest(files) != certify["resume_receipts_sha256"]:
        raise PlanError("Completed resume task receipts changed")
    record = Path(resumed["config"]["root"]) / "resumes" / (resumed["config"]["approval_id"] + ".json")
    if json.loads(record.read_text()) != resume_record(resumed, approval):
        raise PlanError("The resume disclosure record changed")
    return resumed, approval, terminal, files


def certify_binding(resume_control_root, reason):
    """Compute the certification binding for a failed-audit resume (metadata/hash reads only)."""
    control = Path(resume_control_root)
    resumed = json.loads((control / "workflow.json").read_text())
    approval = json.loads((control / "authorization.json").read_text())
    jobs = recorded_jobs(resumed, digest(approval))
    probe = {"config": {"resume": resumed["config"]["resume"], "certify": {
        "resume_control_root": str(control), "resume_workflow_sha256": file_sha256(control / "workflow.json"),
        "resume_authorization_sha256": file_sha256(control / "authorization.json"),
        "resume_terminal_sha256": file_sha256(control / "manifests/terminal.json"),
        "resume_receipts_sha256": digest(_verify_execution_receipts(resumed, approval, jobs, list(SCIENCE_STAGES))),
        "science_jobs": {name: jobs[name] for name in SCIENCE_STAGES}, "reason": reason}}}
    _certified(probe)
    return probe["config"]["certify"]


def _verify_parent_bindings(plan):
    config = plan["config"]
    resume = config["resume"]
    parent, approval, _ = _parent(plan)
    base.verify_launch_inputs(parent)
    parent_config = parent["config"]
    if any(config[name] != parent_config[name] for name in SHARED):
        raise PlanError("A resume must keep the parent's scientific identity and resources")
    if config["deadlines"] != {name: parent_config["deadlines"][name] for name in ("predictions", "evaluation")}:
        raise PlanError("A resume can never extend or change the parent's cutoffs")
    root = Path(config["root"])
    if (file_sha256(root / "execution.json") != resume["execution_sha256"]
            or json.loads((root / "execution.json").read_text()) != base.execution_record(parent, approval)):
        raise PlanError("The parent's execution record changed")
    if file_sha256(root / "evaluation_gate.json") != resume["evaluation_gate_sha256"]:
        raise PlanError("The evaluation gate changed")
    if base.campaign_binding(root) != config["campaign_binding"]:
        raise PlanError("Prepared campaign/role/reuse bytes changed")
    if _parent_receipts(parent, approval)["sha256"] != resume["parent_receipts_sha256"]:
        raise PlanError("Completed parent task receipts changed")
    if _parent_phases(parent) != resume["parent_phases"]:
        raise PlanError("Parent phase freezes changed")
    return parent, approval


def verify_launch_inputs(plan):
    config = plan["config"]
    if config["mode"] == "certify":
        parent, _ = _verify_parent_bindings(plan)
        resumed, _, _, _ = _certified(plan)
        claims = resumed["resource_claims"]
        if (config["external_gpu_minutes"] != claims["workflow_gpu_minutes"] + claims["external_gpu_minutes"]
                or config["external_cpu_minutes"] != claims["workflow_cpu_minutes"] + claims["external_cpu_minutes"]):
            raise PlanError("The parent and resume reservations must count against the certification budget")
        inventory, previous = config["source_sha256"], resumed["config"]["source_sha256"]
        if (set(inventory) != set(previous)
                or any(inventory[name] != sha for name, sha in previous.items() if name not in RESUME_FILES)
                or any(inventory[name] != sha for name, sha in parent["config"]["source_sha256"].items())):
            raise PlanError("Certification may only change the resume orchestration files")
        return
    parent, _ = _verify_parent_bindings(plan)
    if (config["external_gpu_minutes"] != parent["resource_claims"]["workflow_gpu_minutes"]
            or config["external_cpu_minutes"] != parent["resource_claims"]["workflow_cpu_minutes"]):
        raise PlanError("The parent's full reservation must count against the resume budget")
    inventory, parent_inventory = config["source_sha256"], parent["config"]["source_sha256"]
    if (any(inventory.get(name) != sha for name, sha in parent_inventory.items())
            or not set(inventory) - set(parent_inventory) <= RESUME_FILES):
        raise PlanError("A resume may only add its own orchestration files to the parent release")
    for name in RERUN:
        if plan["stages"][name]["tasks"] != parent["stages"][name]["tasks"]:
            raise PlanError("Resumed tasks must be exactly the parent's declared tasks: " + name)


def _refuse_existing_results(plan):
    """Submit-time only: completion evidence re-runs verify_launch_inputs after analysis wrote results."""
    root = Path(plan["config"]["root"])
    if (root / "evaluation/report.json").exists() or (root / "evaluation/complete.json").exists():
        raise PlanError("Evaluation outputs already exist; a resume cannot overwrite results")


def submit_plan(plan, authorization, scheduler=None, *, now=None):
    validate_plan(plan, authorization, now=now)
    verify_release(plan)
    verify_launch_inputs(plan)
    if plan["mode"] == "certify":
        _frozen_json(_control(plan) / "certification.json", certification_record(plan, authorization))
        return _base_submit_plan(plan, authorization, scheduler, now=now)
    _refuse_existing_results(plan)
    _frozen_json(_record_path(plan), resume_record(plan, authorization))
    return _base_submit_plan(plan, authorization, scheduler, now=now)


def certification_record(plan, authorization):
    certify = plan["config"]["certify"]
    return {
        "schema_version": 1, "scope_id": SCOPE, "run_id": plan["run_id"], "mode": "certify",
        "approval_id": plan["approval_id"], "workflow_sha256": digest(plan),
        "authorization_sha256": digest(authorization), "release_sha256": plan["release_sha256"],
        "certified_resume": certify, "parent": plan["config"]["resume"],
        "science_outputs_produced_by": certify["science_jobs"],
        "new_gpu_minutes": 0, "refits": 0, "new_predictions": 0, "outputs_rewritten": False,
        "statement": ("The evaluation outputs were produced by resume jobs "
                      + ", ".join(certify["science_jobs"][name] for name in SCIENCE_STAGES)
                      + "; that resume's guard audit failed on an orchestration bug, so this CPU-only "
                        "workflow re-ran the full completion audit without rewriting any result."),
    }


def required_evidence_paths(plan):
    if plan["mode"] == "certify":
        resumed = json.loads((Path(plan["config"]["certify"]["resume_control_root"]) / "workflow.json").read_text())
        return _base_required_evidence_paths(plan) | {
            "resumes/" + resumed["config"]["approval_id"] + ".json"}
    return _base_required_evidence_paths(plan) | {
        str(_record_path(plan).relative_to(plan["config"]["root"]))}


def evidence_is_complete(plan, evidence):
    return (_base_evidence_is_complete(plan, evidence)
            and isinstance(evidence.get("resume_parent"), dict)
            and evidence["resume_parent"].get("receipts_sha256") == plan["config"]["resume"]["parent_receipts_sha256"]
            and evidence["resume_parent"].get("phases") == plan["config"]["resume"]["parent_phases"]
            and (plan["mode"] != "certify" or (
                isinstance(evidence.get("certified_resume"), dict)
                and evidence["certified_resume"].get("receipts_sha256")
                == plan["config"]["certify"]["resume_receipts_sha256"]
                and evidence["certified_resume"].get("science_jobs")
                == plan["config"]["certify"]["science_jobs"])))


def completion_evidence(plan, authorization):
    parent, approval, _ = _parent(plan)
    certified = None
    if plan["mode"] == "certify":
        if json.loads((_control(plan) / "certification.json").read_text()) != certification_record(plan, authorization):
            raise PlanError("The certification record changed")
        resumed, _, terminal, files = _certified(plan)
        certified = {"workflow_sha256": plan["config"]["certify"]["resume_workflow_sha256"],
                     "terminal_sha256": plan["config"]["certify"]["resume_terminal_sha256"],
                     "terminal_reason": terminal.get("reason"),
                     "science_jobs": plan["config"]["certify"]["science_jobs"],
                     "receipt_files": files, "receipts_sha256": digest(files)}
    elif json.loads(_record_path(plan).read_text()) != resume_record(plan, authorization):
        raise PlanError("The disclosed resume record changed")
    receipts = _parent_receipts(parent, approval)
    evidence = dict(_base_completion_evidence_unchecked(plan, authorization))
    evidence["resume_parent"] = {
        "workflow_sha256": plan["config"]["resume"]["parent_workflow_sha256"],
        "terminal_sha256": plan["config"]["resume"]["parent_terminal_sha256"],
        "reused_stages": receipts["reused_stages"], "receipt_files": receipts["files"],
        "receipts_sha256": receipts["sha256"], "phases": _parent_phases(parent),
    }
    if certified is not None:
        evidence["certified_resume"] = certified
    if not evidence_is_complete(plan, evidence):
        raise PlanError("Independent resume completion evidence is insufficient")
    return evidence


def _base_completion_evidence_unchecked(plan, authorization):
    """Run the cloned benchmark evidence builder, deferring only its final completeness check."""
    global evidence_is_complete
    checker = evidence_is_complete
    evidence_is_complete = _base_evidence_is_complete
    try:
        return _base_completion_evidence(plan, authorization)
    finally:
        evidence_is_complete = checker


def handoff_state(plan, authorization, scheduler=None):
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
            "laptop_close_ready": bool(live and terminal is None),
            "terminal": terminal, "stage_states": stage_states(plan, jobs, states),
            "note": "Resume re-runs only dev_gpu and analysis; no fits"}


if __name__ == "__main__":
    sys.exit(main())  # noqa: F821 - cloned from slurm
