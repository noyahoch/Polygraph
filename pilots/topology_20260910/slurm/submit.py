"""Idempotent submission of a reviewed Slurm dependency graph; never polls."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


def live_dependency(job, mode, started_date, satisfied):
    """Completed jobs can age out of slurmctld while remaining authoritative in sacct."""
    visible = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True)
    if visible.returncode == 0:
        return True
    output = subprocess.check_output(["sacct", "-S", started_date, "-j", job, "-n", "-P",
                                      "--format=JobID,JobName%100,State,ExitCode"], text=True)
    rows = [line.split("|") for line in output.splitlines() if line and "." not in line.split("|")[0]]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
                "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    if not rows or any(row[2].split()[0] not in terminal for row in rows):
        raise SystemExit(f"Cannot reconcile disappeared dependency {job}")
    if mode == "afterok" and any(row[2] != "COMPLETED" or row[3] != "0:0" for row in rows):
        raise SystemExit(f"Required successful dependency {job} failed")
    satisfied.append({"job_id": job, "condition": mode, "accounting": rows,
                      "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stages", nargs="+", required=True)
    a = p.parse_args()
    os.umask(0o077)
    path = a.root / "manifests" / "workflow.json"
    data = path.read_bytes()
    plan = json.loads(data)
    code_root = Path(plan["code_root"]).resolve()
    if code_root.parent != (a.root / "releases").resolve():
        raise SystemExit("Workflow must reference an immutable experiment release")
    sha = hashlib.sha256(data).hexdigest()
    review = json.loads((a.root / "manifests" / "release_review.json").read_text())
    if review.get("decision") != "GO" or review.get("workflow_sha256") != sha:
        raise SystemExit("Exact workflow requires a recorded release GO")
    if not {"engineering", "science"}.issubset(review.get("reviewers", {})):
        raise SystemExit("Both engineering and science reviews are required")
    for stage in a.stages:
        if stage not in review["approved_stages"]:
            raise SystemExit(f"Stage not yet approved: {stage}")
    fits = plan["planned_fits"]
    expected = {(arm, seed) for arm in ["full_graph", "full_rewired", "full_set", "full_endpoint",
                                        "raw_graph", "raw_set", "logit"]
                for seed in [1, 2, 7, 17, 27]}
    if len(fits) != 35 or {(x["arm"], x["seed"]) for x in fits} != expected:
        raise SystemExit("Workflow must preserve the approved 35 arm/seed fits")
    submissions = a.root / "manifests" / "submissions.json"
    with (a.root / "manifests" / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(submissions.read_text()) if submissions.exists() else {}
        for name in a.stages:
            key = sha + ":" + name
            if key in records:
                print(json.dumps({"already_submitted": records[key]}), flush=True)
                continue
            stage = plan["stages"][name]
            if stage.get("final_detector_test", False):
                commands = stage.get("commands", [stage.get("command")])
                for command in commands:
                    module = "pilots.topology_20260910.evaluate"
                    if (not command or module not in command
                            or command[command.index(module) + 1:command.index(module) + 2] != ["score"]
                            or "--freeze" not in command):
                        raise SystemExit("Final detector test may run only through evaluate score with the frozen manifest")
            gpu = int(stage.get("gpus", 0))
            if gpu not in (0, 1):
                raise SystemExit("Only zero/one GPU per independent job allowed")
            concurrent = int(stage.get("concurrency", 1))
            if not 1 <= concurrent <= 8:
                raise SystemExit("Concurrency outside approved 1..8")
            deps = []
            for dep in stage.get("afterok", []):
                if dep.startswith("external:"):
                    deps.append(dep.split(":", 1)[1])
                elif sha + ":" + dep in records:
                    deps.append(records[sha + ":" + dep]["job_id"])
                else:
                    raise SystemExit(f"Dependency must be submitted first: {dep}")
            any_deps = []
            for dep in stage.get("afterany", []):
                if dep.startswith("external:"):
                    any_deps.append(dep.split(":", 1)[1])
                elif sha + ":" + dep in records:
                    any_deps.append(records[sha + ":" + dep]["job_id"])
                else:
                    raise SystemExit(f"Administrative dependency must be submitted first: {dep}")
            # One GPU stage/array at a time, so independently submitted stages cannot
            # silently add their concurrency above the shared eight-GPU ceiling.
            serial_deps = []
            if gpu:
                for record in records.values():
                    if record.get("gpu_per_task", 0) and record["job_id"] not in deps:
                        serial_deps.append(record["job_id"])
            job_name = f"ot-{sha[:7]}-{name}"
            if len(job_name) > 80:
                raise SystemExit("Stage name too long")
            # Reconcile scheduler acceptance if a previous process died before saving its ID.
            history = subprocess.check_output(
                ["sacct", "-S", plan["started_date"], "-n", "-P", "--name", job_name,
                 "--format=JobIDRaw,JobName%100,State,Submit"], text=True)
            found = [line for line in history.splitlines() if line and "." not in line.split("|")[0]]
            queue = subprocess.check_output(["squeue", "--me", "-h", "-n", job_name,
                                             "-o", "%i|%j|%T"], text=True).strip()
            if found or queue:
                raise SystemExit(f"Unrecorded scheduler job for {name}; reconcile before any resubmission")
            cmd = ["sbatch", "--parsable", "--account=gpu-students", "--no-requeue",
                   "--kill-on-invalid-dep=yes", f"--job-name={job_name}",
                   f"--partition={'studentbatch' if gpu else 'cpu-killable'}",
                   f"--time={stage['minutes']}", f"--cpus-per-task={stage.get('cpus', 2)}",
                   f"--mem={stage.get('memory_mb', 32000)}M",
                   f"--output={a.root}/logs/{name}_%A_%a.out",
                   f"--error={a.root}/logs/{name}_%A_%a.err",
                   f"--export=OMRI_WORKFLOW_SHA256={sha}"]
            if gpu:
                cmd += ["--gpus=1", "--constraint=geforce_rtx_2080"]
            if "commands" in stage:
                cmd += [f"--array=0-{len(stage['commands'])-1}%{concurrent}"]
            clauses = []
            satisfied = []
            deps = [job for job in deps if live_dependency(job, "afterok", plan["started_date"], satisfied)]
            other_deps = [job for job in dict.fromkeys(serial_deps + any_deps)
                          if live_dependency(job, "afterany", plan["started_date"], satisfied)]
            if deps:
                clauses.append("afterok:" + ":".join(deps))
            if other_deps:
                clauses.append("afterany:" + ":".join(other_deps))
            if clauses:
                cmd += ["--dependency=" + ",".join(clauses)]
            cmd += [str(code_root / "pilots/topology_20260910/slurm/stage.sbatch"),
                    str(a.root), name, str(code_root)]
            job = subprocess.check_output(cmd, text=True).strip().split(";", 1)[0]
            record = {"job_id": job, "stage": name, "job_name": job_name,
                      "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "workflow_sha256": sha, "command": cmd, "gpu_per_task": gpu,
                      "archived_dependencies_verified": satisfied}
            records[key] = record
            tmp = submissions.with_suffix(".tmp")
            tmp.write_text(json.dumps(records, indent=2) + "\n")
            tmp.replace(submissions)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
