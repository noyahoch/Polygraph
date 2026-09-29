"""Durable accounting/failure receipt after scientific jobs reach terminal states."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--phase", choices=["preparation", "final"], default="final")
a = p.parse_args()
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Allocated Slurm job required")
plan = json.loads((a.root / "manifests/workflow.json").read_text())
submissions = json.loads((a.root / "manifests/submissions.json").read_text())
job_ids = set(plan.get("prior_job_ids", []))
job_ids.update(row["job_id"] for row in submissions.values())
output = subprocess.check_output(
    ["sacct", "-S", plan["started_date"], "-j", ",".join(sorted(job_ids)), "-n", "-P",
     "--format=JobID,JobName%100,State,ElapsedRaw,ExitCode,AllocTRES%150,Start,End"], text=True)
rows = []
gpu_seconds = 0
for line in output.splitlines():
    values = line.split("|")
    if len(values) < 8 or "." in values[0]:
        continue  # .batch/.extern duplicate the allocation, not additional work.
    names = ("job_id", "job_name", "state", "elapsed_seconds", "exit_code", "allocated_tres", "start", "end")
    row = dict(zip(names, values))
    row["elapsed_seconds"] = int(row["elapsed_seconds"] or 0)
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", row["allocated_tres"])
    row["gpu_count"] = int(match.group(1)) if match else 0
    # Array parents may be rendered as ranges for pending members, which have
    # no allocated elapsed time. Concrete array-task rows carry actual usage.
    row["gpu_seconds"] = row["gpu_count"] * row["elapsed_seconds"]
    gpu_seconds += row["gpu_seconds"]
    rows.append(row)
completion = a.root / "results/evaluation_complete.json"
scientific = json.loads(completion.read_text()) if completion.exists() else None
record = {"status": "complete", "meaning": "Administrative collection completed; scientific success is separate",
          "phase": a.phase,
          "job_id": os.environ["SLURM_JOB_ID"], "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "scientific_status": "complete" if scientific and scientific.get("status") == "complete" else "incomplete",
          "evaluation_completion": scientific, "jobs": rows, "allocated_gpu_hours": gpu_seconds / 3600,
          "accounting_caveat": "Includes failed/preflight allocations; a still-running publisher is observed only through this snapshot",
          "active_agent_time": "Tracked separately as team wall-clock union; GPU elapsed is not active-agent time"}
directory = a.root / "status"
directory.mkdir(parents=True, exist_ok=True)
if a.phase == "preparation":
    validation = a.root / "feature_cache/validation.json"
    checked = validation.exists() and json.loads(validation.read_text()).get("passed") is True
    timings = [a.root / "io/one_worker.json"] + [a.root / f"io/two_worker_{index}.json" for index in range(2)]
    ready = checked and all(path.exists() and json.loads(path.read_text()).get("training_batches") for path in timings)
    record["phase_status"] = "ready_for_root_io_admission" if ready else "failed_or_incomplete"
    record["training_submitted_by_this_phase"] = False
    names = ("preparation.json",)
else:
    names = ("remote_summary.json", "terminal.json")
for name in names:
    temporary = directory / (name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(directory / name)
print(json.dumps(record), flush=True)
