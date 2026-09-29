#!/usr/bin/env python3
"""Run a command with live output and append a machine-readable experiment record."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def monitor_gpu_memory(pid: int, stop: threading.Event, peak: list[int]) -> None:
    """Sample this command's CUDA allocation as reported by the driver."""
    while not stop.wait(0.5):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2)
            for line in output.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and int(fields[0]) == pid:
                    peak[0] = max(peak[0], int(fields[1]))
        except (OSError, ValueError, subprocess.SubprocessError):
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=0, help="Timeout in seconds; zero disables it")
    parser.add_argument("--plan-path", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--configuration", default="{}", help="JSON object")
    parser.add_argument("--parameter-count", type=int)
    parser.add_argument("--notes", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    try:
        configuration = json.loads(args.configuration)
    except json.JSONDecodeError as exc:
        parser.error(f"--configuration is not valid JSON: {exc}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    started, t0 = utc_now(), time.monotonic()
    status, reason, returncode = "completed", "", 0
    peak_gpu_memory = [0]
    with args.log.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{started}] $ {' '.join(command)}\n")
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            monitor_stop = threading.Event()
            monitor = threading.Thread(target=monitor_gpu_memory,
                                       args=(proc.pid, monitor_stop, peak_gpu_memory), daemon=True)
            monitor.start()
            assert proc.stdout is not None
            deadline = t0 + args.timeout if args.timeout else None
            while True:
                # Do not block indefinitely in readline(): many training commands only
                # print every few epochs, which previously let them overrun the timeout.
                readable, _, _ = select.select([proc.stdout], [], [], 0.5)
                if readable:
                    line = proc.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                if proc.poll() is not None:
                    for tail in proc.stdout:
                        sys.stdout.write(tail)
                        log.write(tail)
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    status, reason, returncode = "aborted", "timeout", 124
                    break
            monitor_stop.set()
            monitor.join(timeout=3)
            if status != "aborted":
                returncode = int(proc.returncode or 0)
                if returncode:
                    status, reason = "failed", f"exit code {returncode}"
        except (OSError, KeyboardInterrupt) as exc:
            status, reason, returncode = "failed", repr(exc), 1

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = ""
    record = {
        "experiment_id": args.experiment_id,
        "status": status,
        "command": command,
        "start_time_utc": started,
        "end_time_utc": utc_now(),
        "runtime_seconds": round(time.monotonic() - t0, 3),
        "git_commit": commit,
        "plan_path": args.plan_path,
        "seed": args.seed,
        "configuration": configuration,
        "parameter_count": args.parameter_count,
        "peak_gpu_memory_mb": peak_gpu_memory[0] or None,
        "result_files": [str(args.log)],
        "skip_or_failure_reason": reason,
        "notes": args.notes,
    }
    with args.ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
