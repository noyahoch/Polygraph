"""Download an explicit, small result-only inventory from the user's Slurm host."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

BASE = "/home/yandex/MLWG2026/omrifahn/polygraph_september_2026/experiments"
REPLICATION = BASE + "/layer_ensemble_replication_20260916_072636"
PILOT = BASE + "/layer_ensemble_20260914"
TOPOLOGY = BASE + "/topology_20260910"
ARMS = ("block2", "block5", "block8", "block11")
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
ANCHORS = {
    "september10/evaluation_complete.json": "16344ee7bf07f7d2f9456ecd2b4de5d1a1703fc49620f137255b17a322657790",
    "layers/report.json": "b17cbb6da8dd33a28d00d059b6a7bc91ee6b291fc517e2db87e67a38ae7bcf59",
    "layers/complete.json": "3bae5ae5188beb98f956239d60c55affda53578f28a8ceaf5264767353ebf858",
}


def inventory():
    groups = {
        "september10": (TOPOLOGY + "/results",
                        ["summary.json", "results.csv", "bootstrap.json", "evaluation_complete.json"]),
        "layers": (REPLICATION + "/evaluation", ["report.json", "complete.json"]),
    }
    for seed in (7, 17, 27):
        root = PILOT if seed == 7 else REPLICATION + f"/seed{seed}"
        names = ["evaluation/report.json", "evaluation/complete.json", "evaluation/scores.npz",
                 "base_freeze.json", "heads_freeze.json", "heads/stack.json", "heads/last_only.json"]
        for arm in ARMS:
            names += [f"runs/{arm}/seed{seed}/{name}" for name in
                      ("history.json", "complete.json", "config.json")]
        groups[f"layers/seed{seed}"] = (root, names)
    return groups


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise RuntimeError("Result checksum mismatch: " + str(path))


def verify(raw):
    raw = Path(raw)
    for name, expected in ANCHORS.items():
        verify_hash(raw / name, expected)
    old_complete = read(raw / "september10/evaluation_complete.json")
    if old_complete["status"] != "complete":
        raise RuntimeError("Historical evaluation is incomplete")
    for name in ("summary.json", "results.csv", "bootstrap.json"):
        verify_hash(raw / "september10" / name, old_complete["artifacts"][name])
    report = read(raw / "layers/report.json")
    complete = read(raw / "layers/complete.json")
    if report["status"] != "complete" or complete["complete"] is not True or complete["inputs"] != report["inputs"]:
        raise RuntimeError("Current evaluation identity is incomplete or inconsistent")
    verify_hash(raw / "layers/report.json", complete["files"]["report.json"])
    for seed in (7, 17, 27):
        root = raw / f"layers/seed{seed}"
        binding = report["inputs"]["seed_evaluations"][str(seed)]
        verify_hash(root / "evaluation/complete.json", binding["complete_sha256"])
        seed_complete = read(root / "evaluation/complete.json")
        seed_report = read(root / "evaluation/report.json")
        if (seed_complete["complete"] is not True or seed_report["seed"] != seed
                or seed_report["test_evaluated"] is not False
                or seed_complete["inputs"] != binding["inputs"]
                or seed_report["inputs"] != binding["inputs"]):
            raise RuntimeError("Per-seed reporting identity changed")
        for name in ("report.json", "scores.npz"):
            verify_hash(root / "evaluation" / name, seed_complete["files"][name])
        for name in ("base_freeze", "heads_freeze"):
            verify_hash(root / (name + ".json"), binding["inputs"][name + "_sha256"])
        heads = read(root / "heads_freeze.json")
        for name in ("stack", "last_only"):
            verify_hash(root / "heads" / (name + ".json"), heads["files"]["heads/" + name + ".json"])
        bases = read(root / "base_freeze.json")
        for arm in ARMS:
            directory = root / "runs" / arm / f"seed{seed}"
            verify_hash(directory / "complete.json", bases["runs"][arm]["complete_sha256"])
            run_complete = read(directory / "complete.json")
            if (run_complete["complete"] is not True or run_complete["seed"] != seed
                    or run_complete["arm"] != arm or run_complete["completed_epochs"] != 20):
                raise RuntimeError("Wrong base fit identity or training horizon")
            for name in ("history.json", "config.json"):
                verify_hash(directory / name, run_complete["artifacts"][name])
    return report


def remote_sizes(wrapper, root, names):
    stat = ["stat", "-c", "%s", "--", *[root + "/" + name for name in names]]
    sizes = subprocess.run(["bash", str(wrapper), shlex.join(stat)], check=True,
                           capture_output=True, text=True, timeout=90)
    values = [int(value) for value in sizes.stdout.splitlines()]
    if len(values) != len(names) or any(value < 0 or value > MAX_FILE_BYTES for value in values):
        raise RuntimeError("Result download exceeds the explicit per-file ceiling")
    return values


def fetch_group(wrapper, raw, prefix, root, names, sizes):
    if sum(sizes) > MAX_TOTAL_BYTES:
        raise RuntimeError("Result group exceeds the explicit download size ceiling")
    target = raw / prefix
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        command = ["tar", "-C", root, "-cf", "-", "--", *names]
        subprocess.run(["bash", str(wrapper), shlex.join(command)], check=True,
                       stdout=archive, timeout=120)
        if archive.tell() > MAX_TOTAL_BYTES:
            raise RuntimeError("Result archive exceeds the explicit size ceiling")
        archive.seek(0)
        with tarfile.open(fileobj=archive) as contents:
            members = contents.getmembers()
            if (len(members) != len(names) or {m.name for m in members} != set(names)
                    or any(not m.isfile() or m.size > MAX_FILE_BYTES for m in members)):
                raise RuntimeError("Unexpected file or link in result-only archive")
            if {m.name: m.size for m in members} != dict(zip(names, sizes)):
                raise RuntimeError("Result size changed after download preflight")
            for member in members:
                destination = target / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.resolve() != destination:
                    raise RuntimeError("Local result path is redirected")
                stream = contents.extractfile(member)
                if stream is None:
                    raise RuntimeError("Result archive member has no payload")
                content = stream.read()
                if len(content) != member.size:
                    raise RuntimeError("Truncated result download")
                if destination.exists() and destination.read_bytes() != content:
                    raise RuntimeError("Refusing to replace a different result: " + str(destination))
                destination.write_bytes(content)
    return [{"local": prefix + "/" + name, "remote": root + "/" + name,
             "bytes": (target / name).stat().st_size, "sha256": sha(target / name)} for name in names]


def collect(wrapper, raw):
    raw = Path(raw).resolve()
    groups = inventory()
    raw.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        size_futures = {prefix: pool.submit(remote_sizes, wrapper, root, names)
                        for prefix, (root, names) in groups.items()}
        sizes = {prefix: future.result() for prefix, future in size_futures.items()}
        if sum(sum(values) for values in sizes.values()) > MAX_TOTAL_BYTES:
            raise RuntimeError("Result inventory exceeds the total ceiling before download")
        futures = [pool.submit(fetch_group, wrapper, raw, prefix, root, names, sizes[prefix])
                   for prefix, (root, names) in groups.items()]
        files = [row for future in futures for row in future.result()]
    total = sum(row["bytes"] for row in files)
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError("Downloaded result inventory exceeds the total size ceiling")
    verify(raw)
    receipt = {
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "post-run result visualization and descriptive EDA",
        "authorization": "User explicitly permitted fast lightweight local result processing on September16",
        "max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES,
        "files": files, "bytes": total, "verified": True,
        "excluded": ["images", "feature caches", "model weights", "optimizer/RNG state"],
        "historical_test_policy": "Previously computed September10 summary tables only; no test prediction arrays downloaded or rescored",
    }
    (raw / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-wrapper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = collect(args.ssh_wrapper, args.out)
    print(json.dumps({"verified": True, "files": len(receipt["files"]), "bytes": receipt["bytes"],
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
