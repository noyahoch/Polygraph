"""Fetch official CIFAR-100-C on an allocated Slurm CPU, without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
EXPECTED_MD5 = "11f0ed0f1191edbf9fa23466ae6021d3"
CLEAN_HASHES = {
    "test": "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864",
    "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9",
}
BASE_HASH = "ef13a61a72fd5f3b6937ece04e6235159b0d5f9abe1ce00f4e173f14a94af24d"


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--existing-inputs", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("This download/verification must run inside a Slurm allocation")
    os.umask(0o077)
    a.root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(json.dumps({"stage": "data_start", "job_id": os.environ["SLURM_JOB_ID"],
                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
    verified = {}
    for name, expected in CLEAN_HASHES.items():
        source = a.existing_inputs / "cifar-100-python" / name
        actual = digest(source)
        if actual != expected:
            raise RuntimeError(f"Existing clean CIFAR hash mismatch: {name}")
        verified[name] = {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}
    base = a.existing_inputs / "base_model" / "model.safetensors"
    actual = digest(base)
    if actual != BASE_HASH:
        raise RuntimeError("Existing base ViT hash mismatch")
    verified["base_model"] = {"path": str(base), "sha256": actual, "bytes": base.stat().st_size}
    clean_link = a.root / "cifar-100-python"
    if clean_link.exists():
        if clean_link.resolve() != (a.existing_inputs / "cifar-100-python").resolve():
            raise RuntimeError("Existing clean dataset destination points elsewhere")
    else:
        clean_link.symlink_to(a.existing_inputs / "cifar-100-python", target_is_directory=True)

    archive = a.root / "CIFAR-100-C.tar"
    partial = a.root / "CIFAR-100-C.tar.part"
    if not archive.exists():
        print(json.dumps({"stage": "download", "url": URL}), flush=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "2",
                        "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
                        "--continue-at", "-", "--output", str(partial), URL], check=True)
        if digest(partial, "md5") != EXPECTED_MD5:
            raise RuntimeError("Official archive MD5 mismatch; partial retained for inspection")
        partial.replace(archive)
    elif digest(archive, "md5") != EXPECTED_MD5:
        raise RuntimeError("Existing archive does not match official MD5")

    print(json.dumps({"stage": "extract", "bytes": archive.stat().st_size}), flush=True)
    files = []
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "CIFAR-100-C":
                raise RuntimeError(f"Unexpected archive path: {member.name}")
            if member.isdir():
                (a.root / path).mkdir(parents=True, exist_ok=True)
                continue
            allowed_readme = path == Path("CIFAR-100-C/README.txt") and member.size <= 65536
            if not member.isfile() or (path.suffix != ".npy" and not allowed_readme):
                raise RuntimeError(f"Unexpected archive member type: {member.name}")
            target = a.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Missing archive member: {member.name}")
            tmp = target.with_suffix(".npy.part")
            sha = hashlib.sha256()
            with tmp.open("wb") as out:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    out.write(block)
                    sha.update(block)
            if tmp.stat().st_size != member.size:
                raise RuntimeError(f"Wrong extracted length: {member.name}")
            tmp.replace(target)
            files.append({"path": str(path), "bytes": member.size, "sha256": sha.hexdigest()})
            print(json.dumps({"stage": "file_ready", "path": str(path)}), flush=True)
    arrays = [row for row in files if row["path"].endswith(".npy")]
    if len(arrays) != 20 or not (a.root / "CIFAR-100-C" / "labels.npy").is_file():
        raise RuntimeError(f"Expected 19 corruption arrays and labels, found {len(arrays)} arrays")
    result = {"status": "complete", "job_id": os.environ["SLURM_JOB_ID"],
              "source_url": URL, "official_md5": EXPECTED_MD5,
              "archive_bytes": archive.stat().st_size, "archive_sha256": digest(archive),
              "verified_existing": verified, "files": files,
              "elapsed_seconds": time.time() - started,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "classifier_inference_performed": False}
    atomic_json(a.root / "download_complete.json", result)
    print(json.dumps({"stage": "complete", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
