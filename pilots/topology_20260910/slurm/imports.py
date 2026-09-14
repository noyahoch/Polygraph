"""Checksum-bound node-local copies of exact installed Python package trees."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

PACKAGES = {"transformers": "5.16.1", "torch-geometric": "2.6.1"}
TOP_LEVEL = {"transformers", "transformers-5.16.1.dist-info",
             "torch_geometric", "torch_geometric-2.6.1.dist-info"}


def prepare(root, scratch, expected_sha256):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated Slurm job required")
    start = time.monotonic()
    root, scratch = Path(root), Path(scratch)
    metadata = json.loads((root / "import_bundle/complete.json").read_text())
    if (metadata.get("status") != "complete" or metadata.get("sha256") != expected_sha256
            or metadata.get("packages") != PACKAGES):
        raise RuntimeError("Import bundle differs from the reviewed runtime identity")
    scratch.mkdir(parents=True, exist_ok=True)
    local_archive = scratch / "imports.tar"
    original = root / "import_bundle/imports.tar"
    # A single sequential transfer replaces thousands of NFS metadata operations.
    with original.open("rb") as source, local_archive.open("xb") as target:
        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    hasher = hashlib.sha256()
    with local_archive.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != expected_sha256:
        raise RuntimeError("Node-local import archive failed checksum verification")
    destination = scratch / "imports"
    destination.mkdir(exist_ok=False)
    with tarfile.open(local_archive) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] not in TOP_LEVEL or not (member.isfile() or member.isdir())):
                raise RuntimeError("Unexpected path or link in the installed-source archive")
        archive.extractall(destination, filter="data")
    environment = os.environ.copy()
    old = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(destination) + (os.pathsep + old if old else "")
    record = {"archive_sha256": expected_sha256, "archive_bytes": local_archive.stat().st_size,
              "packages": PACKAGES, "local_package_root": str(destination),
              "preparation_seconds": time.monotonic() - start,
              "native_libraries": "Original pinned environment paths; unchanged",
              "precedence": "Experiment cwd first for -m; staged package trees precede installed copies"}
    return environment, record
