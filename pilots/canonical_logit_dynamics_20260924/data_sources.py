"""Fetch only canonical required parquet shards, binding Hub revisions and bytes."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download
from polygraph.config import CIFAR100_REPO, CIFAR100C_REPO
from polygraph.data.sources import clean_path, corruption_path
from .protocol import INPUTS, atomic_json, frozen_json, key_tuple, lock, read, require_slurm, sha256, verify


def stage(bundle, data_root):
    require_slurm()
    bundle, data_root = Path(bundle).resolve(), Path(data_root).resolve()
    relative, wanted = INPUTS["detector_plan"]
    verify(bundle / relative, wanted)
    plan = read(bundle / relative)["splits"]
    keys = [key_tuple(k) for role in ("train", "val", "test") for k in plan[role]]
    pairs = sorted({(source, severity) for source, severity, _index in keys})
    with lock(data_root, "source_download"):
        pin_path = data_root / "dataset_revisions.json"
        if pin_path.exists():
            revisions = read(pin_path)
        else:
            api = HfApi()
            revisions = {repo: api.dataset_info(repo).sha for repo in (CIFAR100_REPO, CIFAR100C_REPO)}
            if any(not isinstance(revision, str) or len(revision) != 40 for revision in revisions.values()):
                raise RuntimeError("Could not resolve immutable source revisions")
            frozen_json(pin_path, revisions)
        existing = read(data_root / "dataset_manifest.json") if (data_root / "dataset_manifest.json").exists() else None
        if existing and existing["revisions"] != revisions:
            raise RuntimeError("Pinned source dataset revisions changed")
        files = {}
        for source, severity in pairs:
            if source == "clean_test":
                repo, filename = CIFAR100_REPO, "cifar100/test-00000-of-00001.parquet"
                path = clean_path(data_root, source)
            else:
                repo, filename = CIFAR100C_REPO, f"data/{source}/severity_{severity}/data-00000.parquet"
                path = corruption_path(data_root, source, severity)
            relative_path = str(path.relative_to(data_root))
            previous = existing["files"].get(relative_path) if existing else None
            if previous is not None:
                verify(path, previous["sha256"])
                files[relative_path] = previous
                continue
            cached = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revisions[repo]))
            if path.exists():
                if sha256(path) != sha256(cached):
                    raise RuntimeError("Pre-existing parquet differs from pinned public data: " + str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".download")
                shutil.copyfile(cached, tmp)
                tmp.replace(path)
            files[relative_path] = {"repo": repo, "revision": revisions[repo], "filename": filename,
                                    "bytes": path.stat().st_size, "sha256": sha256(path)}
            atomic_json(data_root / "dataset_manifest.json", {"complete": False, "revisions": revisions, "files": files})
            print(f"Pinned source available: {source} severity={severity}", flush=True)
        result = {"complete": True, "revisions": revisions, "files": files,
                  "detector_plan_sha256": wanted, "pairs": [list(pair) for pair in pairs]}
        atomic_json(data_root / "dataset_manifest.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    stage(args.bundle, args.data_root)


if __name__ == "__main__":
    main()
