"""Frozen, source-group-disjoint contract and atomic artifact helpers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from pilots.topology_20260910.protocol import (
    MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, require_slurm,
)

SCOPE = "logit_dynamics_20260919"
SEEDS = (7, 17, 27)
METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "label", "pred")
ROLES = ("head_train", "probe_train", "probe_val", "dev_eval")
SPLIT_SALT = "logit_dynamics_20260919:head_split:v1:"
RECIPE = {
    "L": 12, "K": 5, "hidden_indices": list(range(1, 13)),
    "head": {"epochs": 16, "lr": 0.001, "batch_size": 512, "weight_decay": 0.0},
    "probe": {"epochs": 100, "lr": 0.001, "batch_size": 256, "weight_decay": 0.01},
    "head_selection": "fixed final epoch16; separate linear 768-to-100 heads",
    "probe_selection": "strictly greatest probe_val average_precision; earliest tie",
    "normalization": "probe_train population mean/std only; zero std replaced by1",
    "positive_class": "frozen classifier error", "class_weight": "probe_train negative/positive",
    "representation": "raw post-block CLS hidden_states[1..12], no additional layernorm",
    "numeric_competitors": "per depth topK excluding final classifier predicted class",
    "dynamics_topk": "raw per-depth topK including predicted class when present",
    "ties": "descending logit then ascending class ID",
    "storage": "FP16 CLS; FP32 frozen classifier logits; FP32 computation",
    "head_photographs": 1200, "probe_photographs": 800,
    "validation_photographs": 400, "evaluation_photographs": 800,
    "bootstrap_draws": 2000, "bootstrap_seed": 20260919,
    "bootstrap_cluster": "image_id (source photograph), never source_id corruption type",
    "primary": "G_mean minus LogitDynamics mean within-seed paired AUROC",
    "secondary": ["S_mean", "O", "MSP", "entropy"],
    "paper": "https://arxiv.org/html/2604.10643v1",
    "probe_weight_decay_resolution": "0.01 explicitly fixed; omitted in paper, AdamW default",
    "original_test_access": False,
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise RuntimeError("Artifact checksum changed: " + str(path))


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def frozen_json(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError("Refusing to change frozen input: " + str(path))
    else:
        atomic_json(path, value)


def atomic_torch(path, value):
    require_slurm()
    import torch
    import io
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


def atomic_npz(path, **arrays):
    require_slurm()
    import numpy as np
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


@contextmanager
def lock(directory, name):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ("." + name + ".lock")).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sources():
    repository = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    return {str(path.relative_to(repository)): sha256(path) for path in paths}


def make_roles(rows, old_roles):
    """One source-photograph split, shared across seeds, without outcome stratification."""
    base = old_roles["roles"]["base_train"]["photo_ids"]
    ordered = sorted(base, key=lambda x: (hashlib.sha256((SPLIT_SALT + str(x)).encode()).hexdigest(), x))
    photos = {
        "head_train": set(ordered[:1200]),
        "probe_train": set(ordered[1200:]) | set(old_roles["roles"]["meta"]["photo_ids"]),
        "probe_val": set(old_roles["roles"]["checkpoint"]["photo_ids"]),
        "dev_eval": set(old_roles["roles"]["dev_eval"]["photo_ids"]),
    }
    expected_counts = (1200, 800, 400, 800)
    seen = set()
    result = {}
    for (role, photo_ids), count in zip(photos.items(), expected_counts):
        if len(photo_ids) != count or seen & photo_ids:
            raise RuntimeError("Head/probe/validation/evaluation photograph overlap or count mismatch")
        seen.update(photo_ids)
        selected = [row for row in rows if row["image_id"] in photo_ids]
        counts = {image_id: 0 for image_id in photo_ids}
        for row in selected:
            counts[row["image_id"]] += 1
        if len(selected) != 9 * count or set(counts.values()) != {9}:
            raise RuntimeError("Each photograph must retain all nine immutable views")
        result[role] = {"photo_ids": sorted(photo_ids), "record_ids": [r["record_id"] for r in selected],
                        "records": len(selected), "errors": sum(r["y"] for r in selected),
                        "class_counts": {str(c): sum(r["label"] == c for r in selected) for c in range(100)}}
    if len(seen) != 3200 or len(rows) != 28800:
        raise RuntimeError("Expected exactly the existing development cohort")
    if any(count == 0 for count in result["head_train"]["class_counts"].values()):
        raise RuntimeError("Fixed head-training split lacks a ground-truth class; no reshuffling is allowed")
    for role in ("probe_train", "probe_val"):
        if not 0 < result[role]["errors"] < result[role]["records"]:
            raise RuntimeError("Fixed probe role must contain both error outcomes")
    return {"roles": result, "split_salt": SPLIT_SALT, "original_test_access": False}


def prepare(root, baseline_root):
    require_slurm()
    from pilots.final_comparison_20260916.data import cache_index
    root, baseline_root = Path(root).resolve(), Path(baseline_root).resolve()
    old, cache_manifest, rows = cache_index(baseline_root)
    if root == baseline_root or root.is_relative_to(Path(old["cache"])):
        raise RuntimeError("New results must have their own namespace")
    score_path = baseline_root / "evaluation" / "scores.npz"
    complete_path = baseline_root / "evaluation" / "complete.json"
    complete = read(complete_path)
    if complete.get("complete") is not True:
        raise RuntimeError("Existing baseline comparison must already be complete")
    verify(score_path, complete["files"]["scores.npz"])
    with lock(root, "prepare"):
        roles = make_roles(rows, read(baseline_root / "role_map.json"))
        frozen_json(root / "role_map.json", roles)
        campaign = {
            "schema_version": 1, "scope_id": SCOPE, "root": str(root),
            "baseline_root": str(baseline_root), "baseline_campaign_sha256": sha256(baseline_root / "campaign.json"),
            "baseline_complete_sha256": sha256(complete_path), "baseline_scores_sha256": sha256(score_path),
            "cache": old["cache"], "cache_manifest_sha256": sha256(Path(old["cache"]) / "manifest.json"),
            "cache_index_sha256": cache_manifest["index_sha256"],
            "roles_sha256": sha256(root / "role_map.json"), "seeds": list(SEEDS), "recipe": RECIPE,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "processor_id": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
            "sources": sources(), "original_test_access": False,
        }
        frozen_json(root / "campaign.json", campaign)
    return campaign


def campaign(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "campaign.json")
    if (value["scope_id"] != SCOPE or value["root"] != str(root) or value["recipe"] != RECIPE
            or value["seeds"] != list(SEEDS) or value["sources"] != sources()):
        raise RuntimeError("Frozen scientific contract or executed source changed")
    verify(root / "role_map.json", value["roles_sha256"])
    old_root = Path(value["baseline_root"])
    verify(old_root / "campaign.json", value["baseline_campaign_sha256"])
    verify(Path(value["cache"]) / "manifest.json", value["cache_manifest_sha256"])
    return value


def run_dir(root, seed):
    if seed not in SEEDS:
        raise ValueError("Only seeds 7,17,27 are authorized")
    return Path(root) / "runs" / f"seed{seed}"


def freeze(root):
    require_slurm()
    campaign(root)
    files = {}
    for seed in SEEDS:
        directory = run_dir(root, seed)
        for stage in ("heads", "probe"):
            receipt = read(directory / stage / "complete.json")
            if receipt.get("complete") is not True:
                raise RuntimeError("Every seed must complete both training stages before evaluation")
            for name, expected in receipt["files"].items():
                path = directory / stage / name
                verify(path, expected)
                files[str(path.relative_to(root))] = expected
            files[str((directory / stage / "complete.json").relative_to(root))] = sha256(directory / stage / "complete.json")
    value = {"campaign_sha256": sha256(Path(root) / "campaign.json"), "files": files,
             "selection_role": "probe_val", "selection_metric": "average_precision", "complete": True}
    frozen_json(Path(root) / "evaluation_gate.json", value)
    return value


def evaluation_gate(root):
    require_slurm()
    value = read(Path(root) / "evaluation_gate.json")
    if value.get("complete") is not True or value["campaign_sha256"] != sha256(Path(root) / "campaign.json"):
        raise RuntimeError("Evaluation gate campaign changed")
    for relative, expected in value["files"].items():
        verify(Path(root) / relative, expected)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.baseline_root is None:
            parser.error("prepare requires --baseline-root")
        result = prepare(args.root, args.baseline_root)
    else:
        result = freeze(args.root)
    print(json.dumps({"scope_id": SCOPE, "command": args.command, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
