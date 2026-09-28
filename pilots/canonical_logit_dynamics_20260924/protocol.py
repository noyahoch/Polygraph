"""Immutable canonical input identity, source-photo roles, and completion gates."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

SCOPE = "canonical_logit_dynamics_20260924"
SEEDS = (1, 2, 7)
ROLES = ("head_train", "probe_train", "probe_val", "dev_eval")
METADATA = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "label", "pred")
MODEL_ID = "edumunozsala/vit_base-224-in21k-ft-cifar100"
MODEL_REVISION = "b0c51e4a5e5bda35cc922419a28df93bb87e6efa"
PROCESSOR_ID = "google/vit-base-patch16-224-in21k"
PROCESSOR_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
SPLIT_SALT = "canonical-logit-dynamics-20260924:head-split:v1:"
PREFLIGHT_SALT = "canonical-logit-dynamics-20260924:preflight:v1:"
PARITY = {"predictions": "exact", "labels": "exact", "confidence_atol": 2e-4,
          "confidence_rtol": 0.0, "margin_atol": 2e-4, "margin_rtol": 0.0}
RECIPE = {
    "L": 12, "K": 5,
    "head": {"epochs": 16, "lr": 0.001, "batch_size": 512, "weight_decay": 0.0},
    "probe": {"epochs": 100, "lr": 0.001, "batch_size": 256, "weight_decay": 0.01},
    "representation": "raw post-block CLS hidden_states[1..12]; FP16 storage, FP32 heads",
    "probe_selection": "greatest probe_val average_precision; earliest exact tie",
    "head_selection": "final epoch16", "bootstrap_draws": 2000, "bootstrap_seed": 20260924,
    "primary": "mean within-seed Polygraph minus LogitDynamics AUROC",
    "split_salt": SPLIT_SALT, "head_photo_fraction": [3, 5], "parity": PARITY,
}
INPUTS = {
    "detector_plan": ("runs/research_20260830/combiners/strict/main/detector_train_plan.json", "44e1aef95e78be4d82d5582d6629582e09165b996c3f8b4da653ef5bb9be35be"),
    "combiner_plan": ("runs/research_20260830/combiners/strict/main/combiner_eval_plan.json", "e599e06cf2e7a7cad8652e9fc29feae1eca78479c2830263b2a09424d21d2091"),
    "scan": ("data/graph_dataset/scan_records.jsonl", "a999d53f7a71f79794f882809fdfa6541126ec55c8c347752d520b7de88dfe4f"),
    "polygraph_scores": ("runs/topology_depth_last4_20260905/final/best_scores_main.npz", "186a7b76ad21a4f11fe59f46fbdc9763a3362e157e3be592732d4d0c3e74e737"),
}


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("All numerical processing requires a Slurm allocation")


def initialize(seed):
    require_slurm()
    import numpy as np
    import torch
    if seed not in SEEDS or type(seed) is not int:
        raise ValueError("Only frozen seeds 1,2,7 are allowed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Ordinary FP32 CUDA, with TF32 disabled, is fixed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


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
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def frozen_json(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError("Refusing to replace frozen input: " + str(path))
    else:
        atomic_json(path, value)


def atomic_torch(path, value):
    require_slurm()
    import io
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


def atomic_npz(path, **arrays):
    require_slurm()
    import io
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


@contextmanager
def lock(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ("." + name + ".lock")).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sources():
    repo = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [repo / name for name in (
        "pilots/logit_dynamics_20260919/features.py", "pilots/logit_dynamics_20260919/protocol.py",
        "pilots/logit_dynamics_20260919/test_science.py", "pilots/topology_20260910/protocol.py", "polygraph/config.py",
        "polygraph/records.py", "polygraph/data/sources.py", "polygraph/training/evaluate.py")]
    return {str(p.relative_to(repo)): sha256(p) for p in sorted(paths)}


def runtime():
    require_slurm()
    packages = ("torch", "transformers", "numpy", "scikit-learn", "safetensors", "pandas", "pyarrow")
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in packages}}


def key_tuple(values):
    if len(values) != 3 or not isinstance(values[0], str):
        raise RuntimeError("Invalid canonical view key")
    source, severity, index = values
    if type(severity) is not int or type(index) is not int:
        raise RuntimeError("Noninteger canonical severity/index")
    from polygraph.config import ALL_SOURCES
    if (source not in ALL_SOURCES or source == "clean_train" or not 0 <= index < 10000
            or (source == "clean_test" and severity != 0)
            or (source != "clean_test" and not 1 <= severity <= 5)):
        raise RuntimeError("Unexpected source, severity or index in canonical main plan")
    return source, severity, index


def group_id(key):
    return f"test:{key[2]}"  # clean_train explicitly rejected by key_tuple


def make_roles(rows, pools):
    ordered = sorted({group_id(k) for k in pools["train"]},
                     key=lambda value: (hashlib.sha256((SPLIT_SALT + value).encode()).hexdigest(), value))
    cut = 3 * len(ordered) // 5
    head = set(ordered[:cut])
    role_keys = {
        "head_train": [k for k in pools["train"] if group_id(k) in head],
        "probe_train": [k for k in pools["train"] if group_id(k) not in head],
        "probe_val": pools["base_val"], "dev_eval": pools["test"], "unused_meta": pools["meta_val"],
    }
    lookup = {(r["source"], r["severity"], r["image_id"]): r for r in rows}
    seen, result = set(), {}
    for role, keys in role_keys.items():
        groups = {group_id(k) for k in keys}
        if seen & groups:
            raise RuntimeError("A source photograph straddles LD roles")
        seen |= groups
        chosen = [lookup[k] for k in keys]
        if not chosen:
            raise RuntimeError("Empty frozen role")
        result[role] = {"photo_ids": sorted({k[2] for k in keys}), "group_ids": sorted(groups),
                        "record_ids": [r["record_id"] for r in chosen], "records": len(chosen),
                        "errors": sum(r["y"] for r in chosen),
                        "class_counts": {str(c): sum(r["label"] == c for r in chosen) for c in range(100)}}
    if any(v == 0 for v in result["head_train"]["class_counts"].values()):
        raise RuntimeError("Head allocation lacks a true class; no automatic reshuffle")
    for role in ("probe_train", "probe_val", "dev_eval"):
        if not 0 < result[role]["errors"] < result[role]["records"]:
            raise RuntimeError("A role lacks either target class")
    return {"roles": result, "split_salt": SPLIT_SALT, "group_definition": "RecordKey.group_id", "head_fraction": [3, 5]}


def preflight_ids(rows):
    cells = {}
    for row in rows:
        if row["pool"] == "meta_val":
            continue
        cells.setdefault((row["pool"], row["source"], row["severity"]), []).append(row)
    result = set()
    for selected in cells.values():
        ordered = sorted(selected, key=lambda r: (
            hashlib.sha256((PREFLIGHT_SALT + f'{r["source"]}:{r["severity"]}:{r["image_id"]}').encode()).hexdigest(),
            (r["source"], r["severity"], r["image_id"])))
        result.update(r["record_id"] for r in ordered[:2])
    return sorted(result)


def prepare(root, bundle, protocol_file):
    require_slurm()
    import numpy as np
    from polygraph.config import SOURCE_IDS
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    inputs = {name: {"path": str(bundle / relative), "sha256": wanted} for name, (relative, wanted) in INPUTS.items()}
    for spec in inputs.values():
        verify(spec["path"], spec["sha256"])
    detector, combiner = [read(inputs[name]["path"])["splits"] for name in ("detector_plan", "combiner_plan")]
    pools = {"train": list(map(key_tuple, detector["train"])),
             "base_val": list(map(key_tuple, detector["val"])),
             "meta_val": list(map(key_tuple, combiner["val"])),
             "test": list(map(key_tuple, detector["test"]))}
    if pools["base_val"] != list(map(key_tuple, combiner["train"])) or pools["test"] != list(map(key_tuple, combiner["test"])):
        raise RuntimeError("Strict plan ordered role alignment differs")
    if {k: len(v) for k, v in pools.items()} != {"train": 52000, "base_val": 2991, "meta_val": 3009, "test": 17000}:
        raise RuntimeError("Canonical role counts changed")
    ordered_keys = [k for pool in pools.values() for k in pool]
    if len(set(ordered_keys)) != 75000:
        raise RuntimeError("Canonical view keys repeat across pools")
    if {name: len({group_id(k) for k in keys}) for name, keys in pools.items()} != {
            "train": 6985, "base_val": 498, "meta_val": 499, "test": 1998}:
        raise RuntimeError("Canonical pool source-photo counts differ")
    wanted, found = set(ordered_keys), {}
    scan_count = 0
    with Path(inputs["scan"]["path"]).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            scan_count += 1
            item = json.loads(line)
            key = (item["source"], item["severity"], item["base_index"])
            if key not in wanted:
                continue
            if key in found or item["correct"] != int(item["pred"] == item["label"]):
                raise RuntimeError("Duplicate or inconsistent canonical scan record")
            if not (0 <= item["label"] < 100 and 0 <= item["pred"] < 100
                    and np.isfinite([item["confidence"], item["margin"]]).all()
                    and 0 <= item["confidence"] <= 1 and 0 <= item["margin"] <= 1):
                raise RuntimeError("Invalid scan metadata")
            found[key] = item
    if scan_count != 1010000 or set(found) != wanted:
        raise RuntimeError("Canonical scan row count/coverage mismatch")
    rows = []
    for split_id, (pool, keys) in enumerate(pools.items()):
        for key in keys:
            item = found[key]
            rows.append({"record_id": len(rows), "image_id": key[2], "source": key[0], "source_id": SOURCE_IDS[key[0]],
                         "severity": key[1], "split_id": split_id, "pool": pool,
                         "label": item["label"], "pred": item["pred"], "y": 1 - item["correct"],
                         "confidence": item["confidence"], "margin": item["margin"]})
    roles = make_roles(rows, pools)
    if len(roles["roles"]["head_train"]["photo_ids"]) != 4191 or len(roles["roles"]["probe_train"]["photo_ids"]) != 2794:
        raise RuntimeError("Unexpected canonical source-photo counts")
    test_rows = [r for r in rows if r["pool"] == "test"]
    with np.load(inputs["polygraph_scores"]["path"], allow_pickle=False) as scores:
        for key in ("image_id", "source_id", "severity", "y"):
            if not np.array_equal(scores[key], [r[key] for r in test_rows]):
                raise RuntimeError("Canonical Polygraph test metadata differs: " + key)
        for seed in SEEDS:
            if scores[f"score_seed{seed}"].shape != (17000,) or not np.isfinite(scores[f"score_seed{seed}"]).all():
                raise RuntimeError("Invalid frozen Polygraph scores")
        if str(scores["method_name"].item()) != "A_edge_gated_mean" or str(scores["selected_by"].item()) != "mean OOF meta_val AUROC":
            raise RuntimeError("Unexpected canonical method/selection identity")
    protocol_bytes = Path(protocol_file).read_bytes()
    with lock(root, "prepare"):
        frozen_json(root / "records.json", rows)
        frozen_json(root / "role_map.json", roles)
        frozen_json(root / "preflight_records.json", {"record_ids": preflight_ids(rows), "salt": PREFLIGHT_SALT})
        if (root / "PROTOCOL.md").exists():
            if (root / "PROTOCOL.md").read_bytes() != protocol_bytes:
                raise RuntimeError("Scientific protocol changed")
        else:
            atomic_bytes(root / "PROTOCOL.md", protocol_bytes)
        value = {"schema_version": 1, "scope_id": SCOPE, "root": str(root), "inputs": inputs,
                 "records_sha256": sha256(root / "records.json"), "roles_sha256": sha256(root / "role_map.json"),
                 "preflight_sha256": sha256(root / "preflight_records.json"), "protocol_sha256": sha256(root / "PROTOCOL.md"),
                 "seeds": list(SEEDS), "recipe": RECIPE, "sources": sources(), "runtime": runtime(),
                 "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                 "processor_id": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
                 "extraction_records": 71991, "canonical_test_records": 17000}
        frozen_json(root / "campaign.json", value)
    return value


def campaign(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "campaign.json")
    if (value["root"] != str(root) or value["scope_id"] != SCOPE or value["recipe"] != RECIPE
            or value["seeds"] != list(SEEDS) or value["sources"] != sources() or value["runtime"] != runtime()):
        raise RuntimeError("Frozen campaign/source/environment differs")
    for name, key in (("records.json", "records_sha256"), ("role_map.json", "roles_sha256"),
                      ("preflight_records.json", "preflight_sha256"), ("PROTOCOL.md", "protocol_sha256")):
        verify(root / name, value[key])
    return value


def run_dir(root, seed):
    if seed not in SEEDS:
        raise ValueError("Only seeds 1,2,7 are allowed")
    return Path(root) / "runs" / f"seed{seed}"


def training_gate(root):
    value = campaign(root)
    root = Path(root)
    preflight = read(root / "preflight" / "complete.json")
    manifest = read(root / "cls" / "manifest.json")
    if (preflight.get("complete") is not True or preflight["campaign_sha256"] != sha256(root / "campaign.json")
            or manifest.get("complete") is not True or manifest["mode"] != "full"
            or manifest["campaign_sha256"] != sha256(root / "campaign.json")
            or manifest["records"] != value["extraction_records"]):
        raise RuntimeError("Preflight and complete all-row classifier parity must pass before fitting")
    verify(root / "preflight_cls" / "manifest.json", preflight["capture_manifest_sha256"])


def freeze(root):
    require_slurm()
    campaign(root)
    root = Path(root)
    training_gate(root)
    files = {}
    for seed in SEEDS:
        for stage in ("heads", "probe"):
            directory = run_dir(root, seed) / stage
            receipt = read(directory / "complete.json")
            if (receipt.get("complete") is not True or receipt["epochs"] != RECIPE["head" if stage == "heads" else "probe"]["epochs"]
                    or receipt["identity"]["campaign_sha256"] != sha256(root / "campaign.json")
                    or receipt["identity"]["seed"] != seed or receipt["identity"]["stage"] != stage):
                raise RuntimeError("All seed heads/probes must complete the frozen recipe")
            for name, expected in receipt["files"].items():
                path = directory / name
                verify(path, expected)
                files[str(path.relative_to(root))] = expected
            files[str((directory / "complete.json").relative_to(root))] = sha256(directory / "complete.json")
    value = {"campaign_sha256": sha256(root / "campaign.json"), "files": files, "complete": True,
             "selection_role": "probe_val", "selection_metric": "average_precision"}
    frozen_json(root / "evaluation_gate.json", value)
    return value


def evaluation_gate(root):
    require_slurm()
    root = Path(root)
    value = read(root / "evaluation_gate.json")
    if value.get("complete") is not True or value["campaign_sha256"] != sha256(root / "campaign.json"):
        raise RuntimeError("Evaluation gate identity differs")
    for name, expected in value["files"].items():
        verify(root / name, expected)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.bundle is None or args.protocol is None:
            parser.error("prepare requires --bundle and --protocol")
        prepare(args.root, args.bundle, args.protocol)
    else:
        freeze(args.root)
    print(json.dumps({"command": args.command, "complete": True}))


if __name__ == "__main__":
    main()
