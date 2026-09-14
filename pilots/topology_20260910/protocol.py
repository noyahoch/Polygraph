"""Immutable protocol and image-group assignment; no numerical dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA_VERSION = 1
SOURCE_COMMIT = "9af8890405297c0f37dbaa54f6be29a84cb0c37e"
MODEL_ID = "edumunozsala/vit_base-224-in21k-ft-cifar100"
MODEL_REVISION = "b0c51e4a5e5bda35cc922419a28df93bb87e6efa"
PROCESSOR_ID = "google/vit-base-patch16-224-in21k"
PROCESSOR_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
SEEDS = (1, 2, 7, 17, 27)
SOURCES = ("clean_test", "gaussian_noise", "motion_blur", "fog", "jpeg_compression")
VIEWS = (("clean_test", 0),) + tuple((source, severity) for source in SOURCES[1:] for severity in (3, 5))
SPLIT_NAMES = ("train", "val", "test")
ARMS = {
    "full_graph": {"architecture": "edge_gated_mean", "width": 64, "full": True, "rewired": False},
    "full_rewired": {"architecture": "edge_gated_mean", "width": 64, "full": True, "rewired": True},
    "full_set": {"architecture": "m5_node_edge_set", "width": 84, "full": True, "rewired": False},
    "full_endpoint": {"architecture": "m5_endpoint_set", "width": 47, "full": True, "rewired": False},
    "raw_graph": {"architecture": "edge_gated_mean", "width": 64, "full": False, "rewired": False},
    "raw_set": {"architecture": "m5_node_edge_set", "width": 93, "full": False, "rewired": False},
    "logit": {"architecture": "logit", "width": 64, "full": False, "rewired": False},
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Numerical processing is permitted only inside a Slurm allocation (SLURM_JOB_ID required).")


def protocol():
    return {
        "schema_version": SCHEMA_VERSION, "name": "topology_20260910", "source_commit": SOURCE_COMMIT,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "processor_id": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
        "attention_implementation": "eager", "block": 11, "hidden_index": 12,
        "tau": 0.02, "top_k": None, "tokens": 197, "heads": 12,
        "base_node_dim": 16, "full_node_dim": 784, "raw_edge_dim": 12, "full_edge_dim": 36,
        "seeds": list(SEEDS), "arms": ARMS, "views": [list(v) for v in VIEWS],
        "cohort_order": "sha256(UTF8('polygraph-20260911:' + decimal(base_index))), digest ascending, index tie break",
        "photo_counts": {"train": 2400, "val": 800, "test": 800},
        "training": {"epochs": 60, "patience": 8, "min_delta": 0.0, "lr": 0.002,
                     "weight_decay": 0.0001, "dropout": 0.15, "batch_size": 24,
                     "logit_batch_size": 256, "gnn_layers": 2, "minimum_epochs": 0,
                     "loss": "BCEWithLogitsLoss", "class_weight": "train_negative/train_positive"},
        "rewire": {"algorithm": "directed_double_edge_swap", "accepted_swaps_per_edge": 2,
                   "max_attempts_per_edge": 20, "minimum_changed_fraction": 0.80,
                   "allow_self_loops": False, "allow_duplicate_edges": False, "seed": 20260911},
        "data": {"archive": "https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content",
                 "archive_bytes": 2918473216, "archive_md5": "11f0ed0f1191edbf9fa23466ae6021d3",
                 "clean_test_md5": "f0ef6b0ae62326f3e7ffdfab6717acfc",
                 "prevalence": "natural; no outcome-dependent subsampling"},
        "test_policy": "No test scoring before explicit frozen manifest; existing benchmark, not a new benchmark.",
    }


def cohort():
    photos = sorted(range(10000), key=lambda i: (hashlib.sha256(f"polygraph-20260911:{i}".encode()).hexdigest(), i))[:4000]
    records = []
    for rank, photo in enumerate(photos):
        split = "train" if rank < 2400 else "val" if rank < 3200 else "test"
        for source, severity in VIEWS:
            records.append({"record_id": len(records), "image_id": photo, "source": source,
                            "source_id": SOURCES.index(source), "severity": severity, "split": split,
                            "split_id": SPLIT_NAMES.index(split), "key": [source, severity, photo]})
    return {"schema_version": SCHEMA_VERSION, "photo_ids": photos, "records": records,
            "record_counts": {"train": 21600, "val": 7200, "test": 7200}}


def write_frozen(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Refusing to overwrite a different frozen input: {path}")
    else:
        atomic_json(path, value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    write_frozen(args.out_dir / "protocol.json", protocol())
    write_frozen(args.out_dir / "cohort.json", cohort())
    print(json.dumps({"protocol_sha256": digest(protocol()), "cohort_sha256": digest(cohort())}))


if __name__ == "__main__":
    main()
