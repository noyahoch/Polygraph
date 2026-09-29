"""Frozen September 13 screen; old held-out photos are never enumerated here."""
from pathlib import Path
from pilots.topology_20260910.protocol import (
    MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, SOURCES, VIEWS,
    atomic_json, digest, file_sha256, require_slurm, cohort as old_cohort,
    protocol as old_protocol, write_frozen,
)

SEEDS = (1, 7, 17, 27)
ARMS = {"block2": {"layers": [2]}, "block5": {"layers": [5]},
        "block8": {"layers": [8]}, "block11": {"layers": [11]},
        "union4": {"layers": [2, 5, 8, 11]}, "union12": {"layers": list(range(12))}}
MATRIX = [(arm, seed) for arm in ARMS for seed in (7, 17)] + [("block11", 1), ("block11", 27)]

def protocol():
    old = old_protocol()
    return {"schema_version": 1, "name": "layer_screen_20260913", "development_only": True,
            "source_commit": old["source_commit"], "parent_protocol_sha256": digest(old),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "processor_id": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
            "attention_implementation": "eager", "hidden_index": 12, "tokens": 197, "heads": 12,
            "tau": 0.02, "arms": ARMS, "matrix": [list(row) for row in MATRIX],
            "views": [list(v) for v in VIEWS], "photo_counts": {"train": 2400, "val": 800},
            "record_counts": {"train": 21600, "val": 7200},
            "cohort_order": old["cohort_order"],
            "node_features": "row,col,CLS,constant_zero; selected layer diagonals in layer/head order; same final hidden768",
            "edge_features": "12 channels per selected layer; head attention <=tau zero; source=key,target=query; no self edges",
            "edge_presence": "OR over selected layers and heads of original FP32 attention >tau",
            "storage": "one fp16 final hidden, fp16 diagonals, sparse int32 flattened layer/head/query/key and fp16 values",
            "architecture": "edge_gated_mean", "width": 64, "parameter_matching": False,
            "training": {**old["training"], "minimum_epochs": 20},
            "ensemble": "fixed arithmetic mean of sigmoid validation scores; distinct-layer K4 for seed7/17; same-layer block11 seeds1,7,17,27 descriptive only",
            "test_policy": "No original held-out 800 photos, no fresh test, no test CLI or learned stacker",
            "data": old["data"]}

def cohort():
    old = old_cohort()
    records = [row for row in old["records"] if row["split"] in ("train", "val")]
    if len(records) != 28800 or {r["split"] for r in records} != {"train", "val"}:
        raise RuntimeError("Unexpected development cohort")
    return {"schema_version": 1, "parent_cohort_sha256": digest(old),
            "photo_ids": {split: sorted({r["image_id"] for r in records if r["split"] == split}) for split in ("train", "val")},
            "records": records, "record_counts": {"train": 21600, "val": 7200}}

def implementation_identity():
    root = Path(__file__).resolve().parents[2]
    files = ["pilots/layer_screen_20260913/" + n + ".py" for n in ("protocol", "data", "models", "train", "loader_runtime")]
    files += ["polygraph/training/train.py", "polygraph/training/models.py", "polygraph/data/graphs.py",
              "pilots/topology_20260910/data.py", "pilots/topology_20260910/protocol.py"]
    return {name: file_sha256(root / name) for name in files}
