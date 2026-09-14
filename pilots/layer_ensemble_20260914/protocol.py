"""Execution roles for the new study; the September13 feature cache is immutable."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from pilots.layer_screen_20260913.protocol import (
    cohort as cache_cohort, protocol as cache_protocol, digest, file_sha256,
    atomic_json, write_frozen, require_slurm,
)

SCOPE = "layer_ensemble_20260914_core_seed7"
ARMS = ("block2", "block5", "block8", "block11")
SEED = 7
ROLES = ("base_train", "checkpoint", "meta", "dev_eval")
DEADLINES={"base_complete_before":"2026-09-14T23:00:00+03:00",
           "predictions_complete_before":"2026-09-15T04:00:00+03:00"}
TRAINING = {"epochs":20,"minimum_epochs":20,"lr":0.002,"weight_decay":0.0001,
            "dropout":0.15,"batch_size":24,"gnn_layers":2,"width":64,
            "checkpoint_rule":"strictly greatest checkpoint-role AUROC; earliest tie",
            "loss":"BCEWithLogitsLoss","class_weight":"base_train negative/positive"}


def read(path):
    return json.loads(Path(path).read_text())


def deadline_unix(name):
    return dt.datetime.fromisoformat(DEADLINES[name]).timestamp()


def make_roles(group):
    # The cache's photo_ids lists are sorted numerically. Record first occurrence
    # retains the original SHA order and is the authoritative grouping order.
    train = list(dict.fromkeys(r["image_id"] for r in group["records"] if r["split"]=="train"))
    val = list(dict.fromkeys(r["image_id"] for r in group["records"] if r["split"]=="val"))
    if len(train)!=2400 or len(val)!=800 or set(train)&set(val):
        raise RuntimeError("Unexpected immutable development photo partition")
    groups = {"base_train":train[:1600],"checkpoint":train[1600:2000],
              "meta":train[2000:2400],"dev_eval":val}
    roles={}
    for role, photos in groups.items():
        allowed=set(photos)
        rows=[r for r in group["records"] if r["image_id"] in allowed]
        counts=Counter(r["image_id"] for r in rows)
        if len(rows)!=len(photos)*9 or any(counts[p]!=9 for p in photos):
            raise RuntimeError("Every photo must retain all nine immutable views")
        roles[role]={"photo_ids":photos,"record_ids":[r["record_id"] for r in rows],
                     "original_split":"val" if role=="dev_eval" else "train"}
    return {"schema_version":1,"scope_id":SCOPE,"cache_cohort_sha256":digest(group),
            "assignment":"original immutable cohort.records first-photo occurrence; no outcome selection",
            "roles":roles,"original_test_access":False}


def make_execution(cache, roles_path):
    manifest=read(Path(cache)/"manifest.json")
    return {"schema_version":1,"scope_id":SCOPE,"seed":SEED,
            "matrix":[[arm,SEED] for arm in ARMS],"training":TRAINING,"deadlines":DEADLINES,
            "roles_sha256":file_sha256(roles_path),
            "cache_manifest_sha256":file_sha256(Path(cache)/"manifest.json"),
            "cache_protocol_sha256":manifest["protocol_sha256"],
            "cache_cohort_sha256":manifest["cohort_sha256"],
            "training_role":"base_train","selection_role":"checkpoint",
            "head_training_role":"meta","evaluation_role":"dev_eval",
            "heads":{"stack":{"arms":list(ARMS)},"last_only":{"arms":["block11"]}},
            "head_recipe":{"standardize_on":"meta","penalty":"l2","C":1.0,
                           "solver":"lbfgs","max_iter":1000,"tol":0.000001,"class_weight":None},
            "num_workers":0,"precision":"float32","cublas_workspace_config":":4096:8",
            "test_evaluated":False,"future_seeds":"separate explicit authorization required"}


def validate_inputs(cache, execution_path, roles_path):
    cache=Path(cache)
    manifest=read(cache/"manifest.json")
    if manifest.get("complete") is not True or manifest.get("diagnostic_only") is not False or manifest.get("records")!=28800:
        raise RuntimeError("A complete immutable 28800-record cache is required")
    group=read(cache/"cohort.json")
    if digest(group)!=digest(cache_cohort()) or manifest["cohort_sha256"]!=digest(group):
        raise RuntimeError("Cached cohort changed")
    if digest(read(cache/"protocol.json"))!=digest(cache_protocol()) or manifest["protocol_sha256"]!=digest(cache_protocol()):
        raise RuntimeError("Cached feature protocol changed")
    roles=read(roles_path)
    if roles!=make_roles(group): raise RuntimeError("Execution role map changed")
    execution=read(execution_path)
    if execution!=make_execution(cache,roles_path): raise RuntimeError("Execution plan changed")
    return execution,roles


def implementation_identity():
    root=Path(__file__).resolve().parents[2]
    names=["pilots/layer_ensemble_20260914/"+n+".py" for n in ("protocol","data","train","predict")]
    names += ["pilots/layer_screen_20260913/"+n+".py" for n in ("protocol","data","models","train","loader_runtime")]
    names += ["pilots/topology_20260910/"+n+".py" for n in ("protocol","data")]
    names += ["polygraph/training/models.py","polygraph/training/train.py","polygraph/data/graphs.py"]
    return {name:file_sha256(root/name) for name in names}


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out-root",type=Path,required=True)
    args=p.parse_args(); args.out_root.mkdir(parents=True,exist_ok=True)
    role_path=args.out_root/"role_map.json"; execution_path=args.out_root/"execution.json"
    write_frozen(role_path,make_roles(read(args.cache/"cohort.json")))
    write_frozen(execution_path,make_execution(args.cache,role_path))
    validate_inputs(args.cache,execution_path,role_path)
    print(json.dumps({"scope_id":SCOPE,"roles_sha256":file_sha256(role_path),
                      "execution_sha256":file_sha256(execution_path)}),flush=True)


if __name__=="__main__": main()
