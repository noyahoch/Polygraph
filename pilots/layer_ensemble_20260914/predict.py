"""Freeze all four complete bases, then export role-isolated aligned raw logits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch_geometric.data import Batch
from pilots.topology_20260910.data import GraphData
from pilots.layer_screen_20260913.protocol import ARMS as CACHE_ARMS
from pilots.layer_screen_20260913.train import _atomic_npz, _finite, _forward
from .data import RoleDataset, materialize
from .protocol import (ARMS, SEED, SCOPE, atomic_json, file_sha256, read,
                       require_slurm, validate_inputs, write_frozen, deadline_unix)
from .train import initialize, load_run, runtime, verify_complete


def check_prediction_deadline():
    if time.time()>=deadline_unix("predictions_complete_before"):
        raise TimeoutError("Prediction cutoff reached; no incomplete role export is eligible")


def check_execution_deadline(execution):
    if execution["scope_id"] == SCOPE:
        check_prediction_deadline()
    elif time.time() >= deadline_unix("predictions_complete_before",execution):
        raise TimeoutError("Replication prediction cutoff reached; no late export is eligible")


def create_base_freeze(path,run_root,cache,execution_path,roles_path):
    execution,_=validate_inputs(cache,execution_path,roles_path)
    seed=execution["seed"]
    if execution["scope_id"] != SCOPE:
        directory=Path(execution["replication_root"])/f"seed{seed}"
        if Path(path).resolve()!=directory/"base_freeze.json" or Path(run_root).resolve()!=directory/"runs":
            raise RuntimeError("Replication base freeze requires its canonical per-seed paths")
    runs={}
    for arm in ARMS:
        directory=Path(run_root)/arm/f"seed{seed}"
        complete,config=verify_complete(directory,cache,execution_path,roles_path)
        if complete["arm"]!=arm or config["seed"]!=seed: raise RuntimeError("Base freeze identity mismatch")
        runs[arm]={"path":str(directory.resolve()),"completed_epochs":20,"selection_role":"checkpoint",
                   "config_sha256":file_sha256(directory/"config.json"),
                   "best_sha256":file_sha256(directory/"best.safetensors"),
                   "complete_sha256":file_sha256(directory/"complete.json")}
    freeze={"schema_version":1,"scope_id":execution["scope_id"],"complete":True,"seed":seed,"arms":list(ARMS),
            "execution_sha256":file_sha256(execution_path),"roles_sha256":file_sha256(roles_path),
            "cache_manifest_sha256":execution["cache_manifest_sha256"],"runs":runs}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    write_frozen(path,freeze)
    return freeze


def validate_base_freeze(path,run_root,cache,execution_path,roles_path):
    if not Path(path).exists(): raise RuntimeError("All four base checkpoints must be frozen first")
    # Reconstruct expected bindings; write_frozen refuses every mismatch and
    # preserves the same bytes if the existing manifest is valid.
    return create_base_freeze(path,run_root,cache,execution_path,roles_path)


def validate_heads(path,base_path,execution_path,roles_path,cache):
    path=Path(path); value=read(path)
    execution,_=validate_inputs(cache,execution_path,roles_path)
    expected={"execution_sha256":file_sha256(execution_path),"roles_sha256":file_sha256(roles_path),
              "base_freeze_sha256":file_sha256(base_path),
              "cache_manifest_sha256":file_sha256(Path(cache)/"manifest.json")}
    if (value.get("complete") is not True or value.get("seed")!=execution["seed"]
            or value.get("scope_id")!=execution["scope_id"] or value.get("arms")!=list(ARMS)):
        raise RuntimeError("Both stage1 heads must be frozen before dev_eval access")
    if any(value.get(k)!=v for k,v in expected.items()): raise RuntimeError("Head freeze uses other inputs")
    if set(value["files"])!={"heads/stack.json","heads/last_only.json"}:
        raise RuntimeError("Unexpected frozen head files")
    for name,sha in value["files"].items():
        target=(path.parent/name).resolve()
        if not target.is_relative_to(path.parent.resolve()) or file_sha256(target)!=sha:
            raise RuntimeError("Frozen head artifact changed")
    return value


def predict(args):
    require_slurm()
    execution,_=validate_inputs(args.cache,args.execution,args.roles)
    check_execution_deadline(execution); initialize()
    freeze=validate_base_freeze(args.base_freeze,args.run_root,args.cache,args.execution,args.roles)
    joint_sha=None
    if execution["scope_id"] != SCOPE:
        replica_root=Path(execution["replication_root"])
        directory=replica_root/f"seed{execution['seed']}"
        if args.role not in ("meta","dev_eval") or args.out.resolve()!=directory/"predictions"/(args.role+".npz"):
            raise RuntimeError("Replication exports require their canonical per-seed role paths")
    if args.role=="dev_eval":
        if args.heads_freeze is None: raise RuntimeError("dev_eval requires frozen heads")
        validate_heads(args.heads_freeze,args.base_freeze,args.execution,args.roles,args.cache)
        if execution["scope_id"] != SCOPE:
            from .replication import validate_joint_heads_freeze
            if args.heads_freeze.resolve()!=directory/"heads_freeze.json":
                raise RuntimeError("Replication heads path belongs to a different seed")
            validate_joint_heads_freeze(replica_root)
            joint_sha=file_sha256(replica_root/"joint_heads_freeze.json")
    elif args.role!="meta": raise ValueError("Only meta or dev_eval prediction is supported")
    # Dev role dataset construction happens only AFTER the head freeze gate.
    dataset=RoleDataset(args.cache,args.execution,args.roles,args.role,ARMS[0])
    expected={"schema_version":1,"role":args.role,"arms":list(ARMS),"seed":execution["seed"],"rows":len(dataset),
              "execution_sha256":file_sha256(args.execution),"roles_sha256":file_sha256(args.roles),
              "cache_manifest_sha256":file_sha256(args.cache/"manifest.json"),
              "base_freeze_sha256":file_sha256(args.base_freeze)}
    if args.role=="dev_eval": expected["heads_freeze_sha256"]=file_sha256(args.heads_freeze)
    if execution["scope_id"] != SCOPE:
        expected["scope_id"]=execution["scope_id"]
        expected["replication_manifest_sha256"]=execution["replication_manifest_sha256"]
        if joint_sha is not None: expected["joint_heads_freeze_sha256"]=joint_sha
    sidecar=args.out.with_suffix(".json")
    if sidecar.exists():
        previous=read(sidecar)
        if any(previous.get(k)!=v for k,v in expected.items()) or previous["npz_sha256"]!=file_sha256(args.out):
            raise RuntimeError("Existing role export does not match frozen inputs")
        print(json.dumps({"event":"predictions_already_complete","role":args.role}),flush=True); return
    args.out.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda"); models={}; configs={}
    for arm in ARMS:
        models[arm],configs[arm],_=load_run(freeze["runs"][arm]["path"],args.cache,args.execution,args.roles,device)
    predictions=[]; began=time.monotonic()
    with torch.inference_mode():
        for first in range(0,len(dataset),24):
            check_execution_deadline(execution)
            rows=dataset.entries[first:first+24]
            # Retain payload references across all four arm forwards. Adjacent
            # shards crossing this batch boundary do not force four NFS rereads.
            payloads={name:dataset._load(name) for name in dict.fromkeys(row["shard"] for row in rows)}
            columns=[]
            for arm in ARMS:
                check_execution_deadline(execution)
                graphs=[]
                for row in rows:
                    payload=payloads[row["shard"]]; offset=row["offset"]
                    if int(payload["record_id"][offset])!=row["record_id"]: raise RuntimeError("Prediction record alignment changed")
                    x,edge_index,edge_attr=materialize(payload,offset,CACHE_ARMS[arm]["layers"])
                    graphs.append(GraphData(x=x,edge_index=edge_index,edge_attr=edge_attr,
                                           y=torch.tensor([row["y"]],dtype=torch.float32)))
                batch=Batch.from_data_list(graphs)
                score,_=_forward(models[arm],batch,arm,device)
                columns.append(score.detach().cpu().numpy())
                del batch,graphs,score
            predictions.append(np.stack(columns,axis=1).astype(np.float32))
            if first%480==0:
                print(json.dumps({"event":"role_prediction_progress","role":args.role,
                                  "records":first+len(rows),"total":len(dataset),
                                  "elapsed_seconds":time.monotonic()-began}),flush=True)
    logits=np.concatenate(predictions,axis=0); _finite(logits,"role logits")
    if logits.shape!=(len(dataset),4): raise RuntimeError("Expected one raw score from each of four bases")
    metadata=dataset.metadata()
    arrays={k:np.asarray(v,dtype=np.int64) for k,v in metadata.items()
            if k in ("record_id","image_id","source_id","severity","split_id","y","label","pred")}
    arrays["logits"]=logits
    if any(len(v)!=len(dataset) for v in arrays.values()): raise RuntimeError("Prediction metadata length mismatch")
    _atomic_npz(args.out,arrays)
    check_execution_deadline(execution)
    expected.update(npz_sha256=file_sha256(args.out),elapsed_seconds=time.monotonic()-began,
                    numerical_runtime=runtime(),completed_unix=time.time(),job_id=os.environ["SLURM_JOB_ID"])
    atomic_json(sidecar,expected)
    print(json.dumps({"event":"role_predictions_complete",**expected}),flush=True)


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","execution","roles","run-root","base-freeze"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--freeze-base",action="store_true")
    p.add_argument("--role",choices=("meta","dev_eval"))
    p.add_argument("--out",type=Path); p.add_argument("--heads-freeze",type=Path)
    args=p.parse_args()
    if args.freeze_base:
        create_base_freeze(args.base_freeze,args.run_root,args.cache,args.execution,args.roles)
        print(json.dumps({"event":"bases_frozen","sha256":file_sha256(args.base_freeze)}),flush=True)
        if args.role is None: return
    if args.role is None or args.out is None: p.error("Prediction requires --role and --out")
    try:
        predict(args)
    except BaseException as error:
        atomic_json(args.out.with_suffix(".failure.json"),{"complete":False,"role":args.role,
                    "error":repr(error),"failed_unix":time.time(),"job_id":os.environ["SLURM_JOB_ID"]})
        raise


if __name__=="__main__": main()
