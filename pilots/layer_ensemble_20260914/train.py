"""Fresh fixed-20-epoch detectors with checkpoint-only selection and audits."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from safetensors.torch import load_file
from pilots.layer_screen_20260913.models import build_model
from pilots.layer_screen_20260913.loader_runtime import settings
from pilots.layer_screen_20260913.train import (
    _seed, _rng_state, _restore_rng, _cpu_state, _atomic_safetensors, _atomic_npz,
    _loader, _forward, _finite, _check_labels, _run_lock, _numerical_runtime,
    BlockShuffleSampler, compare_scores, auroc,
)
from .data import RoleDataset, atomic_torch
from .protocol import (ARMS, SEED, SCOPE, TRAINING, atomic_json, digest, file_sha256,
                       implementation_identity, read, require_slurm, validate_inputs, deadline_unix)

STOP=False
ARTIFACTS=("config.json","best.safetensors","latest.pt","history.json","checkpoint.npz","checkpoint.json")


def stop_handler(signum, frame):
    global STOP
    STOP=True


def check_base_deadline():
    if time.time()>=deadline_unix("base_complete_before"):
        raise TimeoutError("Base completion cutoff reached; no partial fit comparison")


def initialize(seed=SEED):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise RuntimeError("Pinned CUBLAS_WORKSPACE_CONFIG required before Python")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Scientific throughput uses ordinary CUDA; deterministic mode is audit-only")
    _seed(seed)
    torch.set_float32_matmul_precision("highest")


def runtime():
    return {**_numerical_runtime(),"matmul_precision":torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark":torch.backends.cudnn.benchmark}


def make_config(cache,execution_path,roles_path,dataset,arm,seed):
    execution,_=validate_inputs(cache,execution_path,roles_path)
    if dataset.role!="base_train" or arm not in ARMS or seed!=SEED:
        raise RuntimeError("Only fresh authorized base-role stage1 fits are allowed")
    labels=dataset.labels(); positive=int((labels==1).sum()); negative=int((labels==0).sum())
    if not positive or not negative or positive+negative!=len(labels):
        raise RuntimeError("Base training requires both binary outcomes; no resampling allowed")
    implementation=implementation_identity()
    return {"schema_version":1,"scope_id":SCOPE,"arm":arm,"seed":seed,
            "training":TRAINING,"batch_size":24,"pos_weight":negative/positive,
            "preprocessing":{"kind":"none"},"selection_role":"checkpoint",
            "training_role":"base_train","fresh_initialization":True,
            "training_record_count":len(dataset),"training_labels_sha256":digest(labels.astype(int).tolist()),
            "execution_sha256":file_sha256(execution_path),"roles_sha256":file_sha256(roles_path),
            "cache_manifest_sha256":execution["cache_manifest_sha256"],
            "cache_protocol_sha256":execution["cache_protocol_sha256"],
            "cache_cohort_sha256":execution["cache_cohort_sha256"],
            "implementation":implementation,"implementation_sha256":digest(implementation),
            "loader":settings(0),"numerical_runtime":runtime(),"dtype":"float32","device_type":"cuda",
            "versions":{n:importlib.metadata.version(n) for n in ("torch","numpy","torch-geometric","safetensors")}}


@torch.no_grad()
def collect_checkpoint(model,dataset,config,device,indices=None,batch_size=None):
    if dataset.role!="checkpoint": raise RuntimeError("Training evaluation is checkpoint-role only")
    model.eval(); values=[]
    for batch in _loader(dataset,config,indices=indices,batch_size=batch_size):
        check_base_deadline()
        values.append(_forward(model,batch,config["arm"],device)[0].detach().cpu().numpy())
    result=np.concatenate(values).astype(np.float32)
    _finite(result,"checkpoint scores")
    return result


def verify_complete(directory,cache,execution_path,roles_path):
    directory=Path(directory)
    execution,_=validate_inputs(cache,execution_path,roles_path)
    complete=read(directory/"complete.json"); config=read(directory/"config.json")
    if complete.get("complete") is not True or complete.get("completed_epochs")!=20 or complete.get("selection_role")!="checkpoint":
        raise RuntimeError("Only complete fixed20 checkpoint-selected fits are eligible")
    if not 0<complete.get("completed_unix",0)<deadline_unix("base_complete_before"):
        raise RuntimeError("Base fit did not complete before the frozen cutoff")
    if config.get("scope_id")!=SCOPE or config.get("arm") not in ARMS or config.get("seed")!=SEED:
        raise RuntimeError("Foreign fit identity")
    for key,value in (("execution_sha256",file_sha256(execution_path)),("roles_sha256",file_sha256(roles_path)),
                      ("cache_manifest_sha256",execution["cache_manifest_sha256"]),
                      ("implementation_sha256",digest(implementation_identity()))):
        if config.get(key)!=value: raise RuntimeError("Completed fit binding changed: "+key)
    if config["implementation"]!=implementation_identity() or config["training"]!=TRAINING or config["fresh_initialization"] is not True:
        raise RuntimeError("Completed fit source/recipe changed")
    if complete.get("arm")!=config["arm"] or complete.get("seed")!=config["seed"]:
        raise RuntimeError("Completion identity changed")
    for name in ARTIFACTS:
        if complete["artifacts"].get(name)!=file_sha256(directory/name):
            raise RuntimeError("Completed artifact changed: "+name)
    history=read(directory/"history.json")
    if [r["epoch"] for r in history]!=list(range(1,21)):
        raise RuntimeError("Complete history must contain exactly epochs1..20")
    return complete,config


def load_run(directory,cache,execution_path,roles_path,device):
    complete,config=verify_complete(directory,cache,execution_path,roles_path)
    model=build_model(config["arm"]).to(device)
    model.load_state_dict(load_file(str(Path(directory)/"best.safetensors"),device="cpu"),strict=True)
    if any(p.dtype!=torch.float32 for p in model.parameters()): raise RuntimeError("Expected FP32 parameters")
    model.eval().requires_grad_(False)
    return model,config,complete


def fit(args):
    require_slurm(); check_base_deadline(); initialize(args.seed)
    device=torch.device("cuda")
    train=RoleDataset(args.cache,args.execution,args.roles,"base_train",args.arm)
    checkpoint=RoleDataset(args.cache,args.execution,args.roles,"checkpoint",args.arm)
    _check_labels(train); _check_labels(checkpoint)
    config=make_config(args.cache,args.execution,args.roles,train,args.arm,args.seed)
    directory=args.run_root/args.arm/f"seed{args.seed}"
    directory.mkdir(parents=True,exist_ok=True)
    with _run_lock(directory):
        path=directory/"config.json"; latest=directory/"latest.pt"
        if path.exists():
            if read(path)!=config: raise RuntimeError("Resume config/roles/source/runtime changed")
        else:
            if any((directory/n).exists() for n in ("latest.pt","best.safetensors","complete.json")):
                raise RuntimeError("Refusing unrecognized or old-split weights")
            atomic_json(path,config)
        if (directory/"complete.json").exists():
            verify_complete(directory,args.cache,args.execution,args.roles); return True
        _seed(args.seed)
        model=build_model(args.arm).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"],device=device))
        sampler=BlockShuffleSampler(train,args.seed)
        loader_rng=torch.Generator().manual_seed(args.seed+1000003)
        audit_ids=list(range(min(48,len(checkpoint))))
        history=[]; best_state=None; best_scores=None; best_auc=-float("inf"); best_epoch=0; start_epoch=0
        attempts=read(directory/"attempts.json") if (directory/"attempts.json").exists() else []
        attempt={"started_unix":time.time(),"slurm_job_id":os.environ["SLURM_JOB_ID"],
                 "resume":latest.exists(),"status":"running","selection_role":"checkpoint"}
        attempts.append(attempt); atomic_json(directory/"attempts.json",attempts)
        if latest.exists():
            state=torch.load(latest,map_location="cpu",weights_only=False)
            if state.get("scope_id")!=SCOPE or state.get("config_sha256")!=file_sha256(path):
                raise RuntimeError("Foreign resume state")
            model.load_state_dict(state["model"],strict=True); optimizer.load_state_dict(state["optimizer"])
            sampler.load_state_dict(state["sampler"]); loader_rng.set_state(state["loader_rng"])
            history=state["history"]; start_epoch=state["completed_epochs"]
            if not 0<=start_epoch<=20 or len(history)!=start_epoch or sampler.epoch!=start_epoch:
                raise RuntimeError("Resume completed epoch/sampler/history disagree")
            best_state,best_scores,best_auc,best_epoch=(state[k] for k in ("best_state","best_scores","best_auc","best_epoch"))
            if state["audit"]["indices"]!=audit_ids: raise RuntimeError("Resume audit membership changed")
            restored=collect_checkpoint(model,checkpoint,config,device,indices=audit_ids)
            audit=compare_scores(state["audit"]["scores"],restored)
            audit["alternative_partition"]=compare_scores(restored,collect_checkpoint(model,checkpoint,config,device,indices=audit_ids,batch_size=12))
            atomic_json(directory/"resume_audit.json",audit)
            _restore_rng(state["rng"])
            _atomic_safetensors(directory/"best.safetensors",best_state)
            atomic_json(directory/"history.json",history)
        loader=_loader(train,config,sampler=sampler,generator=loader_rng)
        began=time.monotonic()
        try:
            for epoch in range(start_epoch,20):
                epoch_start=time.monotonic(); model.train(); total_loss=0.0; count=0
                for batch in loader:
                    check_base_deadline()
                    optimizer.zero_grad(set_to_none=True)
                    score,y=_forward(model,batch,args.arm,device)
                    loss=criterion(score,y); _finite(loss,"training loss"); loss.backward()
                    for name,p in model.named_parameters():
                        if p.grad is not None: _finite(p.grad,"gradient "+name)
                    optimizer.step(); total_loss+=float(loss.detach())*len(y); count+=len(y)
                if count!=len(train): raise RuntimeError("Training epoch did not visit every base record once")
                scores=collect_checkpoint(model,checkpoint,config,device)
                auc=auroc(checkpoint.labels(),scores)
                if auc>best_auc:
                    best_auc,best_epoch,best_state,best_scores=auc,epoch+1,_cpu_state(model),scores.copy()
                    _atomic_safetensors(directory/"best.safetensors",best_state)
                history.append({"epoch":epoch+1,"training_loss":total_loss/count,
                                "checkpoint_auroc":auc,"best_epoch":best_epoch,
                                "best_checkpoint_auroc":best_auc,"training_checkpoint_seconds":time.monotonic()-epoch_start})
                audit_scores=collect_checkpoint(model,checkpoint,config,device,indices=audit_ids)
                state={"scope_id":SCOPE,"config_sha256":file_sha256(path),
                       "model":_cpu_state(model),"optimizer":optimizer.state_dict(),
                       "completed_epochs":epoch+1,"history":history,"sampler":sampler.state_dict(),
                       "loader_rng":loader_rng.get_state(),"rng":_rng_state(),
                       "best_state":best_state,"best_scores":best_scores,"best_auc":best_auc,"best_epoch":best_epoch,
                       "audit":{"indices":audit_ids,"scores":audit_scores,"role":"checkpoint"}}
                atomic_torch(latest,state)
                atomic_json(directory/"history.json",history)
                print(json.dumps({"event":"epoch_complete","arm":args.arm,**history[-1],
                                  "attempt_wall_seconds":time.monotonic()-began}),flush=True)
                if STOP and epoch+1<20:
                    attempt.update(status="interrupted_at_epoch_boundary",completed_epochs=epoch+1,ended_unix=time.time())
                    atomic_json(directory/"attempts.json",attempts); return False
            if len(history)!=20 or best_state is None: raise RuntimeError("Incomplete fits cannot produce scientific completion")
            _atomic_safetensors(directory/"best.safetensors",best_state)
            model.load_state_dict(load_file(str(directory/"best.safetensors"),device="cpu"),strict=True)
            restored=collect_checkpoint(model,checkpoint,config,device)
            if not np.allclose(best_scores,restored,atol=1e-5,rtol=1e-5):
                raise RuntimeError("Full checkpoint-role score restoration failed")
            reference=restored[audit_ids]
            audit=compare_scores(reference,collect_checkpoint(model,checkpoint,config,device,indices=audit_ids))
            audit["alternative_partition"]=compare_scores(reference,collect_checkpoint(model,checkpoint,config,device,indices=audit_ids,batch_size=12))
            _atomic_npz(directory/"checkpoint.npz",{**checkpoint.metadata(),"score":restored,"logit":restored})
            atomic_json(directory/"checkpoint.json",{"selection_role":"checkpoint","best_epoch":best_epoch,
                "checkpoint_auroc":auroc(checkpoint.labels(),restored),"restoration_audit":audit,
                "threshold_fitted":False,"dev_eval_accessed":False})
            complete={"schema_version":1,"scope_id":SCOPE,"complete":True,"arm":args.arm,"seed":args.seed,
                      "completed_epochs":20,"best_epoch":best_epoch,"selection_role":"checkpoint",
                      "artifacts":{name:file_sha256(directory/name) for name in ARTIFACTS},
                      "test_evaluated":False,"dev_eval_accessed":False,"completed_unix":time.time()}
            check_base_deadline()
            atomic_json(directory/"complete.json",complete)
            verify_complete(directory,args.cache,args.execution,args.roles)
            attempt.update(status="complete",completed_epochs=20,ended_unix=time.time())
            atomic_json(directory/"attempts.json",attempts)
            return True
        except BaseException as error:
            attempt.update(status="failed",ended_unix=time.time(),error=repr(error),traceback=traceback.format_exc())
            atomic_json(directory/"attempts.json",attempts); raise


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","execution","roles","run-root"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--seed",type=int,choices=(7,),default=7)
    p.add_argument("--device",choices=("cuda",),default="cuda")
    p.add_argument("--num-workers",type=int,choices=(0,),default=0)
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGUSR1,stop_handler)
    if not fit(args): raise SystemExit(75)


if __name__=="__main__": main()
