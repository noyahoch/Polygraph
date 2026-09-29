"""One bounded Slurm readiness check; no scientific fit or meta/dev scoring."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import tempfile
import time
import numpy as np
import torch
from pilots.layer_screen_20260913.models import build_model
from pilots.layer_screen_20260913.profile_workers import deterministic_parity
from pilots.layer_screen_20260913.train import (
    _cpu_state,_finite,_forward,_loader,_restore_rng,_rng_state,_seed,BlockShuffleSampler,
)
from .data import RoleDataset, atomic_torch
from .protocol import ARMS, SCOPE, atomic_json, file_sha256, require_slurm, validate_inputs
from .train import initialize, make_config, runtime


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def compare_tree(a,b,path="state",exact=False):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape!=b.shape: raise AssertionError(path+": tensor layout changed")
        if exact or not a.is_floating_point():
            if not torch.equal(a.cpu(),b.cpu()): raise AssertionError(path+": exact state differs")
        elif not torch.allclose(a.cpu(),b.cpu(),atol=1e-5,rtol=1e-5):
            raise AssertionError(path+": original1e-5 state tolerance exceeded")
    elif isinstance(a,dict):
        if set(a)!=set(b): raise AssertionError(path+": keys changed")
        for key in a: compare_tree(a[key],b[key],path+"."+str(key),exact)
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError(path+": length changed")
        for i,(x,y) in enumerate(zip(a,b)): compare_tree(x,y,path+f"[{i}]",exact)
    elif a!=b: raise AssertionError(path+": value changed")


def cpu_checks():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from .combine import head_scores
    from .evaluate import WeightedAUC, sigmoid
    y=np.array([0,1,0,1,1,0],dtype=np.int64)
    score=np.array([0.2,0.2,0.8,0.8,1.0,0.0])
    weight=np.array([2,1,0,3,1,2],dtype=np.int64)
    actual=WeightedAUC(y,score)
    assert np.isclose(actual(),roc_auc_score(y,score),atol=1e-12)
    assert np.isclose(actual(weight),roc_auc_score(y,score,sample_weight=weight),atol=1e-12)
    repeated=np.repeat(np.arange(len(y)),weight)
    assert np.isclose(actual(weight),roc_auc_score(y[repeated],score[repeated]),atol=1e-12)
    assert WeightedAUC(np.zeros(3,dtype=int),np.arange(3))() is None
    rng=np.random.default_rng(20260914)
    features=rng.normal(size=(60,4)); target=np.tile(np.array([0,1]),30)
    for columns in ([0,1,2,3],[3]):
        scaler=StandardScaler(); x=scaler.fit_transform(features[:,columns])
        model=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,tol=1e-6,random_state=7).fit(x,target)
        head={"columns":columns,"scaler":{"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist()},
              "model":{"coef":model.coef_[0].tolist(),"intercept":float(model.intercept_[0])}}
        serialized=json.loads(json.dumps(head))
        assert np.allclose(head_scores(serialized,features),model.decision_function(x),atol=1e-12,rtol=1e-12)
        assert np.allclose(sigmoid(head_scores(serialized,features)),model.predict_proba(x)[:,1],atol=1e-12,rtol=1e-12)
    groups=np.repeat(np.array([11,23,47]),9); _,membership=np.unique(groups,return_inverse=True)
    weights=np.array([2,0,1])[membership]
    assert all(len(set(weights[groups==group]))==1 for group in np.unique(groups))
    return {"weighted_auc_ties_and_replication":True,"head_json_scores_and_probabilities":True,
            "source_photo_grouping":True,"inputs":"synthetic only; no meta/dev outcomes"}


def io_probe(spec):
    require_slurm(); torch.set_num_threads(1)
    start=time.monotonic()
    ds=RoleDataset(spec["cache"],spec["execution"],spec["roles"],spec["role"],"block11")
    constructed=time.monotonic()-start
    rows=[]
    for offset in (0,len(ds)-24):
        began=time.monotonic()
        graphs=[ds[i] for i in range(offset,offset+24)]
        ids=[int(g.record_id) for g in graphs]
        assert ids==[row["record_id"] for row in ds.entries[offset:offset+24]]
        rows.append({"first_index":offset,"records":24,"seconds":time.monotonic()-began,
                     "shards":list(dict.fromkeys(row["shard"] for row in ds.entries[offset:offset+24]))})
    return {"role":spec["role"],"construction_seconds":constructed,"batches":rows,
            "total_seconds":time.monotonic()-start,"pid":os.getpid(),
            "scope":"two distinct role boundaries; OS/NFS cache state unknown; not four-job throughput proof"}


def step(model,optimizer,batch,config,device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    scores,y=_forward(model,batch,config["arm"],device)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,y,
                pos_weight=torch.tensor(config["pos_weight"],device=device))
    _finite(loss,"smoke loss"); loss.backward()
    grads={}
    for name,param in model.named_parameters():
        if param.grad is not None:
            _finite(param.grad,"smoke gradient "+name); grads[name]=param.grad.detach().cpu().clone()
    optimizer.step()
    return {"loss":loss.detach().cpu().clone(),"grads":grads,"model":_cpu_state(model)}


def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","execution","roles","out"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--max-seconds",type=int,default=1200)
    args=p.parse_args()
    if not 60<=args.max_seconds<=1500: p.error("Bounded smoke duration required")
    def timeout(signum,frame): raise TimeoutError("Readiness allocation internal deadline reached")
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.max_seconds)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result={"scope_id":SCOPE,"passed":False,"diagnostic_only":True,"arms":{},
            "job_id":os.environ["SLURM_JOB_ID"],"source_sha256":file_sha256(__file__),
            "execution_sha256":file_sha256(args.execution),"roles_sha256":file_sha256(args.roles)}
    start=time.monotonic()
    try:
        execution,roles=validate_inputs(args.cache,args.execution,args.roles)
        seed=execution["seed"]
        result.update(scope_id=execution["scope_id"],seed=seed)
        role_sets=[set(roles["roles"][r]["photo_ids"]) for r in roles["roles"]]
        assert sum(map(len,role_sets))==len(set.union(*role_sets))==3200
        result["cpu_checks"]=cpu_checks(); atomic_json(args.out,result)
        # Spawn workers BEFORE this parent initializes CUDA. Workers only read
        # base/checkpoint tensor roles, and their startup/join time is included.
        specs=[{"cache":args.cache,"execution":args.execution,"roles":args.roles,"role":r}
               for r in ("base_train","checkpoint")]
        io_start=time.monotonic()
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context("spawn")) as pool:
            result["two_loader_probes"]=list(pool.map(io_probe,specs))
        result["two_loader_total_wall_seconds"]=time.monotonic()-io_start
        atomic_json(args.out,result)
        initialize(seed); device=torch.device("cuda")
        result["gpu"]=torch.cuda.get_device_name(); result["torch"]=torch.__version__
        result["ordinary_runtime"]=runtime()
        reference=None
        with tempfile.TemporaryDirectory(prefix="smoke_",dir=args.out.parent) as temporary:
            for arm in ARMS:
                ds=RoleDataset(args.cache,args.execution,args.roles,"base_train",arm)
                config=make_config(args.cache,args.execution,args.roles,ds,arm,seed)
                sampler=BlockShuffleSampler(ds,seed)
                first=list(sampler); state=sampler.state_dict(); other=BlockShuffleSampler(ds,seed); other.load_state_dict(state)
                assert sorted(first)==list(range(len(ds))) and list(sampler)==list(other)
                batch=next(iter(_loader(ds,config,generator=torch.Generator().manual_seed(seed+1000003))))
                signature=(batch.record_id.clone(),batch.y.clone(),batch.x[:,-768:].clone(),batch.output_logits.clone())
                if reference is None: reference=signature
                else: assert all(torch.equal(a,b) for a,b in zip(reference,signature))
                assert batch.x.shape==(24*197,784) and batch.edge_attr.shape[1]==12
                _seed(seed); model=build_model(arm).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
                torch.cuda.reset_peak_memory_stats(); began=time.monotonic()
                # Audit uses deterministic CUDA only, with the original1e-5
                # criteria and all parameter leaves. Scientific mode is restored.
                with deterministic_parity():
                    step(model,optimizer,batch,config,device)
                    checkpoint={"model":_cpu_state(model),"optimizer":cpu_tree(optimizer.state_dict()),
                                "rng":_rng_state(),"sampler":state,"config":config}
                    path=Path(temporary)/(arm+".pt"); atomic_torch(path,checkpoint)
                    expected=step(model,optimizer,batch,config,device)
                    restored=torch.load(path,map_location="cpu",weights_only=False)
                    clone=build_model(arm).to(device); clone.load_state_dict(restored["model"],strict=True)
                    clone_optimizer=torch.optim.AdamW(clone.parameters(),lr=.002,weight_decay=.0001)
                    clone_optimizer.load_state_dict(restored["optimizer"])
                    compare_tree(restored["model"],_cpu_state(clone),exact=True)
                    compare_tree(restored["optimizer"],cpu_tree(clone_optimizer.state_dict()),exact=True)
                    _restore_rng(restored["rng"])
                    actual=step(clone,clone_optimizer,batch,config,device)
                    compare_tree(expected,actual)
                torch.cuda.synchronize()
                result["arms"][arm]={"node_dim":784,"edge_dim":12,"finite_gradient":True,
                    "immediate_state_exact":True,"deterministic_resume_original_tolerance_passed":True,
                    "record_and_hidden_parity":True,"sampler_epoch_resume":True,
                    "audit_seconds":time.monotonic()-began,"parameters":sum(p.numel() for p in model.parameters()),
                    "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "scope":"small fixed base-role batch; timing includes initialization and is not an epoch projection"}
                atomic_json(args.out,result)
                print(json.dumps({"event":"smoke_arm_complete","arm":arm,**result["arms"][arm]}),flush=True)
                del model,optimizer,clone,clone_optimizer,batch,expected,actual,restored,checkpoint,ds
                gc.collect(); torch.cuda.empty_cache()
        assert runtime()==result["ordinary_runtime"]
        result.update(passed=True,elapsed_seconds=time.monotonic()-start,completed_unix=time.time(),
                      scientific_fits_created=False,meta_or_dev_predictions=False)
        atomic_json(args.out,result)
    except BaseException as error:
        result.update(error=repr(error),elapsed_seconds=time.monotonic()-start)
        atomic_json(args.out,result); raise
    finally: signal.alarm(0)


if __name__=="__main__": main()
