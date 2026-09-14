"""Allocated-GPU correctness and memory gate; never produces scientific fits."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from torch_geometric.data import Batch
from .data import CachedDataset, materialize
from .models import build_model, parameter_count
from .protocol import ARMS, MATRIX, atomic_json, cohort, digest, protocol, require_slurm
from .train import BlockShuffleSampler, _seed, _rng_state, _restore_rng

def main():
    require_slurm()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=24)
    args=p.parse_args()
    group=cohort()
    assert len(group["records"])==28800 and len(MATRIX)==14
    assert set(group["photo_ids"]["train"]).isdisjoint(group["photo_ids"]["val"])
    assert all(r["split"] in ("train","val") for r in group["records"])
    # Encoded channel=layer5/head3, query7, key9: edge must be source9 -> target7.
    synthetic={"sparse_offsets":torch.tensor([0,1]),
        "sparse_positions":torch.tensor([((5*12+3)*197+7)*197+9],dtype=torch.int32),
        "sparse_values":torch.tensor([.04],dtype=torch.float16),
        "diagonals":torch.zeros(1,12,197,12,dtype=torch.float16),
        "hidden":torch.ones(1,197,768,dtype=torch.float16)}
    x,e,a=materialize(synthetic,0,[2,5,8,11])
    assert e.tolist()==[[9],[7]] and a.shape==(1,48) and a[0,15]>0 and int((a!=0).sum())==1
    assert torch.equal(x[:,-768:],torch.ones(197,768)) and torch.all(x[:,3]==0) and x[0,2]==1
    assert materialize(synthetic,0,[2])[1].numel()==0
    results={"passed":False,"diagnostic_only":True,"batch_size":args.batch_size,
             "protocol_sha256":digest(protocol()),"arms":{},"torch":torch.__version__,
             "gpu":torch.cuda.get_device_name(),"total_gpu_bytes":torch.cuda.get_device_properties(0).total_memory}
    reference=None
    try:
        for arm in ARMS:
            cold_started=time.monotonic()
            ds=CachedDataset(args.cache,"train",arm,diagnostic=True)
            if len(ds)<args.batch_size: raise RuntimeError("Too few diagnostic records")
            # Same record IDs/labels/full hidden/logits are tested across all arms.
            graph0=ds[0]
            if "cold_shard" not in results:
                results["cold_shard"]={"seconds":time.monotonic()-cold_started,
                    "bytes":(args.cache/"shards"/ds.entries[0]["shard"]).stat().st_size,
                    "scope":"First dataset construction, metadata/checksum reads, first shard load and first graph materialization; conservative I/O-inclusive estimate, OS cache state unknown"}
            signature=(graph0.record_id.clone(),graph0.y.clone(),graph0.x[:,-768:].clone(),graph0.output_logits.clone())
            if reference is None: reference=signature
            else:
                assert all(torch.equal(a,b) for a,b in zip(reference,signature))
            sampler=BlockShuffleSampler(ds,7)
            order=list(sampler); state=sampler.state_dict()
            other=BlockShuffleSampler(ds,7); other.load_state_dict(state)
            assert sorted(order)==list(range(len(ds))) and list(sampler)==list(other)
            _seed(7)
            cpu_rng=_rng_state(); expected=torch.rand(4); _restore_rng(cpu_rng); assert torch.equal(expected,torch.rand(4))
            materialization_started=time.monotonic()
            graphs=[ds[i] for i in range(args.batch_size)]
            batch=Batch.from_data_list(graphs).to("cuda")
            torch.cuda.synchronize()
            materialization_seconds=time.monotonic()-materialization_started
            model=build_model(arm).to("cuda")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); started=time.monotonic()
            score,_=model(batch)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(score,batch.y)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            torch.cuda.synchronize()
            seconds=time.monotonic()-started
            model.eval()
            with torch.no_grad(): reference_scores=model(batch)[0].detach()
            state={"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict()}
            from .data import atomic_torch
            tmp=args.out.with_name("preflight_disposable_state.pt")
            tmp.parent.mkdir(parents=True,exist_ok=True)
            atomic_torch(tmp,state)
            restored=torch.load(tmp,map_location="cpu",weights_only=False)
            model.load_state_dict(restored["model"],strict=True); optimizer.load_state_dict(restored["optimizer"])
            with torch.no_grad(): assert torch.allclose(reference_scores,model(batch)[0],atol=1e-5,rtol=1e-5)
            tmp.unlink()
            results["arms"][arm]={"parameters":parameter_count(arm),"node_dim":batch.x.shape[1],
                "edge_dim":batch.edge_attr.shape[1],"edges":batch.edge_index.shape[1],
                "materialization_and_transfer_seconds":materialization_seconds,
                "one_training_step_seconds":seconds,"max_allocated_bytes":torch.cuda.max_memory_allocated(),
                "max_reserved_bytes":torch.cuda.max_memory_reserved(),"finite_gradient":True,"restore_passed":True}
            print(json.dumps({"event":"arm_preflight", "arm":arm, **results["arms"][arm]}),flush=True)
            del batch,model,optimizer,graphs,state,restored,reference_scores
            gc.collect(); torch.cuda.empty_cache()
        results["passed"]=True
    except BaseException as error:
        results["error"]=repr(error)
        atomic_json(args.out,results)
        raise
    atomic_json(args.out,results)
    print(json.dumps(results),flush=True)

if __name__=="__main__": main()
